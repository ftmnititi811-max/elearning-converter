import copy
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from google import genai
from google.genai import types

from elearning_converter.__main__ import run
from elearning_converter.article import ArticleError
from elearning_converter.note_pipeline import (
    AUDIT_SCHEMA, LEDGER_SCHEMA, context_source, generate_pipeline, source_chunks,
    validate_audit, validate_ledger, validate_note,
)
from elearning_converter.output import OutputError
from elearning_converter.render import article_markdown
from elearning_converter.transcribe import Segment
from elearning_converter.note_blocks import STRUCTURED_SCHEMA
from elearning_converter.note_blocks import block_markdown, validate_block
from elearning_converter.note_structure import PLAN_SCHEMA, drafting_jobs, job_data, validate_plan


FIXTURE = Path(__file__).parent / 'fixtures' / 'note_pipeline.json'


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.example = json.loads(FIXTURE.read_text(encoding='utf-8'))
        self.segments = [Segment(s['start'], s['start'], s['text']) for s in self.example['segments']]
        self.ledger = self.example['ledger']
        self.note = self.example['note']
        self.plan = {'sections': [{'heading': '復習ノート', 'fact_ids': [f"C0001.{f['id']}" for f in self.ledger['facts']]}]}
        self.structured = self.example['structured_note']
        self.sources = {s['id']: s for s in source_chunks(self.segments)[0]['source']}

    def audit(self, severity=None):
        return {'issues': [] if severity is None else [
            {'severity': severity, 'source_ids': ['S000001P0001'], 'text': '初期値が欠落'}]}

    @staticmethod
    def block(text, fact_ids, source_ids):
        return {'kind': 'paragraph', 'text': text, 'items': [], 'columns': [], 'rows': [], 'language': '',
                'fact_ids': [f'C0001.{f}' for f in fact_ids], 'source_ids': source_ids}

    def factory(self, responses):
        outputs = [r if isinstance(r, Exception) or hasattr(r, 'text') else SimpleNamespace(text=json.dumps(r))
                   for r in responses]
        generate = Mock(side_effect=outputs)
        return Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=generate))), generate

    def invoke(self, responses, segments=None, **kwargs):
        factory, generate = self.factory(responses)
        result = generate_pipeline(self.segments if segments is None else segments, 'test-secret', 'test-model', factory, lambda _: None, **kwargs)
        return result, generate

    def normal(self):
        return [self.ledger, self.audit(), self.plan, self.structured, self.audit()]

    def test_five_independent_stages_and_rendered_evidence(self):
        result, generate = self.invoke(self.normal())
        self.assertTrue(result.accepted, result.error)
        self.assertEqual((result.calls, result.attempts, generate.call_count), (5, 5, 5))
        requests = [call.kwargs['contents'] for call in generate.call_args_list]
        self.assertNotIn('"note":', requests[0])
        self.assertNotIn('"note":', requests[1])
        self.assertIn('"ledger":', requests[1])
        self.assertIn('"note":', requests[4])
        self.assertTrue(all('test-secret' not in request for request in requests))
        self.assertTrue(all('"source":' in requests[i] for i in (0, 1, 3, 4)))
        markdown = article_markdown(Path('架空.md'), result.parts)
        self.assertIn('| 2回後 | 0 | 2 |', markdown)
        self.assertIn('[00:00:01] [00:00:15] [00:00:25]', markdown)
        self.assertIn('名称は要確認', markdown)
        report = result.markdown('test-secret')
        for expected in self.example['expected']:
            self.assertIn(expected['id'], report)
            self.assertTrue(set(expected['source_ids']) <= set(self.sources))
        self.assertIn('S000007P0001', report)
        self.assertIn('フィラーのみ', report)
        self.assertNotIn(self.segments[-1].text, report)

    def test_source_identity_same_time_long_segments_and_context(self):
        segments = [Segment(1, 1, 'あいうえお' * 100), Segment(1, 1, '別の説明'), Segment(2, 2, '最後')]
        chunks = source_chunks(segments, limit=180)
        sources = [s for chunk in chunks for s in chunk['source']]
        self.assertEqual(len({s['id'] for s in sources}), len(sources))
        self.assertEqual(sources, [s for chunk in source_chunks(segments, limit=180) for s in chunk['source']])
        for index, segment in enumerate(segments, 1):
            pieces = [s for s in sources if s['segment_id'] == f'S{index:06d}']
            self.assertEqual(''.join(s['text'] for s in pieces), segment.text)
            self.assertEqual(pieces[0]['start_offset'], 0)
            self.assertEqual(pieces[-1]['end_offset'], len(segment.text))
        self.assertTrue(chunks[0]['after'])
        self.assertTrue(chunks[-1]['before'])
        for chunk in chunks:
            self.assertLessEqual(sum(len(s['text']) + 64 for s in chunk['source']), 180)
            core = {s['id'] for s in chunk['source']}
            self.assertFalse(core & {s['id'] for s in chunk['before'] + chunk['after']})

    def test_multiple_chunks_keep_global_source_ids_and_local_fact_ids(self):
        segments = [Segment(1, 1, '箱を数える' * 300), Segment(1, 1, '札を並べる' * 300)]
        chunks = source_chunks(segments, limit=1800)
        self.assertEqual(len(chunks), 2)
        responses, notes, qualified_ids = [], [], []
        for index, chunk in enumerate(chunks, 1):
            source_id = chunk['source'][0]['id']
            fact = {'id': 'F001', 'kind': 'procedure', 'text': '架空の操作',
                    'source_ids': [source_id], 'status': 'confirmed'}
            block = self.block('架空の操作', ['F001'], [source_id])
            block['fact_ids'] = [f'C{index:04d}.F001']
            qualified_ids += block['fact_ids']
            notes.append({'blocks': [block], 'checks': []})
            responses += [{'facts': [fact], 'excluded': []}, self.audit()]
        # One chapter can combine facts extracted in different windows.
        combined = {'blocks': [b for note in notes for b in note['blocks']], 'checks': []}
        responses += [{'sections': [{'heading': '操作', 'fact_ids': qualified_ids}]}, combined, self.audit()]
        with patch('elearning_converter.note_workflow.source_chunks', return_value=chunks):
            result, generate = self.invoke(responses, segments)
        self.assertTrue(result.accepted, result.error)
        self.assertEqual(generate.call_count, 7)
        self.assertEqual(result.jobs[0]['source_ids'], ['S000001P0001', 'S000002P0001'])
        # Reference-only context exists but cannot be used as core evidence.
        with self.assertRaises(ArticleError):
            validate_ledger(responses[0], {s['id'] for s in chunks[1]['source']})

    def test_real_sdk_serializes_all_stage_schemas_without_network(self):
        requests = []
        responses = iter(self.normal())

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'content': {'parts': [
                {'text': json.dumps(next(responses))}], 'role': 'model'}, 'finishReason': 'STOP'}]})

        with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
            with genai.Client(api_key='not-a-real-key', http_options=types.HttpOptions(httpx_client=transport)) as client:
                result = generate_pipeline(self.segments, 'not-a-real-key', 'test-model',
                                           client_factory=lambda **kwargs: client)
        self.assertTrue(result.accepted, result.error)
        self.assertEqual([r['generationConfig']['responseJsonSchema'] for r in requests],
                         [LEDGER_SCHEMA, AUDIT_SCHEMA, PLAN_SCHEMA, STRUCTURED_SCHEMA, AUDIT_SCHEMA])
        self.assertNotIn('not-a-real-key', json.dumps(requests))

    def test_ledger_rejects_unprocessed_sources_unknown_duplicate_and_invalid_types(self):
        mutations = [
            lambda v: v['facts'].pop(),
            lambda v: v['facts'].append(copy.deepcopy(v['facts'][0])),
            lambda v: v['facts'][0].update(source_ids=['S999999P0001']),
            lambda v: v['facts'][0].update(source_ids=['S000001P0001'] * 2),
            lambda v: v['facts'][0].update(status=True),
            lambda v: v['facts'][0].update(kind=[]),
            lambda v: v['facts'][0].update(id={}),
            lambda v: v['facts'][0].update(text=' '),
            lambda v: v['excluded'].append(copy.deepcopy(v['excluded'][0])),
            lambda v: v['excluded'][0].update(source_id='S000001P0001'),
        ]
        for mutation in mutations:
            value = copy.deepcopy(self.ledger)
            mutation(value)
            with self.subTest(value=value), self.assertRaises(ArticleError):
                validate_ledger(value, self.sources)
        result, generate = self.invoke([{'facts': [], 'excluded': []}])
        self.assertFalse(result.accepted)
        self.assertEqual(generate.call_count, 1)

    def test_note_rejects_missing_facts_and_false_evidence(self):
        for change in (
            lambda v: v['body'].pop(),
            lambda v: v['body'][0].update(fact_ids=['F999']),
            lambda v: v['body'][0].update(source_ids=['S000001P0001']),
            lambda v: v['body'][0].update(source_ids=['S999999P0001']),
            lambda v: v['body'][0].update(fact_ids=['F001', 'F001']),
            lambda v: v.update(checks=[False]),
        ):
            value = copy.deepcopy(self.note)
            change(value)
            with self.subTest(value=value), self.assertRaises(ArticleError):
                validate_note(value, self.ledger, self.sources)

    def test_degraded_fixtures_distinguish_structure_from_semantics(self):
        for case in self.example['degradations']:
            note = copy.deepcopy(self.note)
            if case['operation'] == 'remove_body':
                note['body'].pop(case['body_index'])
                remaining = {f for b in note['body'] for f in b['fact_ids']}
                self.assertEqual(set(case['expected_missing']), {f['id'] for f in self.ledger['facts']} - remaining)
            else:
                note['body'][case['body_index']]['text'] = case['text']
            with self.subTest(case=case['name']):
                if case['local_reject']:
                    with self.assertRaises(ArticleError):
                        validate_note(note, self.ledger, self.sources)
                else:
                    self.assertEqual(validate_note(note, self.ledger, self.sources), note)
                    # The model's fixed major is a routing test, not detection evidence.
                    structured = {'blocks': [self.block(b['text'], b['fact_ids'], b['source_ids']) for b in note['body']], 'checks': note['checks']}
                    result, generate = self.invoke([self.ledger, self.audit(), self.plan, structured, self.audit('major'), structured, self.audit('major')])
                    self.assertFalse(result.accepted)
                    self.assertEqual(generate.call_count, 7)

    def test_ledger_major_stops_before_drafting_minor_continues(self):
        result, generate = self.invoke([self.ledger, self.audit('major')])
        self.assertFalse(result.accepted)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result.parts, [])
        self.assertIn('台帳照合', result.markdown('test-secret'))
        result, _ = self.invoke([self.ledger, self.audit('minor'), self.plan, self.structured, self.audit()])
        self.assertTrue(result.accepted)

    def test_repair_keeps_fixed_ledger_and_checks_coverage_again(self):
        for severity, accepted in ((None, True), ('minor', True), ('major', False)):
            result, generate = self.invoke([self.ledger, self.audit(), self.plan, self.structured, self.audit('major'), self.structured, self.audit(severity)])
            self.assertEqual(result.accepted, accepted)
            self.assertEqual(generate.call_count, 7)
            request = generate.call_args_list[5].kwargs['contents']
            data = json.loads(request.split('資料JSON:\n')[1])
            self.assertEqual([f['id'] for f in data['ledger']['facts']], self.plan['sections'][0]['fact_ids'])
        bad = copy.deepcopy(self.structured)
        bad['blocks'].pop()
        result, generate = self.invoke([self.ledger, self.audit(), self.plan, self.structured, self.audit('major'), bad])
        self.assertFalse(result.accepted)
        self.assertEqual(generate.call_count, 6)

    def test_failures_safe_and_retry_bounded_at_each_stage(self):
        normal = self.normal()
        failures = [SimpleNamespace(text='private'), {'issues': 'private'},
                    SimpleNamespace(text='{}', candidates=[SimpleNamespace(finish_reason='MAX_TOKENS')])]
        for stage in range(5):
            for failure in failures:
                result, generate = self.invoke(normal[:stage] + [failure])
                self.assertFalse(result.accepted)
                self.assertEqual(generate.call_count, stage + 1)
                self.assertNotIn('private', result.error)
            result, _ = self.invoke(normal[:stage] + [TimeoutError('private')] + normal[stage:])
            self.assertTrue(result.accepted, result.error)
            self.assertEqual((result.calls, result.attempts), (5, 6))
            result, generate = self.invoke(normal[:stage] + [TimeoutError('private')] * 3)
            self.assertFalse(result.accepted)
            self.assertEqual(generate.call_count, stage + 3)
            self.assertNotIn('private', result.error)
        for issue in ({'severity': 'unknown', 'source_ids': ['S000001P0001'], 'text': 'private'},
                      {'severity': 'major', 'source_ids': ['S999999P0001'], 'text': 'private'}):
            with self.assertRaises(ArticleError):
                validate_audit({'issues': [issue]}, self.sources)

    def test_empty_input_and_client_failure_no_calls(self):
        result, generate = self.invoke([], segments=[])
        self.assertFalse(result.accepted)
        generate.assert_not_called()
        factory = Mock(side_effect=RuntimeError('private'))
        result = generate_pipeline(self.segments, 'test-secret', 'test-model', factory)
        self.assertFalse(result.accepted)
        self.assertNotIn('private', result.error)

    def test_later_chunk_failure_and_context_ids_rejected(self):
        chunks = source_chunks(self.segments)
        with patch('elearning_converter.note_workflow.source_chunks', return_value=chunks * 2):
            result, generate = self.invoke([self.ledger, self.audit(), SimpleNamespace(text='bad')])
        self.assertFalse(result.accepted)
        self.assertEqual(len(result.parts), 0)
        self.assertEqual(generate.call_count, 3)
        ledger = copy.deepcopy(self.ledger)
        allowed = {k: v for k, v in self.sources.items() if k != 'S000001P0001'}
        with self.assertRaises(ArticleError):
            validate_ledger(ledger, allowed)

    def test_report_redacts_entire_ledger_and_mapping(self):
        ledger = copy.deepcopy(self.ledger)
        ledger['facts'][0]['text'] = 'test-secret'
        ledger['excluded'][0]['reason'] = 'test-secret'
        result, _ = self.invoke([ledger, self.audit(), self.plan, self.structured, self.audit()])
        result.error = 'test-secret'
        self.assertNotIn('test-secret', result.markdown('test-secret'))
        self.assertIn('[REDACTED]', result.markdown('test-secret'))

    def test_cli_real_pipeline_protects_input_article_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / '日本語 空白.transcript.md'
            source.write_text('\n'.join(f"[00:00:{s['start']:02d}] {s['text']}" if s['start'] < 60 else f"[00:01:00] {s['text']}" for s in self.example['segments']), encoding='utf-8-sig')
            original = source.read_bytes()
            article = Path(folder) / '日本語 空白.md'
            report = Path(folder) / '日本語 空白.quality.md'
            article.write_text('old article', encoding='utf-8')
            factory, generate = self.factory(self.normal())

            def real_pipeline(segments, key, model, **kwargs):
                return generate_pipeline(segments, key, model, factory, lambda _: None, **kwargs)

            with patch('elearning_converter.__main__.gemini_api_key', return_value='test-secret'), patch('elearning_converter.__main__.generate_pipeline', side_effect=real_pipeline), patch('elearning_converter.__main__.extract_audio') as media, patch('elearning_converter.__main__.transcribe') as speech, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                argv = [str(source), '--from-transcript', '--note-pipeline']
                self.assertEqual(run(argv), 2)
                generate.assert_not_called()
                self.assertEqual(run(argv + ['--overwrite']), 0)
                self.assertIn('情報台帳と本文対応', report.read_text(encoding='utf-8'))
                self.assertIn('n=2', article.read_text(encoding='utf-8'))
                saved = article.read_bytes()
                factory2, generate2 = self.factory([self.ledger, self.audit('major')])
                factory.return_value = factory2.return_value
                self.assertEqual(run(argv + ['--overwrite']), 1)
                self.assertEqual(article.read_bytes(), saved)
                self.assertEqual(generate2.call_count, 2)
                article.unlink()
                self.assertEqual(run(argv), 2)  # report alone blocks calls
                self.assertEqual(run(argv + ['--no-article']), 2)
                self.assertEqual(source.read_bytes(), original)
                media.assert_not_called()
                speech.assert_not_called()

    def test_report_failure_and_mp4_failure_never_publish_article(self):
        good, _ = self.invoke(self.normal())
        bad, _ = self.invoke([self.ledger, self.audit('major')])
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / '架空.mp4'
            source.write_bytes(b'synthetic')
            with patch('elearning_converter.__main__.gemini_api_key', return_value='test-secret'), patch('elearning_converter.__main__.extract_audio'), patch('elearning_converter.__main__.transcribe', return_value=self.segments), patch('elearning_converter.__main__.generate_pipeline', return_value=bad), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(run([str(source), '--note-pipeline']), 1)
                self.assertTrue(source.with_suffix('.transcript.failed.md').exists())
                self.assertFalse(source.with_suffix('.md').exists())
                self.assertEqual(source.read_bytes(), b'synthetic')
            transcript = Path(folder) / '別.transcript.md'
            transcript.write_text('[00:00:01] 架空', encoding='utf-8')
            with patch('elearning_converter.__main__.gemini_api_key', return_value='test-secret'), patch('elearning_converter.__main__.generate_pipeline', return_value=good), patch('elearning_converter.__main__.write_atomic', side_effect=OutputError('失敗')) as write, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(run([str(transcript), '--from-transcript', '--note-pipeline']), 1)
                write.assert_called_once()
                self.assertTrue(write.call_args.args[0].name.endswith('.quality.md'))

    def test_cli_rejects_two_quality_modes(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as exc:
            run(['unused.mp4', '--quality-check', '--note-pipeline'])
        self.assertEqual(exc.exception.code, 2)

    def two_sections(self, final_severity=None):
        plan = {'sections': [
            {'heading': '手順', 'fact_ids': self.plan['sections'][0]['fact_ids'][:4]},
            {'heading': '具体例と配置', 'fact_ids': self.plan['sections'][0]['fact_ids'][4:]},
        ]}
        first_ids = set(plan['sections'][0]['fact_ids'])
        first = {'blocks': [b for b in self.structured['blocks'] if set(b['fact_ids']) <= first_ids], 'checks': []}
        second = {'blocks': [b for b in self.structured['blocks'] if not set(b['fact_ids']) <= first_ids], 'checks': self.structured['checks']}
        return [self.ledger, self.audit(), plan, first, self.audit(), second, self.audit(), self.audit(final_severity)]

    def test_sections_seam_review_and_major_refusal(self):
        for severity, accepted in ((None, True), ('minor', True), ('major', False)):
            result, generate = self.invoke(self.two_sections(severity))
            self.assertEqual(result.accepted, accepted, result.error)
            self.assertEqual(generate.call_count, 8)
            request = json.loads(generate.call_args_list[-1].kwargs['contents'].split('資料JSON:\n')[1])
            self.assertEqual(len(request['pair']), 2)
            self.assertIn('source', request)
            self.assertIn('境界照合', result.markdown('test-secret'))
        requests = self.two_sections()
        requests[-1] = {'issues': 'private'}
        result, generate = self.invoke(requests)
        self.assertFalse(result.accepted)
        self.assertNotIn('private', result.error)
        self.assertEqual(generate.call_count, 8)

    def test_plan_rejects_missing_duplicate_unknown_reverse_and_empty_sections(self):
        facts = {f'C0001.{f["id"]}': f for f in self.ledger['facts']}
        for mutate in (
            lambda v: v['sections'][0]['fact_ids'].pop(),
            lambda v: v['sections'][0]['fact_ids'].append('C0001.F001'),
            lambda v: v['sections'][0]['fact_ids'].append('C9999.F001'),
            lambda v: v['sections'][0]['fact_ids'].reverse(),
            lambda v: v['sections'][0].update(heading=' '),
            lambda v: v.update(sections=[]),
        ):
            plan = copy.deepcopy(self.plan)
            mutate(plan)
            with self.subTest(plan=plan), self.assertRaises(ArticleError):
                validate_plan(plan, facts)
            result, generate = self.invoke([self.ledger, self.audit(), plan])
            self.assertFalse(result.accepted)
            self.assertEqual(generate.call_count, 3)

    def test_topic_boundary_prefers_transition_without_losing_source(self):
        segments = [Segment(i, i, ('次に、' if i == 6 else '') + 'あ' * 78) for i in range(9)]
        chunks = source_chunks(segments, limit=1000)
        self.assertEqual(len(chunks[0]['source']), 6)
        self.assertEqual([s['segment_id'] for c in chunks for s in c['source']],
                         [f'S{i:06d}' for i in range(1, 10)])

    def test_reference_context_uses_nearest_text_and_exact_offsets(self):
        source = {'id': 'S000001P0002', 'start_offset': 100, 'end_offset': 1000,
                  'text': '前' * 300 + '後' * 600}
        before, after = context_source(source, before=True), context_source(source)
        self.assertEqual(before['text'], '後' * 600)
        self.assertEqual((before['start_offset'], before['end_offset']), (400, 1000))
        self.assertEqual((after['start_offset'], after['end_offset']), (100, 700))
        self.assertEqual(source['text'][before['start_offset']-100:before['end_offset']-100], before['text'])

    def test_jobs_split_long_chapter_keep_fact_coverage_and_evidence(self):
        sources = {'A': {'id': 'A', 'timestamp': '00:00:01', 'text': 'あ' * 60, 'start_offset': 0, 'end_offset': 60},
                   'B': {'id': 'B', 'timestamp': '00:00:02', 'text': 'い' * 60, 'start_offset': 0, 'end_offset': 60}}
        facts = {'F1': {'id': 'F1', 'source_ids': ['A']}, 'F2': {'id': 'F2', 'source_ids': ['B']}}
        plan = {'sections': [{'heading': '同じ章', 'fact_ids': list(facts)}]}
        jobs = drafting_jobs(plan, facts, sources, limit=200)
        self.assertEqual(len(jobs), 2)
        self.assertEqual([j['fact_ids'] for j in jobs], [['F1'], ['F2']])
        self.assertEqual([j['section_id'] for j in jobs], ['H0001', 'H0001'])
        data = job_data(jobs[0], facts, sources)
        self.assertEqual([s['id'] for s in data['source']], ['A'])
        self.assertEqual([s['id'] for s in data['after']], ['B'])
        with self.assertRaises(ArticleError):
            drafting_jobs(plan, facts, sources, limit=100)

    def test_all_block_kinds_escape_and_reject_malformed_shapes(self):
        base = self.block('内容', ['F001'], ['S000001P0001'])
        paragraph = {**base, 'text': '- [命令](url)\n# 見出し'}
        self.assertTrue(block_markdown(paragraph).startswith('\\- \\['))
        for kind in ('bullets', 'steps'):
            block = {**base, 'kind': kind, 'text': '', 'items': ['初期化', '更新\n停止']}
            rendered = block_markdown(block)
            self.assertIn('更新 停止', rendered)
            self.assertTrue(rendered.startswith('1. ' if kind == 'steps' else '- '))
        table = {**base, 'kind': 'table', 'text': '', 'columns': ['A', 'B'], 'rows': [['a|b', '値\n次'], ['', '未確認']]}
        self.assertIn('a\\|b', block_markdown(table))
        self.assertIn('値 次', block_markdown(table))
        code = {**base, 'kind': 'code', 'text': '```\nx = 1\n```', 'language': 'python'}
        self.assertTrue(block_markdown(code).startswith('````python\n'))
        for value in (
            {**table, 'rows': [['1']]}, {**table, 'rows': [['1', False]]},
            {**base, 'items': ['不要']}, {**base, 'language': 'python'},
            {**code, 'language': 'python\n```'}, {**base, 'kind': []},
            {**base, 'text': False}, {**base, 'extra': 'private'},
        ):
            with self.subTest(value=value), self.assertRaises(ArticleError):
                validate_block(value)

    def test_api_budget_stops_before_next_call_and_does_not_count_retry_as_stage(self):
        for maximum in (1, 2, 3, 4):
            result, generate = self.invoke(self.normal(), max_calls=maximum)
            self.assertFalse(result.accepted)
            self.assertEqual(generate.call_count, maximum)
            self.assertIn('工程上限', result.error)
        result, generate = self.invoke([TimeoutError('private')] + self.normal(), max_calls=5)
        self.assertTrue(result.accepted)
        self.assertEqual((result.calls, result.attempts, generate.call_count), (5, 6, 6))
        for value in (0, -1, True, '5'):
            result, generate = self.invoke([], max_calls=value)
            self.assertFalse(result.accepted)
            generate.assert_not_called()

    def test_usage_progress_and_missing_values_are_reported_without_contents(self):
        responses = [SimpleNamespace(text=json.dumps(value), usage_metadata={
            'prompt_token_count': 10, 'candidates_token_count': 4, 'thoughts_token_count': 2,
            'cached_content_token_count': 0}) for value in self.normal()]
        responses[2].usage_metadata = {'prompt_token_count': -1, 'candidates_token_count': True}
        progress = []
        result, _ = self.invoke([TimeoutError('private')] + responses, progress=progress.append)
        self.assertTrue(result.accepted)
        self.assertEqual(len(progress), 5)
        self.assertTrue(all('工程' in p and 'test-secret' not in p and '入力は' not in p for p in progress))
        self.assertEqual(sum(s['tokens'].get('prompt_token_count', 0) for s in result.stages), 40)
        self.assertEqual(sum('prompt_token_count' not in s['tokens'] for s in result.stages), 2)
        report = result.markdown('test-secret')
        self.assertIn('usage_unknown_attempts_by_field', report)
        self.assertNotIn('private', report)
        self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_plan_size_limit_stops_before_sending_plan(self):
        with patch('elearning_converter.note_structure.PLAN_LIMIT', 1):
            result, generate = self.invoke(self.normal())
        self.assertFalse(result.accepted)
        self.assertEqual(generate.call_count, 2)
        self.assertIn('上限', result.error)

    def test_cli_budget_validation_and_real_budget_failure_keep_article(self):
        with patch('elearning_converter.__main__.gemini_api_key') as key, redirect_stderr(StringIO()):
            for options in (['--max-api-calls', '1'], ['--note-pipeline', '--max-api-calls', '0']):
                self.assertEqual(run(['unused.mp4'] + options), 2)
            key.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / '講義.transcript.md'
            source.write_text('\n'.join(f"[{self.sources[f'S{i:06d}P0001']['timestamp']}] {s.text}"
                                         for i, s in enumerate(self.segments, 1)), encoding='utf-8')
            article = Path(folder) / '講義.md'
            article.write_text('既存', encoding='utf-8')
            factory, generate = self.factory(self.normal())
            def actual(segments, key, model, **kwargs):
                return generate_pipeline(segments, key, model, factory, lambda _: None, **kwargs)
            with patch('elearning_converter.__main__.gemini_api_key', return_value='test-secret'), patch('elearning_converter.__main__.generate_pipeline', side_effect=actual), redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                self.assertEqual(run([str(source), '--from-transcript', '--note-pipeline', '--max-api-calls', '2', '--overwrite']), 1)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(article.read_text(encoding='utf-8'), '既存')
            self.assertIn('工程上限', (Path(folder) / '講義.quality.md').read_text(encoding='utf-8'))

    def test_separate_sections_with_same_title_are_preserved(self):
        responses = self.two_sections()
        for section in responses[2]['sections']:
            section['heading'] = '同じ見出し'
        result, _ = self.invoke(responses)
        self.assertTrue(result.accepted)
        self.assertEqual(article_markdown(Path('架空.md'), result.parts).count('## 同じ見出し'), 2)

    def test_filler_only_transcript_stops_after_audit_without_empty_article(self):
        segments = [Segment(1, 1, 'ええと')]
        ledger = {'facts': [], 'excluded': [{'source_id': 'S000001P0001', 'reason': 'フィラーのみ'}]}
        result, generate = self.invoke([ledger, self.audit()], segments)
        self.assertFalse(result.accepted)
        self.assertIn('本文対象の情報がありません', result.error)
        self.assertEqual(generate.call_count, 2)

    def test_actual_long_input_crosses_windows_then_jobs_then_seams(self):
        segments = [Segment(i * 10, i * 10, '架空の説明' * 600) for i in range(8)]
        requests = []
        def generate(**kwargs):
            data = json.loads(kwargs['contents'].split('資料JSON:\n')[1])
            schema = kwargs['config']['response_json_schema']
            requests.append((schema, data))
            if schema == LEDGER_SCHEMA:
                value = {'facts': [{'id': f'F{i:03d}', 'kind': 'definition', 'text': '架空の説明',
                                    'source_ids': [s['id']], 'status': 'confirmed'}
                                   for i, s in enumerate(data['source'], 1)], 'excluded': []}
            elif schema == PLAN_SCHEMA:
                value = {'sections': [{'heading': '全体をまとめた章', 'fact_ids': [f['id'] for f in data['facts']]}]}
            elif schema == STRUCTURED_SCHEMA:
                blocks = []
                for fact in data['ledger']['facts']:
                    block = self.block('架空の説明', [], fact['source_ids'])
                    block['fact_ids'] = [fact['id']]
                    blocks.append(block)
                value = {'blocks': blocks, 'checks': []}
            else:
                value = {'issues': []}
            return SimpleNamespace(text=json.dumps(value))
        factory = Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
        result = generate_pipeline(segments, 'test-secret', 'test-model', factory, lambda _: None)
        self.assertTrue(result.accepted, result.error)
        windows, jobs = len(result.ledgers), len(result.jobs)
        self.assertGreater(windows, 1)
        self.assertGreater(jobs, 1)
        self.assertEqual(result.calls, 2 * windows + 1 + 2 * jobs + jobs - 1)
        all_sources = [s['id'] for r in result.ledgers for s in r['sources']]
        self.assertEqual(len(all_sources), len(set(all_sources)))
        self.assertEqual(len(all_sources), len(segments))
        all_facts = [f for job in result.jobs for f in job['fact_ids']]
        self.assertEqual(len(all_facts), len(set(all_facts)))
        article = article_markdown(Path('架空.md'), result.parts)
        self.assertEqual(article.count('## 全体をまとめた章'), 1)
        for i in range(8):
            self.assertIn(f'[{i*10//3600:02d}:{i*10//60%60:02d}:{i*10%60:02d}]', article)
        # No draft can happen while further extraction windows remain.
        first_draft = next(i for i, (s, _) in enumerate(requests) if s == STRUCTURED_SCHEMA)
        self.assertEqual(sum(s == LEDGER_SCHEMA for s, _ in requests[:first_draft]), windows)
