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
    AUDIT_SCHEMA, LEDGER_SCHEMA, NOTE_SCHEMA, generate_pipeline, source_chunks,
    validate_audit, validate_ledger, validate_note,
)
from elearning_converter.output import OutputError
from elearning_converter.render import article_markdown
from elearning_converter.transcribe import Segment


FIXTURE = Path(__file__).parent / 'fixtures' / 'note_pipeline.json'


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.example = json.loads(FIXTURE.read_text(encoding='utf-8'))
        self.segments = [Segment(s['start'], s['start'], s['text']) for s in self.example['segments']]
        self.ledger = self.example['ledger']
        self.note = self.example['note']
        self.sources = {s['id']: s for s in source_chunks(self.segments)[0]['source']}

    def audit(self, severity=None):
        return {'issues': [] if severity is None else [
            {'severity': severity, 'source_ids': ['S000001P0001'], 'text': '初期値が欠落'}]}

    def factory(self, responses):
        outputs = [r if isinstance(r, Exception) or hasattr(r, 'text') else SimpleNamespace(text=json.dumps(r))
                   for r in responses]
        generate = Mock(side_effect=outputs)
        return Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=generate))), generate

    def invoke(self, responses, segments=None):
        factory, generate = self.factory(responses)
        result = generate_pipeline(self.segments if segments is None else segments, 'test-secret', 'test-model', factory, lambda _: None)
        return result, generate

    def normal(self):
        return [self.ledger, self.audit(), self.note, self.audit()]

    def test_four_independent_stages_and_rendered_evidence(self):
        result, generate = self.invoke(self.normal())
        self.assertTrue(result.accepted, result.error)
        self.assertEqual((result.calls, result.attempts, generate.call_count), (4, 4, 4))
        requests = [call.kwargs['contents'] for call in generate.call_args_list]
        self.assertNotIn('"note":', requests[0])
        self.assertNotIn('"note":', requests[1])
        self.assertIn('"ledger":', requests[1])
        self.assertIn('"note":', requests[3])
        self.assertTrue(all('"source":' in request and 'test-secret' not in request for request in requests))
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
        responses = []
        for chunk in chunks:
            source_id = chunk['source'][0]['id']
            fact = {'id': 'F001', 'kind': 'procedure', 'text': '架空の操作',
                    'source_ids': [source_id], 'status': 'confirmed'}
            note = {'body': [{'heading': '', 'text': '架空の操作', 'fact_ids': ['F001'],
                              'source_ids': [source_id]}], 'checks': []}
            responses += [{'facts': [fact], 'excluded': []}, self.audit(), note, self.audit()]
        with patch('elearning_converter.note_pipeline.source_chunks', return_value=chunks):
            result, generate = self.invoke(responses, segments)
        self.assertTrue(result.accepted, result.error)
        self.assertEqual(generate.call_count, 8)
        self.assertEqual([r['body_mapping'][0]['source_ids'] for r in result.ledgers],
                         [['S000001P0001'], ['S000002P0001']])
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
                         [LEDGER_SCHEMA, AUDIT_SCHEMA, NOTE_SCHEMA, AUDIT_SCHEMA])
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
                    result, generate = self.invoke([self.ledger, self.audit(), note, self.audit('major'), note, self.audit('major')])
                    self.assertFalse(result.accepted)
                    self.assertEqual(generate.call_count, 6)

    def test_ledger_major_stops_before_drafting_minor_continues(self):
        result, generate = self.invoke([self.ledger, self.audit('major')])
        self.assertFalse(result.accepted)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result.parts, [])
        self.assertIn('台帳照合', result.markdown('test-secret'))
        result, _ = self.invoke([self.ledger, self.audit('minor'), self.note, self.audit()])
        self.assertTrue(result.accepted)

    def test_repair_keeps_fixed_ledger_and_checks_coverage_again(self):
        for severity, accepted in ((None, True), ('minor', True), ('major', False)):
            result, generate = self.invoke([self.ledger, self.audit(), self.note, self.audit('major'), self.note, self.audit(severity)])
            self.assertEqual(result.accepted, accepted)
            self.assertEqual(generate.call_count, 6)
            request = generate.call_args_list[4].kwargs['contents']
            data = json.loads(request.split('資料JSON:\n')[1])
            self.assertEqual(data['ledger'], self.ledger)
        bad = copy.deepcopy(self.note)
        bad['body'].pop()
        result, generate = self.invoke([self.ledger, self.audit(), self.note, self.audit('major'), bad])
        self.assertFalse(result.accepted)
        self.assertEqual(generate.call_count, 5)

    def test_failures_safe_and_retry_bounded_at_each_stage(self):
        normal = self.normal()
        failures = [SimpleNamespace(text='private'), {'issues': 'private'},
                    SimpleNamespace(text='{}', candidates=[SimpleNamespace(finish_reason='MAX_TOKENS')])]
        for stage in range(4):
            for failure in failures:
                result, generate = self.invoke(normal[:stage] + [failure])
                self.assertFalse(result.accepted)
                self.assertEqual(generate.call_count, stage + 1)
                self.assertNotIn('private', result.error)
            result, _ = self.invoke(normal[:stage] + [TimeoutError('private')] + normal[stage:])
            self.assertTrue(result.accepted, result.error)
            self.assertEqual((result.calls, result.attempts), (4, 5))
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
        with patch('elearning_converter.note_pipeline.source_chunks', return_value=chunks * 2):
            result, generate = self.invoke(self.normal() + [SimpleNamespace(text='bad')])
        self.assertFalse(result.accepted)
        self.assertEqual(len(result.parts), 1)
        self.assertEqual(generate.call_count, 5)
        ledger = copy.deepcopy(self.ledger)
        allowed = {k: v for k, v in self.sources.items() if k != 'S000001P0001'}
        with self.assertRaises(ArticleError):
            validate_ledger(ledger, allowed)

    def test_report_redacts_entire_ledger_and_mapping(self):
        ledger = copy.deepcopy(self.ledger)
        ledger['facts'][0]['text'] = 'test-secret'
        ledger['excluded'][0]['reason'] = 'test-secret'
        result, _ = self.invoke([ledger, self.audit(), self.note, self.audit()])
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

            def real_pipeline(segments, key, model):
                return generate_pipeline(segments, key, model, factory, lambda _: None)

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
