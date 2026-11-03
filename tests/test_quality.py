import copy
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from elearning_converter.__main__ import run
from elearning_converter.quality import generate_quality, quality_chunks
from elearning_converter.output import OutputError
from elearning_converter.transcribe import Segment


def draft(time='00:00:01'):
    return {'article': {'body': [{'heading': '', 'text': '箱を数える', 'timestamp': time}], 'checks': []},
            'facts': [{'timestamp': time, 'text': '箱を数える'}]}


def review(severity=None):
    return {'issues': [] if severity is None else [
        {'severity': severity, 'timestamp': '00:00:01', 'text': '初期値が欠落'}]}


class QualityTests(unittest.TestCase):
    def invoke(self, responses, segments=None):
        outputs = [x if isinstance(x, Exception) or hasattr(x, 'text') else SimpleNamespace(text=json.dumps(x)) for x in responses]
        generate = Mock(side_effect=outputs)
        factory = Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
        result = generate_quality(segments or [Segment(1, 2, '箱を数える')], 'test-secret', 'test-model', factory, lambda _: None)
        return result, generate

    def test_two_calls_and_independent_source_review(self):
        result, generate = self.invoke([draft(), review()])
        self.assertTrue(result.accepted)
        self.assertEqual((result.calls, result.attempts), (2, 2))
        second = generate.call_args_list[1].kwargs['contents']
        for key in ('source', 'article', 'facts'):
            self.assertIn(key, second)
        self.assertNotIn('test-secret', second)

    def test_repair_bounded_and_minor_allowed(self):
        for severity, accepted in ((None, True), ('minor', True), ('major', False)):
            with self.subTest(severity=severity):
                result, generate = self.invoke([draft(), review('major'), draft(), review(severity)])
                self.assertEqual(result.accepted, accepted)
                self.assertEqual(generate.call_count, 4)
                self.assertIn('修正後', result.markdown('test-secret'))

    def test_invalid_and_truncated_responses_fail_closed(self):
        invalid = [None, {'issues': 'private'}, {'issues': [{'severity': 'private', 'timestamp': '00:00:01', 'text': 'private'}]},
                   SimpleNamespace(text='private'), SimpleNamespace(text=json.dumps(review()), candidates=[SimpleNamespace(finish_reason='MAX_TOKENS')])]
        for value in invalid:
            with self.subTest(value=value):
                result, generate = self.invoke([draft(), value])
                self.assertFalse(result.accepted)
                self.assertEqual(generate.call_count, 2)
                self.assertNotIn('private', result.error)
        bad = copy.deepcopy(draft())
        bad['article']['body'][0]['timestamp'] = '00:00:59'
        result, generate = self.invoke([bad])
        self.assertFalse(result.accepted)
        self.assertEqual(generate.call_count, 1)

    def test_retries_and_later_chunk_failure(self):
        result, _ = self.invoke([TimeoutError('private'), draft(), review()])
        self.assertTrue(result.accepted)
        self.assertEqual((result.calls, result.attempts), (2, 3))
        result, generate = self.invoke([TimeoutError('private')] * 3)
        self.assertFalse(result.accepted)
        self.assertEqual(generate.call_count, 3)
        self.assertNotIn('private', result.markdown('test-secret'))
        chunks = [{'source': '[00:00:01] 箱', 'before': '', 'after': ''}] * 2
        with patch('elearning_converter.quality.quality_chunks', return_value=chunks):
            result, _ = self.invoke([draft(), review(), SimpleNamespace(text='bad')])
        self.assertFalse(result.accepted)
        self.assertEqual(len(result.parts), 1)

    def test_chunk_core_coverage_and_context(self):
        segments = [Segment(i * 10, i * 10 + 1, f'箱{i}' * 10) for i in range(12)]
        chunks = quality_chunks(segments, limit=160)
        core = '\n'.join(c['source'] for c in chunks)
        for i in range(12):
            self.assertEqual(core.count(f'箱{i}' * 10), 1)
        self.assertTrue(chunks[0]['after'])
        self.assertTrue(chunks[-1]['before'])
        self.assertTrue(all(len(c['source']) <= 160 for c in chunks))

    def test_cli_preserves_article_on_failure_and_report_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / '日本語 空白.transcript.md'
            source.write_text('[00:00:01] 箱', encoding='utf-8')
            article = Path(folder) / '日本語 空白.md'
            report = Path(folder) / '日本語 空白.quality.md'
            article.write_text('old', encoding='utf-8')
            failed, _ = self.invoke([draft(), review('major'), draft(), review('major')])
            with patch('elearning_converter.__main__.gemini_api_key', return_value='test-secret'), patch('elearning_converter.__main__.generate_quality', return_value=failed) as generate, redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                argv = [str(source), '--from-transcript', '--quality-check']
                self.assertEqual(run(argv + ['--overwrite']), 1)
                self.assertEqual(article.read_text(encoding='utf-8'), 'old')
                self.assertIn('未合格', report.read_text(encoding='utf-8'))
                article.unlink()
                generate.reset_mock()
                self.assertEqual(run(argv), 2)  # report alone blocks API
                generate.assert_not_called()
                self.assertEqual(run(argv + ['--no-article']), 2)
                good, _ = self.invoke([draft(), review()])
                generate.return_value = good
                self.assertEqual(run(argv + ['--overwrite']), 0)
                self.assertTrue(article.exists())
                self.assertEqual(source.read_text(encoding='utf-8'), '[00:00:01] 箱')

    def test_secret_redaction_in_report(self):
        result, _ = self.invoke([draft(), review()])
        result.error = 'test-secret'
        self.assertNotIn('test-secret', result.markdown('test-secret'))

    def test_mp4_quality_failure_rescues_transcript(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / '架空.mp4'
            source.write_bytes(b'synthetic')
            failed, _ = self.invoke([draft(), review('major'), draft(), review('major')])
            with patch('elearning_converter.__main__.gemini_api_key', return_value='test-secret'), patch('elearning_converter.__main__.extract_audio'), patch('elearning_converter.__main__.transcribe', return_value=[Segment(1, 2, '箱')]), patch('elearning_converter.__main__.generate_quality', return_value=failed), redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                self.assertEqual(run([str(source), '--quality-check']), 1)
            self.assertTrue(source.with_suffix('.quality.md').exists())
            self.assertTrue((Path(folder) / '架空.transcript.failed.md').exists())
            self.assertFalse(source.with_suffix('.md').exists())
            self.assertEqual(source.read_bytes(), b'synthetic')

    def test_report_write_failure_prevents_article_write(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'sample.transcript.md'
            source.write_text('[00:00:01] 箱', encoding='utf-8')
            good, _ = self.invoke([draft(), review()])
            with patch('elearning_converter.__main__.gemini_api_key', return_value='test-secret'), patch('elearning_converter.__main__.generate_quality', return_value=good), patch('elearning_converter.__main__.write_atomic', side_effect=OutputError('失敗')) as write, redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                self.assertEqual(run([str(source), '--from-transcript', '--quality-check']), 1)
                write.assert_called_once()
                self.assertEqual(write.call_args.args[0].name, 'sample.quality.md')
