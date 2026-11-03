import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from elearning_converter.__main__ import run
from elearning_converter.article import ArticleError
from elearning_converter.config import ConfigurationError
from elearning_converter.transcript_input import read_transcript


class TranscriptInputTests(unittest.TestCase):
    def test_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / '原文.transcript.failed.md'
            for value in ('', '[00:60:00] 箱', '[00:00:00] ', '[00:00:02] 箱\n[00:00:01] 札', '未知の行', '[00:00:01] 箱\n途中の行'):
                source.write_text(value, encoding='utf-8')
                with self.subTest(value=value), self.assertRaises(ConfigurationError):
                    read_transcript(source)
            source.write_bytes(b'\xff')
            with self.assertRaises(ConfigurationError):
                read_transcript(source)
            source.write_text('# 原文\n\n[00:00:01] 箱\n[00:00:01] 札', encoding='utf-8-sig')
            stem, segments = read_transcript(source)
            self.assertEqual(stem, '原文')
            self.assertEqual([s.start_seconds for s in segments], [1, 1])

    def test_cli_preserves_input_and_skips_media(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / '日本語 講義.v2.transcript.md'
            source.write_text('[00:00:01] 架空の箱', encoding='utf-8-sig')
            original = source.read_bytes()
            article = Path(folder) / '日本語 講義.v2.md'
            response = [{'body': [{'heading': '', 'text': '架空の箱', 'timestamp': '00:00:01'}], 'checks': []}]
            with patch('elearning_converter.__main__.gemini_api_key', return_value='test-token'), patch('elearning_converter.__main__.extract_audio') as media, patch('elearning_converter.__main__.transcribe') as speech, patch('elearning_converter.__main__.generate_parts', return_value=response) as generate, redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                argv = [str(source), '--from-transcript', '--article-model', 'test-model']
                self.assertEqual(run(argv), 0)
                self.assertTrue(article.read_text(encoding='utf-8').startswith('# 日本語 講義.v2\n'))
                self.assertEqual(generate.call_args.args[2], 'test-model')
                self.assertEqual(run(argv), 2)
                generate.assert_called_once()
                saved = article.read_bytes()
                generate.side_effect = ArticleError('API障害')
                self.assertEqual(run(argv + ['--overwrite']), 1)
                self.assertEqual(article.read_bytes(), saved)
                generate.reset_mock()
                self.assertEqual(run(argv + ['--no-article']), 2)
                source.write_text('不正な入力', encoding='utf-8')
                self.assertEqual(run(argv + ['--overwrite']), 2)
                generate.assert_not_called()
                media.assert_not_called()
                speech.assert_not_called()
                source.write_bytes(original)
                generate.side_effect = None
                self.assertEqual(run(argv + ['--overwrite']), 0)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(len(list(Path(folder).iterdir())), 2)
