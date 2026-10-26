import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lecture_md.__main__ import run
from lecture_md.article import ArticleError, generate_parts, split_segments
from lecture_md.config import ConfigurationError, Settings, gemini_api_key, validate
from lecture_md.media import MediaError, extract_audio
from lecture_md.output import OutputError, write_atomic
from lecture_md.render import timestamp, transcript_markdown
from lecture_md.transcribe import Segment, TranscriptionError, transcribe


class CoreTests(unittest.TestCase):
    def test_segments_and_markdown(self):
        segments = [Segment(3661.5, 3664, "講義内容")]
        self.assertEqual(timestamp(3661.5), "01:01:01")
        self.assertIn("[01:01:01] 講義内容", transcript_markdown(Path("日本語 講義.mp4"), segments))
        with self.assertRaises(TranscriptionError):
            Segment(-1, 2, "invalid")

    def test_transcription_error_identifies_stage_without_exception_text(self):
        secret = "private-source-text"
        with patch("faster_whisper.WhisperModel", side_effect=RuntimeError(secret)):
            with self.assertRaises(TranscriptionError) as caught:
                transcribe(Path("audio.wav"), "ja", "small")
        self.assertIn("モデル読み込み", str(caught.exception))
        self.assertIn("RuntimeError", str(caught.exception))
        self.assertNotIn(secret, str(caught.exception))

        model = SimpleNamespace(transcribe=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError(secret)))
        with patch("faster_whisper.WhisperModel", return_value=model):
            with self.assertRaises(TranscriptionError) as caught:
                transcribe(Path("audio.wav"), "ja", "small")
        self.assertIn("音声読み込み", str(caught.exception))
        self.assertNotIn(secret, str(caught.exception))

    def test_split_keeps_timestamps(self):
        chunks = split_segments([Segment(0, 1, "あ" * 15), Segment(2, 3, "い" * 15)], limit=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.startswith("[") for chunk in chunks))
        self.assertIn("[00:00:02]", "\n".join(chunks))

    def test_existing_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "日本語 講義.mp4"
            source.write_bytes(b"source")
            output = root / "日本語 講義.transcript.md"
            output.write_text("old", encoding="utf-8")
            settings = Settings(source, root, "ja", "small", "gemini-2.5-flash", True, False)
            with self.assertRaises(ConfigurationError):
                validate(settings, None)
            with self.assertRaises(OutputError):
                write_atomic(output, "new", False)
            self.assertEqual(output.read_text(encoding="utf-8"), "old")
            write_atomic(output, "new", True)
            self.assertEqual(output.read_text(encoding="utf-8"), "new")

    def test_gemini_request_retries_without_key_in_text(self):
        payload = {"summary": ["要点"], "body": [{"heading": "内容", "text": "説明", "timestamp": "00:00:00"}], "terms": [], "checks": []}
        calls = []

        def generate_content(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("temporary")
            return SimpleNamespace(text=json.dumps(payload))

        factory = lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        parts = generate_parts([Segment(0, 1, "本文")], "secret", "model", client_factory=factory, pause=lambda _: None)
        self.assertEqual(parts, [payload])
        self.assertEqual(len(calls), 2)
        self.assertIn("本文", calls[0]["contents"])
        self.assertNotIn("secret", calls[0]["contents"])

    def test_bad_api_response_is_rejected(self):
        factory = lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(text="{}")))
        with self.assertRaises(ArticleError):
            generate_parts([Segment(0, 1, "本文")], "secret", "model", client_factory=factory)

    def test_missing_key_fails_before_processing(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "講義.mp4"
            source.write_bytes(b"source")
            with patch.dict("os.environ", {}, clear=True), patch("lecture_md.__main__.gemini_api_key", return_value=None), patch("lecture_md.__main__.extract_audio") as extractor:
                self.assertEqual(run([str(source)]), 2)
                extractor.assert_not_called()

    def test_env_file_key_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            env_path = Path(folder) / ".env"
            env_path.write_text('# local setting\nGEMINI_API_KEY="file-secret"\n', encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(gemini_api_key(env_path), "file-secret")
            with patch.dict("os.environ", {"GEMINI_API_KEY": "environment-secret"}):
                self.assertEqual(gemini_api_key(env_path), "environment-secret")

    def test_cli_uses_key_from_env_file(self):
        part = {"summary": ["要点"], "body": [{"heading": "内容", "text": "説明", "timestamp": "00:00:00"}], "terms": [], "checks": []}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "日本語 講義.mp4"
            source.write_bytes(b"source")
            env_path = root / ".env"
            env_path.write_text("GEMINI_API_KEY=file-secret\n", encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True), patch("lecture_md.__main__.gemini_api_key", side_effect=lambda: gemini_api_key(env_path)), patch("lecture_md.__main__.extract_audio"), patch("lecture_md.__main__.transcribe", return_value=[Segment(0, 2, "本文")]), patch("lecture_md.__main__.generate_parts", return_value=[part]) as generate:
                self.assertEqual(run([str(source)]), 0)
                self.assertEqual(generate.call_args.args[1], "file-secret")
            article = (root / "日本語 講義.md").read_text(encoding="utf-8")
            self.assertNotIn("file-secret", article)

    def test_ffprobe_no_audio_and_missing_tools(self):
        source = Path("講義.mp4")
        with patch("lecture_md.media.shutil.which", return_value=None):
            with self.assertRaises(MediaError):
                extract_audio(source, Path("audio.wav"))
        with patch("lecture_md.media.shutil.which", side_effect=["ffprobe", "ffmpeg"]), patch("lecture_md.media._run", return_value=SimpleNamespace(returncode=0, stdout='{"streams":[]}')):
            with self.assertRaises(MediaError):
                extract_audio(source, Path("audio.wav"))

    def test_no_article_and_api_failure_rescue(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "日本語 講義.mp4"
            source.write_bytes(b"source")
            with patch.dict("os.environ", {}, clear=True), patch("lecture_md.__main__.extract_audio"), patch("lecture_md.__main__.transcribe", return_value=[Segment(0, 2, "本文")]):
                self.assertEqual(run([str(source), "--no-article"]), 0)
            self.assertTrue((root / "日本語 講義.transcript.md").exists())
            with patch.dict("os.environ", {"GEMINI_API_KEY": "secret"}), patch("lecture_md.__main__.extract_audio"), patch("lecture_md.__main__.transcribe", return_value=[Segment(0, 2, "本文")]), patch("lecture_md.__main__.generate_parts", side_effect=ArticleError("API障害")):
                self.assertEqual(run([str(source), "--overwrite"]), 1)
            self.assertFalse((root / "日本語 講義.md").exists())
            self.assertIn("本文", (root / "日本語 講義.transcript.failed.md").read_text(encoding="utf-8"))

    def test_successful_article_writes_both_files(self):
        part = {"summary": ["要点"], "body": [{"heading": "内容", "text": "説明", "timestamp": "00:00:00"}], "terms": ["用語"], "checks": []}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "日本語 講義.mp4"
            source.write_bytes(b"source")
            with patch.dict("os.environ", {"GEMINI_API_KEY": "secret"}), patch("lecture_md.__main__.extract_audio"), patch("lecture_md.__main__.transcribe", return_value=[Segment(0, 2, "本文")]), patch("lecture_md.__main__.generate_parts", return_value=[part]):
                self.assertEqual(run([str(source)]), 0)
            article = (root / "日本語 講義.md").read_text(encoding="utf-8")
            transcript = (root / "日本語 講義.transcript.md").read_text(encoding="utf-8")
            self.assertIn("## 要点", article)
            self.assertIn("### 内容 [00:00:00]", article)
            self.assertIn("[00:00:00] 本文", transcript)
            self.assertNotIn("secret", article + transcript)


if __name__ == "__main__":
    unittest.main()
