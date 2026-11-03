import copy
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from elearning_converter.__main__ import run
from elearning_converter.article import ArticleError, generate_parts, load_prompt, validate_part
from elearning_converter.render import article_markdown
from elearning_converter.transcribe import Segment


def paragraph(heading="", text="箱に名前を付けると、中身を探しやすくなる", time="00:00:01"):
    return {"heading": heading, "text": text, "timestamp": time}


def part(*body, checks=None):
    return {"body": list(body) or [paragraph()], "checks": checks or []}


class NoteTests(unittest.TestCase):
    def test_fidelity_example_through_real_prompt_parser_and_renderer(self):
        # Authored expectations exercise the plumbing, not model generation quality.
        example = (Path(__file__).resolve().parents[1] / "docs" / "note-fidelity-example.md").read_text(encoding="utf-8")
        transcript = re.search(r"```text\n(.*?)\n```", example, re.S).group(1)
        expected = json.loads(re.search(r"```json\n(.*?)\n```", example, re.S).group(1))
        segments = []
        for line in transcript.splitlines():
            time, text = line[1:].split("] ", 1)
            hours, minutes, seconds = map(int, time.split(":"))
            start = hours * 3600 + minutes * 60 + seconds
            segments.append(Segment(start, start + 1, text))
        generate = Mock(return_value=SimpleNamespace(text=json.dumps(expected)))
        factory = Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
        parts = generate_parts(segments, "test-token", "test-model", client_factory=factory)
        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs["contents"], load_prompt() + transcript)
        self.assertEqual(parts, [expected])
        rendered = article_markdown(Path("架空 残数処理.mp4"), parts)
        for body in expected["body"]:
            self.assertIn(body["text"], rendered)
            self.assertIn(f"[{body['timestamp']}]", rendered)
        for check in expected["checks"]:
            self.assertIn("- " + check, rendered)
        self.assertIn("```\n\n[00:00:35]", rendered)
        self.assertNotIn("[00:00:00]", rendered)
        self.assertEqual(rendered.count("## 箱の残数処理"), 1)

    def test_optional_headings_and_consecutive_merge_preserve_all_content(self):
        parts = [part(paragraph(), paragraph("箱の整理", "**ラベル**は中身を示す名前", "00:00:02")),
                 part(paragraph("箱の整理", "名前には用途を含める", "00:00:03"),
                      paragraph("", "同じ用途の箱をまとめる", "00:00:04"),
                      paragraph("確認", "中身と名前を照合する", "00:00:05"),
                      paragraph("箱の整理", "必要に応じて並べ直す", "00:00:06"))]
        for value in parts:
            validate_part(value)
        result = article_markdown(Path("架空 講義.mp4"), parts)
        self.assertEqual(result.count("## 箱の整理"), 2)
        self.assertNotIn("## 要点", result)
        self.assertNotIn("## 本文", result)
        self.assertNotIn("## 用語", result)
        for value in parts:
            for item in value["body"]:
                self.assertIn(item["text"], result)
                self.assertIn(f"[{item['timestamp']}]", result)
        self.assertLess(result.index("箱に名前"), result.index("## 箱の整理"))
        self.assertNotIn("原文で確認する箇所", result)

    def test_table_list_and_code_references_do_not_break_markdown(self):
        for text in ("| 箱 | 用途 |\n| --- | --- |\n| 青 | 文具 |", "- 文具\n- 衣類", "1. 名前を付ける\n2. 並べる", "```text\n箱\n```", "> 箱の中身を確認"):
            with self.subTest(text=text):
                result = article_markdown(Path("sample.mp4"), [part(paragraph(text=text))])
                self.assertIn(text + "\n\n[00:00:01]\n", result)

    def test_checks_only_exact_duplicates_are_removed(self):
        result = article_markdown(Path("sample.mp4"), [part(checks=["[00:00:01] 箱の名称を確認"]),
                  part(checks=["[00:00:01] 箱の名称を確認", "[00:00:02] 箱の名称を確認"])])
        self.assertEqual(result.count("**原文で確認する箇所**"), 1)
        self.assertEqual(result.count("- [00:00:01] 箱の名称を確認"), 1)
        self.assertIn("- [00:00:02] 箱の名称を確認", result)

    def test_renderer_preserves_meaningful_punctuation_and_examples(self):
        text = "**ラベル**は中身を示す名前。例えば文具箱に『文具』と書くと、開けずに用途が分かる"
        result = article_markdown(Path("sample.mp4"), [part(paragraph(text=text))])
        self.assertIn(text + " [00:00:01]", result)

    def test_invalid_payloads_fail_without_echoing_content(self):
        values = [None, {}, {"body": [], "checks": []}, {"body": "private-text", "checks": []},
                  part(checks=[1]), part(checks=[" "]), part(paragraph(heading=None)),
                  part(paragraph(text=" ")), part(paragraph(time="00:60:00")),
                  part(paragraph(time="00:00:60")), part(paragraph(time="private-text"))]
        extra = copy.deepcopy(part())
        extra["summary"] = ["legacy"]
        values.append(extra)
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ArticleError) as caught:
                    validate_part(value)
                self.assertNotIn("private-text", str(caught.exception))

    def test_prompt_bom_japanese_path_edit_and_cwd_independence(self):
        with tempfile.TemporaryDirectory() as folder:
            prompt = Path(folder) / "日本語 プロンプト.md"
            prompt.write_text("指示A\n文字起こし:\n", encoding="utf-8-sig")
            previous = Path.cwd()
            try:
                os.chdir(folder)
                with patch("elearning_converter.article.PROMPT_PATH", prompt):
                    self.assertEqual(load_prompt(), "指示A\n文字起こし:\n")
                    prompt.write_text("指示B\n", encoding="utf-8")
                    self.assertEqual(load_prompt(), "指示B\n")
                self.assertIn("授業ノート", load_prompt())
            finally:
                os.chdir(previous)

    def test_prompt_is_read_once_and_no_extra_merge_api_is_called(self):
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=json.dumps(part()))
        factory = Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
        segments = [Segment(0, 1, "架空の箱"), Segment(2, 3, "架空のラベル")]
        chunks = ["[00:00:00] 架空の箱", "[00:00:02] 架空のラベル"]
        with patch("elearning_converter.article.load_prompt", return_value="指示\n") as loader, patch("elearning_converter.article.split_segments", return_value=chunks):
            parts = generate_parts(segments, "test-token", "test-model", client_factory=factory)
        loader.assert_called_once_with()
        self.assertEqual(len(parts), 2)
        self.assertEqual([call["contents"] for call in calls], ["指示\n" + chunk for chunk in chunks])
        article_markdown(Path("sample.mp4"), parts)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("test-token", str(calls))

    def test_prompt_errors_prevent_client_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            prompt = Path(folder) / "prompt.md"
            for contents in (None, b" \n", b"\xff"):
                if contents is not None:
                    prompt.write_bytes(contents)
                factory = Mock()
                with self.subTest(contents=contents), patch("elearning_converter.article.PROMPT_PATH", prompt):
                    with self.assertRaises(ArticleError):
                        generate_parts([Segment(0, 1, "箱")], "test-token", "test-model", client_factory=factory)
                factory.assert_not_called()
        with patch("pathlib.Path.read_text", side_effect=PermissionError("private-text")):
            with self.assertRaises(ArticleError) as caught:
                load_prompt()
        self.assertNotIn("private-text", str(caught.exception))

    def test_actual_prompt_edit_affects_next_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            prompt = Path(folder) / "prompt.md"
            generate = Mock(return_value=SimpleNamespace(text=json.dumps(part())))
            factory = Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
            with patch("elearning_converter.article.PROMPT_PATH", prompt):
                for instruction in ("指示A\n", "指示B\n"):
                    prompt.write_text(instruction, encoding="utf-8")
                    generate_parts([Segment(0, 1, "箱")], "test-token", "test-model", client_factory=factory)
                    self.assertTrue(generate.call_args.kwargs["contents"].startswith(instruction))

    def test_prompt_failure_rescues_transcript_and_no_article_skips_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "日本語 講義.mp4"
            source.write_bytes(b"synthetic")
            with patch("elearning_converter.__main__.gemini_api_key", return_value="test-token"), patch("elearning_converter.__main__.extract_audio"), patch("elearning_converter.__main__.transcribe", return_value=[Segment(0, 1, "架空の箱")]), patch("elearning_converter.article.load_prompt", side_effect=ArticleError("プロンプトを読み込めません")) as loader, redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                self.assertEqual(run([str(source)]), 1)
                self.assertTrue((root / "日本語 講義.transcript.failed.md").exists())
                self.assertFalse((root / "日本語 講義.md").exists())
                loader.reset_mock()
                self.assertEqual(run([str(source), "--no-article"]), 0)
                loader.assert_not_called()
            self.assertEqual(source.read_bytes(), b"synthetic")
