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
from elearning_converter.article import ARTICLE_SCHEMA, ArticleError, generate_parts, parse_response
from elearning_converter.transcribe import Segment


PAYLOAD = {"body": [{"heading": "", "text": "架空の箱を整理する", "timestamp": "00:00:01"}], "checks": []}


def sdk_response(text, finish="STOP"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(text=text)]), finish_reason=finish)])


class ResponseTests(unittest.TestCase):
    def test_real_sdk_serializes_schema_and_decodes_response_without_network(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(PAYLOAD)}], "role": "model"}, "finishReason": "STOP"}]})
        with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
            with genai.Client(api_key="not-a-real-key", http_options=types.HttpOptions(httpx_client=transport)) as client:
                result = generate_parts([Segment(1, 2, "架空の箱を整理する")], "not-a-real-key", "test-model", client_factory=lambda **kwargs: client)
        self.assertEqual(result, [PAYLOAD])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["generationConfig"]["responseJsonSchema"], ARTICLE_SCHEMA)
        self.assertNotIn("not-a-real-key", json.dumps(requests))

    def test_distinct_response_failures_are_safe(self):
        invalid = {"body": [{"heading": None, "text": "private-text", "timestamp": "00:00:01"}], "checks": []}
        bad_time = {"body": [{"heading": "", "text": "private-text", "timestamp": "private-text"}], "checks": []}
        cases = [
            (sdk_response(None), "応答が空"),
            (sdk_response(" \n"), "応答が空"),
            (sdk_response('{"private-text":'), "JSONが構文不正"),
            (sdk_response(json.dumps(invalid)), "headingは文字列"),
            (sdk_response(json.dumps(bad_time)), "timestampはHH:MM:SS"),
            (sdk_response(json.dumps(PAYLOAD), "MAX_TOKENS"), "finish_reason=MAX_TOKENS"),
            (sdk_response(None, "SAFETY"), "finish_reason=SAFETY"),
            (types.GenerateContentResponse(prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY", block_reason_message="private-text")), "block_reason=SAFETY"),
            (SimpleNamespace(candidates=[SimpleNamespace(finish_reason="private-text")], text="private-text"), "finish_reason=UNKNOWN"),
            (SimpleNamespace(prompt_feedback=SimpleNamespace(block_reason="private-text"), text="private-text"), "block_reason=UNKNOWN"),
        ]
        for response, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(ArticleError) as caught:
                    parse_response(response, 2, 3)
                message = str(caught.exception)
                self.assertIn(expected, message)
                self.assertIn("分割 2/3", message)
                self.assertNotIn("private-text", message)

    def test_json_error_reports_position_without_source_or_exception_message(self):
        with self.assertRaises(ArticleError) as caught:
            parse_response(sdk_response('{\n"body": private-text}'), 1, 1)
        self.assertIn("行 2", str(caught.exception))
        self.assertIn("列", str(caught.exception))
        self.assertNotIn("private-text", str(caught.exception))

    def test_format_error_is_not_retried_and_identifies_failed_chunk(self):
        generate = Mock(side_effect=[sdk_response(json.dumps(PAYLOAD)), sdk_response('{"private-text":')])
        factory = Mock(return_value=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
        with patch("elearning_converter.article.split_segments", return_value=["first", "second", "third"]):
            with self.assertRaises(ArticleError) as caught:
                generate_parts([Segment(0, 1, "箱")], "not-a-real-key", "test-model", client_factory=factory)
        self.assertIn("分割 2/3", str(caught.exception))
        self.assertEqual(generate.call_count, 2)

    def test_malformed_response_rescues_source_without_article_or_sensitive_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "架空 講義.mp4"
            source.write_bytes(b"synthetic")
            generate = Mock(return_value=sdk_response('{"private-text":'))
            factory = lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=generate))
            def generate_mock(segments, key, model):
                return generate_parts(segments, key, model, client_factory=factory)
            error_output, output = StringIO(), StringIO()
            with patch("elearning_converter.__main__.gemini_api_key", return_value="not-a-real-key"), patch("elearning_converter.__main__.extract_audio"), patch("elearning_converter.__main__.transcribe", return_value=[Segment(1, 2, "private-text")]), patch("elearning_converter.__main__.generate_parts", side_effect=generate_mock), redirect_stderr(error_output), redirect_stdout(output):
                self.assertEqual(run([str(source)]), 1)
            self.assertIn("JSONが構文不正", error_output.getvalue())
            self.assertNotIn("private-text", error_output.getvalue() + output.getvalue())
            self.assertNotIn("not-a-real-key", error_output.getvalue() + output.getvalue())
            self.assertEqual(generate.call_count, 1)
            self.assertFalse(source.with_suffix(".md").exists())
            self.assertIn("private-text", (Path(folder) / "架空 講義.transcript.failed.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
