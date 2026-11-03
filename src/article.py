import json
import re
import time
import httpx
from collections.abc import Callable
from pathlib import Path
from .render import timestamp
from .transcribe import Segment


class ArticleError(Exception):
    pass


LIMIT = 12_000
TIME_RE = re.compile(r"^\d{2,}:[0-5]\d:[0-5]\d$")
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "article.md"
ARTICLE_SCHEMA = {
    "type": "object",
    "properties": {
        "body": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string", "description": "見出し不要なら空文字。null不可"},
                    "text": {"type": "string", "description": "空でない授業ノートのMarkdown本文"},
                    "timestamp": {"type": "string", "description": "根拠の開始時刻HH:MM:SS（例00:01:23）"},
                },
                "required": ["heading", "text", "timestamp"],
                "additionalProperties": False,
            },
        },
        "checks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["body", "checks"],
    "additionalProperties": False,
}

# Only known API enum values may appear in diagnostics. Never echo arbitrary values.
FINISH_REASONS = frozenset({
    "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "LANGUAGE", "OTHER",
    "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL",
    "IMAGE_SAFETY", "UNEXPECTED_TOOL_CALL", "IMAGE_PROHIBITED_CONTENT",
    "NO_IMAGE", "IMAGE_RECITATION", "IMAGE_OTHER", "FINISH_REASON_UNSPECIFIED",
})
BLOCK_REASONS = frozenset({
    "SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY",
    "MODEL_ARMOR", "JAILBREAK", "BLOCKED_REASON_UNSPECIFIED",
})


def _api_failure(exc: Exception) -> tuple[str, bool]:
    code = getattr(exc, "code", None)
    code = code if type(code) is int and 100 <= code <= 599 else None
    name = type(exc).__name__
    name = name if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,79}", name) else "Exception"
    detail = f"HTTP {code}、{name}" if code is not None else name
    retryable = code in (408, 429) or (code is not None and 500 <= code <= 599)
    if code is None:
        retryable = isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError))
    return detail, retryable


def split_segments(segments: list[Segment], limit: int = LIMIT) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for segment in segments:
        prefix = f"[{timestamp(segment.start_seconds)}] "
        text = segment.text
        # A single unusually long segment may cross a boundary; keep its timestamp.
        capacity = max(1, limit - len(prefix) - 1)
        pieces = [text[i:i + capacity] for i in range(0, len(text), capacity)]
        for piece in pieces:
            line = prefix + piece
            if current and size + len(line) + 1 > limit:
                chunks.append("\n".join(current))
                current, size = [], 0
            current.append(line)
            size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def validate_part(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"body", "checks"}:
        raise ArticleError("トップレベルはbody・checksのみのオブジェクトが必要です")
    items = value["checks"]
    if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
        raise ArticleError("checksは空でない文字列の配列が必要です（空配列可）")
    if not isinstance(value["body"], list):
        raise ArticleError("bodyは配列が必要です")
    if not value["body"]:
        raise ArticleError("bodyが空です")
    for item in value["body"]:
        if not isinstance(item, dict) or set(item) != {"heading", "text", "timestamp"}:
            raise ArticleError("bodyの各要素にheading・text・timestampが必要です")
        if not isinstance(item["heading"], str):
            raise ArticleError("headingは文字列が必要です（空文字可）")
        if any(not isinstance(item[k], str) or not item[k].strip() for k in ("text", "timestamp")):
            raise ArticleError("text・timestampは空でない文字列が必要です")
        if not TIME_RE.fullmatch(item["timestamp"]):
            raise ArticleError("timestampはHH:MM:SS形式が必要です（分・秒は00～59）")
    return value


def _known_reason(value: object, allowed: frozenset[str]) -> str | None:
    value = getattr(value, "value", value)
    if value is None:
        return None
    return value if isinstance(value, str) and value in allowed else "UNKNOWN"


def parse_response(response: object, index: int, total: int, validator=validate_part) -> dict:
    context = f"分割 {index}/{total}"
    feedback = getattr(response, "prompt_feedback", None)
    blocked = _known_reason(getattr(feedback, "block_reason", None), BLOCK_REASONS)
    if blocked not in (None, "BLOCKED_REASON_UNSPECIFIED"):
        raise ArticleError(f"Gemini APIの記事生成がブロックされました（{context}、block_reason={blocked}）。")
    candidates = getattr(response, "candidates", None)
    finish = _known_reason(getattr(candidates[0], "finish_reason", None), FINISH_REASONS) if candidates else None
    if finish not in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
        raise ArticleError(f"Gemini APIの記事生成が完了しませんでした（{context}、finish_reason={finish}）。")
    try:
        content = response.text
    except (ValueError, TypeError, AttributeError) as exc:
        raise ArticleError(f"Gemini APIの記事本文を取得できません（{context}）。") from exc
    if not isinstance(content, str) or not content.strip():
        raise ArticleError(f"Gemini APIの記事応答が空です（{context}）。")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ArticleError(f"Gemini APIの記事JSONが構文不正です（{context}、行 {exc.lineno}、列 {exc.colno}）。") from exc
    try:
        return validator(value)
    except ArticleError as exc:
        # validate_part only emits fixed descriptions, never model content.
        raise ArticleError(f"Gemini APIの記事形式が不正です（{context}、{exc}）。") from exc


def load_prompt() -> str:
    try:
        prompt = PROMPT_PATH.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ArticleError("記事化プロンプトを読み込めません。") from exc
    if not prompt.strip():
        raise ArticleError("記事化プロンプトが空です。")
    return prompt


def generate_parts(segments: list[Segment], api_key: str, model_name: str,
                   client_factory: Callable | None = None, pause: Callable[[float], None] = time.sleep) -> list[dict]:
    prompt = load_prompt()
    try:
        if client_factory is None:
            from google import genai
            client_factory = genai.Client
        client = client_factory(api_key=api_key)
    except Exception as exc:
        raise ArticleError("Geminiクライアントを開始できません。") from exc
    parts = []
    chunks = split_segments(segments)
    for index, chunk in enumerate(chunks, start=1):
        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt + chunk,
                    config={"response_mime_type": "application/json", "response_json_schema": ARTICLE_SCHEMA},
                )
                break
            except Exception as exc:
                detail, retryable = _api_failure(exc)
                if not retryable or attempt == 2:
                    raise ArticleError(f"Gemini APIの呼び出しに失敗しました（{detail}）。") from exc
                pause(2 ** attempt)
        part = parse_response(response, index, len(chunks))
        parts.append(part)
    return parts
