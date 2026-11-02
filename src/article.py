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
        raise ArticleError("APIの記事形式が不正です。")
    items = value["checks"]
    if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
        raise ArticleError("APIの記事形式が不正です。")
    if not isinstance(value["body"], list):
        raise ArticleError("APIの記事形式が不正です。")
    if not value["body"]:
        raise ArticleError("APIの記事が空です。")
    for item in value["body"]:
        if not isinstance(item, dict) or set(item) != {"heading", "text", "timestamp"}:
            raise ArticleError("APIの記事形式が不正です。")
        if not isinstance(item["heading"], str):
            raise ArticleError("APIの記事形式が不正です。")
        if any(not isinstance(item[k], str) or not item[k].strip() for k in ("text", "timestamp")):
            raise ArticleError("APIの記事形式が不正です。")
        if not TIME_RE.fullmatch(item["timestamp"]):
            raise ArticleError("APIの記事時刻が不正です。")
    return value


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
    for chunk in split_segments(segments):
        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt + chunk,
                    config={"response_mime_type": "application/json"},
                )
                break
            except Exception as exc:
                detail, retryable = _api_failure(exc)
                if not retryable or attempt == 2:
                    raise ArticleError(f"Gemini APIの呼び出しに失敗しました（{detail}）。") from exc
                pause(2 ** attempt)
        try:
            content = response.text
            part = validate_part(json.loads(content))
        except (ValueError, TypeError, AttributeError, ArticleError) as exc:
            raise ArticleError("Gemini APIの記事応答が空か形式不正です。") from exc
        parts.append(part)
    return parts
