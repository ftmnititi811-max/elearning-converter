from pathlib import Path
import re
from .transcribe import Segment


def timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def transcript_markdown(source: Path, segments: list[Segment]) -> str:
    lines = [f"# {source.stem} 原文", "", "時刻は元の音声に対応します。自動認識結果を原音声と照合してください。", ""]
    lines.extend(f"[{timestamp(s.start_seconds)}] {s.text}" for s in segments)
    return "\n".join(lines) + "\n"


def article_markdown(source: Path, parts: list[dict]) -> str:
    lines = [f"# {source.stem}", "", "> 自動生成の復習用下書き — 原音声・時刻付き原文と要照合", ""]
    previous_heading = None
    for part in parts:
        for body in part["body"]:
            heading = body["heading"].strip()
            if heading and heading != previous_heading:
                lines += [f"## {heading}", ""]
                previous_heading = heading
            text = body["text"].strip()
            reference = f"[{body['timestamp']}]"
            # Keep references out of Markdown tables, lists, and fenced code.
            if re.search(r"(?m)^\s*(?:\||[-+*] |\d+[.)] |`{3}|~{3}|>)", text):
                lines += [text, "", reference, ""]
            else:
                lines += [f"{text} {reference}", ""]
    checks = list(dict.fromkeys(item.strip() for part in parts for item in part["checks"]))
    if checks:
        lines += ["---", "", "**原文で確認する箇所**", ""]
        lines.extend(f"- {item}" for item in checks)
        lines.append("")
    return "\n".join(lines) + "\n"
