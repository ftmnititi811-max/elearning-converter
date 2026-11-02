from pathlib import Path
from .transcribe import Segment


def timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def transcript_markdown(source: Path, segments: list[Segment]) -> str:
    lines = [f"# {source.stem} 原文", "", "時刻は元の音声に対応します。自動認識結果を原音声と照合してください。", ""]
    lines.extend(f"[{timestamp(s.start_seconds)}] {s.text}" for s in segments)
    return "\n".join(lines) + "\n"


def article_markdown(source: Path, parts: list[dict]) -> str:
    lines = [f"# {source.stem}", "", "> 自動生成した復習用下書きです。原音声と時刻付き原文で内容を確認してください。", "", "## 要点", ""]
    for part in parts:
        lines.extend(f"- {item}" for item in part["summary"])
    lines += ["", "## 本文", ""]
    for part in parts:
        for body in part["body"]:
            lines += [f"### {body['heading']} [{body['timestamp']}]", "", body["text"], ""]
    lines += ["## 用語", ""]
    for part in parts:
        lines.extend(f"- {item}" for item in part["terms"])
    lines += ["", "## 要確認事項", ""]
    checks = [item for part in parts for item in part["checks"]]
    lines.extend(f"- {item}" for item in checks or ["特記事項なし。原音声との照合は必要です。"])
    return "\n".join(lines) + "\n"
