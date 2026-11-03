"""Read elconv's saved transcript without loading speech recognition."""
import re
from pathlib import Path

from .config import ConfigurationError
from .transcribe import Segment


def read_transcript(path: Path) -> tuple[str, list[Segment]]:
    suffix = next((s for s in (".transcript.failed.md", ".transcript.md")
                   if path.name.lower().endswith(s)), None)
    if suffix is None or not path.name[:-len(suffix)]:
        raise ConfigurationError("入力は .transcript.md または .transcript.failed.md を指定してください。")
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError("文字起こしファイルをUTF-8で読み込めません。") from exc
    segments = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if not segments and (line.startswith("# ") or line == "時刻は元の音声に対応します。自動認識結果を原音声と照合してください。"):
            continue
        match = re.fullmatch(r"\[(\d{2,}):([0-5]\d):([0-5]\d)\]\s+(.+)", line)
        if not match or not match[4].strip():
            raise ConfigurationError(f"文字起こしの形式が不正です（行{number}）。")
        start = int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3])
        if segments and start < segments[-1].start_seconds:
            raise ConfigurationError(f"文字起こしの時刻が逆順です（行{number}）。")
        segments.append(Segment(start, start, match[4].strip()))
    if not segments:
        raise ConfigurationError("文字起こし本文がありません。")
    return path.name[:-len(suffix)], segments
