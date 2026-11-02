from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import wave


class TranscriptionError(Exception):
    pass


@dataclass(frozen=True)
class Segment:
    start_seconds: float
    end_seconds: float
    text: str

    def __post_init__(self) -> None:
        if not (isfinite(self.start_seconds) and isfinite(self.end_seconds)):
            raise TranscriptionError("認識結果の時刻が不正です。")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise TranscriptionError("認識結果の時刻が不正です。")
        if not self.text.strip():
            raise TranscriptionError("認識結果に空テキストがあります。")


def transcribe(wav_path: Path, language: str, model_name: str) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
        import numpy as np
    except Exception as exc:
        raise TranscriptionError(f"文字起こしのライブラリ読み込みに失敗しました（{type(exc).__name__}）。") from exc
    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as exc:
        raise TranscriptionError(f"文字起こしのモデル読み込みに失敗しました（{type(exc).__name__}）。") from exc
    try:
        with wave.open(str(wav_path), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("unexpected WAV format")
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    except Exception as exc:
        raise TranscriptionError(f"文字起こしの音声読み込みに失敗しました（{type(exc).__name__}）。") from exc
    try:
        raw, _ = model.transcribe(samples, language=language, beam_size=5)
    except Exception as exc:
        raise TranscriptionError(f"文字起こしの認識準備に失敗しました（{type(exc).__name__}）。") from exc
    try:
        segments = [Segment(float(item.start), float(item.end), item.text.strip()) for item in raw if item.text.strip()]
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"文字起こしの推論・結果処理に失敗しました（{type(exc).__name__}）。") from exc
    if not segments:
        raise TranscriptionError("文字起こし結果が空です。")
    return segments
