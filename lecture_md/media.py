import json
import shutil
import subprocess
from pathlib import Path


class MediaError(Exception):
    pass


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, shell=False, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    except OSError as exc:
        raise MediaError("FFmpeg/ffprobeを起動できません。インストールとPATHを確認してください。") from exc


def extract_audio(source: Path, wav_path: Path) -> None:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise MediaError("FFmpegまたはffprobeが見つかりません。インストールとPATHを確認してください。")
    probe = _run([ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "json", str(source)])
    if probe.returncode != 0:
        raise MediaError("MP4の音声情報を検査できません。")
    try:
        streams = json.loads(probe.stdout).get("streams", [])
    except (ValueError, AttributeError) as exc:
        raise MediaError("ffprobeの応答を読み取れません。") from exc
    if not streams:
        raise MediaError("MP4に音声ストリームがありません。")
    result = _run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(wav_path)])
    if result.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size <= 44:
        raise MediaError("音声をWAVへ抽出できません。入力MP4を確認してください。")
