import argparse
import os
import sys
import tempfile
from pathlib import Path

from .article import ArticleError, generate_parts
from .config import ConfigurationError, Settings, validate
from .media import MediaError, extract_audio
from .output import OutputError, write_atomic
from .render import article_markdown, transcript_markdown
from .transcribe import TranscriptionError, transcribe


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="python -m lecture_md",
        description="許可済みMP4の音声を復習用Markdownへ変換します。記事化時は文字起こしテキストをGemini APIへ送信し、従量課金の可能性があります。",
    )
    result.add_argument("input", type=Path, metavar="INPUT.mp4")
    result.add_argument("--output-dir", type=Path, help="出力先（既定: MP4と同じフォルダ）")
    result.add_argument("--language", default="ja", help="認識言語（既定: ja）")
    result.add_argument("--whisper-model", default="small", help="faster-whisperモデル（既定: small）")
    result.add_argument("--article-model", default="gemini-2.5-flash", help="Geminiモデル（既定: gemini-2.5-flash）")
    result.add_argument("--no-article", action="store_true", help="APIを使わず時刻付き原文のみ保存")
    result.add_argument("--overwrite", action="store_true", help="既存出力を置換")
    return result


def run(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = Settings(args.input, args.output_dir or args.input.parent, args.language,
                        args.whisper_model, args.article_model, args.no_article, args.overwrite)
    api_key = None if settings.no_article else os.environ.get("GEMINI_API_KEY")
    try:
        validate(settings, api_key)
    except ConfigurationError as exc:
        print(f"設定エラー: {exc}", file=sys.stderr)
        return 2
    print("音声を検査・抽出しています。", file=sys.stderr)
    try:
        with tempfile.TemporaryDirectory(prefix="lecture-md-") as folder:
            wav_path = Path(folder) / "audio.wav"
            extract_audio(settings.input_path, wav_path)
            print("文字起こしをしています。", file=sys.stderr)
            segments = transcribe(wav_path, settings.language, settings.whisper_model)
    except (MediaError, TranscriptionError) as exc:
        print(f"処理エラー: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("処理エラー: 一時ファイルを作成または削除できません。", file=sys.stderr)
        return 1
    transcript = transcript_markdown(settings.input_path, segments)
    if settings.no_article:
        try:
            write_atomic(settings.transcript_path, transcript, settings.overwrite)
        except OutputError as exc:
            print(f"処理エラー: {exc}", file=sys.stderr)
            return 1
        print(f"原文を保存しました: {settings.transcript_path}")
        return 0
    print("Gemini APIで記事化しています。文字起こしテキストのみ送信します。従量課金の可能性があります。", file=sys.stderr)
    try:
        parts = generate_parts(segments, api_key, settings.article_model)
        article = article_markdown(settings.input_path, parts)
    except ArticleError as exc:
        print(f"処理エラー: {exc}", file=sys.stderr)
        try:
            write_atomic(settings.rescue_path, transcript, settings.overwrite)
            print(f"原文のみ救済しました: {settings.rescue_path}", file=sys.stderr)
        except OutputError:
            print("原文の救済保存にも失敗しました。", file=sys.stderr)
        return 1
    try:
        write_atomic(settings.transcript_path, transcript, settings.overwrite)
        write_atomic(settings.article_path, article, settings.overwrite)
    except OutputError as exc:
        print(f"処理エラー: {exc}。原文だけが保存された可能性があります。", file=sys.stderr)
        return 1
    print(f"記事を保存しました: {settings.article_path}")
    print(f"原文を保存しました: {settings.transcript_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
