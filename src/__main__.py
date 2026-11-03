import argparse
import sys
import tempfile
from pathlib import Path

from .article import ArticleError, generate_parts
from .config import ConfigurationError, Settings, gemini_api_key, validate
from .media import MediaError, extract_audio
from .output import OutputError, write_atomic
from .render import article_markdown, transcript_markdown
from .transcribe import TranscriptionError, transcribe
from .transcript_input import read_transcript


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="elconv",
        description="elearning-converter: 許可済みMP4の音声を復習用Markdownへ変換します。記事化時は文字起こしテキストをGemini APIへ送信し、従量課金の可能性があります。",
    )
    result.add_argument("input", type=Path, metavar="INPUT")
    result.add_argument("--from-transcript", action="store_true", help="既存の .transcript.md / .transcript.failed.md から記事のみ生成（音声処理を省略）")
    result.add_argument("--output-dir", type=Path, help="出力先（既定: MP4と同じフォルダ）")
    result.add_argument("--language", default="ja", help="認識言語（既定: ja）")
    result.add_argument("--whisper-model", default="small", help="faster-whisperモデル（既定: small）")
    result.add_argument("--article-model", default="gemini-3.5-flash-lite", help="Geminiモデル（既定: gemini-3.5-flash-lite）")
    result.add_argument("--no-article", action="store_true", help="APIを使わず時刻付き原文のみ保存")
    result.add_argument("--overwrite", action="store_true", help="既存出力を置換")
    return result


def run(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.from_transcript:
        return run_from_transcript(args)
    settings = Settings(args.input, args.output_dir or args.input.parent, args.language,
                        args.whisper_model, args.article_model, args.no_article, args.overwrite)
    try:
        api_key = None if settings.no_article else gemini_api_key()
        validate(settings, api_key)
    except ConfigurationError as exc:
        print(f"設定エラー: {exc}", file=sys.stderr)
        return 2
    print("音声を検査・抽出しています。", file=sys.stderr)
    try:
        with tempfile.TemporaryDirectory(prefix="elearning-converter-") as folder:
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


def run_from_transcript(args: argparse.Namespace) -> int:
    try:
        if args.no_article:
            raise ConfigurationError("--from-transcript と --no-article は併用できません。")
        stem, segments = read_transcript(args.input)
        target = (args.output_dir or args.input.parent) / f"{stem}.md"
        if target.resolve() == args.input.resolve() or (target.exists() and target.samefile(args.input)):
            raise ConfigurationError("入力原文を出力先に指定できません。")
        if target.exists() and (not args.overwrite or not target.is_file()):
            raise ConfigurationError("既存出力があります。記事の更新には --overwrite を指定してください。")
        if not args.article_model.strip():
            raise ConfigurationError("記事モデル名は空にできません。")
        api_key = gemini_api_key()
        if not api_key:
            raise ConfigurationError("GEMINI_API_KEY が設定されていません。")
        target.parent.mkdir(parents=True, exist_ok=True)
    except (ConfigurationError, OSError) as exc:
        message = str(exc) if isinstance(exc, ConfigurationError) else "入力または出力先を確認できません。"
        print(f"設定エラー: {message}", file=sys.stderr)
        return 2
    print("既存の文字起こしテキストのみGemini APIへ送信して記事化します。従量課金の可能性があります。", file=sys.stderr)
    try:
        parts = generate_parts(segments, api_key, args.article_model)
        write_atomic(target, article_markdown(target, parts), args.overwrite)
    except (ArticleError, OutputError) as exc:
        print(f"処理エラー: {exc}。入力原文は保持されています。", file=sys.stderr)
        return 1
    print(f"記事を保存しました: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
