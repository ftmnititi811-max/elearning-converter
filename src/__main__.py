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
from .quality import generate_quality
from .note_pipeline import generate_pipeline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="elconv",
        description="elearning-converter: 許可済みMP4の音声を復習用Markdownへ変換します。記事化時は文字起こしテキストをGemini APIへ送信し、従量課金の可能性があります。",
    )
    result.add_argument("input", type=Path, metavar="INPUT")
    result.add_argument("--from-transcript", action="store_true", help="既存の .transcript.md / .transcript.failed.md から記事のみ生成（音声処理を省略）")
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--quality-check", action="store_true", help="原文・生成ノート・重要事項・指摘をGeminiへ送り独立照合（通常2回、修正時4回/分割）。品質レポートを保存")
    modes.add_argument("--note-pipeline", action="store_true", help="全台帳から章構成を計画し、構造化本文・境界を照合。台帳・対応・使用量を品質レポートに保存")
    result.add_argument("--max-api-calls", type=int, help="台帳方式のAPI工程上限（既定100、通信再試行別）。料金上限ではありません")
    result.add_argument("--output-dir", type=Path, help="出力先（既定: MP4と同じフォルダ）")
    result.add_argument("--language", default="ja", help="認識言語（既定: ja）")
    result.add_argument("--whisper-model", default="small", help="faster-whisperモデル（既定: small）")
    result.add_argument("--article-model", default="gemini-3.5-flash-lite", help="Geminiモデル（既定: gemini-3.5-flash-lite）")
    result.add_argument("--no-article", action="store_true", help="APIを使わず時刻付き原文のみ保存")
    result.add_argument("--overwrite", action="store_true", help="既存出力を置換")
    return result


def run(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.max_api_calls is not None and (not args.note_pipeline or args.max_api_calls < 1):
        print("設定エラー: --max-api-calls は --note-pipeline と併用し、正の整数を指定してください。", file=sys.stderr)
        return 2
    if (args.quality_check or args.note_pipeline) and args.no_article:
        print("設定エラー: 品質確認・台帳方式と --no-article は併用できません。", file=sys.stderr)
        return 2
    if args.from_transcript:
        return run_from_transcript(args)
    settings = Settings(args.input, args.output_dir or args.input.parent, args.language,
                        args.whisper_model, args.article_model, args.no_article, args.overwrite)
    try:
        api_key = None if settings.no_article else gemini_api_key()
        validate(settings, api_key)
        if args.quality_check or args.note_pipeline:
            validate_quality_target(settings.article_path, args.input, args.overwrite,
                                    [settings.transcript_path, settings.rescue_path])
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
        parts = selected_generation(args, segments, api_key, settings.article_path)
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
        if args.quality_check or args.note_pipeline:
            validate_quality_target(target, args.input, args.overwrite)
    except (ConfigurationError, OSError) as exc:
        message = str(exc) if isinstance(exc, ConfigurationError) else "入力または出力先を確認できません。"
        print(f"設定エラー: {message}", file=sys.stderr)
        return 2
    print("既存の文字起こしテキストのみGemini APIへ送信して記事化します。従量課金の可能性があります。", file=sys.stderr)
    try:
        parts = selected_generation(args, segments, api_key, target)
        write_atomic(target, article_markdown(target, parts), args.overwrite)
    except (ArticleError, OutputError) as exc:
        print(f"処理エラー: {exc}。入力原文は保持されています。", file=sys.stderr)
        return 1
    print(f"記事を保存しました: {target}")
    return 0


def validate_quality_target(target, source, overwrite, other_outputs=()):
    report = target.with_suffix(".quality.md")
    try:
        for protected in (source, target, *other_outputs):
            if report.resolve() == protected.resolve() or (report.exists() and protected.exists() and report.samefile(protected)):
                raise ConfigurationError("品質レポートが入力または他の出力と競合します。")
        if report.exists() and (not overwrite or not report.is_file()):
            raise ConfigurationError("品質レポートが既存です。更新には --overwrite が必要です。")
    except OSError as exc:
        raise ConfigurationError("品質レポートの出力先を確認できません。") from exc


def selected_generation(args, segments, api_key, target):
    if not args.quality_check and not args.note_pipeline:
        return generate_parts(segments, api_key, args.article_model)
    if args.note_pipeline:
        print("台帳方式: 原文・台帳・章構成・本文・指摘をGeminiへ送信します。工程数は章構成に依存し、通信は各工程最大3試行。使用量と対応を品質レポートに保存します。", file=sys.stderr)
        result = generate_pipeline(segments, api_key, args.article_model,
                                   max_calls=args.max_api_calls or 100,
                                   progress=lambda message: print(message, file=sys.stderr))
    else:
        print("品質確認: 原文・生成ノート・重要事項・指摘をGeminiへ送信します。通常2回、修正時4回/分割（通信再試行別）。", file=sys.stderr)
        result = generate_quality(segments, api_key, args.article_model)
    try:
        write_atomic(target.with_suffix(".quality.md"), result.markdown(api_key), args.overwrite)
    except OutputError as exc:
        raise ArticleError("品質レポートを保存できないため記事を確定しません。") from exc
    if not result.accepted:
        raise ArticleError("品質確認が未合格または処理失敗です。品質レポートを確認してください。")
    return result.parts


if __name__ == "__main__":
    raise SystemExit(run())
