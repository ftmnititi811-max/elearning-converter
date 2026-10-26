from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    input_path: Path
    output_dir: Path
    language: str
    whisper_model: str
    article_model: str
    no_article: bool
    overwrite: bool

    @property
    def article_path(self) -> Path:
        return self.output_dir / f"{self.input_path.stem}.md"

    @property
    def transcript_path(self) -> Path:
        return self.output_dir / f"{self.input_path.stem}.transcript.md"

    @property
    def rescue_path(self) -> Path:
        return self.output_dir / f"{self.input_path.stem}.transcript.failed.md"


def validate(settings: Settings, api_key: str | None) -> None:
    source = settings.input_path
    if not source.is_file():
        raise ConfigurationError("入力MP4が見つかりません。")
    if source.suffix.lower() != ".mp4":
        raise ConfigurationError("入力はMP4ファイルを指定してください。")
    if not settings.language.strip() or not settings.whisper_model.strip() or not settings.article_model.strip():
        raise ConfigurationError("言語とモデル名は空にできません。")
    if not settings.no_article and not api_key:
        raise ConfigurationError("GEMINI_API_KEY が設定されていません。")
    try:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError("出力先フォルダを作成できません。") from exc
    if not settings.output_dir.is_dir():
        raise ConfigurationError("出力先はフォルダではありません。")
    targets = [settings.transcript_path]
    if not settings.no_article:
        targets.append(settings.article_path)
    for target in targets:
        if target.exists() and not settings.overwrite:
            raise ConfigurationError(f"既存出力があります。更新するには --overwrite を指定してください: {target.name}")
        if target.resolve() == source.resolve():
            raise ConfigurationError("元MP4を出力先に指定できません。")
