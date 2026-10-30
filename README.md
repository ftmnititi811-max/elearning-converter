# lecture-md

利用権限のあるMP4講義の**音声のみ**を、Windows上で復習用Markdownに変換するCLIです。FFmpegで音声を一時WAVへ抽出し、faster-whisperでローカル文字起こしを行います。通常モードでは、**時刻付き文字起こしテキストのみ**をGemini APIへ送って記事化します。MP4・WAVはAPIに送信しません。

## 導入（Windows 11、Python 3.11以降）

1. Python 3.11以降をインストールし、`py --version` で確認します。
2. FFmpegとffprobeをインストールしてPATHに追加し、`ffmpeg -version` と `ffprobe -version` で確認します。
3. PowerShellでこのフォルダに移動し、仮想環境とパッケージを用意します。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```


## 実行

```powershell
# APIを使わず、時刻付き原文のみ
python -m lecture_md "C:\講義 資料\授業.mp4" --no-article

# 記事と原文を生成。キーは環境変数で渡します
$env:GEMINI_API_KEY = "取得したキー"
python -m lecture_md "C:\講義 資料\授業.mp4" --output-dir "C:\復習 ノート"
```

[.env.example](.env.example) にキーの記入形式を示しています。プロジェクトルートの `.env` に `GEMINI_API_KEY=取得したキー` と書くと、CLIが自動で読み込みます。環境変数が設定されている場合はそちらを優先します。実際の `.env` はGitの管理対象から除外されています。

通常モードは `<stem>.md` と `<stem>.transcript.md` をUTF-8で保存します。`--no-article` は原文だけを保存し、キーは不要です。既存出力を置換する場合だけ `--overwrite` を付けます。記事化失敗時は、可能なら `<stem>.transcript.failed.md` に原文を救済して終了コード1で終了します。既存の救済ファイルも `--overwrite` なしでは置換しません。

オプション: `--language`（既定 `ja`）、`--whisper-model`（既定 `small`、CPU/int8）、`--article-model`（既定 `gemini-3.5-flash-lite`）、`--output-dir`、`--no-article`、`--overwrite`。`python -m lecture_md --help` でも確認できます。終了コードは成功0、入力・設定不備2、処理失敗1です。

## 送信・費用・権利

Gemini APIには、文字起こしテキストを約12,000文字ごとの時刻境界で分割して送ります。分割数だけAPIリクエストが発生し、従量課金やクォータ消費の可能性があります。**料金上限はCLIで設定しません。** 利用前に使用モデルの料金・クォータを確認してください。通信なしで使う場合は `--no-article` を指定します。

既定モデル `gemini-3.5-flash-lite` の料金と利用条件は [Gemini API公式料金表](https://ai.google.dev/gemini-api/docs/pricing) と [モデル仕様](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) で確認してください。モデルを変える場合は `--article-model MODEL` を指定します。

教材の加工・外部送信・共有が利用条件と著作権上許されているか確認してください。機密情報や個人情報を含む可能性がある場合、Geminiへ送る前に内容を確認してください。記事は自動生成した下書きです。専門語、数式、重要な断定は原音声と時刻付き原文で照合してください。キーや文字起こし本文をCLIログには表示しません。

## トラブル対応

- `FFmpegまたはffprobeが見つかりません`: 両コマンドがPATH上にあるか確認します。
- `GEMINI_API_KEY が設定されていません`: プロジェクトルートの `.env` または同じPowerShellセッションの環境変数にキーを設定します。`--no-article` では不要です。
- `文字起こし結果が空です`: 音声が聞こえるMP4か確認します。無音や短すぎる音声は失敗します。
- `既存出力があります`: 別の出力先を選ぶか、更新を意図する場合のみ `--overwrite` を付けます。
- Gemini障害・形式不正: 記事は完成名で保存されません。救済原文がある場合はそこから内容を確認できます。

## 開発・検証

```powershell
python -m unittest discover -s tests -v
```

自動テストは外部APIと講義素材を使わないモックです。権利確認済みの短尺日本語MP4と音声なしMP4を別途用意し、音声抽出・CPU認識・日本語/空白入りパスを確認してください。実APIテストは課金を伴うため明示的に実行してください。モデル、分割長、再試行回数は暫定値です。5～10分の素材で認識誤り、時刻、記事の忠実性、処理時間、API利用量を記録してから調整します。
