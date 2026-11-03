# 設計

構成：CLI・設定検証 → ffprobe・FFmpeg音声抽出 → faster-whisper認識 → 原文Markdown → 分割・Gemini API記事化 → 検証・保存。

情報保持の改善（R-18 / A-18、A-19 / T-24、T-25）：prompts/article.mdに手順の保持、必要な例、概念の区別、不確実性の局所化、実説明への時刻付けと生成内の点検を指示する。既存のbody/checksで疑似コード・表・留保・時刻を表現できるため、PythonのAPI呼出し・スキーマ・rendererは変更しない。意味の正しさはJSON検証だけでは保証できないため、架空の原文・期待応答・レビュー観点をdocs/note-fidelity-example.mdに記録し、モック経路の検証と実生成の評価を分ける。教材はテスト素材へ転記しない。

仮定と検証方法：分割内の原文だけで復元可能な情報を保持する。分割をまたぐ未完の説明は確認事項に残す方式を採り、実生成評価時に欠落を確認する。分割長・モデル・出力トークン上限・保存形式・外部送信範囲は維持する。プロンプトと生成文の増加による費用変動および打切りの可能性は実API評価で測定し、今回のオフライン検証では未確定とする。

授業ノート化（R-04 / A-15、A-16 / T-21）：APIの固定summary・termsを廃止し、bodyとchecksを検証する。body.headingは空文字可、textとtimestampは非空、時刻の分・秒は00～59。プロンプトに必要十分な説明・定型文削減・選択的な構造化・句読点の省略を指示する。rendererは段落ごとの見出しを強制せず、連続する同一headingを省いて本文と時刻を保持する。表・リスト・コードを含む本文の時刻は別段落としMarkdownを壊さない。checksは完全一致のみ重複除去。句読点は生成時に判断し、出力後の一括削除はしない。追加のAPI呼び出しや講義全体の再送信はしない。プロンプト長・生成量による料金の変動はあり、費用の実測は別途行う。

配置修正（R-15、16 / A-13、14 / T-22）：既存srcを移動せず、setuptoolsでelearning_converterパッケージをsrcへ明示的に割り当てる。READMEのeditable installを対象とし、プロンプトはリポジトリ直下から読む。wheel単独配布は今回の対象外。

想定モジュール：elearning_converter/__main__.py（CLI）、config.py（設定）、media.py（音声）、transcribe.py（認識）、article.py（API）、render.py（Markdown）、output.py（保存）。tests/ にテスト、README.md に導入手順。

Segment(start_seconds, end_seconds, text) を基本データとし、時刻妥当性と空テキストを検査する。パスは pathlib.Path で扱い、FFmpegはシェルを介さず引数配列で起動。一時WAVは一時ディレクトリで管理。APIへは文字起こしテキストのみ送る。長い講義は時刻境界で分割・統合する。音声内の命令は講義内容として扱い、操作指示として実行しない。API応答の空・形式不備を検査し、内容は原文と人手で照合する。
文字起こしの import、モデル初期化、WAV読み込み、`transcribe` 呼び出し、遅延評価される結果列の処理を個別に捕捉する。FFmpegの出力はwaveで16 kHz・モノラル・16-bit PCMを確認し、NumPy float32配列（-1～1）へ変換して渡す。これによりPyAVの音声デコーダーに依存しない。表示には固定の段階名と例外クラス名だけを使い、例外メッセージは出さない。Segment の時刻不正は従来の専用エラーを維持する。

Gemini公式 `google-genai` SDKでJSON形式の分割記事を取得する。APIへは文字起こしテキストのみを送る。分割記事の統合はローカルで行い、追加API送信をしない。衝突を処理前に検査し、出力先の一時ファイルへ書いてから置換し、記事を最後に確定。2ファイルの完全な同時確定はできないため障害時の部分成功を明示。記事化失敗時は `<stem>.transcript.failed.md` に原文を救済し、記事の最終名は作成しない。キーは環境変数 GEMINI_API_KEY を優先し、未設定時だけ `__main__.py` の親にあるプロジェクトルートの `.env` から同名の値を読む。`.env` は単純な `KEY=value` 形式として扱い、他の設定やシェル構文は評価しない。キーをコード・引数・ログ・Markdownへ記録しない。本文もログへ出さない。外部送信と従量課金をCLI・READMEで明示し --no-article を提供。

API例外の `code` は数値HTTPステータスとして100～599に限って表示する。表示する例外クラス名も英数字とアンダースコアに限定し、SDKの `message`・`details`・`status`・例外文字列は表示しない。HTTP 408、429、5xx、httpxの通信・タイムアウト例外のみ再試行する。モックで回数と秘密情報非表示を確認し、実APIは送信しない。

既定モデルは `gemini-3.5-flash-lite` とし、CLI引数の `--article-model` で上書きできる。モデルの費用と利用条件はREADMEから公式資料を参照する。API呼び出しはモックで検証し、実APIへの講義テキスト送信は行わない。

命名（R-15 / A-13）：配布名を `elearning-converter` とし、`[project.scripts]` の `elconv` を `elearning_converter.__main__:run` に接続する。Pythonパッケージ、テストのimport・モック先、CLIヘルプ、一時ファイルの接頭辞も名称に合わせる。モジュール起動と短縮コマンドは同じ `run` を実行する。

## テスト計画

応答堅牢化（R-17 / A-17 / T-23）：article.pyにJSON Schemaと応答解析関数を追加する。分割数はsplit_segmentsを一度実行して取得。解析前にprompt_feedback.block_reasonとcandidates[0].finish_reasonを確認し、既知値だけ診断へ採用する。JSONDecodeErrorの生メッセージを出さず行・列だけ表示。validate_partは固定の項目別理由を返し、分割番号を付けてCLIへ伝える。ブロック、打切り、空本文、構文、構造、時刻をSDK実オブジェクトとモックで検証し、正常JSONでもMAX_TOKENSなら拒否する。実API呼出しや応答保存は行わない。

プロンプト外部化（R-16 / A-14 / T-20、T-22）：`article.py` の `load_prompt()` が `Path(__file__).resolve().parent.parent / "prompts" / "article.md"` を `utf-8-sig` で読み込む。空白のみの内容、`OSError`、`UnicodeError` は固定メッセージの `ArticleError` にする。`generate_parts()` 内でクライアント初期化前に一度読み、`contents=prompt + chunk` に使用する。モジュールimport時には読まないため `--no-article` に影響しない。プロンプトはR-04の授業ノート形式へ更新する。API設定・分割・再試行は維持する。ソース配置とREADMEの `pip install -e .` を対象とし、プロンプトはソースと一緒に配置する。

| 層 | 確認 |
| --- | --- |
| 単体 | 引数、時刻、分割境界、Markdown、上書き、空結果 |
| モック結合 | FFmpeg・認識・APIの失敗、再試行、キー欠落、ログ |
| ローカル結合 | 権利確認済み短尺MP4、音声なしMP4、日本語・空白パス、CPU認識 |
| 人手受入 | 5～10分の講義を原音声と照合し、欠落、誤断定、専門語、時刻、可読性を記録 |

実APIテストは課金を伴うため明示的に実行。通常の自動テストはモック。モデル、分割長、再試行、救済名は最初の実装時に選びサンプル評価後に確定する。

## 原文入力経路（R-19 / A-20 / T-26）

専用ローダーで原文行を厳密に読みSegmentへ変換する。終了時刻は記事化に使わないため開始時刻と同値とする。CLIで専用経路へ分岐し、既存generate_parts・article_markdown・write_atomicを再利用する。入力検証と出力衝突検査後にAPIを呼ぶ。原文を再保存せず追加送信も行わない。
