"""Audited ledgers -> global outline -> evidence-grounded blocks -> seam checks."""
import json
import copy
import time
from dataclasses import dataclass, field

from .article import ArticleError, _api_failure, load_prompt, parse_response
from .note_blocks import STRUCTURED_SCHEMA, block_markdown, escape, validate_structured
from .note_pipeline import (AUDIT_LEDGER, AUDIT_SCHEMA, EXTRACT, LEDGER_SCHEMA, POLICY,
                            PipelineResult, source_chunks, validate_audit, validate_ledger)
from .note_structure import PLAN, PLAN_SCHEMA, drafting_jobs, job_data, plan_data, validate_plan


WRITE = """固定した章構成と対象ledgerから授業ノートを作る。根拠はsourceのみ。
全体構成やbefore/afterにある他の情報は本文対象ではなく文脈である。
各ブロックに内容を実際に反映したfact_idsと、その全根拠のsource_idsを付ける。
対象ledgerの全情報を本文に反映し、他ジョブの情報は再掲しない。台帳は変更禁止。
不確かな説明は本文でも留保し、checksへ原文IDと時刻・確認内容を記す。
以下の外部プロンプトの文章方針を使うが、応答構造は最後の構造指示を優先する。
"""
BLOCK_INSTRUCTIONS = """
今回の応答はblocks/checksだけのJSON。body/heading/text/timestamp形式では返さない。
blocksのkindはparagraph/bullets/steps/table/codeから内容に適したものを選ぶ。
paragraph/codeはtext、bullets/stepsはitems、tableはcolumnsとrowsを使う。
codeだけlanguageに短い言語名を入れる。他の未使用欄は空文字/空配列。
見出しやMarkdownの装飾記号はテキストに入れない。整形はローカルで行う。
表の各行は列数を一致させ、原文にない値や行を補わない。
複雑な手順と具体例は適切な手順/表/コードで整理し、短さを目標にしない。
"""
REVIEW = """原文と構造化本文を独立に双方向照合してissuesを返す。
sourceに含まれても対象ledgerに属さない他ジョブの情報は今回の掲載対象ではない。
対象情報の意味・手順・条件・例・停止が残っているか、本文の各主張に根拠があるか検査。
IDがあるだけの概要化、不明語の断定、数値変更、概念混同、創作はmajor。
台帳自体の見落としも原文と照合する。適切な留保は問題にしない。
軽微な文体はminor。source_idsと具体的な問題・修正方針textを返す。
"""
SEAM = """隣接する2つの本文を原文と章構成に照らして独立に照合する。
情報の不必要な重複、別概念の混同、定義の食い違い、根拠のない説明の接続を検査。
長い説明の自然な継続、同じ時刻の別発話、原文の矛盾を適切に留保したものは問題にしない。
重大な問題はmajor、軽微ならminor。issuesに原文のsource_idsと具体的なtextを返す。
"""
TOKEN_FIELDS = ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "cached_content_token_count")


def ledger_schema(source_ids):
    schema = copy.deepcopy(LEDGER_SCHEMA)
    props = schema["properties"]
    props["facts"]["items"]["properties"]["source_ids"]["items"]["enum"] = list(source_ids)
    props["excluded"]["items"]["properties"]["source_id"]["enum"] = list(source_ids)
    return schema


def note_schema(fact_ids, source_ids):
    schema = copy.deepcopy(STRUCTURED_SCHEMA)
    props = schema["properties"]["blocks"]["items"]["properties"]
    props["fact_ids"]["items"]["enum"] = list(fact_ids)
    props["source_ids"]["items"]["enum"] = list(source_ids)
    return schema


@dataclass
class WorkflowResult(PipelineResult):
    review_label = "照合"
    plan: dict = field(default_factory=dict)
    jobs: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    max_calls: int = 100
    elapsed_seconds: float = 0

    def markdown(self, secret):
        text = super().markdown("")
        totals, missing = {}, {}
        for key in TOKEN_FIELDS:
            totals[key] = sum(s["tokens"].get(key, 0) for s in self.stages)
            missing[key] = sum(key not in s["tokens"] for s in self.stages)
        details = {"plan": self.plan, "jobs": self.jobs, "max_api_calls": self.max_calls,
                   "elapsed_seconds": round(self.elapsed_seconds, 3),
                   "observed_tokens": totals, "usage_unknown_attempts_by_field": missing,
                   "stages": self.stages}
        text += "\n## 章構成と実行記録\n\n"
        text += "工程上限は料金上限ではありません。使用量が不明な試行を含む場合、取得分の合計は完全な請求量を表しません。\n\n"
        text += "\n".join("    " + line for line in json.dumps(details, ensure_ascii=False, indent=2).splitlines()) + "\n"
        return text.replace(secret, "[REDACTED]") if secret else text


class Runner:
    def __init__(self, client, result, model, pause, progress):
        self.client, self.result, self.model = client, result, model
        self.pause, self.progress = pause, progress

    def call(self, instruction, data, schema, validator, index, total, stage):
        if self.result.calls >= self.result.max_calls:
            raise ArticleError("API工程上限に達したため追加送信せず停止します。")
        self.result.calls += 1
        if self.progress:
            self.progress(f"台帳方式: {stage} {index}/{total}（工程 {self.result.calls}/{self.result.max_calls}）")
        for attempt in range(3):
            self.result.attempts += 1
            started = time.monotonic()
            entry = {"stage": stage, "index": index, "attempt": attempt+1, "tokens": {}, "received": False}
            self.result.stages.append(entry)
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=POLICY + instruction + "\n資料JSON:\n" + json.dumps(data, ensure_ascii=False),
                    config={"response_mime_type": "application/json", "response_json_schema": schema,
                            "http_options": {"retry_options": {"attempts": 1}}})
                entry["received"] = True
                usage = getattr(response, "usage_metadata", None)
                for key in TOKEN_FIELDS:
                    count = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
                    if type(count) is int and count >= 0:
                        entry["tokens"][key] = count
                break
            except Exception as exc:
                detail, retryable = _api_failure(exc)
                if getattr(exc, "code", None) == 429:
                    raise ArticleError("無料枠・レート制限に達したため追加送信せず停止します（HTTP 429）。") from exc
                if not retryable or attempt == 2:
                    raise ArticleError(f"台帳方式API呼出し失敗（工程={stage}、{detail}）。") from exc
            finally:
                entry["seconds"] = round(time.monotonic() - started, 3)
            self.pause(2 ** attempt)
        try:
            return parse_response(response, index, total, validator)
        except ArticleError as exc:
            raise ArticleError(f"台帳方式の応答失敗（工程={stage}）：{exc}") from exc


def generate_workflow(segments, api_key, model_name, client_factory=None, pause=time.sleep,
                      max_calls=100, progress=None):
    result = WorkflowResult(model_name, max_calls=max_calls)
    started = time.monotonic()
    try:
        if type(max_calls) is not int or max_calls < 1:
            raise ArticleError("API工程上限は正の整数が必要です。")
        prompt = load_prompt()
        chunks = source_chunks(segments)
        if not chunks:
            raise ArticleError("台帳方式の入力が空です")
        try:
            if client_factory is None:
                from google import genai
                client_factory = genai.Client
            client = client_factory(api_key=api_key)
        except Exception as exc:
            raise ArticleError("Geminiクライアントを開始できません。") from exc
        runner = Runner(client, result, model_name, pause, progress)
        facts, sources = {}, {}

        def audit(instruction, data, allowed, index, total, stage):
            response = runner.call(instruction, data, AUDIT_SCHEMA,
                                   lambda v: validate_audit(v, allowed), index, total, stage)
            result.reviews.append({f"{stage} {index}/{total}": [
                {"severity": issue["severity"], "timestamp": allowed[issue["source_ids"][0]]["timestamp"],
                 "text": ", ".join(issue["source_ids"]) + " " + issue["text"]} for issue in response["issues"]]})
            return response

        def major(review):
            return any(i["severity"] == "major" for i in review["issues"])

        # All extraction and audits finish before any prose is generated.
        for index, chunk in enumerate(chunks, 1):
            core = {s["id"]: s for s in chunk["source"]}
            sources.update(core)
            ledger = runner.call(EXTRACT, chunk, ledger_schema(core),
                                 lambda v: validate_ledger(v, core), index, len(chunks), "情報抽出")
            record = {"sources": [{k: v for k, v in s.items() if k != "text"} for s in core.values()],
                      "ledger": ledger}
            result.ledgers.append(record)
            if major(audit(AUDIT_LEDGER, {**chunk, "ledger": ledger}, core, index, len(chunks), "台帳照合")):
                raise ArticleError("台帳に重大な問題があるため記事は保存しません。")
            for fact in ledger["facts"]:
                qualified = {**fact, "id": f"C{index:04d}.{fact['id']}"}
                facts[qualified["id"]] = qualified
        if not facts:
            raise ArticleError("台帳に本文対象の情報がありません。記事は保存しません。")
        positions = {source_id: n for n, source_id in enumerate(sources)}
        facts = dict(sorted(facts.items(), key=lambda item: min(positions[s] for s in item[1]["source_ids"])))
        result.plan = runner.call(PLAN, plan_data(facts), PLAN_SCHEMA,
                                  lambda v: validate_plan(v, facts), 1, 1, "章構成")
        jobs = drafting_jobs(result.plan, facts, sources)
        result.jobs = jobs
        completed = []
        for index, job in enumerate(jobs, 1):
            data = job_data(job, facts, sources)
            core = {s["id"]: s for s in data["source"]}
            # Only headings/IDs are needed as global context during drafting.
            data.update({"outline": result.plan, "target": job})
            validator = lambda v: validate_structured(v, data["ledger"], core)
            schema = note_schema(job["fact_ids"], core)
            instruction = WRITE + prompt + BLOCK_INSTRUCTIONS
            note = runner.call(instruction, data, schema, validator, index, len(jobs), "本文生成")

            def remember():
                job["block_mapping"] = [{"block": n, "fact_ids": b["fact_ids"], "source_ids": b["source_ids"]}
                                        for n, b in enumerate(note["blocks"], 1)]

            remember()
            review = audit(REVIEW, {**data, "note": note}, core, index, len(jobs), "本文照合")
            if review["issues"]:
                note = runner.call(instruction + "\n指摘箇所を修正し、他の対象情報も保持する。",
                                   {**data, "note": note, **review}, schema, validator,
                                   index, len(jobs), "本文修正")
                remember()
                review = audit(REVIEW, {**data, "note": note}, core, index, len(jobs), "本文再照合")
            if major(review):
                raise ArticleError("本文に重大な問題が残るため記事は保存しません。")
            completed.append({"job": job, "note": note})
            body = []
            first_section_job = index == 1 or jobs[index-2]["section_id"] != job["section_id"]
            for block_index, block in enumerate(note["blocks"]):
                references = list(dict.fromkeys(sources[s]["timestamp"] for s in sources if s in block["source_ids"]))
                body.append({"heading": escape(job["heading"]) if first_section_job and block_index == 0 else "",
                             "section_id": job["section_id"], "text": block_markdown(block),
                             "timestamp": references[0], "references": references})
            result.parts.append({"body": body, "checks": note["checks"]})
        for index in range(1, len(completed)):
            pair = completed[index-1:index+1]
            allowed = {s: sources[s] for item in pair for s in item["job"]["source_ids"]}
            review = audit(SEAM, {"source": list(allowed.values()), "outline": result.plan, "pair": pair},
                           allowed, index, len(completed)-1, "境界照合")
            if major(review):
                raise ArticleError("章・本文境界に重大な問題があるため記事は保存しません。")
        result.accepted = True
    except ArticleError as exc:
        result.error = str(exc)
    finally:
        result.elapsed_seconds = time.monotonic() - started
    return result
