"""Plan lecture sections using all audited facts, then retrieve bounded evidence."""
import json

from .article import ArticleError, LIMIT
from .note_pipeline import STRING, array, context_source, fail, ids, is_text, shape
from .quality import object_schema


PLAN_LIMIT = 120_000
PLAN_SCHEMA = object_schema({"sections": array(object_schema({
    "heading": STRING, "fact_ids": array(STRING, True),
}), True)})
PLAN = """全台帳から講義順に章構成を計画する。sectionsのheadingは内容を直接示す短い見出し。
各章のfact_idsに全情報IDを一度ずつ、与えたfactsの順番通りに配置する。
抽出窓の境界にこだわらず、関連する定義・前提・理由・手順・例を一つの章へまとめる。
過度に細かい見出し、各窓ごとの導入やまとめ、空の章を作らない。
長い章はローカルで分割できるので内容を削らない。新しい事実や本文は生成しない。
"""


def validate_plan(value, facts):
    shape(value, ("sections",))
    if not isinstance(value["sections"], list) or not value["sections"]:
        fail()
    ordered = []
    for section in value["sections"]:
        shape(section, ("heading", "fact_ids"))
        if not is_text(section["heading"]):
            fail()
        ids(section["fact_ids"], facts)
        ordered.extend(section["fact_ids"])
    if ordered != list(facts):
        fail()
    return value


def plan_data(facts):
    data = {"facts": list(facts.values())}
    if len(json.dumps(data, ensure_ascii=False)) > PLAN_LIMIT:
        raise ArticleError("章構成の資料が暫定上限120000文字を超えます。短い原文に分けて評価してください。")
    return data


def drafting_jobs(plan, facts, sources, limit=LIMIT):
    jobs = []
    for index, section in enumerate(plan["sections"], 1):
        selected, evidence = [], set()

        def append():
            jobs.append({"section_id": f"H{index:04d}", "heading": section["heading"],
                         "fact_ids": list(selected), "source_ids": [s for s in sources if s in evidence]})

        for fact_id in section["fact_ids"]:
            proposed = evidence | set(facts[fact_id]["source_ids"])
            cost = sum(len(sources[s]["text"]) + 64 for s in proposed)
            if cost > limit and selected:
                append()
                selected, evidence = [], set()
                proposed = set(facts[fact_id]["source_ids"])
                cost = sum(len(sources[s]["text"]) + 64 for s in proposed)
            if cost > limit:
                raise ArticleError("一つの情報の根拠原文が本文生成上限を超えます。")
            selected.append(fact_id)
            evidence = proposed
        if selected:
            append()
    return jobs


def job_data(job, facts, sources):
    ordered = list(sources)
    positions = [ordered.index(s) for s in job["source_ids"]]
    # IDs are global; reference-only context never becomes allowed body evidence.
    before = ordered[max(0, min(positions)-2):min(positions)]
    after = ordered[max(positions)+1:max(positions)+3]
    return {"source": [sources[s] for s in job["source_ids"]],
            "before": [context_source(sources[s], before=True) for s in before],
            "after": [context_source(sources[s]) for s in after],
            "ledger": {"facts": [facts[f] for f in job["fact_ids"]], "excluded": []}}
