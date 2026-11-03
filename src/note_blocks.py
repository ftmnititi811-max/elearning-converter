"""Typed note blocks and deterministic Markdown rendering."""
import re

from .note_pipeline import STRING, array, fail, is_text, shape, validate_note
from .quality import object_schema


KINDS = ("paragraph", "bullets", "steps", "table", "code")
BLOCK_SCHEMA = object_schema({
    "kind": {"type": "string", "enum": list(KINDS)},
    "text": STRING, "items": array(STRING), "columns": array(STRING),
    "rows": array(array(STRING)), "language": STRING,
    "fact_ids": array(STRING, True), "source_ids": array(STRING, True),
})
STRUCTURED_SCHEMA = object_schema({"blocks": array(BLOCK_SCHEMA, True), "checks": array(STRING)})


def escape(text):
    return re.sub(r"([\\`*_{}\[\]<>#!|~+.\-])", r"\\\1", text).replace("\r", "").replace("\n", " ")


def validate_block(block):
    shape(block, BLOCK_SCHEMA["properties"])
    if (block["kind"] not in KINDS or not isinstance(block["text"], str)
            or not isinstance(block["language"], str)
            or any(not isinstance(block[k], list) for k in ("items", "columns", "rows"))):
        fail()
    kind = block["kind"]
    if kind in ("paragraph", "code"):
        if not is_text(block["text"]) or block["items"] or block["columns"] or block["rows"]:
            fail()
    elif block["text"]:
        fail()
    if kind in ("bullets", "steps"):
        if not block["items"] or any(not is_text(x) for x in block["items"]) or block["columns"] or block["rows"]:
            fail()
    elif block["items"]:
        fail()
    if kind == "table":
        if (not block["columns"] or any(not is_text(x) for x in block["columns"]) or not block["rows"]
                or any(not isinstance(row, list) or len(row) != len(block["columns"])
                       or any(not isinstance(cell, str) for cell in row) for row in block["rows"])):
            fail()
    elif block["columns"] or block["rows"]:
        fail()
    if kind == "code":
        if not re.fullmatch(r"[A-Za-z0-9_+-]{0,32}", block["language"]):
            fail()
    elif block["language"]:
        fail()
    return block


def block_markdown(block):
    validate_block(block)
    kind = block["kind"]
    if kind == "paragraph":
        return escape(block["text"])
    if kind in ("bullets", "steps"):
        return "\n".join(f"{str(i) + '.' if kind == 'steps' else '-'} {escape(text)}"
                         for i, text in enumerate(block["items"], 1))
    if kind == "table":
        rows = [block["columns"], ["---"] * len(block["columns"]), *block["rows"]]
        return "\n".join("| " + " | ".join(escape(cell) for cell in row) + " |" for row in rows)
    longest = max((len(x) for x in re.findall(r"`+", block["text"])), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{block['language']}\n{block['text'].rstrip()}\n{fence}"


def validate_structured(value, ledger, allowed):
    shape(value, ("blocks", "checks"))
    if not isinstance(value["blocks"], list) or not value["blocks"]:
        fail()
    body = []
    for block in value["blocks"]:
        validate_block(block)
        body.append({"heading": "", "text": block_markdown(block),
                     "fact_ids": block["fact_ids"], "source_ids": block["source_ids"]})
    validate_note({"body": body, "checks": value["checks"]}, ledger, allowed)
    return value
