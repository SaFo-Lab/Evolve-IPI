"""The audit of Inkling-Small's raw generations.

Inkling-Small's markup is made of special tokens, and a call that the parser cannot read looks exactly like a model
that refused. So every raw generation of a run is read once more after the run: a generation with tool-call markup
that parses to no call is a parser failure, unless the call was cut off at the token limit, and so is a training row
whose recorded count of calls (what the trainer's bridge handed the agent) differs from this reading.

Two kinds of dump are audited:

    training     ``<record-dir>/raw_train.jsonl`` of the trainer; each row carries ``parsed_calls``
                 (``parsed_key="parsed_calls"``)
    model server the raw dump of the model server (evaluation, skill evolution); its rows are only parsed again
                 (``parsed_key=None``)

Reported, never a problem: calls cut off at the token limit, reasoning left open, markup written inside reasoning,
one-token generations of a one-message request (a session's closing request), and text outside every content segment.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evolve_ipi.models.formats.inkling import InklingFormatter

__all__ = ["audit_problems", "audit_raw_dump"]

_FORMATTER = InklingFormatter()
_EXAMPLES = 3


def _rows(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """The generations of a dump, and whether its last line was cut off.

    A missing dump holds none. A last line without its newline is still being written, or was cut off when the
    process that wrote it was stopped; it is not a generation. Any other line that is not a generation raises
    ``ValueError``.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], False
    complete = text[: text.rfind("\n") + 1]
    rows = []
    for number, line in enumerate(complete.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise ValueError(f"line {number} of {path} is not JSON") from None
        if not isinstance(row, dict) or not isinstance(row.get("raw"), str):
            # a bad line of the data, like one that is not JSON: callers catch ValueError for both
            raise ValueError(f"line {number} of {path} is not a generation")  # noqa: TRY004
        rows.append(row)
    return rows, bool(text[len(complete) :].strip())


def audit_raw_dump(path: Path, *, parsed_key: str | None) -> dict[str, Any]:
    """Audit the raw dump at ``path``; ``parsed_key`` names the field of each row that holds the count of calls the
    trainer's bridge parsed (``None``: the rows carry none). Returns the counts and up to three examples."""
    path = Path(path)
    audit: dict[str, Any] = {
        "where": str(path),
        "generations": 0,
        "with_calls": 0,
        "unparsed_markup": 0,
        "truncated_calls": 0,
        "reasoning_unclosed": 0,
        "closure_one_token": 0,
        "markup_in_reasoning": 0,
        "unmarked_text": 0,
        "parse_mismatch": 0,
        "examples": [],
    }
    rows, unfinished = _rows(path)
    audit["unfinished_last_line"] = unfinished
    examples: list[dict[str, Any]] = audit["examples"]
    for row in rows:
        raw = row["raw"]
        calls, _ = _FORMATTER.parse_output(raw)
        audit["generations"] += 1
        audit["with_calls"] += bool(calls)
        if parsed_key is not None and row.get(parsed_key) != len(calls):
            audit["parse_mismatch"] += 1
            if len(examples) < _EXAMPLES:
                examples.append(
                    {"kind": "parse_mismatch", "bridge": row.get(parsed_key), "audit": len(calls), "tail": raw[-800:]}
                )
        if _FORMATTER.call_markup(raw) and not calls:
            if _FORMATTER.truncated_call(raw):
                audit["truncated_calls"] += 1
            else:
                audit["unparsed_markup"] += 1
                if len(examples) < _EXAMPLES:
                    examples.append({"kind": "unparsed", "tail": raw[-1500:]})
        audit["reasoning_unclosed"] += not _FORMATTER.reasoning_closed(raw)
        audit["closure_one_token"] += row.get("n_tokens") == 1 and row.get("messages") == 1
        audit["markup_in_reasoning"] += bool(_FORMATTER.markup_in_reasoning(raw))
        unmarked = _FORMATTER.unmarked_text(raw)
        if unmarked:
            audit["unmarked_text"] += 1
            if len(examples) < _EXAMPLES:
                examples.append({"kind": "unmarked_text", "text": unmarked[:300], "tail": raw[-300:]})
    return audit


def audit_problems(audit: Mapping[str, Any]) -> list[str]:
    """What makes an audited dump unusable; empty when it is clean. A dump with no generation is unusable too: the
    run it belongs to produced nothing that could be checked."""
    where = audit["where"]
    if not audit["generations"]:
        return [f"{where}: no raw generation"]
    if audit["unparsed_markup"] or audit["parse_mismatch"]:
        return [
            f"{where}: {audit['unparsed_markup']} generation(s) with tool-call markup that parsed to nothing and "
            f"{audit['parse_mismatch']} parse mismatch(es): the parser, not the model; examples: "
            + json.dumps(audit["examples"], ensure_ascii=False)[:1500]
        ]
    return []
