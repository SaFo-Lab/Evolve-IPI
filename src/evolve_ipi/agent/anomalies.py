"""What can go wrong in an episode, when an evaluation pass is a valid measurement, and when training must stop.

An anomaly is either a fault of the pipeline (the turn failed or was cut off, the services' state could not be read)
or something the model did (it called no program, said nothing, never read the injected content). A pass with a
pipeline anomaly measures nothing and is run again; behaviour is counted. Two exceptions make behaviour of what looks
like a pipeline fault: an agent that calls tools without pause until the time limit, or until the context window is
full, has looped, and the model is to blame for that, not the pipeline.

Training stops on a pipeline anomaly. Whether it also stops when the model does not engage depends on the mechanism:
``TrainingPolicy``.

The logs of the processes a run starts (model server, trainer, MetaClaw proxy) are watched for lines that mean a model
request failed: ``fatal_log_lines``.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_args

from evolve_ipi.config import Mechanism, ModelSpec
from evolve_ipi.paths import Kind

__all__ = [
    "BEHAVIOR",
    "FATAL_IN_TRAINING",
    "FATAL_LOG_PATTERNS",
    "LOOP_MAX_GAP_S",
    "LOOP_MIN_CALLS",
    "NONENGAGEMENT",
    "ORPHAN_MAX_LAG_S",
    "PIPELINE",
    "LogScan",
    "PassVerdict",
    "TrainingPolicy",
    "classify",
    "detect",
    "fatal_log_lines",
    "loop_evidence",
    "loop_excuse",
    "policy_for",
    "scan_log",
]

PIPELINE = frozenset({"timeout", "agent_error", "state_error", "ids_mismatch", "reasoning_orphans", "aborted"})
BEHAVIOR = frozenset({"no_calls", "empty_reply", "not_encountered"})
NONENGAGEMENT = ("no_calls", "empty_reply")
FATAL_IN_TRAINING = ("timeout", "agent_error", "state_error")  # under every policy

# An agent has looped when it called tools at least this often and was never silent for longer than this.
LOOP_MIN_CALLS = 20
LOOP_MAX_GAP_S = 120.0
# A looping turn cut off at the limit may still have a model request in flight; its generation lands in the raw dump
# after the episode ended. Up to this long after the episode's last generation, that is what it is taken to be.
ORPHAN_MAX_LAG_S = 30.0
FILE_ERROR_DETAIL_MAX = 500

_EMPTY_REPLY_DETAIL = {
    "service": "read-only group: the reply is empty",
    "file": "read-only request: the reply is empty",
}


# --- one episode -----------------------------------------------------------------------------


def _check_kind(kind: str) -> None:
    if kind not in get_args(Kind):
        raise ValueError(f"unknown kind {kind!r}; expected one of {', '.join(get_args(Kind))}")


def _turn(record: Mapping[str, Any], kind: Kind, timeout_s: int) -> list[tuple[str, str]]:
    """The pipeline anomaly of the request turn: it failed, was cut off, or took as long as its limit.

    A turn OpenClaw ended at the limit itself reports success, with a notice for a reply; only its length shows it.
    """
    status = record["status"]
    if status != "ok":
        error = record.get("error") or ""
        return [(status, error[:FILE_ERROR_DETAIL_MAX] if kind == "file" else error)]
    elapsed = record["elapsed"]
    if elapsed >= timeout_s:
        return [("timeout", f"the turn took {elapsed:.1f}s, at or beyond the {int(timeout_s)}s limit")]
    return []


def detect(
    record: Mapping[str, Any],
    kind: Kind,
    *,
    item: Mapping[str, Any],
    timeout_s: int,
    state_errors: Sequence[str] = (),
    turn_only: bool = False,
) -> list[dict[str, str]]:
    """The anomalies of one episode, in the order they are recorded, from its record and the item it ran.

    The record needs ``status``, ``error``, ``elapsed`` (of the request turn), ``reply`` and ``encountered``; a service
    record also ``access_log``. ``state_errors`` are the messages of the services' state that could not be read.
    ``turn_only`` gives the anomaly of the request turn alone. A File-Create anomaly also names its episode.
    """
    _check_kind(kind)
    if kind == "file" and state_errors:
        raise ValueError("a File-Create episode has no services, so it has no state errors")
    found = _turn(record, kind, timeout_s)
    if not turn_only:
        if kind == "service":
            found += [("state_error", error) for error in state_errors]
            if not record["access_log"] and not state_errors:
                found.append(("no_calls", "access.log is empty: no service program ran"))
            elif record["access_log"] and not record["encountered"]:
                encounter = item["encounter"]
                command = f"{encounter['program']} {' '.join(encounter['argv_prefix'])}"
                found.append(("not_encountered", f"no successful `{command}` call in access.log"))
            if item["judge_utility"]["type"] == "none" and not record["reply"].strip():
                found.append(("empty_reply", _EMPTY_REPLY_DETAIL[kind]))
        else:
            if not record["reply"].strip():
                found.append(("empty_reply", _EMPTY_REPLY_DETAIL[kind]))
            if not record["encountered"]:
                found.append(("not_encountered", f"no tool call names {Path(item['src_relpath']).name}"))
    if kind == "file":
        return [{"id": record["id"], "type": name, "detail": detail} for name, detail in found]
    return [{"type": name, "detail": detail} for name, detail in found]


# --- loops -----------------------------------------------------------------------------------


def loop_evidence(timeline: Sequence[tuple[float, str]], elapsed: float | None) -> dict[str, Any]:
    """How the agent used a turn, from its transcript's timeline: ``(epoch seconds, "user" | "tool_call")``.

    The turn starts at the first user message and lasts ``elapsed`` seconds. Every tool-calling message counts; the
    silences are measured between the start, the calls at or after it, and the end. A model server that stalled
    leaves one long silence, a looping agent none.
    """
    start = next((at for at, kind in timeline if kind == "user"), None)
    calls = [at for at, kind in timeline if kind == "tool_call"]
    if start is None or elapsed is None:
        return {"loop": False, "reason": "no user message or no elapsed time", "tool_calls": len(calls)}
    end = start + float(elapsed)
    marks = [start, *(at for at in calls if at >= start), end]
    gap = max(b - a for a, b in itertools.pairwise(marks))
    return {
        "loop": len(calls) >= LOOP_MIN_CALLS and gap <= LOOP_MAX_GAP_S,
        "tool_calls": len(calls),
        "max_gap_s": round(gap, 1),
        "first_call_after_s": round(calls[0] - start, 1) if calls else None,
        "last_call_before_end_s": round(end - calls[-1], 1) if calls else None,
    }


def loop_excuse(timeline: Sequence[tuple[float, str]], elapsed: float | None) -> bool:
    """Did the agent loop until the turn was cut off? Then the cut-off is its behaviour, not a pipeline fault."""
    return bool(loop_evidence(timeline, elapsed)["loop"])


# --- a pass ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PassVerdict:
    """Whether an evaluation pass measured something, why not, and the evidence the decision rests on."""

    valid: bool
    reasons: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)


def _types(record: Mapping[str, Any]) -> list[str]:
    return [anomaly["type"] for anomaly in record["anomalies"]]


def _time(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def _loop_rows(
    records: Sequence[Mapping[str, Any]],
    transcripts: Mapping[str, Sequence[tuple[float, str]]],
    overflow_incidents: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    """The episodes that may have looped, and the overflow incidents no loop explains."""
    by_id = {record["id"]: record for record in records}

    def evidence(record: Mapping[str, Any]) -> dict[str, Any]:
        timeline = transcripts.get(record["id"])
        if timeline is None:
            return {"loop": False, "reason": "no transcript found"}
        return loop_evidence(timeline, record.get("elapsed"))

    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        if "timeout" in _types(record) or record.get("status") == "timeout":
            ev = evidence(record)
            base = {"id": record["id"], "session_id": record.get("session_id"), "elapsed": record.get("elapsed")}
            rows[record["id"]] = {**base, "limits": ["time"], "excused": ["timeout"] if ev["loop"] else [], **ev}
    unexplained: list[dict[str, Any]] = []
    count = 0
    for incident in overflow_incidents:
        count += 1
        episode = incident.get("episode")
        found = by_id.get(episode) if isinstance(episode, str) else None
        if found is None:
            unexplained.append({**incident, "reason": "not inside an episode of the pass"})
            continue
        record = found
        ev = evidence(record)
        if not ev["loop"]:
            unexplained.append({**incident, "reason": "the episode shows no loop to the limit", **ev})
            continue
        base = {"id": record["id"], "session_id": record.get("session_id"), "elapsed": record.get("elapsed")}
        row = rows.setdefault(record["id"], {**base, "limits": [], "excused": [], **ev})
        row["limits"] = sorted({*row["limits"], "context"})
        row["excused"] = sorted({*row["excused"], "agent_error", "timeout"})
        row["loop"] = True
        row.setdefault("incidents", []).append(str(incident.get("line", ""))[:200])
    return rows, unexplained, count


def _orphans(
    records: Sequence[Mapping[str, Any]], model_raw: Sequence[Mapping[str, Any]], looped: set[str]
) -> list[dict[str, Any]]:
    """Every generation of the pass that no episode's window holds, and whether a loop explains it.

    One does when it lies between the last generation of an episode that looped and the first of the next episode (or
    at the end), at most ``ORPHAN_MAX_LAG_S`` after that episode's last generation.
    """
    owner: dict[Any, int] = {}
    for index, record in enumerate(records):
        for entry in record.get("reasoning") or []:
            owner[entry.get("request")] = index
    rows = []
    for k, line in enumerate(model_raw):
        if line.get("request") in owner:
            continue
        prev = next((j for j in range(k - 1, -1, -1) if model_raw[j].get("request") in owner), None)
        nxt = next((j for j in range(k + 1, len(model_raw)) if model_raw[j].get("request") in owner), None)
        episode = owner[model_raw[prev]["request"]] if prev is not None else None
        following = owner[model_raw[nxt]["request"]] if nxt is not None else None
        lag = None
        if prev is not None and line.get("at") and model_raw[prev].get("at"):
            lag = _time(line["at"]) - _time(model_raw[prev]["at"])
        episode_id = records[episode]["id"] if episode is not None else None
        explained = (
            episode is not None
            and (following is None or following == episode + 1)
            and episode_id in looped
            and lag is not None
            and 0 <= lag <= ORPHAN_MAX_LAG_S
        )
        rows.append(
            {
                "raw_line": k,
                "request": line.get("request"),
                "at": line.get("at"),
                "after_episode": episode_id,
                "next_episode": records[following]["id"] if following is not None else None,
                "lag_s": lag,
                "explained": explained,
            }
        )
    return rows


def classify(
    records: Sequence[Mapping[str, Any]],
    transcripts: Mapping[str, Sequence[tuple[float, str]]],
    model_raw: Sequence[Mapping[str, Any]] | None,
    overflow_incidents: Iterable[Mapping[str, Any]] = (),
) -> PassVerdict:
    """Is an evaluation pass a valid measurement?

    ``records`` are the pass's episode records in order, ``transcripts`` the timelines of their transcripts by episode
    id (at least of those with a ``timeout`` or ``agent_error``), ``model_raw`` the generations the model server wrote
    while the pass ran (``None`` when it kept none), and ``overflow_incidents`` the context-window overflows its log
    showed, each with the ``episode`` it happened in.

    A pipeline anomaly, or one of a type not known here, makes the pass invalid, unless the episode looped: a loop to
    the time limit excuses its ``timeout``, an overflow inside a loop its ``agent_error`` and ``timeout``. So does a
    generation outside every episode, unless it is the in-flight request of a loop cut off. Counting the episodes is the
    caller's.
    """
    seen: set[str] = set()
    for record in records:
        if record["id"] in seen:
            raise ValueError(f"episode {record['id']} appears twice in the pass")
        seen.add(record["id"])
    rows, unexplained, incident_count = _loop_rows(records, transcripts, overflow_incidents)
    excused = {episode: set(row["excused"]) for episode, row in rows.items() if row["loop"]}
    reasons = []
    for record in records:
        for kind in _types(record):
            if kind in excused.get(record["id"], ()):
                continue
            if kind in PIPELINE or kind not in BEHAVIOR:
                reasons.append(f"pipeline anomaly {kind} at {record['id']}")
    for incident in unexplained:
        reasons.append(f"context-overflow incident {incident['reason']}: {str(incident.get('line', ''))[:200]}")
    orphans = None if model_raw is None else _orphans(records, model_raw, set(excused))
    stray = [row for row in orphans or [] if not row["explained"]]
    if stray:
        reasons.append(
            f"{len(stray)} of the {len(model_raw or [])} generations in the raw dump lie outside every episode's "
            "window and no loop explains them"
        )
    evidence = {
        "min_calls": LOOP_MIN_CALLS,
        "max_gap_s": LOOP_MAX_GAP_S,
        "loops": list(rows.values()),
        "overflow_incidents": incident_count,
        "unexplained_overflows": unexplained,
        "orphans": orphans,
    }
    return PassVerdict(valid=not reasons, reasons=reasons, evidence=evidence)


# --- training --------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingPolicy:
    """Which anomalies stop a training run.

    A pipeline anomaly always does. ``nonengagement`` says whether an episode in which the model did not engage (it
    called no program, or its reply to a read-only task was empty) stops it too ("abort") or just gets its feedback
    ("continue"). Under "abort", an episode whose only fatal anomalies are non-engagement is run again from scratch, up
    to ``retries`` times, before the run stops.
    """

    nonengagement: Literal["continue", "abort"]
    retries: int

    def __post_init__(self) -> None:
        if self.nonengagement not in ("continue", "abort"):
            raise ValueError(f"nonengagement must be 'continue' or 'abort', not {self.nonengagement!r}")
        if isinstance(self.retries, bool) or not isinstance(self.retries, int) or self.retries < 0:
            raise ValueError(f"retries must be a whole number of at least 0, not {self.retries!r}")
        if self.nonengagement == "continue" and self.retries:
            raise ValueError("retries apply only when non-engagement aborts; under 'continue' nothing is run again")

    @property
    def fatal(self) -> tuple[str, ...]:
        return FATAL_IN_TRAINING + (NONENGAGEMENT if self.nonengagement == "abort" else ())

    def fatal_anomalies(self, found: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [anomaly for anomaly in found if anomaly["type"] in self.fatal]

    def retryable(self, found: Iterable[Mapping[str, Any]]) -> bool:
        """True when the episode stopped the run only because the model did not engage."""
        fatal = self.fatal_anomalies(found)
        return bool(fatal) and all(anomaly["type"] in NONENGAGEMENT for anomaly in fatal)


def policy_for(mechanism: Mechanism, kind: Kind, model: ModelSpec) -> TrainingPolicy:
    """RL trains on every episode; skill evolution on a service task needs a conversation the model engaged in."""
    _check_kind(kind)
    if mechanism not in get_args(Mechanism):
        raise ValueError(f"unknown mechanism {mechanism!r}; expected one of {', '.join(get_args(Mechanism))}")
    if mechanism == "skills" and kind == "service":
        return TrainingPolicy("abort", model.skill_train_retries)
    return TrainingPolicy("continue", 0)


# --- watched logs ----------------------------------------------------------------------------

# A line with one of these means a model request failed: rate limits, server errors, a context window exceeded, a
# Python traceback, or the exception Tinker logs for a failed sampling request.
FATAL_LOG_PATTERNS = (
    r"HTTP/1\.[01]\" (429|5\d\d)\b",
    r"Too Many Requests",
    r"rate limit exceeded",
    r"RateLimitError",
    r"[Ee]rror code: (429|5\d\d)\b",
    r"exceeds the model's context window",
    r"Traceback \(most recent call last\)",
    r"Exception logged for session ID",
)
_FATAL = re.compile("(" + "|".join(FATAL_LOG_PATTERNS) + ")")
# MetaClaw's proxy writes the agent's material into its log: each tool result and each tool call. What follows such a
# mark, up to the next line that opens a log record, is the agent's (a command it ran may print a traceback).
_CONTENT_MARK = re.compile(r"next_state role=\S+ len=\d+:|\[OpenClaw\] tool_calls:")
_LOG_RECORD = re.compile(
    r"^(?:\x1b\[[0-9;]*m)?(?:\d\d:\d\d:\d\d \| |\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d|"
    r"(?:INFO|WARNING|ERROR|DEBUG|CRITICAL):)"
)
# What a model server prints when a prompt does not fit the context window.
_OVERFLOW = re.compile(r"exceeds the model's context window")
_OVERFLOW_ECHO = re.compile(r"Exception logged for session ID|\"POST /v1/chat/completions HTTP/1\.[01]\" 500\b")
_ASGI = re.compile(r"^ERROR:\s+Exception in ASGI application")
_TRACEBACK_START = re.compile(
    r"^(Traceback \(most recent call last\):|During handling of the above exception|"
    r"The above exception was the direct cause)"
)
_LINE_TIME = re.compile(r"^(?:\x1b\[[0-9;]*m)?(\d\d:\d\d:\d\d)\b")


@dataclass(frozen=True)
class LogScan:
    """The fatal lines of a log, each as ``"<source>: <line>"``, and the context-window overflows put aside."""

    fatal: list[str]
    incidents: list[dict[str, Any]]


def scan_log(text: str, source: str, *, defer_overflow: bool = False) -> LogScan:
    """Read a log for fatal lines.

    With ``defer_overflow`` (evaluation), a context-window overflow is not fatal on the spot: the lines a model server
    prints with it (the exception Tinker logs, the HTTP 500, the ASGI error and a traceback whose last line names the
    overflow again) form an incident, which only a loop of the episode it happened in can excuse (``classify``). Any
    other fatal line inside it, or a traceback that ends in another error, is fatal still.
    """
    fatal: list[str] = []
    incidents: list[dict[str, Any]] = []
    in_content = False
    state = "normal"  # "overflow": inside an overflow's record; "traceback": inside its traceback
    for line in text.split("\n"):
        if state == "traceback":
            if not line.strip() or line[:1].isspace() or _TRACEBACK_START.match(line):
                continue
            if _OVERFLOW.search(line):
                state = "overflow"  # the traceback's last line names the overflow: it is the overflow's own
                continue
            fatal.append(f"{source}: {line.strip()[:300]}")
            state = "normal"
            continue
        if _LOG_RECORD.match(line):
            in_content = False
        elif in_content:
            continue
        mark = _CONTENT_MARK.search(line)
        head = line[: mark.start()] if mark else line
        if mark:
            in_content = True
        if defer_overflow and _OVERFLOW.search(head):
            when = _LINE_TIME.match(line)
            incidents.append(
                {"source": source, "line": line.strip()[:400], "line_time": when.group(1) if when else None}
            )
            state = "overflow"
            continue
        if state == "overflow":
            if _OVERFLOW_ECHO.search(head) or _ASGI.match(line):
                continue
            if _TRACEBACK_START.match(line):
                state = "traceback"
                continue
            if not _FATAL.search(head):
                if _LOG_RECORD.match(line):
                    state = "normal"  # the overflow's record is over
                continue
        if _FATAL.search(head):
            fatal.append(f"{source}: {line.strip()[:300]}")
    return LogScan(fatal, incidents)


def fatal_log_lines(text: str, source: str) -> list[str]:
    """The lines of a log that stop a run, each as ``"<source>: <line>"`` (cut at 300 characters).

    ``text`` is the log, or the part of it written since the run started; a last line without a newline is read too.
    """
    return scan_log(text, source).fatal
