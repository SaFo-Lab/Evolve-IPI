"""Checks of MetaClaw's skill evolver: one of its endpoint before an evolution, two of what it was shown after one.

    python metaclaw_guards.py evolver <max_completion_tokens>
    python metaclaw_guards.py no-turn-drop <evolution_history.jsonl> [more ...]
    python metaclaw_guards.py saw-conversation <evolution_history.jsonl> <conversations.jsonl>

evolver            sends one request through MetaClaw's own evolver client, built as the evolving proxy builds it from
                   OPENAI_BASE_URL, OPENAI_API_KEY and SKILL_EVOLVER_MODEL (the key comes from the environment only,
                   never from argv). Any successful answer passes; a reasoning model may answer without text. A failure
                   names the error's type and its HTTP status, and its code when that is a short identifier without the
                   key; never the error's message, which can quote the key. The caller names the endpoint and the model.

Every evolution writes a line to the library's ``evolution_history.jsonl``: ``num_failures_analyzed``, the turns of the
conversation the evolver was handed, and ``failure_prompts``, the excerpt it rendered of each.

no-turn-drop       requires the two to match on every line: every turn of a conversation reached the evolver's
                   prompt. MetaClaw as released rendered only the first six turns and dropped the end of a longer
                   conversation, where its outcome and the user's feedback are; the patch removes that cap. The check
                   also reports the longest conversation, since nothing else bounds the prompt.
saw-conversation   requires every excerpt to be the last 300 characters of the conversation of a turn MetaClaw recorded
                   (``conversations.jsonl`` of the proxy that evolved the library), never of the turn's whole model
                   input. As released, the excerpt was cut from the model input, system prompt included, so on a short
                   conversation it showed OpenClaw's system prompt, or the skills the evolver itself had written. The
                   test is exact set membership; it can only tell the two apart on a conversation of at most 300
                   characters, and it says so when no turn was that short.

A check that holds prints its report and exits 0; one that fails exits 1 with the reason on stderr. This script runs
with the MetaClaw environment's Python: ``evolver`` uses MetaClaw's ``SkillEvolver`` and ``openai``,
``saw-conversation`` rebuilds the conversations with MetaClaw's own ``_flatten_message_content``, and ``no-turn-drop``
needs only the standard library.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

WINDOW = 300  # the evolver keeps the last 300 characters of each turn's conversation as its excerpt
CHARS_PER_TURN = 1100  # about what one turn adds to the evolver's prompt
USAGE = (
    "usage: metaclaw_guards.py evolver <max_completion_tokens>\n"
    "       metaclaw_guards.py no-turn-drop <evolution_history.jsonl> [more ...]\n"
    "       metaclaw_guards.py saw-conversation <evolution_history.jsonl> <conversations.jsonl>"
)
PREFLIGHT_PROMPT = "Reply with the single word OK."
_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}")  # what a provider's error code looks like: invalid_api_key, 429, ...

Flatten = Callable[[Any], str]


def _objects(path: Path, check: str) -> list[tuple[int, dict[str, Any]]]:
    """The non-blank lines of a JSON-lines file, each a JSON object, with their line numbers."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"{check}: no such file: {path}") from None
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise SystemExit(f"{check}: {path} line {number} is not JSON: {error}") from None
        if not isinstance(row, dict):
            raise SystemExit(f"{check}: {path} line {number} is not a JSON object")
        rows.append((number, row))
    return rows


# --- no-turn-drop ------------------------------------------------------------------------------------


def _turns_of(path: Path) -> tuple[int, int]:
    """(events, the most turns in one event). Exits on a dropped turn."""
    if not path.is_file():
        raise SystemExit(f"no-turn-drop: no such history: {path}")
    events, worst = 0, 0
    for number, event in _objects(path, "no-turn-drop"):
        fed = event.get("num_failures_analyzed")
        if fed is None:
            raise SystemExit(
                f"no-turn-drop: {path} line {number} has no num_failures_analyzed; this is not a MetaClaw evolution "
                "history"
            )
        rendered = len(event.get("failure_prompts") or [])
        if fed != rendered:
            raise SystemExit(
                f"no-turn-drop: {path} line {number} -- {fed} turns were fed to the evolver but only {rendered} were "
                f"rendered into its prompt, so {fed - rendered} turn(s) were silently dropped.\n"
                "The `failed_samples[:6]` caps are supposed to be gone (see "
                "third_party/patches/metaclaw-skill-evolver.patch). Either the patch is not live in the process that "
                "ran this, or a cap was re-introduced. Do NOT proceed: the skills in this library were evolved from a "
                "truncated view of the episode."
            )
        events += 1
        worst = max(worst, fed)
    if events == 0:
        raise SystemExit(f"no-turn-drop: {path} holds no evolution events")
    return events, worst


def no_turn_drop(histories: Sequence[Path]) -> str:
    """The report of the check over ``histories``; exits with the reason when a turn was dropped."""
    total, worst = 0, 0
    for path in histories:
        events, longest = _turns_of(Path(path))
        total += events
        worst = max(worst, longest)
    return (
        f"  evolver input: {total} event(s), every turn rendered, longest episode {worst} turn(s) "
        f"(~{worst * CHARS_PER_TURN} chars of evolver prompt)"
    )


# --- saw-conversation --------------------------------------------------------------------------------


def conversation_text(messages: Sequence[Any], flatten: Flatten) -> str:
    """What MetaClaw records as a turn's conversation: every message but the system prompt, flattened."""
    lines = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"a message is not a JSON object: {message!r}"[:200])
        if message.get("role") != "system":
            lines.append(f"{message.get('role', '?')}: {flatten(message.get('content', ''))}")
    return "\n".join(lines)


def legitimate_tails(records: Path, flatten: Flatten) -> tuple[set[str], int]:
    """The excerpts the evolver may have recorded, and how many recorded turns could tell the two sources apart."""
    allowed: set[str] = set()
    discriminating = 0
    for number, record in _objects(records, "saw-conversation"):
        messages = record.get("messages")
        if messages is None:
            raise SystemExit(
                f"saw-conversation: {records} line {number} has no `messages`, so what the evolver should have seen "
                "cannot be reconstructed. This is not a MetaClaw conversation record."
            )
        if not isinstance(messages, list):
            raise SystemExit(f"saw-conversation: {records} line {number}: `messages` is not a list")
        try:
            conversation = conversation_text(messages, flatten)
        except TypeError as error:
            raise SystemExit(f"saw-conversation: {records} line {number}: {error}") from None
        allowed.add(conversation[-WINDOW:])
        if len(conversation) <= WINDOW:
            # On a short conversation the model input's tail would reach into the system prompt: here the two differ.
            discriminating += 1
    if not allowed:
        raise SystemExit(f"saw-conversation: {records} holds no turns")
    return allowed, discriminating


def saw_conversation(history: Path, records: Path, flatten: Flatten) -> list[str]:
    """The report of the check; exits with the reason when an excerpt is not the end of a recorded conversation."""
    allowed, discriminating = legitimate_tails(Path(records), flatten)
    events = entries = 0
    for number, event in _objects(Path(history), "saw-conversation"):
        prompts = event.get("failure_prompts")
        if prompts is None:
            raise SystemExit(
                f"saw-conversation: {history} line {number} has no failure_prompts; this is not a MetaClaw evolution "
                "history"
            )
        for index, excerpt in enumerate(prompts):
            if excerpt not in allowed:
                raise SystemExit(
                    f"saw-conversation: {history} line {number}, failure {index + 1} was NOT the tail of any recorded "
                    f"conversation. It BEGINS:\n\n  {str(excerpt)[:200]!r}\n\n"
                    "-- which is system prompt, not conversation. (The entry's own tail looks innocent because the "
                    "conversation is at the END of it; the contamination is at the front, shown above.)\n"
                    "It is the tail of the full model input instead -- system prompt included. So the evolver read "
                    "OpenClaw's runtime footer (or its own previously injected skills) where its own label promises "
                    "the conversation. Either the patch at third_party/patches/ is not live in the process that ran "
                    "this, or a stale proxy served the run. Do NOT proceed: the skills in this library were evolved "
                    "from the prompt's boilerplate, not from the episode."
                )
            entries += 1
        events += 1
    if events == 0:
        raise SystemExit(f"saw-conversation: {history} holds no evolution events")
    report = [
        (
            f"  evolver input: {events} event(s), {entries} turn excerpt(s), all from the conversation and not the "
            "system prompt"
        )
    ]
    if discriminating:
        report.append(
            f"                 ({discriminating} recorded turn(s) had a conversation short enough for the two "
            "sources to differ, so the check has teeth)"
        )
    else:
        report.append(
            f"                 WARNING: no recorded turn had a conversation of {WINDOW} characters or fewer, so this "
            "run could not distinguish a patched evolver from an unpatched one. The pass above is vacuous."
        )
    return report


# --- evolver ---------------------------------------------------------------------------------------


def error_code(code: object, key: str) -> str:
    """A provider's error code as it may be printed: a short identifier that does not hold the key, else withheld."""
    text = str(code)
    return text if _CODE.fullmatch(text) and not (key and key in text) else "<code withheld>"


def evolver(max_completion_tokens: int) -> str:
    """One request through MetaClaw's own evolver client, in the shape of the evolver's calls: the model, one user
    message, ``max_completion_tokens``, nothing else. Returns the report; exits naming what failed."""
    import openai  # type: ignore[import-not-found]
    from metaclaw.skill_evolver import SkillEvolver  # type: ignore[import-not-found]

    where = f"{os.environ.get('OPENAI_BASE_URL', '')} (model {os.environ.get('SKILL_EVOLVER_MODEL', '')})"
    try:
        # the client reads the three variables, as the evolving proxy's own evolver does
        client = SkillEvolver(max_completion_tokens=max_completion_tokens)
    except OSError as error:  # MetaClaw's EnvironmentError: a variable is missing
        raise SystemExit(f"evolver: {error}") from None
    # Only types, statuses and codes from here on: an error's message can quote the key.
    try:
        client._call_llm(PREFLIGHT_PROMPT)
    except openai.APIStatusError as error:
        code = f", {error_code(error.code, os.environ.get('OPENAI_API_KEY', ''))}" if error.code else ""
        name = type(error).__name__
        raise SystemExit(f"evolver: refused the request: {name} (HTTP {error.status_code}{code})") from None
    except openai.APIConnectionError as error:
        raise SystemExit(f"evolver: could not be reached: {type(error).__name__}") from None
    except Exception as error:  # noqa: BLE001 -- e.g. an answer without choices, which the evolver could not read
        raise SystemExit(f"evolver: failed: {type(error).__name__}") from None
    return f"  evolver: {where} answered"


# --- command line ------------------------------------------------------------------------------------


def main(argv: Sequence[str]) -> int:
    if len(argv) == 2 and argv[0] == "evolver" and argv[1].isdigit():
        print(evolver(int(argv[1])))
        return 0
    if len(argv) >= 2 and argv[0] == "no-turn-drop":
        print(no_turn_drop([Path(a) for a in argv[1:]]))
        return 0
    if len(argv) == 3 and argv[0] == "saw-conversation":
        # MetaClaw's own rule for the text of a message
        from metaclaw.api_server import _flatten_message_content  # type: ignore[import-not-found]

        for line in saw_conversation(Path(argv[1]), Path(argv[2]), _flatten_message_content):
            print(line)
        return 0
    raise SystemExit(USAGE)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
