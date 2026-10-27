"""One episode: the agent gets one request, is judged, and in training hears what the user thinks of what it did.

Every episode starts from the same environment, whatever the episode before it did:

    service tasks   reset the agent's session, move earlier transcripts out of its reach, put the workspace back,
                    write the services' initial state for the item
    File-Create     put the workspace back, reset the session, move earlier transcripts away; the file the item's
                    document asks for must not exist yet

Then the request is sent and judged: a service task by the services' state and the access log of the programs, File-
Create by whether the requested file exists right after the turn. In training, the user's feedback follows in the
same session, and the skill line closes the session so that MetaClaw evolves on it.

A pipeline fault (a reset that failed, leftover state, a feedback turn that failed) raises ``EpisodeAbort``; an
episode is never recorded as something it was not. In evaluation the anomalies of an episode are recorded and judged
for the whole pass (``anomalies.classify``); in training the ``TrainingPolicy`` decides which of them stop the run.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, get_args

from evolve_ipi import tasks
from evolve_ipi.agent import anomalies
from evolve_ipi.agent.anomalies import TrainingPolicy
from evolve_ipi.agent.openclaw import (
    GATEWAY_SPAN,
    OpenClawError,
    Profile,
    TurnResult,
    archive_deleted_transcripts,
    delete_session,
    profile_for,
    run_turn,
    transcript_timeline,
    transcript_tool_calls,
    write_profile,
)
from evolve_ipi.config import PortsSpec
from evolve_ipi.environment import state, verifiers, workspace
from evolve_ipi.environment.state import StateError
from evolve_ipi.environment.workspace import Snapshot, WorkspaceError
from evolve_ipi.paths import Line, RuntimePaths
from evolve_ipi.procs import require_free
from evolve_ipi.records import now
from evolve_ipi.tasks import Task

__all__ = [
    "Closure",
    "EpisodeAbort",
    "EpisodeContext",
    "LeftoverState",
    "Mode",
    "RawDumpError",
    "episode_timeline",
    "file_feedback",
    "prepare_environment",
    "read_raw_dump",
    "reasoning_of",
    "run_episode",
    "run_file_episode",
    "run_service_episode",
    "service_feedback",
]

Mode = Literal["eval", "train"]
# Closes a training episode's session (the skill line: it makes MetaClaw evolve); called with the session id, it
# returns what the record keeps of it.
Closure = Callable[[str], dict[str, Any]]


def _no_verdict() -> dict[str, Any]:
    """The verdict of a judge that could not read the services' state."""
    return {"hit": None, "params_exact": None, "matches": [], "new_records": None}


class EpisodeAbort(RuntimeError):
    """The episode could not be run or recorded as the method requires.

    ``record`` is what the attempt that stopped got to, if anything; ``retried_attempts`` are the records of the
    attempts given up on before it (the record holds them too). ``retryable=False`` marks a fault of the pipeline that
    a policy's retry must never run past, whatever the record's anomalies say.
    """

    def __init__(
        self,
        reason: str,
        record: dict[str, Any] | None = None,
        retried_attempts: list[dict[str, Any]] | None = None,
        *,
        retryable: bool = True,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.record = record
        self.retried_attempts: list[dict[str, Any]] = list(retried_attempts or [])
        self.retryable = retryable


class RawDumpError(ValueError):
    """A line of the model server's raw dump is not a generation."""


class LeftoverState(RuntimeError):
    """Something an earlier run left where the agent works. Running again would meet it again, so it is not retried."""


@dataclass(frozen=True)
class EpisodeContext:
    """Everything an episode of one task on one line runs in.

    ``state_dir`` is the services' state (service tasks), ``target_dir`` the directory of the agent's home that the
    File-Create documents point at. ``raw_dump`` is the model server's raw dump: when it is set, every record gets the
    generations written while its episode ran.
    """

    profile: Profile
    task: Task
    line: Line
    workspace_snapshot: Snapshot
    state_dir: Path | None
    target_dir: Path | None
    transcript_archive: Path
    timeout_s: int
    raw_dump: Path | None = None

    def with_raw_dump(self, raw_dump: Path | None) -> EpisodeContext:
        return replace(self, raw_dump=raw_dump)


# --- the environment -------------------------------------------------------------------------


def _remove_earlier(directory: Path, root: Path) -> None:
    """Remove what an earlier pass left of a directory this run keeps below its runtime root."""
    if not os.path.lexists(directory):
        return
    if directory.is_symlink() or not directory.is_dir():
        raise LeftoverState(f"{directory} is not a directory this run made; remove it and start again")
    if root.resolve() not in directory.resolve().parents:
        raise ValueError(f"{directory} is not below the runtime directory {root}")
    shutil.rmtree(directory)


def _target_dir(task: Task, line: Line, home: Path) -> Path:
    written = tasks.file_target_dir(task, line)  # "~/<dir>/", as the documents write it
    relative = written.removeprefix("~/").strip("/")
    if not written.startswith("~/") or not relative or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError(f"the target directory {written!r} of {task.name} is not a directory of the home")
    return home / relative


def _prepare_target_dir(task: Task, line: Line, home: Path) -> Path:
    """The directory the documents ask the agent to write into, empty; anything in it is an earlier run's."""
    directory = _target_dir(task, line, home)
    if directory.is_symlink() or (os.path.lexists(directory) and not directory.is_dir()):
        raise LeftoverState(
            f"{directory} is a link or a file, not a directory: the files the documents ask for would land elsewhere"
        )
    directory.mkdir(parents=True, exist_ok=True)
    left = sorted(entry.name for entry in directory.iterdir())
    if left:
        raise LeftoverState(
            f"{directory} holds {', '.join(left[:10])}{' ...' if len(left) > 10 else ''}: an earlier run left them, "
            "and an episode whose file already exists cannot be judged. Remove them and start again."
        )
    return directory


def prepare_environment(
    task: Task, line: Line, paths: RuntimePaths, ports: PortsSpec, provider_port: int, timeout_s: int
) -> EpisodeContext:
    """Build the environment of ``task`` on ``line``: a fresh workspace, the services or the target directory, and
    the OpenClaw profile whose agent talks to ``provider_port``.

    The workspace is built before a gateway first sees it (OpenClaw would set up one it finds empty itself), so no
    gateway of the profile may run; its ports must be free. File-Create gets the 88 documents and no services, and
    the agent's target directory must be empty. A service task gets the programs, with their state outside the
    workspace.
    """
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s < 1:
        raise ValueError(f"timeout_s must be a whole number of seconds, at least 1, not {timeout_s!r}")
    kind = task.kind
    profile = profile_for(line, kind, paths, ports.gateway, provider_port)
    require_free(range(profile.gateway_port, profile.gateway_port + GATEWAY_SPAN))
    target_dir = _prepare_target_dir(task, line, paths.agent_home) if kind == "file" else None
    state_dir = paths.service_state(line, "service") if kind == "service" else None
    _remove_earlier(profile.workspace, paths.root)
    if state_dir is not None:
        _remove_earlier(state_dir, paths.root)
        workspace.build(profile.workspace, "service", None, state_dir)
    else:
        workspace.build(profile.workspace, "file", tasks.file_workspace_documents(task, line), None)
    workspace.guard(profile.workspace)
    snapshot = workspace.snapshot(profile.workspace)
    write_profile(profile)
    return EpisodeContext(
        profile=profile,
        task=task,
        line=line,
        workspace_snapshot=snapshot,
        state_dir=state_dir,
        target_dir=target_dir,
        transcript_archive=paths.transcript_archive(line, kind),
        timeout_s=timeout_s,
    )


# --- feedback --------------------------------------------------------------------------------


def service_feedback(
    task: Task, item: Mapping[str, Any], asr: Mapping[str, Any], utility: Mapping[str, Any]
) -> tuple[str, str]:
    """What the user says after a service episode, and whether it is praise ("pos") or a correction ("neg")."""
    text = verifiers.feedback(dict(item), task.meta["feedback_group"], bool(asr["hit"]), utility["hit"])
    return text, "pos" if text.startswith("Great") else "neg"


def file_feedback(task: Task, created: bool) -> tuple[str, str]:
    """What the user says after a File-Create episode: praise when the requested file was created."""
    return task.meta["feedback"]["positive" if created else "negative"], "pos" if created else "neg"


# --- generations -----------------------------------------------------------------------------


def read_raw_dump(path: Path) -> list[dict[str, Any]]:
    """The generations a model server has dumped so far, oldest first; a last line still being written is not one yet.

    Raises ``OSError`` when the dump cannot be read and ``RawDumpError`` for a line that is not a generation.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = []
    for number, line in enumerate(text[: text.rfind("\n") + 1].split("\n"), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            raise RawDumpError(f"line {number} of the raw dump {path} is not JSON") from None
        if not isinstance(entry, dict) or not isinstance(entry.get("raw"), str):
            raise RawDumpError(f"line {number} of the raw dump {path} is not a generation")
        lines.append(entry)
    return lines


def _raw_lines(path: Path) -> list[dict[str, Any]]:
    try:
        return read_raw_dump(path)
    except (OSError, RawDumpError) as error:
        raise EpisodeAbort(f"the raw dump of the model server cannot be read: {error}") from error


def reasoning_of(raw_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per generation: the reasoning before the first ``</think>`` and the output after it.

    A generation without ``</think>`` is all output, and its reasoning is ``None``, not an empty string.
    """
    out = []
    for line in raw_lines:
        thinking, end, rest = line["raw"].partition("</think>")
        out.append(
            {
                "request": line.get("request"),
                "n_tokens": line.get("n_tokens"),
                "thinking": thinking if end else None,
                "output": rest if end else line["raw"],
            }
        )
    return out


def _raw_mark(ctx: EpisodeContext) -> int | None:
    return None if ctx.raw_dump is None else len(_raw_lines(ctx.raw_dump))


def _attach_reasoning(ctx: EpisodeContext, record: dict[str, Any], mark: int | None) -> None:
    """Episodes run one after the other through one model server, so the generations written between an episode's
    start and its end are its own."""
    if ctx.raw_dump is None or mark is None:
        return
    lines = _raw_lines(ctx.raw_dump)
    if len(lines) < mark:
        # It also runs while an abort propagates, whose record may hold anomalies a policy retries: never this one.
        raise EpisodeAbort(
            f"the raw dump {ctx.raw_dump} lost generations during episode {record['id']}", record, retryable=False
        )
    record["reasoning"] = reasoning_of(lines[mark:])


# --- shared steps ----------------------------------------------------------------------------


def _check_call(
    ctx: EpisodeContext, mode: str, authorized: bool, closure: Closure | None, policy: TrainingPolicy | None
) -> None:
    if mode not in get_args(Mode):
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(get_args(Mode))}")
    if mode == "eval":
        if policy is not None:
            raise ValueError("an evaluation episode has no training policy")
        if closure is not None:
            raise ValueError("an evaluation episode has no closure: nothing evolves on it")
        return
    if policy is None:
        raise ValueError("a training episode needs the training policy of its line")
    if not isinstance(policy, TrainingPolicy):
        raise TypeError(f"policy must be a TrainingPolicy, not {type(policy).__name__}")
    if not authorized:
        raise ValueError("a training episode sends the authorizing request: authorized must be True")
    if ctx.task.kind == "file" and policy.nonengagement != "continue":
        raise ValueError("File-Create training goes on when the model does not engage: its policy is 'continue'")


def _feedback_turn(
    ctx: EpisodeContext, item: Mapping[str, Any], turn: TurnResult, feedback: TurnResult, record: dict[str, Any]
) -> None:
    """The feedback turn has to be a normal turn of the episode's own session."""
    if feedback.status != "ok":
        raise EpisodeAbort(f"{item['id']}: feedback turn failed: {feedback.status} {feedback.error}", record)
    if feedback.session_id != turn.session_id:
        raise EpisodeAbort(
            f"{item['id']}: the feedback turn landed in session {feedback.session_id}, not the episode's "
            f"{turn.session_id}",
            record,
        )
    if feedback.limit_reached:
        raise EpisodeAbort(
            f"{item['id']}: the feedback turn took {feedback.elapsed:.1f}s, at or beyond the {ctx.timeout_s}s limit",
            record,
        )


def _close(closure: Closure, session_id: str | None, item: Mapping[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    if session_id is None:
        raise EpisodeAbort(f"{item['id']}: the episode has no session to close", record)
    try:
        result = closure(session_id)
    except Exception as error:  # whatever stops the session from closing aborts the episode
        raise EpisodeAbort(
            f"{item['id']}: closing session {session_id} failed: {type(error).__name__}: {error}", record
        ) from error
    if not isinstance(result, dict):
        raise TypeError(f"a closure returns a dict for the record, not {type(result).__name__}")
    return result


def _fatal_reason(item: Mapping[str, Any], found: Sequence[Mapping[str, Any]]) -> str:
    return f"{item['id']}: " + "; ".join(f"{anomaly['type']}: {anomaly['detail']}" for anomaly in found)


# --- service tasks ---------------------------------------------------------------------------


def _service_attempt(
    ctx: EpisodeContext,
    item: dict[str, Any],
    mode: Mode,
    request: str,
    closure: Closure | None,
    policy: TrainingPolicy | None,
) -> dict[str, Any]:
    profile, state_dir = ctx.profile, ctx.state_dir
    assert state_dir is not None
    mark = _raw_mark(ctx)
    try:
        delete_session(profile)
        archive_deleted_transcripts(profile, ctx.transcript_archive)
        reverted = workspace.restore(profile.workspace, ctx.workspace_snapshot)
        workspace.guard(profile.workspace)
        state.reset(item, state_dir)
    except (OpenClawError, WorkspaceError, StateError) as error:
        raise EpisodeAbort(f"{item['id']}: the episode cannot start from a clean environment: {error}") from error
    turn = run_turn(profile, request, ctx.timeout_s)
    state_errors: list[str] = []
    access_log: list[dict[str, Any]] = []
    met = False
    try:
        access_log = state.read_access_log(state_dir)
        met = verifiers.encountered(item, access_log)
    except StateError as error:
        state_errors.append(str(error))
    asr: dict[str, Any] | None = None
    utility: dict[str, Any] | None = None
    try:
        asr = verifiers.judge_asr(item, state_dir)
        utility = verifiers.judge_utility(item, state_dir)
    except StateError as error:
        state_errors.append(str(error))
    record: dict[str, Any] = {
        "id": item["id"],
        "mode": mode,
        "request": request,
        "status": turn.status,
        "error": turn.error,
        "elapsed": turn.elapsed,
        "session_id": turn.session_id,
        "reply": turn.reply,
        "asr": asr if asr is not None else _no_verdict(),
        "utility": utility if utility is not None else _no_verdict(),
        "encountered": met,
        "access_log": access_log,
        "anomalies": [],
        "workspace_reverted": reverted,
    }
    record["anomalies"] = anomalies.detect(
        record, "service", item=item, timeout_s=ctx.timeout_s, state_errors=state_errors
    )
    try:
        if mode == "train":
            assert policy is not None
            fatal = policy.fatal_anomalies(record["anomalies"])
            if fatal:
                raise EpisodeAbort(_fatal_reason(item, fatal), record)
            text, polarity = service_feedback(ctx.task, item, record["asr"], record["utility"])
            feedback = run_turn(profile, text, ctx.timeout_s)
            record.update(
                feedback=text,
                feedback_polarity=polarity,
                fb_status=feedback.status,
                fb_reply=feedback.reply,
                fb_elapsed=feedback.elapsed,
            )
            _feedback_turn(ctx, item, turn, feedback, record)
            if closure is not None:
                record["closure"] = _close(closure, turn.session_id, item, record)
    except EpisodeAbort:
        _attach_reasoning(ctx, record, mark)
        raise
    _attach_reasoning(ctx, record, mark)
    return record


def run_service_episode(
    ctx: EpisodeContext,
    item: dict[str, Any],
    mode: Mode,
    authorized: bool,
    closure: Closure | None = None,
    *,
    policy: TrainingPolicy | None = None,
) -> dict[str, Any]:
    """One episode of a service task; see ``run_episode``.

    Under a policy with retries, an episode that stopped training only because the model did not engage is run again
    from the start (a new session, the workspace and the services' state reset), and each attempt given up on is kept
    in the record's ``retried_attempts``.
    """
    if ctx.task.kind != "service":
        raise ValueError(f"{ctx.task.name} is not a service task")
    _check_call(ctx, mode, authorized, closure, policy)
    request = tasks.request(ctx.task, item, authorized)
    retried: list[dict[str, Any]] = []
    while True:
        try:
            record = _service_attempt(ctx, item, mode, request, closure, policy)
        except EpisodeAbort as abort:
            if mode != "train":
                raise
            assert policy is not None
            if (
                abort.retryable
                and abort.record is not None
                and policy.retryable(abort.record["anomalies"])
                and len(retried) < policy.retries
            ):
                retried.append(abort.record)
                continue
            if abort.record is not None:
                abort.record["retried_attempts"] = retried
            if not retried:
                raise
            raise EpisodeAbort(f"{abort.reason} (after {len(retried)} retries)", abort.record, retried) from abort
        if mode == "train":
            record["retried_attempts"] = retried
        return record


# --- File-Create -----------------------------------------------------------------------------


def _transcript_path(profile: Profile, session_id: str) -> Path | None:
    live = profile.sessions_dir / f"{session_id}.jsonl"
    if live.is_file():
        return live
    return next((p for p in sorted(profile.sessions_dir.glob(f"{session_id}.jsonl.deleted.*")) if p.is_file()), None)


def _remove_target(target: Path) -> None:
    """Remove whatever the agent made at the target's name; a link is removed, never what it points at."""
    if target.is_symlink() or (os.path.lexists(target) and not target.is_dir()):
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)


def run_file_episode(
    ctx: EpisodeContext,
    item: dict[str, Any],
    mode: Mode,
    authorized: bool,
    closure: Closure | None = None,
    *,
    policy: TrainingPolicy | None = None,
) -> dict[str, Any]:
    """One episode of File-Create; see ``run_episode``.

    The attack succeeded when the file the document asks for exists right after the request turn. The tool calls are
    read from the session's transcript after the last turn, so in training they cover the feedback turn too. The file
    is removed after the episode, also when the episode stops the run; one found before the episode refuses it.
    """
    if ctx.task.kind != "file":
        raise ValueError(f"{ctx.task.name} is not the File-Create task")
    _check_call(ctx, mode, authorized, closure, policy)
    request = tasks.request(ctx.task, item, authorized)
    profile, target_dir = ctx.profile, ctx.target_dir
    assert target_dir is not None
    target = target_dir / item["tgt_name"]
    mark = _raw_mark(ctx)
    try:
        reverted = workspace.restore(profile.workspace, ctx.workspace_snapshot)
        workspace.guard(profile.workspace)
        delete_session(profile)
        archive_deleted_transcripts(profile, ctx.transcript_archive)
    except (OpenClawError, WorkspaceError) as error:
        raise EpisodeAbort(f"{item['id']}: the episode cannot start from a clean environment: {error}") from error
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise EpisodeAbort(
            f"{item['id']}: the target directory {target_dir} is missing or no longer a directory: an earlier "
            "episode removed or replaced it"
        )
    if os.path.lexists(target):
        raise EpisodeAbort(f"{item['id']}: target {target} exists before the episode (isolation or cleanup failure)")
    record: dict[str, Any] = {
        "id": item["id"],
        "mode": mode,
        "started": now(),
        "workspace_reverted": reverted,
        "request": request,
    }
    try:
        turn = run_turn(profile, request, ctx.timeout_s)
        hit = verifiers.file_attack_success(target)
        record.update(
            status=turn.status,
            error=turn.error,
            reply=turn.reply,
            session_id=turn.session_id,
            elapsed=turn.elapsed,
            attack_success=hit,
        )
        if mode == "train":
            assert policy is not None
            failed = anomalies.detect(record, "file", item=item, timeout_s=ctx.timeout_s, turn_only=True)
            fatal = policy.fatal_anomalies(failed)
            if fatal:
                record["anomalies"] = failed
                raise EpisodeAbort(_fatal_reason(item, fatal), record)
            text, polarity = file_feedback(ctx.task, hit)
            feedback = run_turn(profile, text, ctx.timeout_s)
            record.update(
                created=hit,
                feedback=polarity,
                fb_status=feedback.status,
                fb_error=feedback.error,
                fb_reply=feedback.reply,
                fb_session_id=feedback.session_id,
                elapsed=round(turn.elapsed + feedback.elapsed, 1),
            )
            _feedback_turn(ctx, item, turn, feedback, record)
        calls: list[dict[str, Any]] = []
        transcript = None
        if turn.session_id:
            try:
                calls = transcript_tool_calls(profile, turn.session_id)
            except (OpenClawError, ValueError) as error:
                raise EpisodeAbort(f"{item['id']}: {error}", record) from error
            path = _transcript_path(profile, turn.session_id)
            if path is None:
                raise EpisodeAbort(f"{item['id']}: the transcript of session {turn.session_id} disappeared", record)
            transcript = str(path)
        met, acted = verifiers.file_encountered_acted(calls, item["src_relpath"], item["tgt_name"])
        record.update(tool_calls=calls, transcript=transcript, encountered=met, acted=acted)
        if closure is not None:
            record["closure"] = _close(closure, turn.session_id, item, record)
        record["anomalies"] = (
            anomalies.detect(record, "file", item=item, timeout_s=ctx.timeout_s) if mode == "eval" else []
        )
    except EpisodeAbort:
        record.setdefault("anomalies", [])
        record["target_after_episode"] = os.path.exists(target)
        _remove_target(target)
        _attach_reasoning(ctx, record, mark)
        raise
    except BaseException:
        _remove_target(target)
        raise
    record["target_after_episode"] = os.path.exists(target)
    _remove_target(target)
    _attach_reasoning(ctx, record, mark)
    return record


# --- any task --------------------------------------------------------------------------------


def run_episode(
    ctx: EpisodeContext,
    item: dict[str, Any],
    mode: Mode,
    authorized: bool,
    closure: Closure | None = None,
    *,
    policy: TrainingPolicy | None = None,
) -> dict[str, Any]:
    """Run one episode of ``item`` and return its record.

    ``authorized`` sends the request that also asks the agent to do what the content asks for (adaptation, and
    setting 1 of the evaluation); training always does. A training episode (``mode="train"``) needs the line's
    ``policy``; on the skill line its ``closure`` closes the session after the feedback turn (File-Create: after the
    transcript was read) and the record keeps what it returns. Raises ``EpisodeAbort`` for a pipeline fault, and in
    training for every anomaly the policy makes fatal.
    """
    if ctx.task.kind == "file":
        return run_file_episode(ctx, item, mode, authorized, closure, policy=policy)
    return run_service_episode(ctx, item, mode, authorized, closure, policy=policy)


def episode_timeline(ctx: EpisodeContext, record: Mapping[str, Any]) -> list[tuple[float, str]] | None:
    """The timeline of an episode's transcript, which ``anomalies.classify`` reads to tell a loop from a stall.

    Read it right after the episode: the next one deletes the session and moves its transcript away. ``None`` when the
    turn produced no session.
    """
    session_id = record.get("session_id")
    if not session_id:
        return None
    return transcript_timeline(ctx.profile, session_id)
