"""Skill evolution: MetaClaw's skill evolver writes a skill library from 12 conversations of the agent.

    adapt_skills    one library for a model and a task: the conversations, the drain, the two guards
    start_proxy     MetaClaw's skills_only proxy, evolving a library or serving a frozen one
    close_session   tells MetaClaw a conversation ended, which makes its evolver run on it

A finished library is held to its hash, ``adapt.library.lib_sha``.

The agent talks to MetaClaw's proxy, which forwards every request to the model server. Each conversation is the
authorized request of one adaptation item, judged, then the user's feedback in the same session; then the session is
closed and MetaClaw evolves on it in the background, without being waited for. After the twelfth conversation the
library's evolution history is drained until all twelve evolutions have landed. Two guards then check that the evolver
saw every turn of each conversation, and the conversation itself rather than the system prompt.

An evolution that stops part way (a failed episode, a crashed process, a fatal line in a watched log, a failed guard,
a wrong number of evolutions) is moved to ``adapt/_void/`` and started again from an empty library, up to
``skills.max_restarts`` times. A finished one is marked by ``adapt/library.json`` and never run again.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx

from evolve_ipi.adapt import AdaptAbort
from evolve_ipi.adapt.library import lib_sha
from evolve_ipi.agent.anomalies import fatal_log_lines, policy_for
from evolve_ipi.agent.episode import Closure, EpisodeAbort, prepare_environment, run_episode
from evolve_ipi.agent.openclaw import Gateway, OpenClawError
from evolve_ipi.children import (
    StartFailed,
    Stoppable,
    metaclaw_env,
    metaclaw_python,
    model_server_env,
    starting,
    stop_all,
    stop_failures,
)
from evolve_ipi.config import ConfigError, Configs, ModelSpec, PortsSpec, Secrets, SkillsSpec
from evolve_ipi.models.launch import HOST, server_files, start_model_server
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import ManagedProcess, ProcessError, require_free
from evolve_ipi.records import RunDir, now, write_json
from evolve_ipi.tasks import Task, load_skill_order
from evolve_ipi.thirdparty import verify_patches

__all__ = [
    "GUARDS_SCRIPT",
    "MARKER",
    "PROXY_SCRIPT",
    "LibraryError",
    "StartFailed",
    "adapt_skills",
    "check_inputs",
    "close_session",
    "drain",
    "evolver_preflight",
    "history_lines",
    "proxy_env",
    "ready_line",
    "run_guards",
    "start_proxy",
]

logger = logging.getLogger(__name__)

PROXY_SCRIPT = Path(__file__).with_name("metaclaw_proxy.py")
GUARDS_SCRIPT = Path(__file__).with_name("metaclaw_guards.py")
HISTORY_FILE = "evolution_history.jsonl"  # MetaClaw's evolver appends one line per evolution
MARKER = "library.json"  # in adapt/: the evolution finished
PATCHED_LINE = "METACLAW patched "
PATCHED_TIMEOUT_S = 180
READY_TIMEOUT_S = 300
PORT_TIMEOUT_S = 60
CLOSE_TIMEOUT_S = 120
GUARD_TIMEOUT_S = 600
# MetaClaw's evolver client is openai's, with its defaults: a request that fails to connect, times out or gets a 408,
# 409, 429 or 5xx answer is retried twice (three tries in all), each try waiting up to 600 s for its answer.
EVOLVER_PREFLIGHT_TIMEOUT_S = 900
GUARDS = ("no-turn-drop", "saw-conversation")
_VOID = "_void"
_SECRET_NAMES = {
    "tinker_api_key": "TINKER_API_KEY",
    "evolver_base_url": "EVOLVER_BASE_URL",
    "evolver_api_key": "EVOLVER_API_KEY",
    "evolver_model": "EVOLVER_MODEL",
}
_EVOLVER_SECRETS = ("evolver_base_url", "evolver_api_key", "evolver_model")


class LibraryError(RuntimeError):
    """A finished skill library does not match its record; it is never evolved again, so a person has to look."""


# --- the library -------------------------------------------------------------------------------------


def history_lines(path: Path) -> int:
    """The evolutions recorded so far: the non-blank lines of the history; a history not written yet has none."""
    try:
        with open(path, encoding="utf-8") as history:
            return sum(1 for line in history if line.strip())
    except FileNotFoundError:
        return 0


def drain(
    history: Path,
    expected: int,
    *,
    poll_s: float,
    stable_polls: int,
    max_polls: int,
    sleep: Callable[[float], None] = time.sleep,
    check: Callable[[], None] | None = None,
) -> int:
    """Wait for the evolutions still running in the background; returns how many the history holds.

    It reads the history every ``poll_s`` seconds and stops once the count has been the same for ``stable_polls``
    readings in a row and is at least ``expected``, or after ``max_polls`` readings. The caller requires exactly
    ``expected``. ``check`` runs before every reading and stops the drain by raising (a proxy that died, a fatal line
    in a watched log).
    """
    previous, stable, count = -1, 0, 0
    for _ in range(max_polls):
        if check is not None:
            check()
        count = history_lines(history)
        stable = stable + 1 if count == previous else 0
        if stable >= stable_polls and count >= expected:
            break
        previous = count
        sleep(poll_s)
    return count


# --- the proxy ---------------------------------------------------------------------------------------


def close_session(proxy_port: int, served_name: str, session_id: str) -> None:
    """End the agent's session ``session_id`` at the proxy, which then evolves on the session's turns.

    The request is a side turn, so MetaClaw does not buffer it as a turn of the conversation; it still forwards it to
    the model server, for one token. Raises for a status that is not a success.
    """
    if not session_id.strip():
        raise ValueError("refusing to close an empty session id: the agent's reply carried none")
    response = httpx.post(
        f"http://{HOST}:{proxy_port}/v1/chat/completions",
        json={"model": served_name, "messages": [{"role": "user", "content": "done"}], "max_tokens": 1},
        headers={"X-Session-Id": session_id, "X-Turn-Type": "side", "X-Session-Done": "true"},
        timeout=CLOSE_TIMEOUT_S,
        trust_env=False,  # a local call: never through a proxy of the environment
    )
    response.raise_for_status()


def ready_line(port: int) -> str:
    """What the proxy logs once it serves; the space after the port keeps 3001 from matching 300."""
    return f"skills_only proxy READY on {HOST}:{port} "


def _missing_secrets(secrets: Secrets, fields: tuple[str, ...]) -> list[str]:
    return [_SECRET_NAMES[name] for name in fields if not getattr(secrets, name)]


def _require_secrets(secrets: Secrets, fields: tuple[str, ...], purpose: str) -> None:
    missing = _missing_secrets(secrets, fields)
    if missing:
        raise ConfigError(
            f"{purpose} needs the secret(s) {', '.join(missing)}; set each one in the environment or the .env file "
            "(see .env.example). An empty value counts as missing."
        )


def proxy_env(
    model: ModelSpec,
    skills_dir: Path,
    evolve: bool,
    require_empty: bool,
    record_dir: Path,
    ports: PortsSpec,
    secrets: Secrets,
    *,
    skills: SkillsSpec,
    paths: RuntimePaths,
) -> dict[str, str]:
    """The whole environment of the proxy script. Only an evolving proxy gets the evolver's endpoint, key and model;
    no proxy gets the Tinker key or anything else of this process's environment."""
    if evolve:
        _require_secrets(secrets, _EVOLVER_SECRETS, "a MetaClaw proxy that evolves its library")
    sums = verify_patches()
    env = metaclaw_env(paths)
    env.update(
        SKILLS_DIR=str(Path(skills_dir).absolute()),
        AUTO_EVOLVE="1" if evolve else "0",
        REQUIRE_EMPTY="1" if require_empty else "0",
        LLM_BASE=f"http://{HOST}:{ports.model}/v1",
        LLM_MODEL=model.skill_served_name,
        PROXY_PORT=str(ports.proxy),
        RECORD_DIR=str(Path(record_dir).absolute()),
        EXPECT_SHA256_SKILL_EVOLVER=sums["MetaClaw/metaclaw/skill_evolver.py"],
        EXPECT_SHA256_API_SERVER=sums["MetaClaw/metaclaw/api_server.py"],
        SKILL_TOP_K=str(skills.skill_top_k),
        TASK_SPECIFIC_TOP_K=str(skills.task_specific_top_k),
        EVOLVE_EVERY=str(skills.evolution_every_n_turns),
        MAX_NEW_SKILLS=str(skills.max_new_skills),
        EVOLVER_MAX_COMPLETION_TOKENS=str(skills.evolver_max_completion_tokens),
    )
    if evolve:
        assert secrets.evolver_base_url and secrets.evolver_api_key and secrets.evolver_model
        env.update(
            OPENAI_BASE_URL=secrets.evolver_base_url,
            OPENAI_API_KEY=secrets.evolver_api_key,
            SKILL_EVOLVER_MODEL=secrets.evolver_model,
        )
    return env


def start_proxy(
    model: ModelSpec,
    skills_dir: Path,
    evolve: bool,
    require_empty: bool,
    record_dir: Path,
    ports: PortsSpec,
    secrets: Secrets,
    *,
    skills: SkillsSpec,
    paths: RuntimePaths | None = None,
) -> ManagedProcess:
    """Start MetaClaw's proxy on ``ports.proxy`` in front of the model server on ``ports.model``, and wait until it
    serves.

    ``record_dir`` must not exist: it is made here and gets MetaClaw's records and the proxy's log (``proxy.log``);
    MetaClaw empties its ``conversations.jsonl`` whenever it starts. ``require_empty`` makes the proxy refuse a
    library that has a skill or a history. Raises ``PortInUse`` before starting anything when the port is taken, and
    ``ProcessError`` with the end of the log when the proxy does not get ready (it is stopped then).
    """
    paths = RuntimePaths() if paths is None else paths
    python = metaclaw_python(paths)
    env = proxy_env(model, skills_dir, evolve, require_empty, record_dir, ports, secrets, skills=skills, paths=paths)
    require_free([ports.proxy])
    record_dir = Path(record_dir).absolute()
    if os.path.lexists(record_dir):
        raise ValueError(
            f"{record_dir} exists; every proxy start needs a fresh record directory, since MetaClaw empties the "
            "records it finds there"
        )
    record_dir.mkdir(parents=True)
    Path(skills_dir).mkdir(parents=True, exist_ok=True)
    paths.service_home.mkdir(parents=True, exist_ok=True)
    process = ManagedProcess("MetaClaw proxy", [python, PROXY_SCRIPT], env, record_dir / "proxy.log", cwd=record_dir)
    process.start()
    try:
        process.wait_for_line(PATCHED_LINE, PATCHED_TIMEOUT_S)
        process.wait_for_line(ready_line(ports.proxy), READY_TIMEOUT_S)
        process.wait_for_port(ports.proxy, PORT_TIMEOUT_S)
    except BaseException:
        process.stop()
        raise
    return process


# --- after the conversations ---------------------------------------------------------------------------


def run_guards(history: Path, records: Path, log_dir: Path, *, paths: RuntimePaths) -> None:
    """Run both guards of ``metaclaw_guards.py`` with MetaClaw's Python; their output goes to ``log_dir``.

    Raises ``AdaptAbort`` with the end of the output when one fails.
    """
    python = metaclaw_python(paths)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = metaclaw_env(paths)
    for guard, args in zip(GUARDS, ([history], [history, records]), strict=True):
        try:
            done = subprocess.run(
                [str(python), str(GUARDS_SCRIPT), guard, *map(str, args)],
                capture_output=True,
                text=True,
                env=env,
                cwd=log_dir,
                timeout=GUARD_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            output = _text(expired.stdout) + _text(expired.stderr)
            (log_dir / f"{guard}.log").write_text(output, encoding="utf-8")
            raise AdaptAbort(
                f"the guard {guard} did not finish within {GUARD_TIMEOUT_S} s; it had written: {output.strip()[-500:]}"
            ) from None
        output = done.stdout + done.stderr
        (log_dir / f"{guard}.log").write_text(output, encoding="utf-8")
        if done.returncode != 0:
            raise AdaptAbort(f"the guard {guard} failed (exit {done.returncode}): {output.strip()[-500:]}")
        for line in output.splitlines():
            if line.strip().startswith("WARNING:"):  # a pass that could not tell the evolver's sources apart
                logger.warning("the guard %s passed vacuously: %s", guard, line.strip())


def _text(output: bytes | str | None) -> str:
    """What a process wrote; a timed-out run hands it over as bytes."""
    return output.decode(errors="replace") if isinstance(output, bytes) else (output or "")


def evolver_preflight(secrets: Secrets, skills: SkillsSpec, *, paths: RuntimePaths) -> None:
    """One request to the skill evolver before an evolution, made as MetaClaw's evolver makes its own (its client,
    one user message, ``max_completion_tokens``), with MetaClaw's Python.

    A URL, key or model the evolver could not use fails here, before anything is made or paid for, and not at the
    first evolution, after paid conversations. Any successful answer passes. Raises ``ConfigError`` naming the
    endpoint and the model once, never the key.
    """
    _require_secrets(secrets, _EVOLVER_SECRETS, "skill evolution")
    assert secrets.evolver_base_url and secrets.evolver_api_key and secrets.evolver_model
    python = metaclaw_python(paths)
    env = metaclaw_env(paths)
    env.update(
        OPENAI_BASE_URL=secrets.evolver_base_url,
        OPENAI_API_KEY=secrets.evolver_api_key,
        SKILL_EVOLVER_MODEL=secrets.evolver_model,
    )
    where = f"the skill evolver at {secrets.evolver_base_url} (model {secrets.evolver_model})"
    argv = [str(python), str(GUARDS_SCRIPT), "evolver", str(skills.evolver_max_completion_tokens)]
    with tempfile.TemporaryDirectory(prefix="evolver-preflight-") as cwd:
        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, env=env, cwd=cwd, timeout=EVOLVER_PREFLIGHT_TIMEOUT_S, check=False
            )
        except subprocess.TimeoutExpired:
            raise ConfigError(f"{where} did not answer within {EVOLVER_PREFLIGHT_TIMEOUT_S} s") from None
    if done.returncode != 0:
        # the key is taken out before the text is cut, so that no piece of it is left at the cut
        said = (done.stderr.strip() or done.stdout.strip()).replace(secrets.evolver_api_key, "<the evolver key>")
        raise ConfigError(f"{where} cannot be used: {said[-800:]}")
    logger.info("the skill evolver at %s (model %s) answered", secrets.evolver_base_url, secrets.evolver_model)


class _LogWatch:
    """The fatal lines (``anomalies.fatal_log_lines``) the watched logs get from the first conversation on."""

    def __init__(self, *logs: Path) -> None:
        self._starts = {log: log.stat().st_size for log in logs}

    def check(self) -> None:
        fatal: list[str] = []
        for log, start in self._starts.items():
            try:
                with open(log, "rb") as handle:
                    handle.seek(start)
                    text = handle.read().decode(errors="replace")
            except FileNotFoundError:
                raise AdaptAbort(f"the watched log {log} disappeared") from None
            fatal += fatal_log_lines(text, log.name)
        if fatal:
            more = f" (and {len(fatal) - 3} more)" if len(fatal) > 3 else ""
            raise AdaptAbort("a watched log has a fatal line: " + "; ".join(fatal[:3]) + more)


def _require_running(*processes: ManagedProcess) -> None:
    for process in processes:
        if not process.alive():
            raise AdaptAbort(
                f"the {process.name} exited (status {process.returncode}); the end of its log:\n{process.log_tail()}"
            )


# --- one evolution -----------------------------------------------------------------------------------


def _closure(history: Path, proxy_port: int, served_name: str) -> Closure:
    def close(session_id: str) -> dict[str, Any]:
        before = history_lines(history)
        close_session(proxy_port, served_name, session_id)
        return {"at": now(), "history_before": before}

    return close


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")


def _executed(record: Mapping[str, Any]) -> bool:
    if "asr" in record:
        return bool(record["asr"].get("hit"))
    return bool(record.get("attack_success"))


def _praised(record: Mapping[str, Any]) -> bool:
    return (record.get("feedback_polarity") or record.get("feedback")) == "pos"


def _evolve(
    model: ModelSpec,
    task: Task,
    run: RunDir,
    cfgs: Configs,
    secrets: Secrets,
    paths: RuntimePaths,
    order: list[str],
    attempt: int,
) -> dict[str, Any]:
    adapt = run.adapt_dir
    library, proxy_dir, server_dir = adapt / "library", adapt / "proxy", adapt / "model_server"
    history = library / HISTORY_FILE
    spec = cfgs.skills
    ctx = prepare_environment(task, "skills", paths, cfgs.ports, cfgs.ports.proxy, cfgs.eval.turn_timeout_s[task.name])
    policy = policy_for("skills", task.kind, model)
    close = _closure(history, cfgs.ports.proxy, model.skill_served_name)
    library.mkdir(parents=True)
    paths.service_home.mkdir(parents=True, exist_ok=True)
    paths.hf_home.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    # The gateway first, as an evaluation starts it: a profile or an OpenClaw version it refuses fails before a model
    # is loaded. Everything goes on ``started`` the moment it has started, so whatever ends the attempt stops it.
    gateway = Gateway(ctx.profile)
    started: list[Stoppable] = [gateway]
    try:
        with starting("gateway"):
            gateway.start()
        with starting("model server"):
            server = start_model_server(
                model.family,
                model.tinker_id,
                model.skill_served_name,
                cfgs.ports.model,
                server_dir,
                None,
                model_server_env(paths, secrets),
                temperature=cfgs.eval.temperature,
            )
            started.append(server)
        with starting("proxy"):
            proxy = start_proxy(model, library, True, True, proxy_dir, cfgs.ports, secrets, skills=spec, paths=paths)
            started.append(proxy)
        watch = _LogWatch(proxy.log_path, server_files(server_dir).log)
        for number, item_id in enumerate(order, 1):
            restarted = gateway.restart_if_needed() if number > 1 else None
            record = run_episode(ctx, task.adaptation[item_id], "train", True, close, policy=policy)
            record["conversation"] = number
            if restarted is not None:
                record["gateway_restarted"] = restarted
            _append_jsonl(adapt / "episodes.jsonl", record)
            records.append(record)
            logger.info("conversation %d/%d (%s) closed", number, len(order), item_id)
            watch.check()
            _require_running(proxy, server)
        started.remove(gateway)  # the agent is done; the evolutions still running need the proxy alone
        gateway.stop()

        def still_fine() -> None:  # before every poll: a fatal line of the evolver's own requests, a proxy that died
            watch.check()
            _require_running(proxy)

        count = drain(
            history,
            spec.conversations,
            poll_s=spec.drain_poll_s,
            stable_polls=spec.drain_stable_polls,
            max_polls=spec.drain_max_polls,
            check=still_fine,
        )
        still_fine()  # and after the last poll
    except BaseException as error:
        stop_all(started, error)
        raise
    stop_all(started)
    if count != spec.conversations:
        raise AdaptAbort(f"{spec.conversations} conversations were closed but {count} evolutions landed in {history}")
    run_guards(history, proxy_dir / "conversations.jsonl", adapt / "guards", paths=paths)
    if model.family == "inkling":
        _inkling_audit(server_files(server_dir).raw, adapt / "raw_audit.json")
    events = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    sha = lib_sha(library)
    skill_count = len(list(library.rglob("SKILL.md")))
    write_json(
        adapt / "stats.json",
        {
            "conversations": [record["id"] for record in records],
            "executions": sum(map(_executed, records)),
            "praised": sum(map(_praised, records)),
            "attempt": attempt,
            "evolver_model": secrets.evolver_model,
        },
    )
    write_json(
        adapt / MARKER,
        {
            "library": library.name,
            "sha": sha,
            "evolutions": count,
            "skills": skill_count,
            "zero_skill_events": sum(1 for event in events if event.get("num_skills_generated") == 0),
            "at": now(),
        },
    )
    return {"library": library, "sha": sha, "evolutions": count, "skills": skill_count}


def _shared_inkling_audit() -> ModuleType:
    """The raw audit of Inkling-Small's generations, shared with the RL line and the evaluation; imported only when
    Inkling-Small needs it."""
    return importlib.import_module("evolve_ipi.models.inkling_audit")


def _inkling_audit(raw: Path, out: Path) -> None:
    """Inkling-Small: the raw audit of the evolution's generations, shared with the RL line and the evaluation. Tool-call
    markup that parsed to no call is a parser failure, which would pass for a model that did not act. The audit goes to
    ``out``; a problem, or a dump that cannot be read, raises ``AdaptAbort``."""
    shared = _shared_inkling_audit()
    try:
        audit = shared.audit_raw_dump(raw, parsed_key=None)  # a model server's dump: the calls are parsed again
    except (OSError, ValueError) as error:
        raise AdaptAbort(f"the raw dump {raw} of the model server cannot be read: {error}") from error
    write_json(out, {**audit, "at": now()})
    if audit.get("unfinished_last_line"):
        logger.warning(
            "the raw dump %s ends in a line that was cut off when its writer stopped; it was not audited", raw
        )
    problems = list(shared.audit_problems(audit))  # a dump without any generation is one of them
    if problems:
        raise AdaptAbort("the raw audit of the model server's generations failed: " + "; ".join(problems))


# --- restarts ----------------------------------------------------------------------------------------

# What stops an evolution part way and is worth another try from an empty library: a crash after the start, a failed
# episode, a fatal log line, a failed guard. Anything else (a process that does not start, leftover state in the agent's
# home, a port in use, a configuration mistake) would stop the next attempt the same way.
_RESTARTABLE = (AdaptAbort, EpisodeAbort, ProcessError, OpenClawError)


def _void(adapt: Path, reason: str, details: Mapping[str, Any] | None = None) -> Path | None:
    """Move everything in ``adapt`` but earlier voids into ``adapt/_void/<time>/``, with ``why.json``."""
    if not adapt.is_dir():
        return None
    entries = sorted(entry for entry in adapt.iterdir() if entry.name != _VOID)
    if not entries:
        return None
    stamp = time.strftime("%Y%m%dT%H%M%S")
    target, suffix = adapt / _VOID / stamp, 1
    while target.exists():
        suffix += 1
        target = adapt / _VOID / f"{stamp}-{suffix}"
    target.mkdir(parents=True)
    for entry in entries:
        shutil.move(entry, target / entry.name)
    write_json(target / "why.json", {"at": now(), "reason": reason, **(details or {})})
    return target


def _why(error: BaseException, attempt: int) -> tuple[str, dict[str, Any]]:
    details: dict[str, Any] = {"attempt": attempt, **stop_failures(error)}
    if isinstance(error, EpisodeAbort):
        details.update(episode=error.record, retried_attempts=error.retried_attempts)
    return f"{type(error).__name__}: {error}", details


def _finished(adapt: Path) -> dict[str, Any] | None:
    marker = adapt / MARKER
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        library, recorded = adapt / data["library"], data["sha"]
        result = {"library": library, "sha": recorded, "evolutions": data["evolutions"], "skills": data["skills"]}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise LibraryError(f"{marker} cannot be read ({error}); a person has to look at {adapt}") from None
    if not library.is_dir():
        raise LibraryError(f"{marker} records a finished library, but {library} is gone; a person has to look")
    found = lib_sha(library)
    if found != recorded:
        raise LibraryError(
            f"{library} has changed since its evolution finished: its sha256 is {found}, {marker} records {recorded}. "
            f"A finished evolution is never run again: restore the library, or move {adapt} away to evolve a new one."
        )
    return result


def check_inputs(model: ModelSpec, cfgs: Configs, secrets: Secrets, paths: RuntimePaths, order: list[str]) -> None:
    """What skill evolution of ``model`` needs, checked before anything starts: the model takes part, the secrets, MetaClaw's
    Python, the patched frameworks, the Inkling audit, the length of the skill order. Raises ``ConfigError`` or
    ``PatchError``. The runner calls it for every combination before the first one starts, and ``adapt_skills`` again."""
    if "skills" not in model.mechanisms:
        raise ConfigError(
            f"model {model.name} does not take part in skill evolution; models.yaml gives it the mechanisms "
            f"{', '.join(model.mechanisms)}"
        )
    _require_secrets(secrets, tuple(_SECRET_NAMES), "skill evolution")
    metaclaw_python(paths)
    verify_patches()
    if model.family == "inkling":
        _shared_inkling_audit()  # its audit runs after the evolution; a missing module must stop it before it starts
    if len(order) != cfgs.skills.conversations:
        raise ConfigError(
            f"the skill order holds {len(order)} conversations, but skills.yaml says {cfgs.skills.conversations}"
        )


def adapt_skills(
    model: ModelSpec,
    task: Task,
    run: RunDir,
    cfgs: Configs,
    secrets: Secrets,
    *,
    paths: RuntimePaths | None = None,
    preflight: bool = True,
) -> dict[str, Any]:
    """Evolve the skill library of ``model`` on ``task`` into ``run.adapt_dir / "library"``.

    Returns ``{"library": path, "sha": ..., "evolutions": 12, "skills": n}``. Everything that can be checked is checked
    before anything starts: the secrets (``TINKER_API_KEY`` and the evolver's), MetaClaw's Python, the patched
    frameworks, the skill order, and (unless the evolution is finished already) the evolver itself, with one request;
    ``preflight=False`` leaves that request out, for a caller that has just made it. A finished evolution is returned as
    it is; an unfinished one left by an earlier run is moved to ``adapt/_void/`` first. Writes ``adapt/episodes.jsonl``
    (one record per conversation), the library, the proxy's and the model server's directories, the guards' logs,
    ``stats.json`` and, last, ``library.json``. An interruption is never turned into another attempt: a process that
    does not stop while one propagates is attached to it as a note.
    """
    paths = RuntimePaths() if paths is None else paths
    order = load_skill_order(task.name)
    check_inputs(model, cfgs, secrets, paths, order)
    adapt = run.adapt_dir
    finished = _finished(adapt)
    if finished is not None:
        return finished
    if preflight:
        evolver_preflight(secrets, cfgs.skills, paths=paths)  # before anything is made or paid for
    _void(adapt, "an earlier run stopped before its evolution finished (interrupted or crashed)")
    failures: list[str] = []
    attempts = cfgs.skills.max_restarts + 1
    for attempt in range(1, attempts + 1):
        try:
            return _evolve(model, task, run, cfgs, secrets, paths, order, attempt)
        except _RESTARTABLE as error:
            reason, details = _why(error, attempt)
            _void(adapt, reason, details)
            failures.append(f"attempt {attempt}: {reason}")
            logger.warning("skill evolution of %s on %s, attempt %d: %s", model.name, task.name, attempt, reason)
            if attempt == attempts:
                raise AdaptAbort(
                    f"the skill evolution of {model.name} on {task.name} failed {attempts} times, which is the limit "
                    f"of 1 + skills.max_restarts={cfgs.skills.max_restarts}:\n  " + "\n  ".join(failures)
                ) from error
        except Exception as error:
            reason, details = _why(error, attempt)
            _void(adapt, reason, details)
            raise
    raise AssertionError("unreachable")
