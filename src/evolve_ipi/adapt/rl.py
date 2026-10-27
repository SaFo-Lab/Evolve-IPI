"""RL adaptation: OpenClaw-RL trains the agent's model on the agent's own episodes, in two stages of 32 updates.

    adapt/stage1/                  updates 1 to 32, from the base model
    adapt/stage2/                  updates 33 to 64, resumed from update 32's checkpoint with its optimizer state
        trainer.log                the trainer process's output
        records/                   the trainer's own records (turns, reward-model scores; Inkling: raw generations)
        episodes.jsonl             one record per episode, with its epoch and the trainer's progress after it
        stats.json                 episodes, executions, samples, mean reward per update, items used
        checkpoint.json            the stage's checkpoint, found on Tinker from the trainer log
        fingerprint.json           the checkpoint is not the base model
    adapt/checkpoints.json         {"base_model", "update_32", "update_64"}
    adapt/_void/stage<k>-<time>/   a stage attempt that failed or was interrupted, with why.json

A stage starts a trainer of its own and runs the stage's adaptation items in order, one episode at a time: the
authorizing request, the judges and the user's feedback. The trainer makes samples of the agent's turns and updates
the model on every batch of them. Once batch_size x updates_per_stage samples were submitted, no further episode
starts and the stage waits for the trainer's last update. In stage 2 a watcher stops the trainer right after that
update, so that no update past 64 can begin; an episode it cuts short is recorded with a ``cap_cut`` anomaly. The
stage's checkpoint is then looked up on Tinker and fingerprinted, each in a process of its own.

A stage that fails is moved to ``_void/`` and run again from its starting point, at most ``rl.max_restarts`` times:
an episode that stops the run (``EpisodeAbort``), a trainer that dies, does not get ready or logs a fatal line, an
order that runs out before the last update, a checkpoint that is not found or not trained. A stage left unfinished
by Ctrl-C or a crash is moved there when the next run starts. A finished stage is never run again. Leftover state in
the agent's home, a busy port and configuration mistakes stop the run at once.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evolve_ipi.adapt import AdaptAbort
from evolve_ipi.adapt.rl_entry import ENTRY_LINE, RAW_DUMP_NAME, RESTORED_LINE, family_of
from evolve_ipi.agent.anomalies import NONENGAGEMENT, fatal_log_lines, policy_for
from evolve_ipi.agent.episode import EpisodeAbort, EpisodeContext, prepare_environment, run_episode
from evolve_ipi.agent.openclaw import Gateway
from evolve_ipi.children import trainer_env
from evolve_ipi.config import ConfigError, Configs, ModelSpec, Secrets
from evolve_ipi.models import checkpoints, inkling_audit
from evolve_ipi.paths import Kind, RuntimePaths
from evolve_ipi.procs import ManagedProcess, ProcessError, require_free
from evolve_ipi.records import RunDir, now, write_json
from evolve_ipi.tasks import RLOrder, Task, TaskDataError, load_rl_order
from evolve_ipi.thirdparty import PATCHES, PatchError, verify_patches

__all__ = [
    "CAP_POLL_S",
    "STEP_POLL_S",
    "TrainerProgress",
    "adapt_rl",
    "check_setup",
    "epochs",
    "stage_stats",
    "stop_launching",
    "trainer_flags",
    "trainer_progress",
]

logger = logging.getLogger(__name__)

SERVED_MODEL_NAME = "policy"  # the model the RL line's OpenClaw profile asks its endpoint for
TRAINER_HOST = "127.0.0.1"
TRAINER_READY_TIMEOUT_S = 1800  # the trainer loads the tokenizer, connects to Tinker and, in stage 2, loads the state
PORT_TIMEOUT_S = 120
CAP_POLL_S = 0.25  # how often stage 2's watcher reads the trainer log for the last update
STEP_POLL_S = 10  # how often the trainer log is read while the stage waits for its last update
CHECKPOINT_TOOL_TIMEOUT_S = 1800
CHECKPOINTS_FILE = "checkpoints.json"
ROLLOUT_PATCH = next(patch.path for patch in PATCHES if patch.target == "openclaw-tinker/rollout.py")

# What the trainer logs per sample it queues ("[Server] submitted ..."; the combined method tags its samples), per
# update ("[Trainer] step N done | batch=... mean_reward=..."), and when it starts collecting a batch.
_SUBMITTED = re.compile(r"\[Server\] submitted (?:(?:OPD\+RL|OPD|RL) )?session=")
_SUBMITTED_SCORE = re.compile(r"\[Server\] submitted (?:(?:OPD\+RL|OPD|RL) )?session=\S+ idx=\d+ score=(-?[0-9.]+)")
_STEP_DONE = re.compile(r"\[Trainer\] step (\d+) done")
_STEP_REWARD = re.compile(r"step \d+ done \| batch=\d+ mean_reward=(-?[0-9.]+)")
_BATCH = re.compile(r"collecting batch \(size=(\d+)\)")


# --- the trainer's command and environment ---------------------------------------------------------------------


def _float_flag(value: float) -> str:
    """A rate as the recipe writes it: 1e-4, not 0.0001."""
    mantissa, exponent = f"{value:.12e}".split("e")
    text = f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    return text if float(text) == value else repr(value)


def trainer_flags(model: ModelSpec, cfgs: Configs, record_dir: Path, resume: str | None = None) -> list[str]:
    """The arguments of OpenClaw-RL's ``run.py`` for one stage: the recipe of ``rl.yaml``, the policy port, and the
    stage-1 checkpoint a stage-2 trainer resumes from."""
    rl = cfgs.rl
    flags = [
        "--method", "rl", "--model-name", model.tinker_id, "--batch-size", str(rl.batch_size),
        "--prm-m", str(rl.prm_m), "--lora-rank", str(rl.lora_rank), "--learning-rate", _float_flag(rl.learning_rate),
        "--max-steps", str(rl.max_steps), "--save-interval", str(rl.save_interval), "--proxy-host", TRAINER_HOST,
        "--proxy-port", str(cfgs.ports.policy), "--served-model-name", SERVED_MODEL_NAME,
        "--record-dir", str(record_dir),
    ]  # fmt: skip
    if resume:
        flags += ["--resume-from-ckpt", resume]
    return flags


def _trainer_command(flags: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", "evolve_ipi.adapt.rl_entry", *flags]


def _checkpoints_command(args: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", "evolve_ipi.models.checkpoints", *args]


# --- reading the trainer log -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainerProgress:
    """What the trainer has taken in and trained, read from its log."""

    submitted: int  # samples queued for training
    steps_done: int  # the last update completed
    batch_size: int | None  # from "collecting batch (size=N)"; None before the trainer logged it
    read_at: str

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def trainer_progress(log_text: str) -> TrainerProgress:
    text = checkpoints.strip_ansi(log_text)
    steps = [int(step) for step in _STEP_DONE.findall(text)]
    batch = _BATCH.findall(text)
    return TrainerProgress(
        submitted=len(_SUBMITTED.findall(text)),
        steps_done=max(steps) if steps else 0,
        batch_size=int(batch[0]) if batch else None,
        read_at=now(),
    )


def stop_launching(progress: TrainerProgress, updates: int) -> bool:
    """No further episode starts once the trainer was given enough samples for ``updates`` updates."""
    if progress.batch_size is None:
        raise AdaptAbort("the trainer log has no 'collecting batch (size=N)' line")
    return progress.submitted >= progress.batch_size * updates


def _fatal_lines(trainer: ManagedProcess) -> list[str]:
    return fatal_log_lines(trainer.log_text(), "trainer.log")


# --- epochs and statistics ---------------------------------------------------------------------------------------


def epochs(order: Sequence[str], kind: Kind) -> list[int]:
    """The epoch of each position of a stage's order.

    A File-Create order is a sequence of passes over its items, each in an order of its own. A service task's order
    is one sequence of the items played twice; its epoch is the play.
    """
    if not order:
        raise TaskDataError("an RL stage's order is empty")
    if kind == "file":
        size = len(set(order))
        for start in range(0, len(order), size):
            chunk = order[start : start + size]
            if len(chunk) != size or set(chunk) != set(order):
                raise TaskDataError(
                    f"positions {start}.. of a File-Create RL order are not a pass over its {size} items"
                )
    else:
        size = len(order) // 2
        if len(order) % 2 or list(order[:size]) != list(order[size:]):
            raise TaskDataError("a service task's RL order is not one sequence of items played twice")
    return [position // size for position in range(len(order))]


def _cap_cut(record: Mapping[str, Any]) -> bool:
    return any(anomaly.get("type") == "cap_cut" for anomaly in record.get("anomalies") or [])


def stage_stats(records: Sequence[Mapping[str, Any]], kind: Kind, log_text: str) -> dict[str, Any]:
    """What a stage trained on: its episodes, how often the injected task was carried out under the authorizing
    request, the samples and their rewards, and how often each item was used.

    A non-engaged episode called no program (service tasks), or called no tool or gave an empty reply (File-Create);
    under RL it got its feedback and training went on. An episode the cap cut short is not counted as one.
    """
    text = checkpoints.strip_ansi(log_text)
    scores = [float(score) for score in _SUBMITTED_SCORE.findall(text)]
    rewards = [float(reward) for reward in _STEP_REWARD.findall(text)]
    whole = [record for record in records if not _cap_cut(record)]
    if kind == "file":
        executed = sum(1 for record in records if record.get("created"))
        nonengaged = sum(
            1 for record in whole if not record.get("tool_calls") or not (record.get("reply") or "").strip()
        )
    else:
        executed = sum(1 for record in records if (record.get("asr") or {}).get("hit"))
        nonengaged = sum(
            1
            for record in whole
            if any(anomaly.get("type") in NONENGAGEMENT for anomaly in record.get("anomalies") or [])
        )
    uses: dict[str, int] = {}
    for record in records:
        uses[record["id"]] = uses.get(record["id"], 0) + 1
    return {
        "episodes": len(records),
        "executions_under_authorization": executed,
        "execution_rate": round(executed / len(records), 4) if records else None,
        "samples_submitted": len(scores),
        "samples_positive": sum(1 for score in scores if score > 0),
        "samples_negative": sum(1 for score in scores if score < 0),
        "samples_zero": sum(1 for score in scores if score == 0),
        "mean_reward_per_update": rewards,
        "items_used": len(uses),
        "uses_per_item": uses,
        "nonengaged_episodes_continued": nonengaged,
        "cap_cut_episodes": len(records) - len(whole),
    }


# --- files -------------------------------------------------------------------------------------------------------


def _append(path: Path, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_record(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"there is no {path.name}"
    except (OSError, ValueError) as error:
        return None, f"{path.name} cannot be read ({error})"
    if not isinstance(value, dict):
        return None, f"{path.name} is not a JSON object"
    return value, ""


def _void(directory: Path, adapt: Path, reason: str) -> Path:
    """Move a stage attempt out of the way, with the reason, so that nothing reads it as a result."""
    root = adapt / "_void"
    root.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    target, number = root / f"{directory.name}-{stamp}", 2
    while target.exists():
        target, number = root / f"{directory.name}-{stamp}-{number}", number + 1
    directory.rename(target)
    write_json(target / "why.json", {"reason": reason, "at": now()})
    logger.warning("%s moved to %s: %s", directory, target, reason)
    return target


# --- one stage ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Stage:
    number: int
    ids: list[str]
    epochs: list[int]
    resume: str | None  # the checkpoint the stage starts from; None for the base model
    model: ModelSpec
    task: Task
    cfgs: Configs
    secrets: Secrets
    paths: RuntimePaths
    adapt: Path

    @property
    def directory(self) -> Path:
        return self.adapt / f"stage{self.number}"

    @property
    def updates(self) -> int:
        return self.cfgs.rl.updates_per_stage


@dataclass
class _Driven:
    records: list[dict[str, Any]]
    at_stop: TrainerProgress
    at_end: TrainerProgress | None = None
    cut: str | None = None  # the episode the cap cut short


def _finished(stage: _Stage) -> tuple[dict[str, Any] | None, str]:
    """The stage's checkpoint record when the stage finished, else why it did not."""
    checkpoint, why = _read_record(stage.directory / "checkpoint.json")
    if checkpoint is None:
        return None, why
    fingerprint, why = _read_record(stage.directory / "fingerprint.json")
    if fingerprint is None:
        return None, why
    final = checkpoint.get("final")
    problems = {
        f"checkpoint.json is not stage {stage.number}'s": checkpoint.get("stage") != stage.number,
        f"the stage did not start from {stage.resume or 'the base model'}": checkpoint.get("resumed_from")
        != stage.resume,
        "checkpoint.json names no loadable final checkpoint": not isinstance(final, str)
        or not final.endswith(f"/step_{stage.updates:04d}")
        or checkpoint.get("verified_loadable") is not True,
        "fingerprint.json is not a passing fingerprint of that checkpoint": fingerprint.get("path") != final
        or fingerprint.get("base_model") != stage.model.tinker_id
        or fingerprint.get("non_base") is not True
        or fingerprint.get("min_diff") != stage.cfgs.rl.fingerprint_min_diff,
    }
    found = [problem for problem, holds in problems.items() if holds]
    return (None, "; ".join(found)) if found else (checkpoint, "")


class _CapWatcher:
    """Stops the stage-2 trainer as soon as its log shows the stage's last update, before it can begin another."""

    def __init__(self, trainer: ManagedProcess, marker: str) -> None:
        self.trainer = trainer
        self.marker = marker.encode()
        self.fired_at: str | None = None
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="cap watcher", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._done.set()
        self._thread.join(timeout=10)

    def _watch(self) -> None:
        offset, tail, keep = 0, b"", len(self.marker) - 1
        while self.trainer.alive() and not self._done.is_set():
            with open(self.trainer.log_path, "rb") as log:
                log.seek(offset)
                chunk = log.read()
                offset = log.tell()
            window = tail + chunk
            if self.marker in window:
                pid = self.trainer.pid
                assert pid is not None  # the watcher starts after the trainer did
                self.fired_at = now()
                try:
                    os.killpg(pid, signal.SIGTERM)  # the trainer leads a session (and process group) of its own
                except (ProcessLookupError, PermissionError):  # it has exited already
                    pass
                return
            tail = window[-keep:]
            self._done.wait(CAP_POLL_S)


def _checkpoint_tool(stage: _Stage, args: list[str], env: Mapping[str, str]) -> None:
    """Run ``python -m evolve_ipi.models.checkpoints ARGS``; a failure aborts the stage."""
    log_path = stage.directory / "checkpoints.log"
    with open(log_path, "ab") as log:
        try:
            done = subprocess.run(
                _checkpoints_command(args),
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=stage.directory,
                timeout=CHECKPOINT_TOOL_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise AdaptAbort(f"the checkpoint {args[0]} did not finish within {CHECKPOINT_TOOL_TIMEOUT_S} s") from error
    if done.returncode != 0:
        tail = " | ".join(log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:])
        raise AdaptAbort(f"the checkpoint {args[0]} of {args[1]} failed (exit status {done.returncode}): {tail}")


def _fingerprint(stage: _Stage, path: str, out: Path, env: Mapping[str, str]) -> None:
    min_diff = stage.cfgs.rl.fingerprint_min_diff
    args = ["fingerprint", path, "--base-model", stage.model.tinker_id, "--out", str(out), "--min-diff", str(min_diff)]
    _checkpoint_tool(stage, args, env)
    record, why = _read_record(out)
    if record is None or record.get("path") != path or record.get("non_base") is not True:
        raise AdaptAbort(f"the fingerprint of {path} did not pass: {why or record}")


def _wait_until_ready(stage: _Stage, trainer: ManagedProcess) -> None:
    """The trainer collects its first batch on the policy port, came up through the trainer entry and, in stage 2,
    resumed from the stage-1 checkpoint with its optimizer state."""
    try:
        trainer.wait_for_line(f"collecting batch (size={stage.cfgs.rl.batch_size})", TRAINER_READY_TIMEOUT_S)
        trainer.wait_for_port(stage.cfgs.ports.policy, PORT_TIMEOUT_S)
    except ProcessError as error:
        raise AdaptAbort(f"the stage-{stage.number} trainer did not get ready: {error}") from error
    text = trainer.log_text()
    entry = ENTRY_LINE.format(family=stage.model.family, model=stage.model.tinker_id, resume=stage.resume or "none")
    if entry not in checkpoints.strip_ansi(text).splitlines():
        raise AdaptAbort(f"the trainer log {trainer.log_path} lacks {entry!r}: the trainer did not start as it should")
    if stage.resume is not None:
        log = checkpoints.parse_trainer_log(text)
        resumed = [path for _, path in log.resumed_from]
        restored = [path for _, path in log.restored_from]
        if resumed != [stage.resume]:
            raise AdaptAbort(
                f"the stage-2 trainer log {trainer.log_path} shows 'resuming from checkpoint: ' {resumed}; "
                f"expected {stage.resume} exactly once"
            )
        if restored != [stage.resume]:
            raise AdaptAbort(
                f"the stage-2 trainer log {trainer.log_path} does not show {RESTORED_LINE % stage.resume!r} exactly "
                f"once (it shows {restored}): the optimizer state was not restored"
            )


def _cut_record(
    abort: EpisodeAbort, item_id: str, epoch: int, progress: TrainerProgress, absolute: int
) -> dict[str, Any]:
    record: dict[str, Any] = dict(abort.record) if abort.record is not None else {"id": item_id, "mode": "train"}
    anomalies = list(record.get("anomalies") or [])
    anomalies.append(
        {
            "type": "cap_cut",
            "detail": f"the trainer was stopped after update {absolute} while this episode ran ({abort.reason}); "
            "its later turns were not trained on",
        }
    )
    record.update(anomalies=anomalies, epoch=epoch, trainer=progress.as_record())
    return record


def _drive(stage: _Stage, ctx: EpisodeContext, trainer: ManagedProcess, watcher: _CapWatcher | None) -> _Driven:
    """Run episodes until the trainer was given enough samples, the cap cut an episode short, or the order ran out."""
    policy = policy_for("rl", stage.task.kind, stage.model)
    path = stage.directory / "episodes.jsonl"
    path.touch()
    records: list[dict[str, Any]] = []
    for index, item_id in enumerate(stage.ids):
        if not trainer.alive():
            raise AdaptAbort(f"the trainer exited (status {trainer.returncode}) before update {stage.updates}")
        epoch = stage.epochs[index]
        try:
            record = run_episode(ctx, stage.task.adaptation[item_id], "train", True, policy=policy)
        except EpisodeAbort as abort:
            text = trainer.log_text()
            marker = f"[Trainer] step {stage.updates} done"
            if watcher is not None and watcher.fired_at is not None and marker in text:
                progress = trainer_progress(text)
                cut = _cut_record(abort, item_id, epoch, progress, stage.updates * stage.number)
                _append(path, cut)
                records.append(cut)
                logger.info("stage %d: the cap stopped the trainer during %s", stage.number, item_id)
                return _Driven(records, progress, cut=item_id)
            if abort.record is not None:
                _append(path, dict(abort.record, epoch=epoch, aborted=abort.reason))
            raise
        record["epoch"] = epoch
        progress = trainer_progress(trainer.log_text())
        record["trainer"] = progress.as_record()
        _append(path, record)
        records.append(record)
        logger.info(
            "stage %d episode %d %s: %d samples submitted, update %d",
            stage.number,
            index + 1,
            item_id,
            progress.submitted,
            progress.steps_done,
        )
        fatal = _fatal_lines(trainer)
        if fatal:
            raise AdaptAbort(f"a fatal line in the trainer log: {fatal[0]}")
        if stop_launching(progress, stage.updates):
            return _Driven(records, progress)
    progress = trainer_progress(trainer.log_text())
    raise AdaptAbort(
        f"update {stage.updates} not reached: {progress.submitted} samples submitted after {len(records)} episodes; "
        "the order is exhausted"
    )


def _wait_for_update(stage: _Stage, trainer: ManagedProcess) -> TrainerProgress:
    """Wait until the trainer has completed the stage's last update."""
    timeout = stage.cfgs.rl.until_step_timeout_s
    deadline = time.monotonic() + timeout
    while True:
        progress = trainer_progress(trainer.log_text())
        if progress.steps_done >= stage.updates:
            return progress
        fatal = _fatal_lines(trainer)
        if fatal:
            raise AdaptAbort(f"a fatal line in the trainer log: {fatal[0]}")
        if not trainer.alive():
            raise AdaptAbort(
                f"the trainer exited (status {trainer.returncode}) at update {progress.steps_done}, before update "
                f"{stage.updates}"
            )
        if time.monotonic() >= deadline:
            raise AdaptAbort(
                f"the trainer did not reach update {stage.updates} within {timeout} s (at update {progress.steps_done})"
            )
        time.sleep(min(STEP_POLL_S, max(1, timeout)))


def _train(stage: _Stage, ctx: EpisodeContext, env: Mapping[str, str]) -> tuple[_Driven, _CapWatcher | None]:
    """Start the stage's trainer, run the episodes, wait for the last update, and stop the trainer."""
    flags = trainer_flags(stage.model, stage.cfgs, stage.directory / "records", stage.resume)
    trainer = ManagedProcess(
        f"trainer (stage {stage.number})",
        _trainer_command(flags),
        env,
        stage.directory / "trainer.log",
        cwd=stage.directory,
    )
    trainer.start()
    watcher: _CapWatcher | None = None
    try:
        _wait_until_ready(stage, trainer)
        if stage.number == 2:
            watcher = _CapWatcher(trainer, f"[Trainer] step {stage.updates} done")
            watcher.start()
        driven = _drive(stage, ctx, trainer, watcher)
        driven.at_end = _wait_for_update(stage, trainer)
        fatal = _fatal_lines(trainer)
        if fatal:
            raise AdaptAbort(f"a fatal line in the trainer log: {fatal[0]}")
    finally:
        if watcher is not None:
            watcher.stop()
        trainer.stop()
    return driven, watcher


def _run_stage(stage: _Stage) -> dict[str, Any]:
    """One attempt at a stage, in a fresh directory; returns the stage's checkpoint record."""
    cfgs, directory = stage.cfgs, stage.directory
    require_free([cfgs.ports.policy])
    directory.mkdir(parents=True)
    env = trainer_env(stage.paths, stage.secrets)
    for home in (stage.paths.service_home, stage.paths.hf_home):
        home.mkdir(parents=True, exist_ok=True)
    if stage.resume is not None:
        # the checkpoint the stage resumes from must still load and differ from the base model
        _fingerprint(stage, stage.resume, directory / "resume_fingerprint.json", env)
    ctx = prepare_environment(
        stage.task, "rl", stage.paths, cfgs.ports, cfgs.ports.policy, cfgs.eval.turn_timeout_s[stage.task.name]
    )
    with Gateway(ctx.profile):
        driven, watcher = _train(stage, ctx, env)

    log_text = (directory / "trainer.log").read_text(encoding="utf-8", errors="replace")
    stats = stage_stats(driven.records, stage.task.kind, log_text)
    stats["until"] = {
        "target": stage.updates,
        "trainer_at_stop": driven.at_stop.as_record(),
        "trainer_at_end": driven.at_end.as_record() if driven.at_end else None,
    }
    if watcher is not None:
        stats["cap"] = {"fired_at": watcher.fired_at, "cut_episode": driven.cut}
    write_json(directory / "stats.json", stats)
    if stage.model.family == "inkling":
        dump = directory / "records" / RAW_DUMP_NAME
        try:
            audit = inkling_audit.audit_raw_dump(dump, parsed_key="parsed_calls")
        except (OSError, ValueError) as error:
            raise AdaptAbort(f"the trainer's raw dump {dump} cannot be audited: {error}") from error
        write_json(directory / "raw_audit.json", audit)
        if audit.get("unfinished_last_line"):
            logger.warning(
                "the trainer's raw dump %s ends in a line that was cut off when the trainer stopped; it was not "
                "audited",
                dump,
            )
        problems = inkling_audit.audit_problems(audit)
        if problems:
            raise AdaptAbort("raw audit: " + "; ".join(problems))

    collect = ["collect", str(directory / "trainer.log"), "--stage", str(stage.number)]
    collect += ["--base-model", stage.model.tinker_id, "--out", str(directory / "checkpoint.json")]
    if stage.resume is not None:
        collect += ["--resume-from", stage.resume]
    _checkpoint_tool(stage, collect, env)
    checkpoint, why = _read_record(directory / "checkpoint.json")
    if checkpoint is None or not isinstance(checkpoint.get("final"), str):
        raise AdaptAbort(f"the checkpoint collection left no checkpoint record: {why or checkpoint}")
    _fingerprint(stage, checkpoint["final"], directory / "fingerprint.json", env)
    record, why = _finished(stage)
    if record is None:
        raise AdaptAbort(f"stage {stage.number} did not finish as it should: {why}")
    logger.info("stage %d: %s", stage.number, record["final"])
    return record


def _run_with_restarts(stage: _Stage, unfinished: str) -> dict[str, Any]:
    """Run a stage until it finishes, moving every failed or interrupted attempt to ``_void/``."""
    reason = f"an earlier run left this stage unfinished (interrupted, stopped or failed): {unfinished}"
    failures = 0
    while True:
        if stage.directory.exists():
            _void(stage.directory, stage.adapt, reason)
        try:
            return _run_stage(stage)
        except (AdaptAbort, EpisodeAbort) as error:
            failures += 1
            reason = f"{type(error).__name__}: {error}"
            if failures > stage.cfgs.rl.max_restarts:
                if stage.directory.exists():
                    _void(stage.directory, stage.adapt, reason)
                raise AdaptAbort(
                    f"stage {stage.number} failed {failures} time{'s' if failures != 1 else ''}; the last failure: "
                    f"{reason}"
                ) from error
            logger.warning("stage %d failed (%s); starting it again from its starting point", stage.number, reason)


# --- the run -----------------------------------------------------------------------------------------------------


def check_setup(model: ModelSpec, task: Task, cfgs: Configs, secrets: Secrets) -> tuple[RLOrder, list[int], list[int]]:
    """Everything that can be checked before anything starts; returns the order and the epochs of its two stages.

    Raises ``ConfigError`` (the model, the secret, the recipe of ``rl.yaml``), ``PatchError`` or ``TaskDataError``. The
    runner calls it for every combination before the first one starts, and ``adapt_rl`` calls it again."""
    if "rl" not in model.mechanisms:
        raise ConfigError(
            f"models.yaml: {model.name} does not take part in RL adaptation (mechanisms: {', '.join(model.mechanisms)})"
        )
    try:
        family = family_of(model.tinker_id)
    except ValueError as error:
        raise ConfigError(f"models.yaml: {model.name}: {error}") from None
    if family != model.family:
        raise ConfigError(
            f"models.yaml: {model.name} has family {model.family!r}, but the trainer reads {model.tinker_id} as "
            f"{family!r}"
        )
    if not secrets.tinker_api_key:
        raise ConfigError("TINKER_API_KEY is required for RL adaptation")
    rl = cfgs.rl
    if rl.updates_per_stage != checkpoints.UPDATES_PER_STAGE:
        raise ConfigError(
            f"rl.yaml: updates_per_stage is {rl.updates_per_stage}; the checkpoint check takes a stage's checkpoint "
            f"at update {checkpoints.UPDATES_PER_STAGE}"
        )
    if rl.save_interval != rl.updates_per_stage:
        raise ConfigError(
            f"rl.yaml: save_interval is {rl.save_interval}; it must equal updates_per_stage, so that a stage saves "
            "exactly one checkpoint, after its last update"
        )
    if rl.max_steps < 2 * rl.updates_per_stage:
        raise ConfigError(
            f"rl.yaml: max_steps is {rl.max_steps}; it must be at least 2 x updates_per_stage, so that the trainer "
            "neither stops by itself nor saves again before the stage stops it"
        )
    sums = verify_patches()
    if ROLLOUT_PATCH not in sums:
        raise PatchError(f"{ROLLOUT_PATCH} is not among the patched files")
    order = load_rl_order(task.name)
    return order, epochs(order.stage1, task.kind), epochs(order.stage2, task.kind)


def adapt_rl(
    model: ModelSpec,
    task: Task,
    run: RunDir,
    cfgs: Configs,
    secrets: Secrets,
    *,
    paths: RuntimePaths | None = None,
) -> dict[str, str]:
    """Adapt ``model`` on ``task`` with RL; return the checkpoints of updates 32 and 64.

    Finished stages are kept; a stage that is not finished is run (again) from its starting point. See the module
    docstring for the files and the rules.
    """
    paths = paths if paths is not None else RuntimePaths()
    order, epochs1, epochs2 = check_setup(model, task, cfgs, secrets)
    adapt = run.adapt_dir
    adapt.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    resume: str | None = None
    for number, ids, stage_epochs in ((1, order.stage1, epochs1), (2, order.stage2, epochs2)):
        stage = _Stage(number, ids, stage_epochs, resume, model, task, cfgs, secrets, paths, adapt)
        record, why = _finished(stage)
        if record is None:
            (adapt / CHECKPOINTS_FILE).unlink(missing_ok=True)
            record = _run_with_restarts(stage, why)
        resume = record["final"]
        results[f"update_{stage.updates * number}"] = record["final"]
    write_json(adapt / CHECKPOINTS_FILE, {"base_model": model.tinker_id, **results})
    return results
