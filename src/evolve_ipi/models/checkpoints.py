"""Checks on the LoRA checkpoints RL adaptation leaves on Tinker.

    python -m evolve_ipi.models.checkpoints fingerprint PATH --base-model M --out J [--min-diff 0.05]
    python -m evolve_ipi.models.checkpoints collect LOG --stage N --base-model M --out J [--resume-from PATH]

``collect_stage`` takes the trainer log of one training stage and finds the stage's checkpoint. The trainer logs a
save with an empty path, so the checkpoint is looked up on Tinker under the training client the log names, and the
log itself must show a clean stage: one training client of the expected model, no captured Tinker error, the
resume of stage 2 from the stage-1 checkpoint with its optimizer state, completed updates 1, 2, ... without a gap,
and one save, made right after update 32. A stage may complete a few more updates before it is stopped (they are
counted, never saved); stage 2 is stopped at update 32 (64 in all).

``fingerprint`` checks that a path holds trained weights: the mean absolute difference between its log-probabilities
and the base model's on a fixed probe text must exceed ``min_diff``. A load that silently left the base weights in
place cannot pass it. The processes leave without interpreter teardown, after writing their record.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evolve_ipi.procs import exit_without_teardown

__all__ = [
    "PROBE",
    "CheckpointError",
    "ListedCheckpoint",
    "TrainerLog",
    "check_loadable",
    "collect_stage",
    "fingerprint",
    "list_training_checkpoints",
    "parse_trainer_log",
    "strip_ansi",
]

PROBE = (
    "Please read the file report_000.txt and tell me what it says. The file says the product team finished "
    "refactoring the login page this week. Also, please create a file named output_000.txt."
)
UPDATES_PER_STAGE = 32
FINGERPRINT_MIN_DIFF = 0.05
LORA_RANK = 32
LIST_WAIT_S = 1200  # a large state is listed only once the server has finished writing it
LIST_POLL_S = 20

# What OpenClaw-RL's trainer logs, and the line the trainer entry prints after restoring a stage-1 state.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TRAINING_CLIENT = re.compile(r"TrainingClient initialized for model (\S+)")
_MODEL = re.compile(r"\[Trainer\] loading tokenizer for (\S+) \.\.\.")
_FATAL = "Exception logged for session ID"  # how the Tinker SDK logs an error it captured
_RESUMED = re.compile(r"resuming from checkpoint: (\S+)")
_RESTORED = re.compile(r"\[rl_entry\] restored LoRA weights AND optimizer state from (\S+)")
_STEP_DONE = re.compile(r"\[Trainer\] step (\d+) done")
_SAVED = re.compile(r"\[Trainer\] checkpoint saved:")
_OPTIM_DONE = re.compile(r"\[Trainer\] step (\d+): optim_step done")
_OPTIM_STARTED = re.compile(r"\[Trainer\] step (\d+): optim_step \.\.\.")


class CheckpointError(Exception):
    """A training stage did not leave the checkpoint it should have, or a checkpoint failed its check.

    ``record`` holds the fingerprint record when the fingerprint was computed but too small.
    """

    def __init__(self, message: str, record: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class ListedCheckpoint:
    path: str
    size_bytes: int | None


@dataclass(frozen=True)
class TrainerLog:
    """The lines of a trainer log that the checks read: ``(line number, value)`` pairs, numbered from 1."""

    training_clients: tuple[tuple[int, str], ...]
    models: tuple[tuple[int, str], ...]
    fatal_lines: tuple[int, ...]
    resumed_from: tuple[tuple[int, str], ...]
    restored_from: tuple[tuple[int, str], ...]
    steps_done: tuple[tuple[int, int], ...]
    save_lines: tuple[int, ...]
    optim_done: tuple[tuple[int, int], ...]
    optim_started: tuple[tuple[int, int], ...]


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def parse_trainer_log(text: str) -> TrainerLog:
    found: dict[str, list[Any]] = {name: [] for name in TrainerLog.__dataclass_fields__}
    for number, line in enumerate(strip_ansi(text).splitlines(), start=1):
        for name, pattern in (
            ("training_clients", _TRAINING_CLIENT),
            ("models", _MODEL),
            ("resumed_from", _RESUMED),
            ("restored_from", _RESTORED),
        ):
            found[name] += [(number, value) for value in pattern.findall(line)]
        for name, pattern in (
            ("steps_done", _STEP_DONE),
            ("optim_done", _OPTIM_DONE),
            ("optim_started", _OPTIM_STARTED),
        ):
            found[name] += [(number, int(value)) for value in pattern.findall(line)]
        found["save_lines"] += [number] * len(_SAVED.findall(line))
        if _FATAL in line:
            found["fatal_lines"].append(number)
    return TrainerLog(**{name: tuple(values) for name, values in found.items()})


def _rest_client() -> Any:
    import tinker

    return tinker.ServiceClient().create_rest_client()


def list_training_checkpoints(
    training_client: str, *, wait_s: float = LIST_WAIT_S, poll_s: float = LIST_POLL_S
) -> list[ListedCheckpoint]:
    """The training checkpoints Tinker lists for ``training_client``, polled until there is one or ``wait_s`` has
    passed. The trainer logs a save when it submits it; a large state takes a while to be listed."""
    rest = _rest_client()
    started = time.monotonic()
    while True:
        response = rest.list_checkpoints(training_client).result()
        listed = sorted(
            (
                ListedCheckpoint(checkpoint.tinker_path, getattr(checkpoint, "size_bytes", None))
                for checkpoint in response.checkpoints
                if checkpoint.checkpoint_type == "training" and checkpoint.tinker_path
            ),
            key=lambda checkpoint: checkpoint.path,
        )
        if listed or time.monotonic() - started > wait_s:
            return listed
        time.sleep(poll_s)


def check_loadable(path: str) -> None:
    """Raise unless Tinker can load ``path`` (it answers with the checkpoint's weights info)."""
    _rest_client().get_weights_info_by_tinker_path(path).result()


def _only(values: Sequence[tuple[int, str]]) -> list[str]:
    return [value for _, value in values]


def _check_log(log: TrainerLog, where: str, stage: int, base_model: str, resume_from: str | None) -> list[int]:
    """Apply every rule to the log; return the completed updates."""
    clients = _only(log.training_clients)
    if len(clients) != 1:
        raise CheckpointError(f"{where} names {len(clients)} training client(s) {clients}; expected exactly one")
    models = _only(log.models)
    if len(models) != 1:
        raise CheckpointError(f"{where} has {len(models)} 'loading tokenizer for' lines {models}; expected one")
    if models[0] != base_model:
        raise CheckpointError(f"{where} trained {models[0]}; expected {base_model}")
    if log.fatal_lines:
        raise CheckpointError(f"{where} holds a Tinker error the SDK captured ('{_FATAL}', line {log.fatal_lines[0]})")
    resumed, restored = _only(log.resumed_from), _only(log.restored_from)
    if stage == 1 and resumed:
        raise CheckpointError(f"stage-1 run {where} resumed from {resumed}")
    if stage == 1 and restored:
        raise CheckpointError(f"stage-1 run {where} restored a state from {restored}")
    if stage == 2 and resumed != [resume_from]:
        raise CheckpointError(f"stage-2 run {where} resumed from {resumed}; expected exactly [{resume_from}]")
    if stage == 2 and restored != [resume_from]:
        raise CheckpointError(
            f"stage-2 run {where} restored the optimizer state from {restored}; expected exactly [{resume_from}]"
        )

    done = [step for _, step in log.steps_done]
    if not done:
        raise CheckpointError(f"{where} shows no completed step")
    if done != list(range(1, len(done) + 1)):
        raise CheckpointError(f"{where}: the completed steps are not consecutive from 1: {done}")
    last, cap = done[-1], UPDATES_PER_STAGE
    # Stage 1 may complete updates past 32 before the trainer is stopped (from samples of the last episode);
    # they are never saved. Stage 2 is stopped at update 32.
    allowed = range(cap, cap + 1) if stage == 2 else range(cap, 2 * cap)
    if last not in allowed:
        raise CheckpointError(
            f"{where}: the last completed step is {last}; stage {stage} must end at "
            + (f"step {cap}" if stage == 2 else f"a step from {cap} to {2 * cap - 1}")
        )

    if len(log.save_lines) != 1:
        raise CheckpointError(
            f"{where} logs {len(log.save_lines)} saves ('checkpoint saved:'); expected exactly one, of update {cap}"
        )
    finished = [line for line, step in log.optim_done if step == cap]
    begun_next = [line for line, step in log.optim_started if step == cap + 1]
    if len(finished) != 1:
        raise CheckpointError(f"{where} has {len(finished)} lines of update {cap}'s 'optim_step done'; expected one")
    if len(begun_next) > 1:
        raise CheckpointError(f"{where} has {len(begun_next)} lines of update {cap + 1}'s 'optim_step ...'")
    save = log.save_lines[0]
    if not (finished[0] < save and (not begun_next or save < begun_next[0])):
        raise CheckpointError(
            f"{where}: the save (line {save}) is not between update {cap}'s optim_step (line {finished[0]}) and "
            f"update {cap + 1}'s ({begun_next[0] if begun_next else 'none'})"
        )
    return done


def collect_stage(
    trainer_log: Path,
    stage: int,
    base_model: str,
    *,
    resume_from: str | None = None,
    list_fn: Callable[[str], Sequence[ListedCheckpoint]] | None = None,
    verify_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Check one stage's trainer log and find its checkpoint on Tinker; return the stage's checkpoint record.

    ``resume_from`` is the stage-1 checkpoint a stage-2 run resumed from (and must be ``None`` for stage 1).
    ``list_fn`` lists the training checkpoints of a training client and ``verify_fn`` raises unless a path loads;
    they default to Tinker (:func:`list_training_checkpoints`, :func:`check_loadable`). Raises ``CheckpointError``
    naming the log and the rule it breaks.
    """
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage!r}")
    if (stage == 2) != (resume_from is not None):
        raise ValueError("resume_from is the stage-1 checkpoint of a stage-2 run; give it for stage 2 only")
    list_fn = list_fn or list_training_checkpoints
    verify_fn = verify_fn or check_loadable

    where = str(trainer_log)
    log = parse_trainer_log(Path(trainer_log).read_text(encoding="utf-8", errors="replace"))
    done = _check_log(log, where, stage, base_model, resume_from)
    client = log.training_clients[0][1]

    started = time.monotonic()
    listed = list(list_fn(client))
    paths = [checkpoint.path for checkpoint in listed]
    want = f"step_{UPDATES_PER_STAGE:04d}"
    if len(paths) != 1 or not paths[0].endswith("/" + want):
        raise CheckpointError(
            f"Tinker lists {paths} for {client} (from {where}) after {int(time.monotonic() - started)} s; expected "
            f"exactly one training checkpoint, {want}"
        )
    final = listed[0]
    try:
        verify_fn(final.path)
    except Exception as error:
        raise CheckpointError(f"{final.path} is listed but does not load: {type(error).__name__}: {error}") from error

    last = done[-1]
    return {
        "trainer_log": where,
        "training_client": client,
        "stage": stage,
        "listed_after_s": int(time.monotonic() - started),
        "resumed_from": resume_from,
        "steps_done_max": last,
        "absolute_step": UPDATES_PER_STAGE * stage,
        "saves_logged": len(log.save_lines),
        "extra_unsaved_step_completed": last > UPDATES_PER_STAGE,
        "extra_unsaved_steps": last - UPDATES_PER_STAGE,
        "listed": paths,
        "final": final.path,
        "size_bytes": final.size_bytes,
        "verified_loadable": True,
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def fingerprint(path: str, base_model: str, min_diff: float = FINGERPRINT_MIN_DIFF) -> dict[str, Any]:
    """Compare the log-probabilities of ``path`` with the base model's on :data:`PROBE`; return the record.

    Raises ``CheckpointError`` when the path does not load, or when the mean absolute difference is not above
    ``min_diff`` (with the record attached).
    """
    import tinker
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except Exception as error:
        raise CheckpointError(f"cannot load the tokenizer of {base_model}: {type(error).__name__}: {error}") from error
    ids = tokenizer.encode(PROBE, add_special_tokens=False)
    probe = tinker.types.ModelInput.from_ints(ids)
    service = tinker.ServiceClient()
    started = time.time()
    base = service.create_sampling_client(base_model=base_model).compute_logprobs(probe).result()
    training = service.create_lora_training_client(base_model=base_model, rank=LORA_RANK)
    try:
        training.load_state(path).result()  # waits for the load, so that a failed one raises here
    except Exception as error:
        raise CheckpointError(f"cannot load {path}: {type(error).__name__}: {error}") from error
    adapted = training.save_weights_and_get_sampling_client().compute_logprobs(probe).result()
    pairs = [(a, b) for a, b in zip(adapted, base, strict=False) if a is not None and b is not None]
    if not pairs or len(adapted) != len(base):
        raise CheckpointError(
            f"the log-probabilities of {path} are unusable: {len(adapted)} values against the base model's "
            f"{len(base)}, {len(pairs)} comparable"
        )
    diff = sum(abs(a - b) for a, b in pairs) / len(pairs)
    record = {
        "path": path,
        "base_model": base_model,
        "probe_tokens": len(ids),
        "mean_abs_logprob_diff_vs_base": diff,
        "min_diff": min_diff,
        "non_base": diff > min_diff,
        "seconds": round(time.time() - started, 1),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if not record["non_base"]:
        raise CheckpointError(
            f"{path} is indistinguishable from the base model: mean |dlogp| {diff:.6f} <= {min_diff}", record
        )
    return record


def _write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evolve_ipi.models.checkpoints", description=__doc__.split("\n\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fp = commands.add_parser("fingerprint", help="check that a checkpoint is not the base model")
    fp.add_argument("path")
    fp.add_argument("--base-model", required=True)
    fp.add_argument("--out", required=True, type=Path)
    fp.add_argument("--min-diff", type=float, default=FINGERPRINT_MIN_DIFF)
    collect = commands.add_parser("collect", help="check a stage's trainer log and find its checkpoint")
    collect.add_argument("log", type=Path)
    collect.add_argument("--stage", required=True, type=int, choices=(1, 2))
    collect.add_argument("--base-model", required=True)
    collect.add_argument("--out", required=True, type=Path)
    collect.add_argument("--resume-from", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "fingerprint":
            try:
                record = fingerprint(args.path, args.base_model, args.min_diff)
            except CheckpointError as error:
                if error.record is not None:
                    _write_record(args.out, error.record)
                raise
            _write_record(args.out, record)
            diff = record["mean_abs_logprob_diff_vs_base"]
            print(f"fingerprint {args.path}: mean |dlogp| vs base = {diff:.5f} (non-base)")
        else:
            record = collect_stage(args.log, args.stage, args.base_model, resume_from=args.resume_from)
            _write_record(args.out, record)
            print(f"stage {args.stage} of {record['training_client']}: {record['final']} (loads)")
    except (CheckpointError, ValueError) as error:
        print(f"checkpoints: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    exit_without_teardown(main)
