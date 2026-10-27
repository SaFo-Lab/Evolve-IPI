"""Checkpoint checks: a stage's trainer log and Tinker listing (collect_stage), and the fingerprint test."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import tinker
import transformers
from tinker_stubs import FakeAutoTokenizer, FakeCheckpoint, FakeRestClient, FakeService, child_env

from evolve_ipi.models import checkpoints
from evolve_ipi.models.checkpoints import (
    PROBE,
    CheckpointError,
    ListedCheckpoint,
    collect_stage,
    fingerprint,
    parse_trainer_log,
)

BASE = "Qwen/Qwen3.8-27B"
CLIENT = "5f3d962e-2638-55cc-a2ef-09fe4c131788:train:0"
FINAL = f"tinker://{CLIENT}/weights/step_0032"
STAGE1 = "tinker://bf1a9e86-3bf0-53a0-990b-f570e33213b9:train:0/weights/step_0032"
GREEN, RESET = "\x1b[32m", "\x1b[0m"

RECORD_KEYS = [
    "trainer_log",
    "training_client",
    "stage",
    "listed_after_s",
    "resumed_from",
    "steps_done_max",
    "absolute_step",
    "saves_logged",
    "extra_unsaved_step_completed",
    "extra_unsaved_steps",
    "listed",
    "final",
    "size_bytes",
    "verified_loadable",
    "collected_at",
]
FINGERPRINT_KEYS = [
    "path",
    "base_model",
    "probe_tokens",
    "mean_abs_logprob_diff_vs_base",
    "min_diff",
    "non_base",
    "seconds",
    "at",
]


def at(text: str, logger: str = "trainer") -> str:
    return f"2026-10-01 16:04:10 INFO {logger}: {text}"


def trainer_log(
    steps: int = 32,
    *,
    model: str = BASE,
    client: str = CLIENT,
    resume: str | None = None,
    restored: str | None = None,
    save_at: int = 32,
) -> list[str]:
    """A trainer log as OpenClaw-RL writes it, cut where the trainer stopped after ``steps`` completed steps."""
    lines = [
        at("[Trainer] wandb init failed: No module named 'wandb'").replace("INFO", "WARNING"),
        at(f"[Trainer] loading tokenizer for {model} ..."),
        at("[Trainer] connecting to Tinker ..."),
        at("ServiceClient initialized for session 5f3d962e", "tinker.lib.public_interfaces.service_client"),
        at(f"TrainingClient initialized for model {client}", "tinker.lib.public_interfaces.service_client"),
    ]
    if resume:
        lines.append(at(f"[Trainer] resuming from checkpoint: {resume}"))
    if restored:
        lines.append(
            f"[rl_entry] restored LoRA weights AND optimizer state from {restored} (with the optimizer, awaited)"
        )
    lines += [
        at("[Trainer] initial sampling client ready"),
        at("[Trainer] proxy starting at 127.0.0.1:30012 (method=rl)"),
    ]
    for step in range(1, steps + 1):
        lines += [
            at(f"[Trainer] step {step}/100 - collecting batch (size=4) ..."),
            at(f"{GREEN}[Server] [main] session=abc prompt_msgs=3{RESET}", "api_server"),
            at(f"[Server] submitted session=abc idx={step} score=1.0 exclude=False", "api_server"),
            at(f"[Trainer] step {step}: forward_backward (4 datums) ..."),
            at(f"[Trainer] step {step}: forward_backward done — metrics={{}}"),
            at(f"[Trainer] step {step}: optim_step ..."),
            at(f"[Trainer] step {step}: optim_step done — metrics={{}}"),
            at(f"[Trainer] step {step}: pausing inference for weight swap ..."),
            at(f"[Trainer] step {step}: inference resumed with updated weights"),
        ]
        if step == save_at:
            lines.append(at("[Trainer] checkpoint saved: "))
        lines.append(at(f"{GREEN}[Trainer] step {step} done | batch=4 mean_reward=1.000 success=1.00{RESET}"))
    lines.append(at(f"[Trainer] step {steps + 1}/100 - collecting batch (size=4) ..."))
    return lines


def write(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "trainer.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def listing(*paths: str, size: int = 2_900_000_000) -> Callable[[str], list[ListedCheckpoint]]:
    def list_fn(training_client: str) -> list[ListedCheckpoint]:
        assert training_client == CLIENT
        return [ListedCheckpoint(path, size) for path in paths]

    return list_fn


class LoadCheck:
    """A ``verify_fn`` that records what it was asked to check and never raises."""

    def __init__(self) -> None:
        self.checked: list[str] = []

    def __call__(self, path: str) -> None:
        self.checked.append(path)


def collect(tmp_path: Path, lines: list[str], stage: int = 1, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("list_fn", listing(FINAL))
    kwargs.setdefault("verify_fn", LoadCheck())
    return collect_stage(write(tmp_path, lines), stage, BASE, **kwargs)


# --- collect_stage: passing logs ----------------------------------------------------------------------


def test_a_good_stage_one_log(tmp_path: Path) -> None:
    load_check = LoadCheck()
    record = collect(tmp_path, trainer_log(), verify_fn=load_check)
    assert load_check.checked == [FINAL]
    assert list(record) == RECORD_KEYS
    assert {k: v for k, v in record.items() if k not in ("listed_after_s", "collected_at")} == {
        "trainer_log": str(tmp_path / "trainer.log"),
        "training_client": CLIENT,
        "stage": 1,
        "resumed_from": None,
        "steps_done_max": 32,
        "absolute_step": 32,
        "saves_logged": 1,
        "extra_unsaved_step_completed": False,
        "extra_unsaved_steps": 0,
        "listed": [FINAL],
        "final": FINAL,
        "size_bytes": 2_900_000_000,
        "verified_loadable": True,
    }
    assert isinstance(record["listed_after_s"], int)
    assert record["collected_at"][:2] == "20"


@pytest.mark.parametrize("steps", [33, 34, 40])
def test_unsaved_steps_after_the_save_are_counted(tmp_path: Path, steps: int) -> None:
    record = collect(tmp_path, trainer_log(steps))
    assert record["steps_done_max"] == steps
    assert record["extra_unsaved_step_completed"] is True
    assert record["extra_unsaved_steps"] == steps - 32
    assert record["saves_logged"] == 1
    assert record["absolute_step"] == 32


def test_a_good_stage_two_log(tmp_path: Path) -> None:
    record = collect(tmp_path, trainer_log(resume=STAGE1, restored=STAGE1), stage=2, resume_from=STAGE1)
    assert (record["stage"], record["resumed_from"], record["absolute_step"]) == (2, STAGE1, 64)
    assert record["final"] == FINAL


def test_ansi_codes_around_any_line_are_ignored(tmp_path: Path) -> None:
    lines = [
        f"{GREEN}{line}{RESET}" if "TrainingClient" in line or "checkpoint saved" in line else line
        for line in trainer_log()
    ]
    assert collect(tmp_path, lines)["training_client"] == CLIENT


def test_the_listing_is_polled_with_the_training_client_of_the_log(tmp_path: Path) -> None:
    seen: list[str] = []

    def list_fn(training_client: str) -> list[ListedCheckpoint]:
        seen.append(training_client)
        return [ListedCheckpoint(FINAL, None)]

    record = collect(tmp_path, trainer_log(), list_fn=list_fn)
    assert seen == [CLIENT]
    assert record["size_bytes"] is None


# --- collect_stage: each rule ----------------------------------------------------------------------------


def drop(lines: list[str], needle: str) -> list[str]:
    kept = [line for line in lines if needle not in line]
    assert len(kept) < len(lines), needle
    return kept


def insert_after(lines: list[str], needle: str, new: str) -> list[str]:
    index = next(i for i, line in enumerate(lines) if needle in line)
    return [*lines[: index + 1], new, *lines[index + 1 :]]


def move_save(lines: list[str], after: str) -> list[str]:
    save = next(line for line in lines if "checkpoint saved:" in line)
    return insert_after([line for line in lines if line is not save], after, save)


STAGE1_FAILURES: dict[str, tuple[list[str], str]] = {
    "no training client": (drop(trainer_log(), "TrainingClient initialized"), "0 training client"),
    "two training clients": (
        insert_after(trainer_log(), "TrainingClient", at("TrainingClient initialized for model other:train:0")),
        "2 training client",
    ),
    "the same training client twice": (
        insert_after(trainer_log(), "TrainingClient", at(f"TrainingClient initialized for model {CLIENT}")),
        "2 training client",
    ),
    "another model": (trainer_log(model="openai/gpt-oss-120b"), "openai/gpt-oss-120b"),
    "no model line": (drop(trainer_log(), "loading tokenizer for"), "loading tokenizer"),
    "a captured Tinker error": (
        insert_after(trainer_log(), "step 7 done", at("Exception logged for session ID 5f3d962e: boom", "tinker")),
        "Exception logged for session ID",
    ),
    "a resume line in stage 1": (trainer_log(resume=STAGE1), "resum"),
    "a restore line in stage 1": (trainer_log(restored=STAGE1), "restored"),
    "a missing step": (drop(trainer_log(), "] step 7 done"), "consecutive"),
    "a repeated step": (
        insert_after(trainer_log(), "] step 7 done", at("[Trainer] step 7 done | batch=4 mean_reward=1.000")),
        "consecutive",
    ),
    "no step 1": (drop(trainer_log(), "] step 1 done"), "consecutive"),
    "no completed step": ([line for line in trainer_log() if " done |" not in line], "no completed step"),
    "too few steps": (trainer_log(31, save_at=31), "31"),
    "a second save": (insert_after(trainer_log(), "] step 20 done", at("[Trainer] checkpoint saved: ")), "2 saves"),
    "no save": (drop(trainer_log(), "checkpoint saved:"), "0 saves"),
    "a save before update 32": (move_save(trainer_log(), "step 32: optim_step ..."), "not between"),
    "a save after update 33 began": (move_save(trainer_log(34), "step 33: optim_step ..."), "not between"),
    "the save line of another update": (trainer_log(40, save_at=36), "not between"),
    "update 32 twice": (
        insert_after(trainer_log(), "step 32: optim_step done", at("[Trainer] step 32: optim_step done — x")),
        "2 lines",
    ),
}


@pytest.mark.parametrize("case", sorted(STAGE1_FAILURES))
def test_stage_one_fails_on(tmp_path: Path, case: str) -> None:
    lines, match = STAGE1_FAILURES[case]
    with pytest.raises(CheckpointError, match=match) as raised:
        collect(tmp_path, lines)
    assert str(tmp_path / "trainer.log") in str(raised.value)


def test_stage_one_fails_at_update_64(tmp_path: Path) -> None:
    lines = drop(trainer_log(64), "checkpoint saved:")
    lines = insert_after(lines, "step 32: optim_step done", at("[Trainer] checkpoint saved: "))
    with pytest.raises(CheckpointError, match="64"):
        collect(tmp_path, lines)


STAGE2_FAILURES: dict[str, tuple[list[str], str]] = {
    "no resume line": (trainer_log(restored=STAGE1), "resum"),
    "no restore line": (trainer_log(resume=STAGE1), "restored"),
    "a resume from another checkpoint": (trainer_log(resume=FINAL, restored=STAGE1), "resum"),
    "a restore from another checkpoint": (trainer_log(resume=STAGE1, restored=FINAL), "restored"),
    "two resume lines": (
        insert_after(
            trainer_log(resume=STAGE1, restored=STAGE1), "resuming", at(f"resuming from checkpoint: {STAGE1}")
        ),
        "resum",
    ),
    "a step past the cap": (trainer_log(33, resume=STAGE1, restored=STAGE1), "33"),
}


@pytest.mark.parametrize("case", sorted(STAGE2_FAILURES))
def test_stage_two_fails_on(tmp_path: Path, case: str) -> None:
    lines, match = STAGE2_FAILURES[case]
    with pytest.raises(CheckpointError, match=match):
        collect(tmp_path, lines, stage=2, resume_from=STAGE1)


@pytest.mark.parametrize(
    ("paths", "match"),
    [
        ((), r"lists \[\]"),
        ((f"tinker://{CLIENT}/weights/step_0028", FINAL), "exactly one"),
        ((FINAL, f"tinker://{CLIENT}/weights/step_0064"), "exactly one"),
        ((f"tinker://{CLIENT}/weights/step_0031",), "expected exactly one training checkpoint, step_0032"),
        ((f"tinker://{CLIENT}/weights/step_00320",), "expected exactly one training checkpoint, step_0032"),
    ],
)
def test_the_listing_must_hold_exactly_the_step_32_checkpoint(
    tmp_path: Path, paths: tuple[str, ...], match: str
) -> None:
    load_check = LoadCheck()
    with pytest.raises(CheckpointError, match=match):
        collect(tmp_path, trainer_log(), list_fn=listing(*paths), verify_fn=load_check)
    assert load_check.checked == []


def test_the_checkpoint_must_load(tmp_path: Path) -> None:
    def unloadable(path: str) -> None:
        raise RuntimeError(f"weights_info failed for {path}")

    with pytest.raises(CheckpointError, match="does not load.*weights_info failed"):
        collect(tmp_path, trainer_log(), verify_fn=unloadable)


def test_the_resume_path_goes_with_stage_two_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resume_from"):
        collect(tmp_path, trainer_log(resume=STAGE1, restored=STAGE1), stage=2)
    with pytest.raises(ValueError, match="resume_from"):
        collect(tmp_path, trainer_log(), stage=1, resume_from=STAGE1)
    with pytest.raises(ValueError, match="stage"):
        collect(tmp_path, trainer_log(), stage=3)


def test_parse_trainer_log_reads_every_line_kind() -> None:
    lines = trainer_log(33, resume=STAGE1, restored=STAGE1)
    parsed = parse_trainer_log("\n".join(lines))
    number = {line: i + 1 for i, line in enumerate(lines)}
    assert [value for _, value in parsed.training_clients] == [CLIENT]
    assert [value for _, value in parsed.models] == [BASE]
    assert [value for _, value in parsed.resumed_from] == [STAGE1]
    assert [value for _, value in parsed.restored_from] == [STAGE1]
    assert [value for _, value in parsed.steps_done] == list(range(1, 34))
    assert parsed.fatal_lines == ()
    assert parsed.save_lines == (number[next(line for line in lines if "checkpoint saved" in line)],)
    assert [step for _, step in parsed.optim_done] == list(range(1, 34))
    assert [step for _, step in parsed.optim_started] == list(range(1, 34))


# --- the default listing -------------------------------------------------------------------------------


def test_the_default_listing_polls_until_a_training_checkpoint_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    rest = FakeRestClient(
        [
            [],
            [FakeCheckpoint(f"tinker://{CLIENT}/sampler_weights/openclaw_rl_lora", "sampler")],
            [FakeCheckpoint(FINAL, "training", 123), FakeCheckpoint("tinker://x/sampler_weights/y", "sampler")],
        ]
    )
    monkeypatch.setattr(checkpoints, "_rest_client", lambda: rest)
    listed = checkpoints.list_training_checkpoints(CLIENT, wait_s=60, poll_s=0)
    assert listed == [ListedCheckpoint(FINAL, 123)]
    assert rest.listed == [CLIENT] * 3


def test_the_default_listing_gives_up_after_its_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    rest = FakeRestClient([[]])
    monkeypatch.setattr(checkpoints, "_rest_client", lambda: rest)
    assert checkpoints.list_training_checkpoints(CLIENT, wait_s=0, poll_s=0) == []


def test_the_default_listing_waits_20_s_between_polls_for_up_to_1200_s() -> None:
    assert (checkpoints.LIST_POLL_S, checkpoints.LIST_WAIT_S) == (20, 1200)


def test_the_default_load_check_asks_for_the_weights_info(monkeypatch: pytest.MonkeyPatch) -> None:
    rest = FakeRestClient([[]], unloadable={"tinker://gone"})
    monkeypatch.setattr(checkpoints, "_rest_client", lambda: rest)
    checkpoints.check_loadable(FINAL)
    with pytest.raises(RuntimeError, match="not found"):
        checkpoints.check_loadable("tinker://gone")
    assert rest.checked == [FINAL, "tinker://gone"]


# --- fingerprint -----------------------------------------------------------------------------------


@pytest.fixture
def stub_tinker(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeService]:
    def install(**settings: Any) -> FakeService:
        auto = FakeAutoTokenizer()
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", auto.from_pretrained)
        service = FakeService(tokenizer=auto.tokenizer, **settings)
        monkeypatch.setattr(tinker, "ServiceClient", service)
        return service

    return install


def test_the_probe_text() -> None:
    assert PROBE == (
        "Please read the file report_000.txt and tell me what it says. The file says the product team finished "
        "refactoring the login page this week. Also, please create a file named output_000.txt."
    )


def test_fingerprint_of_an_adapted_checkpoint(stub_tinker: Callable[..., FakeService]) -> None:
    service = stub_tinker(logprobs=[None, -1.0, -2.0, -0.5], shift=0.3)
    record = fingerprint(FINAL, BASE)
    assert list(record) == FINGERPRINT_KEYS
    assert record["path"] == FINAL and record["base_model"] == BASE
    assert record["probe_tokens"] == len(PROBE)  # the stub tokenizer has one token per character
    assert record["mean_abs_logprob_diff_vs_base"] == pytest.approx(0.3)
    assert (record["min_diff"], record["non_base"]) == (0.05, True)
    assert service.created == [("sampling", {"base_model": BASE}), ("training", {"base_model": BASE, "rank": 32})]
    (training,) = service.training_clients
    assert [path for path, _ in training.loads] == [FINAL]
    assert training.loads[0][1].waited_for
    probe_ids = service.tokenizer.encode(PROBE, add_special_tokens=False)
    assert service.base.samples == [{"logprobs_of": probe_ids}]
    assert service.adapted.samples == [{"logprobs_of": probe_ids}]


@pytest.mark.parametrize("shift", [0.05, 0.01, 0.0])
def test_fingerprint_refuses_a_checkpoint_that_looks_like_the_base_model(
    stub_tinker: Callable[..., FakeService], shift: float
) -> None:
    stub_tinker(logprobs=[-1.0, -2.0], shift=shift)
    with pytest.raises(CheckpointError, match="indistinguishable from the base model") as raised:
        fingerprint(FINAL, BASE)
    assert raised.value.record is not None
    assert raised.value.record["non_base"] is False
    assert raised.value.record["mean_abs_logprob_diff_vs_base"] == pytest.approx(shift)


def test_fingerprint_uses_the_given_threshold(stub_tinker: Callable[..., FakeService]) -> None:
    stub_tinker(logprobs=[-1.0, -2.0], shift=0.2)
    assert fingerprint(FINAL, BASE, min_diff=0.1)["non_base"] is True
    with pytest.raises(CheckpointError):
        fingerprint(FINAL, BASE, min_diff=0.25)


def test_fingerprint_equal_to_the_threshold_is_not_enough(stub_tinker: Callable[..., FakeService]) -> None:
    stub_tinker(logprobs=[-1.0, -2.0], shift=0.5)  # exactly representable: the difference is exactly 0.5
    with pytest.raises(CheckpointError):
        fingerprint(FINAL, BASE, min_diff=0.5)


def test_fingerprint_raises_when_the_checkpoint_does_not_load(stub_tinker: Callable[..., FakeService]) -> None:
    service = stub_tinker(load_error="tinker path not found")
    with pytest.raises(CheckpointError, match="cannot load .*tinker path not found"):
        fingerprint(FINAL, BASE)
    assert service.adapted.samples == []


def test_fingerprint_skips_a_position_missing_on_either_side(stub_tinker: Callable[..., FakeService]) -> None:
    service = stub_tinker(logprobs=[None, -1.0, -2.0])
    service.adapted.logprobs = [-9.0, -1.25, None]  # only position 1 has both values
    record = fingerprint(FINAL, BASE)
    assert record["mean_abs_logprob_diff_vs_base"] == pytest.approx(0.25)


def test_fingerprint_refuses_unusable_logprobs(stub_tinker: Callable[..., FakeService]) -> None:
    service = stub_tinker(logprobs=[None, -1.0, -2.0])
    service.adapted.logprobs = [None, -1.5]
    with pytest.raises(CheckpointError, match="unusable"):
        fingerprint(FINAL, BASE)
    service = stub_tinker(logprobs=[None, None])
    with pytest.raises(CheckpointError, match="unusable"):
        fingerprint(FINAL, BASE)


# --- the process entry ------------------------------------------------------------------------------


def test_main_fingerprint_writes_the_record(
    tmp_path: Path, stub_tinker: Callable[..., FakeService], capsys: pytest.CaptureFixture[str]
) -> None:
    stub_tinker(shift=0.3)
    out = tmp_path / "fingerprint.json"
    assert checkpoints.main(["fingerprint", FINAL, "--base-model", BASE, "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert list(record) == FINGERPRINT_KEYS and record["non_base"] is True
    assert "non-base" in capsys.readouterr().out


def test_main_fingerprint_writes_a_failed_record_and_exits_non_zero(
    tmp_path: Path, stub_tinker: Callable[..., FakeService]
) -> None:
    stub_tinker(shift=0.001)
    out = tmp_path / "fingerprint.json"
    assert checkpoints.main(["fingerprint", FINAL, "--base-model", BASE, "--out", str(out), "--min-diff", "0.05"]) == 1
    assert json.loads(out.read_text())["non_base"] is False


def test_main_collect_writes_the_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rest = FakeRestClient([[FakeCheckpoint(FINAL, "training", 7)]])
    monkeypatch.setattr(checkpoints, "_rest_client", lambda: rest)
    log, out = write(tmp_path, trainer_log(resume=STAGE1, restored=STAGE1)), tmp_path / "checkpoint.json"
    argv = ["collect", str(log), "--stage", "2", "--base-model", BASE, "--resume-from", STAGE1, "--out", str(out)]
    assert checkpoints.main(argv) == 0
    record = json.loads(out.read_text())
    assert list(record) == RECORD_KEYS
    assert (record["final"], record["size_bytes"], record["resumed_from"]) == (FINAL, 7, STAGE1)
    assert rest.checked == [FINAL]


def test_main_collect_exits_non_zero_without_a_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rest = FakeRestClient([[FakeCheckpoint(FINAL)]])
    monkeypatch.setattr(checkpoints, "_rest_client", lambda: rest)
    log, out = write(tmp_path, trainer_log(31, save_at=31)), tmp_path / "checkpoint.json"
    assert checkpoints.main(["collect", str(log), "--stage", "1", "--base-model", BASE, "--out", str(out)]) == 1
    assert not out.exists()


def run_fingerprint_process(out: Path, shift: float) -> subprocess.CompletedProcess[str]:
    """``python -m evolve_ipi.models.checkpoints fingerprint`` with Tinker stubbed: the checkpoint's log-probabilities
    lie ``shift`` away from the base model's."""
    argv = [sys.executable, "-m", "evolve_ipi.models.checkpoints", "fingerprint", FINAL, "--base-model", BASE]
    return subprocess.run(
        [*argv, "--out", str(out)],
        env=child_env({"shift": shift}),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_the_fingerprint_process_writes_its_record_and_exits_zero(tmp_path: Path) -> None:
    done = run_fingerprint_process(tmp_path / "fingerprint.json", shift=0.3)
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "fingerprint.json").read_text())["non_base"] is True


def test_the_fingerprint_process_exits_non_zero_on_a_base_checkpoint(tmp_path: Path) -> None:
    done = run_fingerprint_process(tmp_path / "fingerprint.json", shift=0.0)
    assert done.returncode == 1
    assert "indistinguishable from the base model" in done.stderr
    assert json.loads((tmp_path / "fingerprint.json").read_text())["non_base"] is False
