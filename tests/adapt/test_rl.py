"""RL adaptation: the trainer's command and environment, the stop rule, the cap at update 64, the restarts, and the
records, against the fake trainer and a scripted agent (no Tinker, no OpenClaw)."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rl_harness import FakeGateway, Harness, Step, episodes, make_harness, wait_until

from evolve_ipi import children
from evolve_ipi.adapt import AdaptAbort, rl
from evolve_ipi.agent.anomalies import TrainingPolicy
from evolve_ipi.agent.episode import EpisodeAbort, LeftoverState
from evolve_ipi.config import ConfigError, load_configs
from evolve_ipi.models.checkpoints import strip_ansi
from evolve_ipi.procs import PortInUse, port_in_use
from evolve_ipi.tasks import RLOrder, TaskDataError, load_rl_order
from evolve_ipi.thirdparty import PatchError

STAGE1, STAGE2 = "stage1", "stage2"


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_port: Callable[[], int]) -> Harness:
    return make_harness(tmp_path, monkeypatch, free_port(), "drive-create")


def short_order(harness: Harness, stage1: list[str], stage2: list[str]) -> None:
    harness.monkeypatch.setattr(rl, "load_rl_order", lambda task: RLOrder(stage1=stage1, stage2=stage2))


def adapt_ids(harness: Harness, count: int) -> list[str]:
    return list(harness.task.adaptation)[:count]


# --- configuration ---------------------------------------------------------------------------------------------


def test_max_restarts_is_a_key_of_rl_yaml(configs_dir: Path, tmp_path: Path) -> None:
    assert load_configs(configs_dir).rl.max_restarts == 3
    copy = tmp_path / "configs"
    copy.mkdir()
    for name in ("models", "rl", "skills", "eval", "runtime"):
        text = (configs_dir / f"{name}.yaml").read_text(encoding="utf-8")
        if name == "rl":
            text = "\n".join(line for line in text.splitlines() if not line.startswith("max_restarts:"))
        (copy / f"{name}.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="max_restarts: missing key"):
        load_configs(copy)


def test_the_trainer_gets_the_recipe_flags(harness: Harness) -> None:
    model = harness.cfgs.models["qwen3.8-27b"]
    record_dir = Path("/runs/stage1/records")
    assert rl.trainer_flags(model, harness.cfgs, record_dir) == [
        "--method", "rl", "--model-name", "Qwen/Qwen3.8-27B", "--batch-size", "4", "--prm-m", "3",
        "--lora-rank", "32", "--learning-rate", "1e-4", "--max-steps", "100", "--save-interval", "32",
        "--proxy-host", "127.0.0.1", "--proxy-port", str(harness.cfgs.ports.policy), "--served-model-name", "policy",
        "--record-dir", "/runs/stage1/records",
    ]  # fmt: skip
    resumed = rl.trainer_flags(model, harness.cfgs, record_dir, resume="tinker://a:train:0/weights/step_0032")
    assert resumed[-2:] == ["--resume-from-ckpt", "tinker://a:train:0/weights/step_0032"]


def test_the_trainer_command_runs_the_entry_with_this_python() -> None:
    command = rl._trainer_command(["--method", "rl"])
    assert command[1:] == ["-m", "evolve_ipi.adapt.rl_entry", "--method", "rl"]
    assert os.path.realpath(command[0]) == os.path.realpath(__import__("sys").executable)


def test_the_trainer_environment_holds_no_other_secret(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "EVOLVER_API_KEY": "evolver-secret",
        "EVOLVER_BASE_URL": "https://evolver.example/v1",
        "OPENAI_API_KEY": "openai-secret",
        "OPENCLAW_GATEWAY_TOKEN": "gateway-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "TZ": "America/New_York",
        "LANG": "C.UTF-8",
    }.items():
        monkeypatch.setenv(name, value)
    env = children.trainer_env(harness.paths, harness.secrets)
    assert env["HOME"] == str(harness.paths.service_home)
    assert env["HF_HOME"] == str(harness.paths.hf_home)
    assert env["TINKER_API_KEY"] == "test-key-not-real"
    assert env["WANDB_DISABLED"] == "1"
    assert env["TZ"] == "America/New_York" and env["LANG"] == "C.UTF-8" and env["PATH"] == os.environ["PATH"]
    assert "HF_TOKEN" not in env
    assert set(env) <= {
        "PATH",
        "HOME",
        "LANG",
        "TZ",
        "HF_HOME",
        "HF_TOKEN",
        "TINKER_API_KEY",
        "WANDB_DISABLED",
        "PYTHONPATH",
    }
    assert not any("secret" in value for value in env.values())
    with_token = children.trainer_env(
        harness.paths, harness.secrets.__class__(tinker_api_key="k", hf_token="hf-token")
    )
    assert with_token["HF_TOKEN"] == "hf-token"


# --- reading the trainer's log ---------------------------------------------------------------------------------

LOG = """\
2026-10-01 16:04:28 INFO trainer: [Trainer] step 1/100 - collecting batch (size=4) ...
2026-10-01 16:04:42 INFO api_server: [Server] submitted session=s1 idx=0 score=1.0 exclude=False
2026-10-01 16:04:48 INFO api_server: [Server] submitted session=s1 idx=1 score=-1.0 exclude=False
2026-10-01 16:04:48 INFO api_server: [Server] submitted RL session=s1 idx=2 score=0.0 exclude=True
2026-10-01 16:04:50 INFO api_server: [Server] submitted OPD+RL session=s2 idx=3 score=1.0 exclude=False
2026-10-01 16:04:51 INFO trainer: \x1b[32m[Trainer] step 1 done | batch=4 mean_reward=0.250 success=0.50\x1b[0m
2026-10-01 16:04:51 INFO trainer: [Trainer] step 2/100 - collecting batch (size=4) ...
"""


def test_progress_is_read_from_the_trainer_log() -> None:
    progress = rl.trainer_progress(LOG)
    assert (progress.submitted, progress.steps_done, progress.batch_size) == (4, 1, 4)
    assert set(progress.as_record()) == {"submitted", "steps_done", "batch_size", "read_at"}
    empty = rl.trainer_progress("")
    assert (empty.submitted, empty.steps_done, empty.batch_size) == (0, 0, None)


def test_launching_stops_once_batch_times_updates_samples_were_submitted() -> None:
    def progress(submitted: int, batch: int | None = 4) -> rl.TrainerProgress:
        return rl.TrainerProgress(submitted=submitted, steps_done=0, batch_size=batch, read_at="")

    assert not rl.stop_launching(progress(127), 32)
    assert rl.stop_launching(progress(128), 32)
    assert rl.stop_launching(progress(131), 32)
    with pytest.raises(AdaptAbort, match="collecting batch"):
        rl.stop_launching(progress(200, None), 32)


def test_epochs_follow_the_order_files() -> None:
    order = load_rl_order("file-create")
    assert rl.epochs(order.stage1, "file") == [i // 12 for i in range(96)]
    assert rl.epochs(order.stage2, "file") == [i // 36 for i in range(216)]
    service = load_rl_order("chat-invite")
    assert rl.epochs(service.stage1, "service") == [i // 72 for i in range(144)]
    assert rl.epochs(service.stage2, "service") == [i // 108 for i in range(216)]
    with pytest.raises(TaskDataError):
        rl.epochs(["a", "b", "a", "a"], "file")  # an epoch that is not a pass over the items
    with pytest.raises(TaskDataError):
        rl.epochs(["a", "b", "b", "a"], "service")  # the two passes differ


def test_stage_statistics() -> None:
    records = [
        {"id": "x-1", "asr": {"hit": True}, "anomalies": []},
        {"id": "x-2", "asr": {"hit": False}, "anomalies": [{"type": "no_calls", "detail": ""}]},
        {"id": "x-1", "asr": {"hit": False}, "anomalies": [{"type": "empty_reply", "detail": ""}]},
        {"id": "x-3", "asr": {"hit": True}, "anomalies": [{"type": "agent_error"}, {"type": "no_calls"}, {"type": "cap_cut"}]},
    ]  # fmt: skip
    stats = rl.stage_stats(records, "service", LOG)
    assert stats == {
        "episodes": 4,
        "executions_under_authorization": 2,
        "execution_rate": 0.5,
        "samples_submitted": 4,
        "samples_positive": 2,
        "samples_negative": 1,
        "samples_zero": 1,
        "mean_reward_per_update": [0.25],
        "items_used": 3,
        "uses_per_item": {"x-1": 2, "x-2": 1, "x-3": 1},
        "nonengaged_episodes_continued": 2,
        "cap_cut_episodes": 1,
    }
    files = [
        {"id": "f-1", "created": True, "tool_calls": [{"name": "read"}], "reply": "ok", "anomalies": []},
        {"id": "f-2", "created": False, "tool_calls": [], "reply": "ok", "anomalies": []},
        {"id": "f-3", "created": False, "tool_calls": [{"name": "read"}], "reply": "  ", "anomalies": []},
    ]
    file_stats = rl.stage_stats(files, "file", LOG)
    assert file_stats["executions_under_authorization"] == 1
    assert file_stats["nonengaged_episodes_continued"] == 2


# --- Inkling-Small's raw generations (the audit itself is tested in tests/models/test_inkling_audit.py) ----------

INKLING_CALL = (
    '<|content_thinking|>ok<|end_message|><|message_model|>read<|content_invoke_tool_json|>{"name": "read", '
    '"args": {"path": "a.txt"}}<|end_message|><|content_model_end_sampling|>'
)
INKLING_TEXT = "<|content_text|>Here it is.<|end_message|><|content_model_end_sampling|>"
INKLING_UNPARSED = (
    "<|message_model|>read<|content_invoke_tool_text|>read a.txt<|end_message|><|content_model_end_sampling|>"
)


def raw_row(raw: str, parsed: int) -> dict[str, Any]:
    return {"at": "t", "n_tokens": 10, "stop": ["<|content_model_end_sampling|>"], "finish_reason": "stop",
            "parsed_calls": parsed, "raw": raw}  # fmt: skip


# --- refusals before anything starts --------------------------------------------------------------------------


def test_a_model_without_rl_is_refused(harness: Harness) -> None:
    with pytest.raises(ConfigError, match="gpt-oss-20b"):
        harness.adapt("gpt-oss-20b")
    assert harness.events("start") == [] and harness.prepared == []


def test_a_model_whose_family_the_trainer_would_not_take_is_refused(harness: Harness) -> None:
    spec = harness.cfgs.models["qwen3.8-27b"].model_copy(update={"family": "harmony"})
    with pytest.raises(ConfigError, match="family"):
        rl.adapt_rl(spec, harness.task, harness.run_dir(), harness.cfgs, harness.secrets, paths=harness.paths)


def test_a_missing_tinker_key_is_refused(harness: Harness) -> None:
    secrets = harness.secrets.__class__()
    with pytest.raises(ConfigError, match="TINKER_API_KEY"):
        rl.adapt_rl(
            harness.cfgs.models["qwen3.8-27b"],
            harness.task,
            harness.run_dir(),
            harness.cfgs,
            secrets,
            paths=harness.paths,
        )


def test_an_unpatched_rollout_is_refused(harness: Harness) -> None:
    def unpatched() -> dict[str, str]:
        raise PatchError("OpenClaw-RL/openclaw-tinker/rollout.py has sha256 0000, SHA256SUMS says 1a89")

    harness.monkeypatch.setattr(rl, "verify_patches", unpatched)
    with pytest.raises(PatchError, match="rollout.py"):
        harness.adapt()
    assert harness.events("start") == [] and harness.prepared == []


@pytest.mark.parametrize(("key", "value"), [("updates_per_stage", 16), ("save_interval", 4), ("max_steps", 40)])
def test_a_recipe_the_collector_cannot_check_is_refused(harness: Harness, key: str, value: int) -> None:
    harness.set_rl(**{key: value})
    with pytest.raises(ConfigError, match=key):
        harness.adapt()


# --- the two stages ---------------------------------------------------------------------------------------------


def test_two_stages_from_the_base_model_to_update_64(harness: Harness) -> None:
    result = harness.adapt()
    adapt = harness.run_dir().adapt_dir
    stage1 = json.loads((adapt / STAGE1 / "checkpoint.json").read_text(encoding="utf-8"))
    stage2 = json.loads((adapt / STAGE2 / "checkpoint.json").read_text(encoding="utf-8"))
    assert result == {"update_32": stage1["final"], "update_64": stage2["final"]}
    assert json.loads((adapt / "checkpoints.json").read_text(encoding="utf-8")) == {
        "base_model": "Qwen/Qwen3.8-27B",
        **result,
    }
    assert stage1["stage"] == 1 and stage1["resumed_from"] is None
    assert stage2["stage"] == 2 and stage2["resumed_from"] == result["update_32"] and stage2["absolute_step"] == 64

    starts = harness.events("start")
    assert [start["argv"] for start in starts] == [
        rl.trainer_flags(harness.cfgs.models["qwen3.8-27b"], harness.cfgs, adapt / STAGE1 / "records"),
        rl.trainer_flags(harness.cfgs.models["qwen3.8-27b"], harness.cfgs, adapt / STAGE2 / "records", resume=result["update_32"]),
    ]  # fmt: skip
    # the resume path is fingerprinted, in a process of its own, before the stage-2 trainer starts
    kinds = [(row["event"], row.get("path")) for row in harness.events() if row["event"] in ("start", "fingerprint")]
    assert kinds == [
        ("start", None),
        ("fingerprint", result["update_32"]),
        ("fingerprint", result["update_32"]),
        ("start", None),
        ("fingerprint", result["update_64"]),
    ]
    resume_fp = json.loads((adapt / STAGE2 / "resume_fingerprint.json").read_text(encoding="utf-8"))
    assert resume_fp["path"] == result["update_32"] and resume_fp["non_base"]
    # one environment and one gateway per stage, the gateway stopped at the end of each
    assert [entry[:2] for entry in harness.prepared] == [("drive-create", "rl"), ("drive-create", "rl")]
    assert harness.prepared[0][4] == harness.cfgs.ports.policy and harness.prepared[0][5] == 600
    assert [event for event, _ in FakeGateway.events] == ["start", "stop", "start", "stop"]
    assert all(policy == TrainingPolicy("continue", 0) for policy in harness.agent.policies)

    order = load_rl_order("drive-create")
    for stage, ids in ((STAGE1, order.stage1), (STAGE2, order.stage2)):
        directory = adapt / stage
        for name in ("trainer.log", "episodes.jsonl", "stats.json", "checkpoint.json", "fingerprint.json"):
            assert (directory / name).is_file(), name
        records = episodes(directory / "episodes.jsonl")
        assert [r["id"] for r in records] == ids[:4]  # 32 samples an episode: 4 episodes reach 128
        assert [r["epoch"] for r in records] == [0, 0, 0, 0]
        assert [r["trainer"]["submitted"] for r in records] == [32, 64, 96, 128]
        stats = json.loads((directory / "stats.json").read_text(encoding="utf-8"))
        assert (
            stats["episodes"] == 4 and stats["samples_submitted"] == 128 and len(stats["mean_reward_per_update"]) == 32
        )
        assert stats["until"]["target"] == 32 and stats["until"]["trainer_at_end"]["steps_done"] == 32
        fingerprint = json.loads((directory / "fingerprint.json").read_text(encoding="utf-8"))
        assert fingerprint["non_base"] and fingerprint["min_diff"] == 0.05
    assert harness.agent.calls == order.stage1[:4] + order.stage2[:4]
    # the trainers are gone and their port is free
    assert len(harness.events("sigterm")) == 2
    assert not port_in_use(harness.cfgs.ports.policy)


def test_the_trainer_runs_with_the_allowlisted_environment(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLVER_API_KEY", "evolver-secret")
    monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    harness.adapt()
    for start in harness.events("start"):
        assert start["env"]["TINKER_API_KEY"] == "test-key-not-real"
        assert start["env"]["HOME"] == str(harness.paths.service_home)
        assert start["env"]["HTTPS_PROXY"] == "http://proxy.example:3128", "Tinker is reached as the run reaches it"
        assert start["env"]["NO_PROXY"] == start["env"]["no_proxy"] == "127.0.0.1,localhost"
        assert not any("secret" in value for value in start["env"].values())
    for collect in harness.events("collect"):
        assert not any("secret" in value for value in collect["env"].values())


def test_launching_stops_at_128_submitted_samples(harness: Harness) -> None:
    harness.agent.default = Step(samples=5)
    harness.adapt()
    order = load_rl_order("drive-create")
    adapt = harness.run_dir().adapt_dir
    assert harness.agent.calls == order.stage1[:26] + order.stage2[:26]  # 25 x 5 = 125 < 128 <= 26 x 5
    for stage in (STAGE1, STAGE2):
        records = episodes(adapt / stage / "episodes.jsonl")
        assert [r["trainer"]["submitted"] for r in records][-2:] == [125, 130]
        stats = json.loads((adapt / stage / "stats.json").read_text(encoding="utf-8"))
        assert stats["samples_submitted"] == 130 and stats["until"]["trainer_at_stop"]["submitted"] == 130


def test_the_stage_waits_for_update_32_after_the_last_episode(harness: Harness) -> None:
    harness.behaviors = [{"step_delay_s": 0.01}, {"step_delay_s": 0.01}]
    harness.adapt()
    for stage in (STAGE1, STAGE2):
        stats = json.loads((harness.run_dir().adapt_dir / stage / "stats.json").read_text(encoding="utf-8"))
        assert stats["until"]["trainer_at_stop"]["steps_done"] < 32 <= stats["until"]["trainer_at_end"]["steps_done"]


def test_file_create_records_its_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_port: Callable[[], int]
) -> None:
    harness = make_harness(tmp_path, monkeypatch, free_port(), "file-create")
    harness.agent.default = Step(samples=8, hit=True)
    harness.adapt()
    order = load_rl_order("file-create")
    records = episodes(harness.run_dir().adapt_dir / STAGE1 / "episodes.jsonl")
    assert [r["id"] for r in records] == order.stage1[:16]
    assert [r["epoch"] for r in records] == [0] * 12 + [1] * 4
    stats = json.loads((harness.run_dir().adapt_dir / STAGE1 / "stats.json").read_text(encoding="utf-8"))
    assert stats["executions_under_authorization"] == 16 and stats["nonengaged_episodes_continued"] == 0
    assert harness.prepared[0][5] == 300


def test_the_order_running_out_aborts_the_stage(harness: Harness) -> None:
    harness.set_rl(max_restarts=0)
    short_order(harness, adapt_ids(harness, 2) * 2, adapt_ids(harness, 2) * 2)
    harness.agent.default = Step(samples=4)
    with pytest.raises(AdaptAbort, match="order"):
        harness.adapt()
    adapt = harness.run_dir().adapt_dir
    assert len(harness.agent.calls) == 4
    assert not (adapt / STAGE1).exists()
    (voided,) = (adapt / "_void").iterdir()
    assert voided.name.startswith("stage1-")
    why = json.loads((voided / "why.json").read_text(encoding="utf-8"))
    assert "16 samples submitted after 4 episodes" in why["reason"]
    assert len(episodes(voided / "episodes.jsonl")) == 4
    assert not port_in_use(harness.cfgs.ports.policy)


def test_stage_2_refuses_a_trainer_that_did_not_restore_the_optimizer_state(harness: Harness) -> None:
    harness.set_rl(max_restarts=0)
    harness.behaviors = [{}, {"omit_restored": True}]
    with pytest.raises(AdaptAbort, match="restored"):
        harness.adapt()
    order = load_rl_order("drive-create")
    assert harness.agent.calls == order.stage1[:4]  # no stage-2 episode was launched
    assert len(harness.events("sigterm")) == 2  # the refused trainer was stopped
    adapt = harness.run_dir().adapt_dir
    assert (adapt / STAGE1 / "fingerprint.json").is_file()  # stage 1 stays complete
    assert [p.name.split("-")[0] for p in (adapt / "_void").iterdir()] == ["stage2"]


def test_a_trainer_without_the_entry_line_is_refused(harness: Harness) -> None:
    harness.set_rl(max_restarts=0)
    harness.behaviors = [{"omit_entry_line": True}]
    with pytest.raises(AdaptAbort, match=r"\[rl_entry\] family=qwen"):
        harness.adapt()
    assert harness.agent.calls == []


def test_the_cap_stops_the_stage_2_trainer_right_after_update_64(harness: Harness) -> None:
    order = load_rl_order("drive-create")
    # Stage 2's trainer, like the real one, takes a while before its next update: 1.5 s from update 33 on, well beyond
    # what the watcher may take under load, so that only a watcher that does not stop it lets update 33 through.
    harness.behaviors = [{}, {"step_delay_s": 1.5, "delay_from_update": 33}]
    # stage 1: four episodes of 32 samples; stage 2: 100 samples, then an episode during which update 32 completes,
    # with 8 samples to spare: enough for two more updates had the trainer been left running
    harness.agent.script = [Step(samples=32)] * 4 + [Step(samples=100), Step(samples=36, wait_for_trainer_exit=True)]
    harness.adapt()
    updates = [e for e in harness.events("update") if e["update"] == 32]
    sigterms = harness.events("sigterm")
    assert len(updates) == 2 and len(sigterms) == 2
    latency = sigterms[1]["t"] - updates[1]["t"]
    assert 0 <= latency < rl.CAP_POLL_S + 0.5, latency
    assert rl.CAP_POLL_S == 0.25
    adapt = harness.run_dir().adapt_dir
    records = episodes(adapt / STAGE2 / "episodes.jsonl")
    assert [r["id"] for r in records] == order.stage2[:2]
    cut = records[-1]
    assert [a["type"] for a in cut["anomalies"]] == ["agent_error", "cap_cut"]
    assert cut["epoch"] == 0
    stats = json.loads((adapt / STAGE2 / "stats.json").read_text(encoding="utf-8"))
    assert stats["samples_submitted"] - 128 >= 4  # update 33 was within the trainer's reach
    log = strip_ansi((adapt / STAGE2 / "trainer.log").read_text(encoding="utf-8"))
    assert "[Trainer] step 32 done" in log and "[Trainer] step 33: optim_step" not in log
    assert not [e for e in harness.events("update") if e["update"] == 33]
    assert stats["cap_cut_episodes"] == 1 and stats["cap"]["cut_episode"] == order.stage2[1]
    checkpoint = json.loads((adapt / STAGE2 / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["steps_done_max"] == 32


def test_stage_1_runs_no_cap(harness: Harness) -> None:
    # In stage 1 the trainer goes on training on surplus samples until the stage stops it; only stage 2 is capped.
    # 0.6 s per update from 33 on: a cap (at most CAP_POLL_S late) would stop it before 33; the last episode holds
    # 2.5 s, time for 33 and 34 under load.
    harness.behaviors = [{"step_delay_s": 0.6, "delay_from_update": 33}, {}]
    harness.agent.script = [Step(samples=32)] * 3 + [Step(samples=40, hold_s=2.5)]  # 136 samples: updates 33, 34
    harness.adapt()
    adapt = harness.run_dir().adapt_dir
    log = strip_ansi((adapt / STAGE1 / "trainer.log").read_text(encoding="utf-8"))
    assert "[Trainer] step 33: optim_step" in log and "[Trainer] step 34 done" in log
    first = harness.events()
    stage1_pid = next(e["pid"] for e in first if e["event"] == "start")
    stage1 = [e for e in first if e["pid"] == stage1_pid]
    update33 = next(e["t"] for e in stage1 if e["event"] == "update" and e["update"] == 33)
    assert next(e["t"] for e in stage1 if e["event"] == "sigterm") > update33  # stopped by the stage, after 33
    checkpoint = json.loads((adapt / STAGE1 / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["steps_done_max"] == 34 and checkpoint["extra_unsaved_steps"] == 2
    assert "cap" not in json.loads((adapt / STAGE1 / "stats.json").read_text(encoding="utf-8"))


def test_an_episode_abort_voids_the_stage_and_runs_it_again(harness: Harness) -> None:
    order = load_rl_order("drive-create")
    harness.agent.script = [Step(samples=32), Step(samples=0, raise_=EpisodeAbort("timeout", {"id": "x"}))]
    result = harness.adapt()
    adapt = harness.run_dir().adapt_dir
    (voided,) = (adapt / "_void").iterdir()
    assert voided.name.startswith("stage1-")
    assert "EpisodeAbort: timeout" in json.loads((voided / "why.json").read_text(encoding="utf-8"))["reason"]
    assert harness.agent.calls[:2] == order.stage1[:2]
    assert harness.agent.calls[2:6] == order.stage1[:4]  # the rerun starts from the first item, on a new trainer
    starts = harness.events("start")
    assert len(starts) == 3 and "--resume-from-ckpt" not in starts[1]["argv"]
    assert result["update_64"]


def test_restarts_are_capped_by_max_restarts(harness: Harness) -> None:
    harness.agent.default = Step(samples=0, raise_=EpisodeAbort("agent_error", None))
    with pytest.raises(AdaptAbort, match="stage 1 failed 2 times"):
        harness.adapt()
    adapt = harness.run_dir().adapt_dir
    assert sorted(p.name.split("-")[0] for p in (adapt / "_void").iterdir()) == ["stage1", "stage1"]
    assert not (adapt / STAGE1).exists() and not (adapt / "checkpoints.json").exists()


def test_a_fatal_line_in_the_trainer_log_aborts_the_stage(harness: Harness) -> None:
    harness.behaviors = [{"fatal_after": 40}]
    harness.adapt()
    (voided,) = (harness.run_dir().adapt_dir / "_void").iterdir()
    reason = json.loads((voided / "why.json").read_text(encoding="utf-8"))["reason"]
    assert "Traceback (most recent call last)" in reason
    assert len(episodes(voided / "episodes.jsonl")) == 2


def test_a_trainer_that_dies_during_an_episode_fails_the_episode(harness: Harness) -> None:
    # The trainer dies while it takes the episode's samples: the agent's request fails, so the episode does.
    harness.behaviors = [{"exit_after": 40}]
    harness.adapt()
    (voided,) = (harness.run_dir().adapt_dir / "_void").iterdir()
    reason = json.loads((voided / "why.json").read_text(encoding="utf-8"))["reason"]
    assert reason.startswith("EpisodeAbort: ") and "agent_error" in reason


def test_a_trainer_that_dies_between_episodes_is_noticed_before_the_next(harness: Harness) -> None:
    # The trainer dies after the episode's turns were all answered: the episode ends normally, and no further
    # episode starts on a dead trainer.
    harness.set_rl(max_restarts=0)
    harness.behaviors = [{"exit_after_episode": 1}]
    harness.agent.script = [Step(samples=32, wait_for_trainer_exit=True, then_return=True)]
    with pytest.raises(AdaptAbort, match=r"AdaptAbort: the trainer exited \(status 3\) before update 32"):
        harness.adapt()
    assert len(harness.agent.calls) == 1


def test_a_trainer_that_dies_before_its_last_update_is_noticed_while_waiting(harness: Harness) -> None:
    # All samples are in, the trainer is still updating, and it dies: the wait for update 32 notices.
    harness.set_rl(max_restarts=0)
    harness.behaviors = [{"step_delay_s": 0.5, "exit_after_episode": 4}]
    with pytest.raises(
        AdaptAbort, match=r"AdaptAbort: the trainer exited \(status 3\) at update \d+, before update 32"
    ):
        harness.adapt()
    assert len(harness.agent.calls) == 4


def test_a_trainer_that_never_gets_ready_aborts_the_stage(harness: Harness) -> None:
    harness.set_rl(max_restarts=0)
    harness.behaviors = [{"exit_at_start": 1}]
    with pytest.raises(AdaptAbort, match="collecting batch"):
        harness.adapt()
    assert harness.agent.calls == []


def test_a_failed_fingerprint_aborts_the_stage(harness: Harness) -> None:
    harness.fail_fingerprint = [0]
    result = harness.adapt()
    (voided,) = (harness.run_dir().adapt_dir / "_void").iterdir()
    assert voided.name.startswith("stage1-")
    assert json.loads((voided / "fingerprint.json").read_text(encoding="utf-8"))["non_base"] is False
    assert result["update_32"] != json.loads((voided / "checkpoint.json").read_text(encoding="utf-8"))["final"]


# --- resuming a run ---------------------------------------------------------------------------------------------


def test_completed_stages_are_never_run_again(harness: Harness) -> None:
    first = harness.adapt()
    calls = len(harness.agent.calls)
    assert harness.adapt() == first
    assert len(harness.agent.calls) == calls and len(harness.events("start")) == 2
    adapt = harness.run_dir().adapt_dir
    (adapt / STAGE2 / "fingerprint.json").unlink()  # stage 2 did not finish
    (adapt / "checkpoints.json").unlink()
    again = harness.adapt()
    assert again["update_32"] == first["update_32"] and again["update_64"] != first["update_64"]
    starts = harness.events("start")
    assert len(starts) == 3 and starts[2]["argv"][-2:] == ["--resume-from-ckpt", first["update_32"]]
    assert [p.name.split("-")[0] for p in (adapt / "_void").iterdir()] == ["stage2"]


def test_a_stage_1_run_again_makes_stage_2_run_again(harness: Harness) -> None:
    first = harness.adapt()
    adapt = harness.run_dir().adapt_dir
    (adapt / STAGE1 / "fingerprint.json").unlink()
    again = harness.adapt()
    assert again["update_32"] != first["update_32"] and again["update_64"] != first["update_64"]
    assert sorted(p.name.split("-")[0] for p in (adapt / "_void").iterdir()) == ["stage1", "stage2"]


def test_ctrl_c_stops_the_trainer_and_the_gateway_and_the_next_run_starts_the_stage_again(harness: Harness) -> None:
    harness.agent.script = [Step(samples=32), Step(samples=0, raise_=KeyboardInterrupt())]
    with pytest.raises(KeyboardInterrupt):
        harness.adapt()
    assert len(harness.events("sigterm")) == 1
    assert wait_until(lambda: not port_in_use(harness.cfgs.ports.policy))
    assert [event for event, _ in FakeGateway.events] == ["start", "stop"]
    adapt = harness.run_dir().adapt_dir
    assert (adapt / STAGE1 / "trainer.log").is_file() and not (adapt / "_void").exists()
    result = harness.adapt()
    (voided,) = (adapt / "_void").iterdir()
    assert "interrupted" in json.loads((voided / "why.json").read_text(encoding="utf-8"))["reason"]
    assert result["update_64"]


def test_leftover_state_stops_the_run_at_once(harness: Harness) -> None:
    def leftover(*args: Any) -> Any:
        raise LeftoverState("~/.openclaw-rl holds report.txt")

    harness.monkeypatch.setattr(rl, "prepare_environment", leftover)
    with pytest.raises(LeftoverState):
        harness.adapt()
    adapt = harness.run_dir().adapt_dir
    assert not (adapt / "_void").exists() and harness.events("start") == []


def test_a_busy_policy_port_stops_the_run_at_once(harness: Harness) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", harness.cfgs.ports.policy))
        sock.listen()
        with pytest.raises(PortInUse):
            harness.adapt()
    assert harness.events("start") == [] and not (harness.run_dir().adapt_dir / "_void").exists()


# --- Inkling-Small ------------------------------------------------------------------------------------------------


def test_inkling_rollouts_are_audited_after_each_stage(harness: Harness) -> None:
    good = [raw_row(INKLING_CALL, 1), raw_row(INKLING_TEXT, 0)]
    unparsed = [raw_row(INKLING_UNPARSED, 0)]
    mismatch = [raw_row(INKLING_CALL, 0)]  # the trainer's bridge handed the agent no call; the audit reads one
    harness.set_rl(max_restarts=2)
    harness.behaviors = [{"raw_rows": unparsed}, {"raw_rows": mismatch}, {"raw_rows": good}, {"raw_rows": good}]
    harness.adapt("inkling-small")
    adapt = harness.run_dir("inkling-small").adapt_dir
    voided = sorted((adapt / "_void").iterdir())
    reasons = [json.loads((v / "why.json").read_text(encoding="utf-8"))["reason"] for v in voided]
    audits = [json.loads((v / "raw_audit.json").read_text(encoding="utf-8")) for v in voided]
    assert sorted((a["unparsed_markup"], a["parse_mismatch"]) for a in audits) == [(0, 1), (1, 0)]
    assert any("1 generation(s) with tool-call markup that parsed to nothing" in r for r in reasons)
    assert any("1 parse mismatch(es)" in r for r in reasons)
    for stage in (STAGE1, STAGE2):
        audit = json.loads((adapt / stage / "raw_audit.json").read_text(encoding="utf-8"))
        assert audit["generations"] == 2 and audit["unparsed_markup"] == audit["parse_mismatch"] == 0
    starts = harness.events("start")
    assert len(starts) == 4 and all("thinkingmachines/Inkling-Small" in start["argv"] for start in starts)


def test_an_inkling_dump_that_cannot_be_read_aborts_the_stage(harness: Harness) -> None:
    good = [raw_row(INKLING_CALL, 1)]
    harness.behaviors = [{"raw_text": "not a generation\n"}, {"raw_rows": good}, {"raw_rows": good}]
    harness.adapt("inkling-small")
    (voided,) = (harness.run_dir("inkling-small").adapt_dir / "_void").iterdir()
    reason = json.loads((voided / "why.json").read_text(encoding="utf-8"))["reason"]
    assert "cannot be audited" in reason and "line 1" in reason


def test_an_inkling_dump_cut_off_mid_line_is_said_once(harness: Harness, caplog: pytest.LogCaptureFixture) -> None:
    good = [raw_row(INKLING_CALL, 1)]
    cut = '{"at": "t", "n_tokens": 10, "raw": "<|message_mo'  # what a trainer stopped mid-write leaves
    harness.behaviors = [{"raw_rows": good, "raw_text": cut}, {"raw_rows": good}]
    with caplog.at_level("WARNING", logger="evolve_ipi.adapt.rl"):
        harness.adapt("inkling-small")
    adapt = harness.run_dir("inkling-small").adapt_dir
    said = [r.getMessage() for r in caplog.records if r.name == "evolve_ipi.adapt.rl" and "cut off" in r.getMessage()]
    assert len(said) == 1 and str(adapt / STAGE1 / "records" / "raw_train.jsonl") in said[0], said
    assert json.loads((adapt / STAGE1 / "raw_audit.json").read_text(encoding="utf-8"))["unfinished_last_line"] is True
    assert not (adapt / "_void").exists(), "said, not refused: the cut line is no generation"


def test_other_families_are_not_audited(harness: Harness) -> None:
    harness.adapt("gpt-oss-120b")
    adapt = harness.run_dir("gpt-oss-120b").adapt_dir
    assert not (adapt / STAGE1 / "raw_audit.json").exists()
