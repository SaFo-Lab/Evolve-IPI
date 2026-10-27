"""What the tests of ``evolve_ipi.adapt.rl`` run RL adaptation against: the fake trainer and checkpoint tool as real
processes, a scripted agent in place of OpenClaw episodes, and stand-ins for the environment and the gateway."""

from __future__ import annotations

import dataclasses
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import pytest

from evolve_ipi.adapt import rl
from evolve_ipi.agent.anomalies import TrainingPolicy
from evolve_ipi.agent.episode import EpisodeAbort
from evolve_ipi.config import Configs, PortsSpec, Secrets, load_configs
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.records import RunDir
from evolve_ipi.tasks import Task, load_task

HERE = Path(__file__).resolve().parent
FAKE_TRAINER = HERE / "fake_trainer.py"
FAKE_CHECKPOINTS = HERE / "rl_fake_checkpoints.py"
CONFIGS = HERE.parents[1] / "configs"


class FakeGateway:
    """Stands in for ``agent.openclaw.Gateway``: records when it is started and stopped."""

    events: ClassVar[list[tuple[str, Any]]] = []

    def __init__(self, profile: Any) -> None:
        self.profile = profile

    def __enter__(self) -> Self:
        FakeGateway.events.append(("start", self.profile))
        return self

    def __exit__(self, *exc: object) -> None:
        FakeGateway.events.append(("stop", self.profile))


@dataclass
class Step:
    """What one scripted episode does: submit samples, then return its record or raise.

    ``hold_s`` keeps the episode running that long after its samples went in. ``wait_for_trainer_exit`` keeps it
    running until the trainer is gone, and then fails it as the agent would (an agent error), or with
    ``then_return`` ends it normally (its turns were all done before the trainer went).
    """

    samples: int = 32
    hit: bool = False
    raise_: BaseException | None = None
    hold_s: float = 0
    wait_for_trainer_exit: bool = False
    then_return: bool = False
    anomalies: list[dict[str, Any]] = field(default_factory=list)


class ScriptedAgent:
    """Replaces ``run_episode``: each call takes the next ``Step`` (or ``default``) and tells the fake trainer how
    many samples the episode submitted, as the trainer's server would have seen them."""

    def __init__(self, port: int, task: Task) -> None:
        self.port = port
        self.task = task
        self.script: list[Step] = []
        self.default = Step()
        self.calls: list[str] = []
        self.policies: list[Any] = []

    def _post(self, samples: int) -> dict[str, Any] | None:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/fake/episode",
            data=json.dumps({"samples": samples}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                answer: dict[str, Any] = json.loads(response.read())
                return answer
        except (urllib.error.URLError, ConnectionError):
            return None

    def _alive(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/healthz", timeout=2):
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return False

    def record(self, item: dict[str, Any], step: Step) -> dict[str, Any]:
        if self.task.kind == "file":
            return {
                "id": item["id"],
                "mode": "train",
                "status": "ok",
                "attack_success": step.hit,
                "created": step.hit,
                "reply": "Done.",
                "tool_calls": [{"name": "read", "arguments": {"path": item["src_relpath"]}}],
                "anomalies": list(step.anomalies),
            }
        return {
            "id": item["id"],
            "mode": "train",
            "status": "ok",
            "reply": "Done.",
            "asr": {"hit": step.hit},
            "utility": {"hit": True},
            "anomalies": list(step.anomalies),
        }

    def __call__(
        self,
        ctx: Any,
        item: dict[str, Any],
        mode: str,
        authorized: bool,
        closure: Any = None,
        *,
        policy: TrainingPolicy | None = None,
    ) -> dict[str, Any]:
        assert (mode, authorized, closure) == ("train", True, None)
        assert item is self.task.adaptation[item["id"]]
        self.calls.append(item["id"])
        self.policies.append(policy)
        step = self.script.pop(0) if self.script else self.default
        if step.raise_ is not None and not step.samples:
            raise step.raise_
        answer = self._post(step.samples) if step.samples else {}
        failed = {"type": "agent_error", "detail": "the model endpoint does not answer"}
        if answer is None:
            raise EpisodeAbort(f"{item['id']}: agent_error", dict(self.record(item, step), anomalies=[failed]))
        if step.hold_s:
            time.sleep(step.hold_s)
        if step.wait_for_trainer_exit:
            deadline = time.monotonic() + 10
            while self._alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not step.then_return:
                raise EpisodeAbort(f"{item['id']}: agent_error", dict(self.record(item, step), anomalies=[failed]))
            time.sleep(0.05)  # its port is closed; let the process finish exiting too
        if step.raise_ is not None:
            raise step.raise_
        return self.record(item, step)


@dataclass
class Harness:
    tmp: Path
    control: Path
    events_path: Path
    cfgs: Configs
    secrets: Secrets
    paths: RuntimePaths
    agent: ScriptedAgent
    task: Task
    prepared: list[tuple[Any, ...]]
    monkeypatch: pytest.MonkeyPatch
    behaviors: list[dict[str, Any]] = field(default_factory=list)
    fail_fingerprint: list[int] = field(default_factory=list)

    def run_dir(self, model: str = "qwen3.8-27b") -> RunDir:
        return RunDir.for_combo(self.tmp / "outputs", model, self.task.name, "rl")

    def write_control(self) -> None:
        self.control.write_text(
            json.dumps(
                {
                    "events": str(self.events_path),
                    "behaviors": self.behaviors,
                    "fail_fingerprint": self.fail_fingerprint,
                }
            ),
            encoding="utf-8",
        )

    def adapt(self, model: str = "qwen3.8-27b") -> dict[str, str]:
        self.write_control()
        return rl.adapt_rl(
            self.cfgs.models[model], self.task, self.run_dir(model), self.cfgs, self.secrets, paths=self.paths
        )

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        rows = [json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines()]
        return [row for row in rows if kind is None or row["event"] == kind]

    def set_rl(self, **values: Any) -> None:
        self.cfgs = dataclasses.replace(self.cfgs, rl=self.cfgs.rl.model_copy(update=values))

    def use_task(self, name: str) -> None:
        self.task = load_task(name)
        self.agent.task = self.task


def make_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy_port: int, task: str) -> Harness:
    cfgs = load_configs(CONFIGS)
    ports = PortsSpec(gateway=811, policy=policy_port, proxy=821, model=831)  # only the policy port is ever bound
    # a short wait for the last update: a test whose trainer never gets there fails in seconds, not half an hour
    cfgs = dataclasses.replace(
        cfgs, ports=ports, rl=cfgs.rl.model_copy(update={"max_restarts": 1, "until_step_timeout_s": 30})
    )
    secrets = Secrets(tinker_api_key="test-key-not-real", evolver_api_key="evolver-secret", evolver_model="m")
    paths = RuntimePaths(
        root=tmp_path / "runtime",
        agent_home=tmp_path / "home",
        outputs=tmp_path / "outputs",
        metaclaw_python=tmp_path / "none",
    )
    loaded = load_task(task)
    agent = ScriptedAgent(policy_port, loaded)
    prepared: list[tuple[Any, ...]] = []

    def prepare_environment(
        task: Task, line: str, paths_: RuntimePaths, ports_: PortsSpec, port: int, timeout: int
    ) -> Any:
        prepared.append((task.name, line, paths_, ports_, port, timeout))
        return SimpleNamespace(profile=SimpleNamespace(name=f"profile-{len(prepared)}"), task=task)

    FakeGateway.events = []
    control = tmp_path / "control.json"
    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")
    monkeypatch.setattr(rl, "prepare_environment", prepare_environment)
    monkeypatch.setattr(rl, "Gateway", FakeGateway)
    monkeypatch.setattr(rl, "run_episode", agent)
    monkeypatch.setattr(rl, "verify_patches", lambda: {"OpenClaw-RL/openclaw-tinker/rollout.py": "patched"})
    monkeypatch.setattr(
        rl, "_trainer_command", lambda flags: [sys.executable, str(FAKE_TRAINER), "--control", str(control), *flags]
    )
    monkeypatch.setattr(
        rl,
        "_checkpoints_command",
        lambda args: [sys.executable, str(FAKE_CHECKPOINTS), "--control", str(control), *args],
    )
    monkeypatch.setattr(rl, "STEP_POLL_S", 0.05)
    harness = Harness(
        tmp=tmp_path,
        control=control,
        events_path=events,
        cfgs=cfgs,
        secrets=secrets,
        paths=paths,
        agent=agent,
        task=loaded,
        prepared=prepared,
        monkeypatch=monkeypatch,
    )
    harness.write_control()
    return harness


def episodes(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def wait_until(condition: Callable[[], bool], timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()
