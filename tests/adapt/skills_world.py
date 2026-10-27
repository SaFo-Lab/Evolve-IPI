"""A stand-in world for ``adapt_skills``: the model server, the MetaClaw proxy, the gateway, the environment, the
episodes and the guards are scripted, so a whole evolution runs in a fraction of a second and starts nothing.

Each attempt follows a ``Script`` (the first attempt ``scripts[0]``, and so on; attempts beyond the list are clean).
``calls`` lists, in order, what the code under test did to the stand-ins.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from evolve_ipi.adapt import AdaptAbort, skills
from evolve_ipi.config import Configs, ModelSpec, Secrets, load_configs
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.records import RunDir
from evolve_ipi.tasks import Task, load_task

REPO = Path(__file__).resolve().parents[2]
SECRETS = Secrets(
    tinker_api_key="tinker-test-key",
    evolver_base_url="https://evolver.invalid/v1",
    evolver_api_key="evolver-test-key",
    evolver_model="evolver-test-model",
)
SHAS = {
    "MetaClaw/metaclaw/api_server.py": "a" * 64,
    "MetaClaw/metaclaw/skill_evolver.py": "b" * 64,
    "OpenClaw-RL/openclaw-tinker/rollout.py": "c" * 64,
}
CLEAN_GENERATION = {"request": 1, "messages": 2, "n_tokens": 5, "raw": "<|content_text|>ok<|end_message|>"}
SHARED_AUDIT = "evolve_ipi.models.inkling_audit"


@dataclass
class Script:
    """What happens in one attempt; the defaults make a clean evolution."""

    prepare_raises: BaseException | None = None
    raise_at: dict[int, BaseException] = field(default_factory=dict)  # conversation -> raised by run_episode
    proxy_log_at: dict[int, str] = field(default_factory=dict)  # conversation -> appended to the proxy log
    model_log_at: dict[int, str] = field(default_factory=dict)  # conversation -> appended to the model server log
    proxy_dies_at: int | None = None
    evolutions: dict[int, int] = field(default_factory=dict)  # conversation -> history lines its closing adds
    zero_skill_at: set[int] = field(default_factory=set)
    guard_fails: str | None = None
    audit_problems: list[str] = field(default_factory=list)  # what the shared Inkling audit reports
    audit_raises: BaseException | None = None
    raw_text: str | None = str(CLEAN_GENERATION["raw"])  # the one generation the model server dumps (None: none)
    gateway_restart_at: dict[int, Any] = field(default_factory=dict)  # a reason, or an exception the restart raises
    start_error: dict[str, BaseException] = field(default_factory=dict)  # "model server" / "proxy" / "gateway"
    stop_error: dict[str, BaseException] = field(default_factory=dict)  # raised by that process's stop(), once stopped
    proxy_alive_checks: int | None = None  # the proxy dies once it has been asked this many times whether it runs
    proxy_log_at_check: dict[int, str] = field(default_factory=dict)  # n-th time the proxy is asked -> its log gets
    audit_unfinished: bool = False  # the shared Inkling audit finds the dump's last line cut off


class FakeProcess:
    def __init__(self, world: World, name: str, log_path: Path, alive_checks: int | None = None) -> None:
        self.world, self.name, self.log_path = world, name, log_path
        self.returncode: int | None = None
        self.alive_checks, self.asked = alive_checks, 0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"00:00:00 | INFO | {name} | started\n", encoding="utf-8")

    def alive(self) -> bool:
        self.asked += 1
        if self.asked in self.world.script.proxy_log_at_check and self.name == "proxy":
            self.write(self.world.script.proxy_log_at_check[self.asked])
        if self.alive_checks is not None and self.asked > self.alive_checks and self.returncode is None:
            self.returncode = 1
            self.write("12:00:00 | ERROR | proxy | killed\n")
        return self.returncode is None

    def write(self, text: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(text)

    def log_tail(self, lines: int = 40) -> str:
        return "\n".join(self.log_path.read_text(encoding="utf-8").splitlines()[-lines:])

    def stop(self) -> None:
        self.world.calls.append(("stop", self.name))
        if self.returncode is None:
            self.returncode = -15
        if self.name in self.world.script.stop_error:
            raise self.world.script.stop_error[self.name]


class FakeGateway:
    def __init__(self, world: World, profile: Any) -> None:
        self.world, self.profile = world, profile

    def start(self) -> None:
        self.world.calls.append(("start", "gateway"))
        if "gateway" in self.world.script.start_error:
            raise self.world.script.start_error["gateway"]
        self.world.gateway = self

    def stop(self) -> None:
        self.world.calls.append(("stop", "gateway"))
        if "gateway" in self.world.script.stop_error:
            raise self.world.script.stop_error["gateway"]

    def restart_if_needed(self) -> str | None:
        reason = self.world.script.gateway_restart_at.get(self.world.conversation + 1)
        self.world.calls.append(("restart_if_needed", reason))
        if isinstance(reason, BaseException):
            raise reason
        return reason


class World:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.calls: list[tuple[Any, ...]] = []
        self.scripts: list[Script] = []
        self.preflight_error: BaseException | None = None
        self.proxies: list[FakeProcess] = []
        self.attempt = 0
        self.conversation = 0
        self.proxy: FakeProcess | None = None
        self.server: FakeProcess | None = None
        self.gateway: FakeGateway | None = None
        self.history: Path | None = None
        self.proxy_args: list[dict[str, Any]] = []
        self.server_args: list[dict[str, Any]] = []
        self.episode_args: list[dict[str, Any]] = []
        base = load_configs(REPO / "configs")
        self.cfgs: Configs = dataclasses.replace(base, skills=base.skills.model_copy(update={"drain_poll_s": 0.001}))
        home = tmp_path / "home"
        home.mkdir()
        self.paths = RuntimePaths(
            root=tmp_path / "runtime",
            agent_home=home,
            outputs=tmp_path / "outputs",
            metaclaw_python=Path(sys.executable),
        )
        for name, value in {
            "verify_patches": lambda: dict(SHAS),
            "prepare_environment": self.prepare_environment,
            "start_model_server": self.start_model_server,
            "start_proxy": self.start_proxy,
            "Gateway": lambda profile: FakeGateway(self, profile),
            "run_episode": self.run_episode,
            "close_session": self.close_session,
            "run_guards": self.run_guards,
            "evolver_preflight": self.evolver_preflight,
        }.items():
            monkeypatch.setattr(skills, name, value)
        self.audits: list[tuple[Path, Any]] = []
        shared = ModuleType(SHARED_AUDIT)
        shared.audit_raw_dump = self.audit_raw_dump  # type: ignore[attr-defined]
        shared.audit_problems = self.audit_problems  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, SHARED_AUDIT, shared)

    # --- the stand-ins -----------------------------------------------------------------------------

    @property
    def script(self) -> Script:
        return self.scripts[self.attempt - 1] if self.attempt <= len(self.scripts) else Script()

    def prepare_environment(
        self, task: Task, line: str, paths: RuntimePaths, ports: Any, provider_port: int, timeout_s: int
    ) -> Any:
        self.attempt += 1
        self.conversation = 0
        self.calls.append(("prepare", task.name, line, provider_port, timeout_s))
        if self.script.prepare_raises is not None:
            raise self.script.prepare_raises
        return SimpleNamespace(profile=f"profile-{task.name}", task=task)

    def start_model_server(
        self,
        family: str,
        base_model: str,
        served_name: str,
        port: int,
        out_dir: Path,
        checkpoint: str | None,
        env: Mapping[str, str],
        *,
        temperature: float,
    ) -> FakeProcess:
        assert not out_dir.exists(), "every model server start gets a fresh directory"
        self.calls.append(("start", "model server"))
        if "model server" in self.script.start_error:
            raise self.script.start_error["model server"]
        self.server_args.append(
            {
                "family": family,
                "base_model": base_model,
                "served_name": served_name,
                "port": port,
                "out_dir": out_dir,
                "checkpoint": checkpoint,
                "env": dict(env),
                "temperature": temperature,
            }
        )
        self.server = FakeProcess(self, "model server", out_dir / "model_server.log")
        row = {**CLEAN_GENERATION, "raw": self.script.raw_text}
        dumped = "" if self.script.raw_text is None else json.dumps(row) + "\n"
        (out_dir / "raw.jsonl").write_text(dumped, encoding="utf-8")
        return self.server

    def start_proxy(
        self,
        model: ModelSpec,
        skills_dir: Path,
        evolve: bool,
        require_empty: bool,
        record_dir: Path,
        ports: Any,
        secrets: Secrets,
        *,
        skills: Any,
        paths: RuntimePaths,
    ) -> FakeProcess:
        assert not record_dir.exists(), "every proxy start gets a fresh record directory"
        assert skills_dir.is_dir() and not any(skills_dir.iterdir()), "evolution starts from an empty library"
        self.calls.append(("start", "proxy"))
        if "proxy" in self.script.start_error:
            raise self.script.start_error["proxy"]
        self.proxy_args.append(
            {
                "model": model,
                "skills_dir": skills_dir,
                "evolve": evolve,
                "require_empty": require_empty,
                "record_dir": record_dir,
                "ports": ports,
                "secrets": secrets,
                "skills": skills,
                "paths": paths,
            }
        )
        self.history = skills_dir / "evolution_history.jsonl"
        self.proxy = FakeProcess(self, "proxy", record_dir / "proxy.log", self.script.proxy_alive_checks)
        self.proxies.append(self.proxy)
        (record_dir / "conversations.jsonl").write_text("", encoding="utf-8")
        return self.proxy

    def run_episode(
        self, ctx: Any, item: dict[str, Any], mode: str, authorized: bool, closure: Any, *, policy: Any
    ) -> dict[str, Any]:
        self.conversation += 1
        number = self.conversation
        self.calls.append(("episode", item["id"]))
        self.episode_args.append({"ctx": ctx, "mode": mode, "authorized": authorized, "policy": policy})
        if number in self.script.raise_at:
            raise self.script.raise_at[number]
        session = f"session-{self.attempt}-{number}"
        self.calls.append(("feedback", item["id"]))
        record: dict[str, Any] = {"id": item["id"], "mode": mode, "session_id": session}
        if ctx.task.kind == "service":
            record.update(asr={"hit": number % 3 == 0}, feedback_polarity="pos" if number % 3 == 0 else "neg")
        else:
            record.update(attack_success=number % 4 == 0, feedback="pos" if number % 4 == 0 else "neg")
        record["closure"] = closure(session)
        assert self.proxy is not None and self.server is not None
        if number in self.script.proxy_log_at:
            self.proxy.write(self.script.proxy_log_at[number])
        if number in self.script.model_log_at:
            self.server.write(self.script.model_log_at[number])
        if number == self.script.proxy_dies_at:
            self.proxy.returncode = 1
        return record

    def close_session(self, proxy_port: int, served_name: str, session_id: str) -> None:
        self.calls.append(("close", session_id, proxy_port, served_name))
        assert self.history is not None
        number = self.conversation
        library = self.history.parent
        with open(self.history, "a", encoding="utf-8") as history:
            for k in range(self.script.evolutions.get(number, 1)):
                generated = 0 if number in self.script.zero_skill_at else 1
                if generated:
                    skill = library / f"skill-{self.attempt}-{number}-{k}"
                    skill.mkdir()
                    (skill / "SKILL.md").write_text(f"# skill {number}\n", encoding="utf-8")
                line = {"num_failures_analyzed": 2, "failure_prompts": ["a", "b"], "num_skills_generated": generated}
                history.write(json.dumps(line) + "\n")

    def evolver_preflight(self, secrets: Secrets, skills: Any, *, paths: RuntimePaths) -> None:
        self.calls.append(("preflight", secrets.evolver_model, skills.evolver_max_completion_tokens))
        if self.preflight_error is not None:
            raise self.preflight_error

    def run_guards(self, history: Path, records: Path, log_dir: Path, *, paths: RuntimePaths) -> None:
        self.calls.append(("guards", history, records))
        if self.script.guard_fails:
            raise AdaptAbort(f"the {self.script.guard_fails} guard failed")

    def audit_raw_dump(self, path: Path, *, parsed_key: str | None) -> dict[str, Any]:
        self.audits.append((path, parsed_key))
        self.calls.append(("audit", path))
        if self.script.audit_raises is not None:
            raise self.script.audit_raises
        audit = {"where": str(path), "generations": 1, "unparsed_markup": 0}
        return {**audit, "unfinished_last_line": True} if self.script.audit_unfinished else audit

    def audit_problems(self, audit: Mapping[str, Any]) -> list[str]:
        return list(self.script.audit_problems)

    # --- running --------------------------------------------------------------------------------

    def run_dir(self, model: str, task: str) -> RunDir:
        return RunDir.for_combo(self.paths.outputs, model, task, "skills")

    def adapt(
        self, model: str = "qwen3.8-27b", task: str = "drive-create", secrets: Secrets = SECRETS, **options: Any
    ) -> Any:
        return skills.adapt_skills(
            self.cfgs.models[model],
            load_task(task),
            self.run_dir(model, task),
            self.cfgs,
            secrets,
            paths=self.paths,
            **options,
        )

    def named(self, kind: str) -> list[Any]:
        return [call[1] for call in self.calls if call[0] == kind]

    def started(self) -> list[Any]:
        return [call[1] for call in self.calls if call[0] in ("start", "prepare")]


def episodes(run: RunDir) -> list[dict[str, Any]]:
    path = run.adapt_dir / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def voids(run: RunDir) -> list[Path]:
    void = run.adapt_dir / "_void"
    return sorted(void.iterdir()) if void.is_dir() else []


def why(void: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((void / "why.json").read_text(encoding="utf-8"))
    return data


def abort_at(number: int, error: Callable[[], BaseException]) -> Script:
    return Script(raise_at={number: error()})
