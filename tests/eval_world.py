"""A stand-in for everything an evaluation pass starts and runs: the model server, MetaClaw's proxy, the OpenClaw gateway,
the environment, the checkpoint fingerprint and the agent's episodes. A pass runs in milliseconds and starts nothing.

Every start of a model server begins a new attempt, numbered from 1 over the whole test. What the agent does in an
episode is ``World.acts[(attempt, item id)]`` and otherwise ``World.default``: a clean episode in which the agent reads
the content, writes one generation and does not carry out the injected request. ``World.calls`` lists, in order, what
the code under test did to the stand-ins.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evolve_ipi import evaluate, tasks
from evolve_ipi.adapt.library import lib_sha
from evolve_ipi.agent.episode import EpisodeAbort
from evolve_ipi.config import Secrets, load_configs
from evolve_ipi.models.launch import ready_line, server_files
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import ProcessError
from evolve_ipi.records import RunDir

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = "tinker://0000-train:0/weights/step_0032"
STAMP = "%Y-%m-%dT%H:%M:%S%z"


@dataclass
class Act:
    """What the agent does in one episode, and what the processes write meanwhile."""

    hit: bool = False
    encountered: bool = True
    utility: bool = True  # the verdict of the user's own task, where the task has one
    anomalies: list[dict[str, str]] = field(default_factory=list)
    status: str = "ok"
    elapsed: float = 3.0
    reply: str = "The document says the team finished the refactoring."
    generations: list[str] = field(default_factory=lambda: ["<think>read it</think>Done."])
    server_log: str = ""  # appended to the model server's log during the episode
    proxy_log: str = ""
    abort: str | None = None  # the episode raises EpisodeAbort
    raise_: BaseException | None = None
    effect: Callable[[World], None] | None = None
    timeline: list[tuple[float, str]] | None = None
    timeline_error: BaseException | None = None  # raised when the episode's transcript timeline is read
    request: str | None = None  # what the record says was sent, instead of the right request
    extra: dict[str, Any] = field(default_factory=dict)  # more fields of the record


@dataclass
class FakeProcess:
    name: str
    log_path: Path
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    stop_error: BaseException | None = None  # raised by stop(), after the process has stopped
    stopped: bool = False
    exited: int | None = None

    @property
    def returncode(self) -> int | None:
        return self.exited

    def alive(self) -> bool:
        return not self.stopped and self.exited is None

    def stop(self) -> None:
        self.calls.append(("stop", self.name))
        self.stopped = True
        if self.stop_error is not None:
            raise self.stop_error

    def log_tail(self, lines: int = 40) -> str:
        return "\n".join(self.log_path.read_text().splitlines()[-lines:])

    def write(self, text: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(text)


@dataclass(frozen=True)
class FakeProfile:
    line: str
    model_id: str
    provider_port: int


@dataclass(frozen=True)
class FakeContext:
    profile: FakeProfile
    raw_dump: Path | None = None

    def with_raw_dump(self, raw_dump: Path | None) -> FakeContext:
        return dataclasses.replace(self, raw_dump=raw_dump)


class FakeGateway:
    """Stands in for ``agent.openclaw.Gateway``: ``World.gateway_fails`` is raised by ``start``, and the n-th check
    between episodes of a gateway returns (or raises) ``World.restarts[n]``, once."""

    def __init__(self, world: World, profile: FakeProfile) -> None:
        self.world, self.profile = world, profile
        self.running = False
        self.checks = 0

    def start(self) -> None:
        self.world.calls.append(("gateway", "start", self.profile.line))
        self.world.gateways.append(self)
        if self.world.gateway_fails is not None:
            raise self.world.gateway_fails
        self.running = True

    def stop(self) -> None:
        self.world.calls.append(("gateway", "stop"))
        self.running = False

    def restart_if_needed(self) -> str | None:
        self.checks += 1
        reason = self.world.restarts.pop(self.checks, None)
        if isinstance(reason, BaseException):
            raise reason
        if reason is not None:
            self.world.calls.append(("gateway", "restart", reason))
        return reason


class World:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        task: str = "drive-create",
        model: str = "qwen3.8-27b",
        mechanism: str = "rl",
    ) -> None:
        self.cfgs = load_configs(REPO / "configs")
        self.task = tasks.load_task(task)
        self.model = self.cfgs.models[model]
        self.mechanism = mechanism
        self.paths = RuntimePaths(
            root=tmp_path / "runtime",
            agent_home=tmp_path / "home",
            outputs=tmp_path / "outputs",
            metaclaw_python=tmp_path / "venv" / "bin" / "python",
        )
        self.secrets = Secrets(tinker_api_key="tinker-test-key")
        self.run = RunDir.for_combo(tmp_path / "outputs", model, task, mechanism)
        self.calls: list[tuple[Any, ...]] = []
        self.acts: dict[tuple[int, str], Act] = {}
        self.default = Act()
        self.attempt = 0
        self.servers: list[FakeProcess] = []
        self.proxies: list[FakeProcess] = []
        self.gateways: list[FakeGateway] = []
        self.server_env: dict[str, str] = {}
        self.server_fails: dict[int, BaseException] = {}  # attempt -> raised by start_model_server
        self.prepare_raises: BaseException | None = None
        self.gateway_fails: BaseException | None = None
        self.proxy_fails: BaseException | None = None  # raised by start_proxy
        self.restarts: dict[int, str | BaseException] = {}  # n-th check between episodes -> its reason, or a failure
        self.stop_errors: dict[str, BaseException] = {}  # process name -> raised when the next one is stopped
        self.fingerprint_runs = 0
        self.fingerprint_status = 0
        self.fingerprint_diff = 0.31
        self._requests = 0
        self._sessions = 0
        monkeypatch.setattr(evaluate, "start_model_server", self.start_model_server)
        monkeypatch.setattr(evaluate, "prepare_environment", self.prepare_environment)
        monkeypatch.setattr(evaluate, "Gateway", lambda profile: FakeGateway(self, profile))
        monkeypatch.setattr(evaluate, "run_episode", self.run_episode)
        monkeypatch.setattr(evaluate, "episode_timeline", self.episode_timeline)
        monkeypatch.setattr(evaluate, "_run_tool", self.run_tool)
        self.skills = SimpleNamespace(start_proxy=self.start_proxy)
        monkeypatch.setattr(evaluate, "_skills", lambda: self.skills)
        self.audits: list[tuple[Path, str | None]] = []
        self.inkling_audit = SimpleNamespace(audit_raw_dump=self.audit_raw_dump, audit_problems=audit_problems)
        monkeypatch.setattr(evaluate, "_inkling_audit", lambda: self.inkling_audit)

    # --- the adaptation a test evaluates ---

    def finish_rl(self, stage1: list[str] | None = None, stage2: list[str] | None = None) -> None:
        adapt = self.run.adapt_dir
        ids = sorted(self.task.adaptation)
        for number, chosen in ((1, stage1 or ids[:12]), (2, stage2 or ids[12:])):
            directory = adapt / f"stage{number}"
            directory.mkdir(parents=True, exist_ok=True)
            rows = [json.dumps({"id": item, "mode": "train"}) for item in chosen]
            (directory / "episodes.jsonl").write_text("\n".join(rows) + "\n")
        record = {"stage": 2, "absolute_step": 64, "verified_loadable": True, "final": CHECKPOINT}
        (adapt / "stage2" / "checkpoint.json").write_text(json.dumps(record))
        marker = {"base_model": self.model.tinker_id, "update_32": "tinker://x/step_0032", "update_64": CHECKPOINT}
        (adapt / "checkpoints.json").write_text(json.dumps(marker))

    def finish_skills(self, skills: dict[str, str] | None = None, order: list[str] | None = None) -> Path:
        adapt = self.run.adapt_dir
        library = adapt / "library"
        for name, text in (skills or {"read-before-acting": "Read first.", "follow-the-document": "Do it."}).items():
            (library / name).mkdir(parents=True)
            (library / name / "SKILL.md").write_text(text)
        (library / "evolution_history.jsonl").write_text("{}\n" * 12)
        rows = [json.dumps({"id": item, "mode": "train"}) for item in (order or sorted(self.task.adaptation)[:12])]
        (adapt / "episodes.jsonl").write_text("\n".join(rows) + "\n")
        marker = {"library": "library", "sha": lib_sha(library), "evolutions": 12, "skills": 2, "zero_skill_events": 0,
                  "at": "now"}  # fmt: skip
        (adapt / "library.json").write_text(json.dumps(marker))
        return library

    # --- the stand-ins ---

    def act(self, item_id: str) -> Act:
        return self.acts.get((self.attempt, item_id), self.default)

    def prepare_environment(
        self, task: Any, line: str, paths: RuntimePaths, ports: Any, provider_port: int, timeout_s: int
    ) -> FakeContext:
        self.calls.append(("prepare", task.name, line, provider_port, timeout_s))
        if self.prepare_raises is not None:
            raise self.prepare_raises
        model_id = "policy" if line == "rl" else "qwen3.8-27b"
        return FakeContext(FakeProfile(line, model_id, provider_port))

    def start_model_server(
        self,
        family: str,
        base_model: str,
        served_name: str,
        port: int,
        out_dir: Path,
        checkpoint: str | None,
        env: dict[str, str],
        *,
        temperature: float,
    ) -> FakeProcess:
        self.attempt += 1
        self._requests = 0
        self.calls.append(("server", family, base_model, served_name, port, out_dir, checkpoint, temperature))
        self.server_env = dict(env)
        if self.attempt in self.server_fails:
            raise self.server_fails[self.attempt]
        out_dir.mkdir(parents=True)
        files = server_files(out_dir)
        files.log.write_text(
            "INFO starting\nTraceback (most recent call last):\n  a handled startup problem\nValueError: x\n"
            + ready_line("127.0.0.1", port, base_model, checkpoint)
            + "\n"
        )
        files.raw.write_text("")
        process = FakeProcess("model server", files.log, self.calls, self.stop_errors.pop("model server", None))
        self.servers.append(process)
        return process

    def start_proxy(
        self,
        model: Any,
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
        self.calls.append(("proxy", Path(skills_dir), evolve, require_empty, Path(record_dir)))
        if self.proxy_fails is not None:
            raise self.proxy_fails
        assert not record_dir.exists(), "every proxy start gets a fresh record directory"
        if require_empty:
            assert not any(Path(skills_dir).iterdir()), "a proxy that requires an empty library got another"
        record_dir.mkdir(parents=True)
        log = record_dir / "proxy.log"
        log.write_text(
            "Traceback (most recent call last):\n  OSError: the tokenizer is not on the Hub\n"
            "METACLAW patched x\nskills_only proxy READY on 127.0.0.1:1\n"
        )
        process = FakeProcess("MetaClaw proxy", log, self.calls, self.stop_errors.pop("MetaClaw proxy", None))
        self.proxies.append(process)
        return process

    def run_tool(self, argv: list[str], env: dict[str, str], log: Path) -> int:
        self.fingerprint_runs += 1
        self.calls.append(("fingerprint", argv))
        out = Path(argv[argv.index("--out") + 1])
        path = argv[argv.index("fingerprint") + 1]
        min_diff = float(argv[argv.index("--min-diff") + 1])
        record = {"path": path, "base_model": argv[argv.index("--base-model") + 1], "probe_tokens": 40,
                  "mean_abs_logprob_diff_vs_base": self.fingerprint_diff, "min_diff": min_diff,
                  "non_base": self.fingerprint_diff > min_diff, "seconds": 1.0, "at": "now"}  # fmt: skip
        out.write_text(json.dumps(record))
        log.write_text("fingerprint\n")
        return self.fingerprint_status

    def audit_raw_dump(self, path: Path, *, parsed_key: str | None) -> dict[str, Any]:
        """Stands in for the shared raw audit: a generation that holds UNPARSED counts as unparsed markup."""
        self.audits.append((Path(path), parsed_key))
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return {"generations": len(rows), "unparsed_markup": sum("UNPARSED" in row["raw"] for row in rows)}

    def episode_timeline(self, ctx: FakeContext, record: dict[str, Any]) -> list[tuple[float, str]] | None:
        act = self.act(record["id"])
        if act.timeline_error is not None:
            raise act.timeline_error
        return act.timeline

    def run_episode(
        self,
        ctx: FakeContext,
        item: dict[str, Any],
        mode: str,
        authorized: bool,
        closure: Any = None,
        *,
        policy: Any = None,
    ) -> dict[str, Any]:
        assert mode == "eval" and closure is None and policy is None
        assert ctx.raw_dump is not None
        act = self.act(item["id"])
        self.calls.append(("episode", item["id"], authorized))
        requests = []
        for text in act.generations:
            self._requests += 1
            requests.append(self._requests)
            line = {"at": time.strftime(STAMP), "request": self._requests, "messages": 3, "n_tokens": 7, "raw": text}
            with open(ctx.raw_dump, "a", encoding="utf-8") as dump:
                dump.write(json.dumps(line) + "\n")
        if act.server_log:
            self.servers[-1].write(act.server_log)
        if act.proxy_log:
            self.proxies[-1].write(act.proxy_log)
        if act.effect is not None:
            act.effect(self)
        if act.raise_ is not None:
            raise act.raise_
        record = self.record(item, authorized, act, requests)
        if act.abort is not None:
            raise EpisodeAbort(act.abort, record)
        return record

    def record(self, item: dict[str, Any], authorized: bool, act: Act, requests: list[int]) -> dict[str, Any]:
        self._sessions += 1
        record: dict[str, Any] = {
            "id": item["id"],
            "mode": "eval",
            "request": act.request or tasks.request(self.task, item, authorized),
            "status": act.status,
            "error": None if act.status == "ok" else "the turn failed",
            "elapsed": act.elapsed,
            "session_id": f"session-{self._sessions}",
            "reply": act.reply,
            "anomalies": [dict(anomaly) for anomaly in act.anomalies],
            "workspace_reverted": [],
        }
        if self.task.kind == "service":
            encounter = item["encounter"]
            access = [{"program": encounter["program"], "argv": list(encounter["argv_prefix"]), "rc": 0}]
            utility = None if item["judge_utility"]["type"] == "none" else act.utility
            record.update(
                asr={"hit": act.hit, "params_exact": None, "matches": [], "new_records": 0},
                utility={"hit": utility, "params_exact": None, "matches": [], "new_records": 0},
                encountered=act.encountered,
                access_log=access if act.encountered else [],
            )
        else:
            calls = [{"name": "read", "arguments": {"path": item["src_relpath"]}}] if act.encountered else []
            record.update(
                attack_success=act.hit,
                tool_calls=calls,
                transcript=None,
                encountered=act.encountered,
                acted=act.hit,
                target_after_episode=False,
            )
        record["reasoning"] = [
            {"request": number, "n_tokens": 7, "thinking": None, "output": "Done."} for number in requests
        ]
        record.update(act.extra)
        return record

    # --- what happened ---

    def episodes(self) -> list[tuple[str, bool]]:
        return [(call[1], call[2]) for call in self.calls if call[0] == "episode"]

    def server_starts(self) -> list[tuple[Any, ...]]:
        return [call for call in self.calls if call[0] == "server"]

    def all_stopped(self) -> bool:
        return all(not process.alive() for process in self.servers + self.proxies) and not any(
            gateway.running for gateway in self.gateways
        )


def audit_problems(audit: dict[str, Any]) -> list[str]:
    count = audit["unparsed_markup"]
    return [f"{count} generation(s) hold tool-call markup that parsed to no call"] if count else []


def server_fails(message: str = "the model server exited with status 1 before logging its ready line") -> ProcessError:
    return ProcessError(message)
