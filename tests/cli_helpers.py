"""Helpers of the command-line tests: configurations with ports of their own, stand-ins for the paid work, and probes
for the programs the integration tests need."""

from __future__ import annotations

import json
import os
import random
import shutil
import socket
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from evolve_ipi import runner
from evolve_ipi.adapt.library import lib_sha
from evolve_ipi.agent.openclaw import GATEWAY_SPAN, OPENCLAW_VERSION, OpenClawError, installed_version
from evolve_ipi.config import Configs, ModelSpec, PortsSpec, Secrets
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import port_in_use
from evolve_ipi.records import RunDir, write_json
from evolve_ipi.tasks import Task
from evolve_ipi.thirdparty import PATCHES

REPO = Path(__file__).resolve().parents[1]
REAL_PORTS = {18789, 30011, 30012, 30013}  # the ports of a real run; no test may use one
PATCHED = {patch.path: "0" * 64 for patch in PATCHES}
SECRET_VALUES = {
    "TINKER_API_KEY": "tinker-secret-value-0001",
    "EVOLVER_BASE_URL": "https://evolver.invalid/v1",
    "EVOLVER_API_KEY": "evolver-secret-value-0002",
    "EVOLVER_MODEL": "evolver-model-test",
}


# --- ports and configurations -----------------------------------------------------------------------------------------


def _bindable(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return not port_in_use(port)


def free_ports() -> PortsSpec:
    """Four distinct free ports, the gateway's with the five above it free as well. They are taken below the ports the
    system hands out for outgoing connections, and never are a port of a real run."""
    rng = random.Random()
    for _ in range(500):
        first = rng.randrange(20000, 32000 - GATEWAY_SPAN - 3)
        ports = range(first, first + GATEWAY_SPAN + 3)
        if not REAL_PORTS & set(ports) and all(_bindable(port) for port in ports):
            return PortsSpec(gateway=first, policy=ports[-3], proxy=ports[-2], model=ports[-1])
    raise RuntimeError("no free ports found")


def write_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def edit_yaml(path: Path, change: Callable[[Any], None]) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    write_yaml(path, data)


def combo(model: str, task: str, mechanism: str) -> dict[str, str]:
    return {"model": model, "task": task, "mechanism": mechanism}


def write_configs(
    root: Path,
    *,
    combos: Sequence[dict[str, str]] | None = None,
    settings: Sequence[int] = (1, 2, 3),
    ports: PortsSpec | None = None,
    name: str = "configs",
) -> Path:
    """A copy of ``configs/`` in ``root``, with ports that are free and the experiment's combinations as given."""
    destination = root / name
    shutil.copytree(REPO / "configs", destination)
    ports = ports or free_ports()
    write_yaml(
        destination / "runtime.yaml",
        {"ports": {"gateway": ports.gateway, "policy": ports.policy, "proxy": ports.proxy, "model": ports.model}},
    )
    if combos is not None:
        write_yaml(destination / "main.yaml", {"settings": list(settings), "combos": list(combos)})
    return destination


# --- the environment of a command -------------------------------------------------------------------------------------


def set_secrets(monkeypatch: Any, *names: str) -> None:
    """Put the named secrets in the environment (all four of them when none is named)."""
    for name in names or tuple(SECRET_VALUES):
        monkeypatch.setenv(name, SECRET_VALUES[name])


def runtime_env(monkeypatch: Any, tmp_path: Path, metaclaw_python: Path | None = None) -> RuntimePaths:
    """Point the runtime directories, the agent's home and the outputs at ``tmp_path``, as the environment does."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.delenv(
        "EVOLVE_IPI_IMAGE", raising=False
    )  # whatever machine the tests run on, they run outside the image
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("EVOLVE_IPI_RUNTIME", str(tmp_path / "runtime"))
    monkeypatch.setenv("EVOLVE_IPI_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("EVOLVE_IPI_METACLAW_PYTHON", str(metaclaw_python or tmp_path / "no-metaclaw" / "python"))
    return RuntimePaths()


def in_image(
    monkeypatch: Any, tmp_path: Path, *, hostname: str = "evolve-ipi", mounted: Sequence[Path] = (), first: bool = False
) -> Path:
    """Make the process look like the one in the image: the variable the image sets, the host name, whether it is the
    first process of the container, and what is mounted (a ``/proc/self/mountinfo`` of its own). Returns that file."""
    lines = ["22 1 0:21 / / rw - overlay overlay rw", "23 22 0:5 / /proc rw - proc proc rw"]
    lines += [
        f"{30 + number} 22 8:1 / {str(path).replace(' ', chr(92) + '040')} rw - ext4 /dev/sda1 rw"
        for number, path in enumerate(mounted)
    ]
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("EVOLVE_IPI_IMAGE", "1")
    monkeypatch.setattr(runner, "_hostname", lambda: hostname)
    monkeypatch.setattr(runner, "_is_first_process", lambda: first)
    monkeypatch.setattr(runner, "MOUNTINFO", mountinfo)
    return mountinfo


def make_executable(path: Path) -> Path:
    """A file that passes for the MetaClaw Python where the test does not run it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


# --- stand-ins for the paid work --------------------------------------------------------------------------------------


class Work:
    """Stands in for ``adapt_rl``, ``adapt_skills``, ``evaluate_combo`` and ``run_pass``.

    Each call is recorded as a tuple, and each adaptation leaves the files the real one leaves for the next step to
    find: the records and the markers of a finished adaptation, and the summary. A stand-in checks that its caller
    holds the run directory.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.settings_seen: list[Sequence[int] | None] = []
        self.ports_checked: list[list[int]] = []
        self.checked: list[tuple[str, str, str]] = []  # (what, model, task) of the checks made before a combination
        self.preflights: list[str] = []  # the evolver models asked, one entry per request
        self.skills_preflight: list[bool] = []  # what each skill evolution was told about asking the evolver itself
        self._made: Counter[str] = Counter()  # the calls of each kind so far, whatever was cleared from ``calls``
        self._errors: dict[tuple[str, int], BaseException] = {}

    def raise_on(self, kind: str, number: int, error: BaseException) -> None:
        """Raise ``error``, once, instead of doing the ``number``-th (from 1) call of ``kind`` (adapt-rl, adapt-skills,
        evaluate or pass), counted over the whole life of this object."""
        self._errors[(kind, number)] = error

    def _enter(self, kind: str, run: RunDir, paths: Any, *details: Any) -> None:
        self.calls.append((kind, *details))
        self._made[kind] += 1
        assert run.holds_lock(), f"{kind} was called without the run directory {run.root} locked"
        assert paths.holds_lock(), f"{kind} was called without the runtime directory and the agent's home locked"
        error = self._errors.pop((kind, self._made[kind]), None)
        if error is not None:
            raise error

    def adapt_rl(
        self, model: ModelSpec, task: Task, run: RunDir, cfgs: Configs, secrets: Secrets, *, paths: Any = None
    ) -> dict[str, str]:
        self._enter("adapt-rl", run, paths, model.name, task.name)
        assert secrets.tinker_api_key
        ids = sorted(task.adaptation)
        for stage, items in ((1, ids[:3]), (2, ids[3:6])):
            directory = run.adapt_dir / f"stage{stage}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "episodes.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in items), "utf-8")
        results = {"update_32": f"tinker://{model.name}/{task.name}/32", "update_64": f"tinker://{model.name}/64"}
        write_json(
            run.adapt_dir / "stage2" / "checkpoint.json",
            {"absolute_step": 64, "verified_loadable": True, "final": results["update_64"]},
        )
        write_json(run.adapt_dir / "checkpoints.json", {"base_model": model.tinker_id, **results})
        return results

    def adapt_skills(
        self,
        model: ModelSpec,
        task: Task,
        run: RunDir,
        cfgs: Configs,
        secrets: Secrets,
        *,
        paths: Any = None,
        preflight: bool = True,
    ) -> dict[str, Any]:
        self._enter("adapt-skills", run, paths, model.name, task.name)
        self.skills_preflight.append(preflight)
        if preflight:  # as the real one does
            self.evolver_preflight(secrets, cfgs.skills, paths=paths)
        assert secrets.tinker_api_key and secrets.evolver_model and secrets.evolver_api_key
        library = run.adapt_dir / "library"
        library.mkdir(parents=True, exist_ok=True)
        (run.adapt_dir / "episodes.jsonl").write_text(
            "".join(json.dumps({"id": i}) + "\n" for i in sorted(task.adaptation)[:12]), encoding="utf-8"
        )
        sha = lib_sha(library)
        write_json(run.adapt_dir / "library.json", {"library": "library", "sha": sha, "evolutions": 12, "skills": 0})
        return {"library": library, "sha": sha, "evolutions": 12, "skills": 0}

    def evaluate_combo(
        self,
        run: RunDir,
        model: ModelSpec,
        task: Task,
        mechanism: str,
        cfgs: Configs,
        secrets: Secrets,
        *,
        paths: Any = None,
        settings: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        self._enter("evaluate", run, paths, model.name, task.name, mechanism)
        self.settings_seen.append(None if settings is None else list(settings))
        summary = {"model": model.name, "task": task.name, "mechanism": mechanism}
        write_json(run.summary_path, summary)
        return summary

    def run_pass(
        self,
        run: RunDir,
        variant: str,
        setting: int,
        k: int,
        agent: Any,
        model: ModelSpec,
        task: Task,
        cfgs: Configs,
        secrets: Secrets,
        *,
        paths: Any = None,
        items: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        assert items is not None and secrets.tinker_api_key
        self._enter("pass", run, paths, model.name, task.name, variant, setting, k, type(agent).__name__, tuple(items))
        hits = 1 if variant == "base" else 2
        record = {"hits": hits, "n": len(items), "rate": round(hits / len(items), 4)}
        return {"pass": k, "valid": True, "asr": record, "asr_encountered": dict(record), "encounter_rate": 1.0}

    def check_setup(self, model: ModelSpec, task: Task, cfgs: Configs, secrets: Secrets) -> None:
        self.checked.append(("rl", model.name, task.name))

    def check_inputs(self, model: ModelSpec, cfgs: Configs, secrets: Secrets, paths: Any, order: list[str]) -> None:
        self.checked.append(("skills", model.name, ""))

    def evolver_preflight(self, secrets: Secrets, skills: Any, *, paths: Any) -> None:
        self.preflights.append(secrets.evolver_model or "")

    def install(self, monkeypatch: Any) -> Work:
        """Replace the paid work in ``runner``, and the checks of the machine that a test cannot make: the OpenClaw
        version, the patched frameworks, the ports (which are recorded in ``ports_checked``), what each adaptation
        checks before it starts (``checked``) and the request to the evolver (``preflights``)."""
        monkeypatch.setattr(runner, "adapt_rl", self.adapt_rl)
        monkeypatch.setattr(runner, "adapt_skills", self.adapt_skills)
        monkeypatch.setattr(runner, "evaluate_combo", self.evaluate_combo)
        monkeypatch.setattr(runner, "run_pass", self.run_pass)
        monkeypatch.setattr(runner, "check_setup", self.check_setup)
        monkeypatch.setattr(runner, "check_inputs", self.check_inputs)
        monkeypatch.setattr(runner, "evolver_preflight", self.evolver_preflight)
        monkeypatch.setattr(runner, "check_openclaw", lambda: OPENCLAW_VERSION)
        monkeypatch.setattr(runner, "verify_patches", lambda: dict(PATCHED))
        monkeypatch.setattr(runner, "require_free", lambda ports: self.ports_checked.append(sorted(ports)))
        return self


def install_world(monkeypatch: Any, tmp_path: Path) -> Work:
    """A runtime in ``tmp_path`` with every secret set and the paid work replaced."""
    runtime_env(monkeypatch, tmp_path, make_executable(tmp_path / "metaclaw" / "bin" / "python"))
    set_secrets(monkeypatch)
    return Work().install(monkeypatch)


# --- the programs the integration tests need --------------------------------------------------------------------------


def openclaw_problem() -> str | None:
    """Why OpenClaw 2026.3.7 cannot be run here, or None."""
    with tempfile.TemporaryDirectory() as probe:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": probe,
            "LANG": "C.UTF-8",
            "OPENCLAW_STATE_DIR": str(Path(probe) / "state"),
            "OPENCLAW_CONFIG_PATH": str(Path(probe) / "openclaw.json"),
        }
        try:
            version = installed_version(env)
        except OpenClawError as error:
            return f"OpenClaw cannot be run here: {error}"
    if version != OPENCLAW_VERSION:
        return f"these tests need OpenClaw {OPENCLAW_VERSION}; {version} is installed"
    return None


def metaclaw_problem(python: Path) -> str | None:
    """Why the MetaClaw environment cannot be used here, or None."""
    if not (python.is_file() and os.access(python, os.X_OK)):
        return f"no MetaClaw Python at {python}; set EVOLVE_IPI_METACLAW_PYTHON to run this test"
    with tempfile.TemporaryDirectory() as home:
        done = subprocess.run(
            [str(python), "-c", "import metaclaw.api_server"],
            capture_output=True,
            text=True,
            timeout=300,
            env={"PATH": os.environ.get("PATH", ""), "HOME": home, "PYTHONSAFEPATH": "1"},
            check=False,
        )
    return None if done.returncode == 0 else f"{python} cannot import metaclaw: {done.stderr.strip()[-300:]}"
