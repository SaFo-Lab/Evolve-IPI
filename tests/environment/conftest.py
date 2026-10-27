"""Fixtures for the environment tests: the service tasks and a workspace with the services installed."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from evolve_ipi.environment import state
from evolve_ipi.tasks import TASKS, Task, load_task

SYSTEM_PYTHON = Path("/usr/bin/python3")

# What the agent's exec tool offers: a bare PATH, so that `env python3` finds the system interpreter.
AGENT_ENV = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/")}


class SvcEnv:
    """A workspace whose ``svc/`` holds the installed programs, with the state directory next to it."""

    def __init__(self, workspace: Path, state_dir: Path) -> None:
        self.workspace = workspace
        self.state = state_dir

    def run(self, program: str, *args: object, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        """Run ``svc/<program>`` the way the agent does: from the workspace root, through its shebang."""
        return subprocess.run(
            [str(self.workspace / "svc" / program), *map(str, args)],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            env=AGENT_ENV if env is None else env,
        )

    def write(self, name: str, obj: Any) -> None:
        (self.state / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")

    def read(self, name: str) -> Any:
        return json.loads((self.state / f"{name}.json").read_text(encoding="utf-8"))

    def access_log(self) -> list[dict[str, Any]]:
        path = self.state / "access.log"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def svc_env(tmp_path: Path) -> SvcEnv:
    if not SYSTEM_PYTHON.exists():
        pytest.skip("the programs are run with the system interpreter, /usr/bin/python3, which is missing")
    workspace, state_dir = tmp_path / "ws", tmp_path / "state"
    workspace.mkdir()
    state.install(workspace, state_dir)
    return SvcEnv(workspace, state_dir)


@pytest.fixture(scope="session")
def service_tasks() -> dict[str, Task]:
    """The four tasks whose world is the fake services, by name."""
    return {name: load_task(name) for name in TASKS if name != "file-create"}


@pytest.fixture(scope="session")
def file_task() -> Task:
    return load_task("file-create")
