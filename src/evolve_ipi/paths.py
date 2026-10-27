"""Where a run keeps its files: runtime directories, the agent's home and the output tree.

Only paths are computed here; creating the directories is left to the code that uses them.
"""

from __future__ import annotations

import errno
import fcntl
import os
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args

from evolve_ipi.config import ConfigError
from evolve_ipi.records import RunDirError, RunLocked

Line = Literal["rl", "skills"]
Kind = Literal["file", "service"]

# The agent each line runs as in OpenClaw. It names the agent's workspace directory, and OpenClaw's own state for it.
AGENT_IDS: Mapping[Line, str] = MappingProxyType({"rl": "rl", "skills": "mc"})


# What a file system answers when it cannot lock files at all (some network and container volumes); some network file
# systems answer a lock on a directory with EINVAL or EISDIR.
_NO_LOCKS = frozenset({errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL, errno.EISDIR})
_HELD: dict[str, int] = {}  # the directories this interpreter holds (real paths) -> the pid that took them


def _env_path(variable: str, default: str) -> Path:
    """The path ``variable`` names, or ``default`` when it is unset. Set but empty is refused: it is a mistake, and
    taking the default for it would quietly put the run's files somewhere else."""
    value = os.environ.get(variable)
    if value is not None and not value.strip():
        raise ConfigError(f"{variable} is set but empty; unset it to use the default {default}, or name a path")
    # absolute(), not resolve(): resolving a virtualenv's python symlink would leave the virtualenv.
    return Path(default if value is None else value).expanduser().absolute()


def _check(line: str, kind: str) -> None:
    if line not in get_args(Line):
        raise ValueError(f"unknown line {line!r}; expected one of {', '.join(get_args(Line))}")
    if kind not in get_args(Kind):
        raise ValueError(f"unknown kind {kind!r}; expected one of {', '.join(get_args(Kind))}")


def agent_id(line: Line) -> str:
    """The OpenClaw agent of ``line``: ``rl`` for the RL line, ``mc`` for the skill line."""
    if line not in AGENT_IDS:
        raise ValueError(f"unknown line {line!r}; expected one of {', '.join(AGENT_IDS)}")
    return AGENT_IDS[line]


def _refuse_git_checkout(workspace: Path) -> None:
    """OpenClaw prints the git root into the system prompt, so a workspace inside a checkout would
    change what the model sees."""
    for parent in workspace.resolve().parents:
        if (parent / ".git").exists():
            raise ConfigError(
                f"the agent workspace {workspace} lies inside the git checkout {parent}; "
                "set EVOLVE_IPI_RUNTIME to a directory outside every git checkout"
            )


@contextmanager
def _hold(directory: Path, what: str) -> Iterator[None]:
    """An exclusive ``flock`` on the directory itself, taken without waiting. Nothing is written into it: the agent can
    read its own home, and the runtime directory holds what the run must not leak to it."""
    directory.mkdir(parents=True, exist_ok=True)
    key = os.path.realpath(directory)
    descriptor = os.open(directory, os.O_RDONLY | os.O_CLOEXEC)  # close-on-exec: no child process inherits the lock
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RunLocked(
            f"{what} {directory} is in use by another run. Two runs at once must not share the runtime directory (the "
            "OpenClaw profiles, workspaces and service state) or the agent's home (where File-Create's target lies): "
            "give each run its own ports, its own EVOLVE_IPI_RUNTIME and its own HOME, or run each in its own "
            "container."
        ) from None
    except OSError as error:
        os.close(descriptor)
        if error.errno in _NO_LOCKS:
            raise RunDirError(
                f"{what} {directory}: the file system cannot lock files ({os.strerror(error.errno or 0)}). Without "
                "file locks two runs could share this directory and clobber each other."
            ) from error
        raise
    except BaseException:
        os.close(descriptor)
        raise
    _HELD[key] = os.getpid()
    try:
        yield
    finally:
        _HELD.pop(key, None)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True)
class RuntimePaths:
    """Everything a run writes besides its results.

    ``root`` holds one OpenClaw profile, workspace, service state and transcript archive per
    (line, kind). ``agent_home`` is the home of the agent, which is where the injected requests of the
    File-Create task point. The model server, the trainer and MetaClaw run with ``service_home`` as
    their HOME, so nothing they write lands next to the agent's own state.

    A workspace is ``<root>/workspaces/<line>-<kind>/workspace-<agent>``. The agent can search upward from its
    workspace, and the two directories above it hold nothing but workspaces: not the profile with the live
    transcripts, not the transcript archive, not the service state.
    """

    root: Path = field(default_factory=lambda: _env_path("EVOLVE_IPI_RUNTIME", "/var/lib/evolve-ipi"))
    agent_home: Path = field(default_factory=Path.home)
    outputs: Path = field(default_factory=lambda: _env_path("EVOLVE_IPI_OUTPUTS", "outputs"))
    metaclaw_python: Path = field(
        default_factory=lambda: _env_path(
            "EVOLVE_IPI_METACLAW_PYTHON", "/opt/evolve-ipi/venv-metaclaw/bin/python"
        )
    )

    def _slot(self, area: str, line: Line, kind: Kind) -> Path:
        _check(line, kind)
        return self.root / area / f"{line}-{kind}"

    def profile(self, line: Line, kind: Kind) -> Path:
        return self._slot("openclaw", line, kind)

    def workspace(self, line: Line, kind: Kind) -> Path:
        workspace = self._slot("workspaces", line, kind) / f"workspace-{agent_id(line)}"
        _refuse_git_checkout(workspace)
        return workspace

    def service_state(self, line: Line, kind: Kind) -> Path:
        return self._slot("services", line, kind)

    def transcript_archive(self, line: Line, kind: Kind) -> Path:
        return self._slot("transcripts", line, kind)

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold the runtime directory and the agent's home while the block runs, so that no second run uses either.

        Every run of a line and a kind shares ``<root>/{openclaw,workspaces,services,transcripts}/<line>-<kind>`` and
        the agent's ``~/.openclaw-rl`` (or ``~/.metaclaw``), and rebuilds them before it starts; two runs at once, even
        on other ports, would clobber each other. Raises ``RunLocked`` naming the directory another run holds.
        """
        with ExitStack() as stack:
            stack.enter_context(_hold(self.root, "the runtime directory"))
            if os.path.realpath(self.agent_home) != os.path.realpath(self.root):
                stack.enter_context(_hold(self.agent_home, "the agent's home"))
            yield

    def holds_lock(self) -> bool:
        """Whether this process holds the runtime directory and the agent's home (``lock``)."""
        pid = os.getpid()
        return all(_HELD.get(os.path.realpath(path)) == pid for path in (self.root, self.agent_home))

    @property
    def service_home(self) -> Path:
        return self.root / "service-home"

    @property
    def hf_home(self) -> Path:
        return self.root / "hf"
