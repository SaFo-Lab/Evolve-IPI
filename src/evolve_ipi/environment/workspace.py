"""The agent's workspace: built, snapshotted, put back between episodes, and checked for leftovers.

OpenClaw pastes six files of the workspace into every system prompt, so a run starts from exactly the files its
framework version writes into a new workspace. They are packaged in ``workspace_files/`` and pinned by hash: a workspace
built from different files would be a different environment.

* ``build`` writes the prompt files, ``TOOLS.md`` (with the services section when the task has services) and, for
  File-Create, the documents; for the service tasks it installs the programs too.
* ``snapshot`` records every file; ``restore`` puts the workspace back to it before each episode, so that nothing the
  agent wrote in one episode is read in the next, and reports what it had to change.
* ``guard`` refuses a workspace that carries anything an earlier agent left: notes, an edited prompt file, skills,
  commits in the workspace repository, a changed ``TOOLS.md``, an edited or no longer executable service program.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from evolve_ipi.environment import state
from evolve_ipi.paths import Kind

__all__ = [
    "FACTORY_PROMPT_SHA256",
    "FACTORY_TOOLS_SHA256",
    "FILES_DIR",
    "PROMPT_FILES",
    "Snapshot",
    "TOOLS_SECTION_SHA256",
    "WorkspaceError",
    "build",
    "guard",
    "restore",
    "snapshot",
]

FILES_DIR = Path(__file__).resolve().parent / "workspace_files"

# The files OpenClaw pastes verbatim into every system prompt, and the sha256 each must have. They are OpenClaw
# 2026.3.7's own templates minus the one leading blank line it strips. An upgrade that changes a template changes the
# environment, so the pins are re-set by hand, never by the code that checks them.
PROMPT_FILES = ("AGENTS.md", "SOUL.md", "IDENTITY.md", "USER.md", "HEARTBEAT.md", "BOOTSTRAP.md")
FACTORY_PROMPT_SHA256 = {
    "AGENTS.md": "3ecae6a107899f284728bf1aa406c75d141773f9f380ba39ba378b5e9fa28f4c",
    "SOUL.md": "95c467f710183b2c0397c29ec4ef886bb723671a39ee767dc4d59165f723b68f",
    "IDENTITY.md": "1379f924cf4b4d6d0a306ecc2544a913dab706a1325fcf8eb9b267686ee8551c",
    "USER.md": "178a5807bd2e6ace25b80a366f0f7d8d3f354a47f577c4ab0092397456b8738f",
    "HEARTBEAT.md": "dac422286075ba178e9bae58154cc0ae8749a66854e438c9c58e8eb91f22c126",
    "BOOTSTRAP.md": "5c4cb81099167dc7901bf16d7d4b8fd4bdbb37d2ec57437952f9409f19b88e39",
}
# The TOOLS.md OpenClaw writes into a new workspace, and the section the service tasks append after a newline.
FACTORY_TOOLS_SHA256 = "78f3e26b8625ea283c615cec291d4978da5f9f01df730628c5c42094418b8dc2"
TOOLS_SECTION_SHA256 = "d35b8bc970c5432f093daa231f030353823de6aa186cc12bbb01e5396b2096e2"

# <workspace>/skills/*/SKILL.md and <workspace>/.agents/skills/*/SKILL.md are loaded into the system prompt as
# available skills. Nothing here writes one, so any that exists was written by the agent.
_SKILL_DIRS = ("skills", ".agents/skills")
# The framework's own bookkeeping (.git, .openclaw) is not touched, and running the programs creates __pycache__.
_SKIP = (".git", ".openclaw", "__pycache__")
_RESERVED = {*PROMPT_FILES, "TOOLS.md", "svc", *_SKIP}


class WorkspaceError(Exception):
    """The workspace is not what the run needs. ``problems`` lists each thing found wrong, when there are several."""

    def __init__(self, message: str, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.problems: list[str] = list(problems or [])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _packaged(name: str, expected: str) -> bytes:
    """A packaged file, refused when it is not the pinned one."""
    data = (FILES_DIR / name).read_bytes()
    if _sha256(data) != expected:
        raise WorkspaceError(
            f"the packaged {name} (sha256 {_sha256(data)[:12]}) is not the pinned file (sha256 {expected[:12]}); "
            "a workspace built from it would be a different environment"
        )
    return data


# --- build -----------------------------------------------------------------------------------


def _check_document_path(rel: str) -> None:
    path = PurePosixPath(rel)
    problem = None
    if not rel or str(path) != rel or path.is_absolute() or not path.parts:
        problem = "it is not a normalised relative path"
    elif ".." in path.parts:
        problem = "it leaves the workspace"
    elif path.parts[0] in _RESERVED or any(part in _SKIP for part in path.parts):
        problem = "it lies where the workspace keeps its own files"
    if problem:
        raise WorkspaceError(f"document path {rel!r} is not allowed: {problem}")


def build(ws: Path, kind: Kind, docs: dict[str, str] | None, state_dir: Path | None) -> None:
    """Build a fresh workspace at ``ws``, which must not exist yet or be empty.

    ``kind`` "file" is File-Create's: the prompt files, the factory ``TOOLS.md`` and ``docs`` (path inside the
    workspace -> text, written as given), with no services. ``kind`` "service" is every other task's: the prompt
    files, ``TOOLS.md`` with the services section, and the installed programs pointing at ``state_dir``, which must lie
    outside ``ws``.
    """
    ws = Path(ws)
    if kind == "file":
        if not docs:
            raise ValueError("a file workspace needs the documents of its task")
        if state_dir is not None:
            raise ValueError("a file workspace has no services, so it has no state directory")
    elif kind == "service":
        if docs is not None:
            raise ValueError("a service workspace has no documents")
        if state_dir is None:
            raise ValueError("a service workspace needs a state directory")
    else:
        raise ValueError(f"unknown kind {kind!r}; expected 'file' or 'service'")
    for rel in docs or {}:
        _check_document_path(rel)
        for parent in PurePosixPath(rel).parents:
            if str(parent) in (docs or {}):
                raise WorkspaceError(
                    f"document path {rel!r} is not allowed: {str(parent)!r} is a document, not a directory"
                )
    prompts = {name: _packaged(name, FACTORY_PROMPT_SHA256[name]) for name in PROMPT_FILES}
    tools = _packaged("TOOLS.factory.md", FACTORY_TOOLS_SHA256)
    if kind == "service":
        tools += b"\n" + _packaged("tools_section.md", TOOLS_SECTION_SHA256)
    if ws.exists() and not ws.is_dir():
        raise WorkspaceError(f"workspace {ws} is not a directory")
    if ws.is_dir() and any(ws.iterdir()):
        raise WorkspaceError(f"workspace {ws} is not empty; a workspace is built fresh, so remove it first")
    ws.mkdir(parents=True, exist_ok=True)
    if state_dir is not None:
        state.install(ws, state_dir)
    for name, data in prompts.items():
        (ws / name).write_bytes(data)
    (ws / "TOOLS.md").write_bytes(tools)
    for rel, text in (docs or {}).items():
        path = ws / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))


# --- snapshot and restore --------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """Every file of the workspace as the run found it: its bytes, its permission bits and its sha256.

    ``pinned`` names the files that must never change, ``TOOLS.md`` and the installed programs: a change there is
    corruption of the measured environment, not the agent keeping notes.
    """

    contents: dict[str, bytes]
    modes: dict[str, int]
    sha256: dict[str, str]
    pinned: tuple[str, ...]


def _files(ws: Path) -> list[str]:
    """Every file below ``ws``, relative to it, except the framework's own bookkeeping."""
    found = []
    for path in ws.rglob("*"):
        if path.is_file():
            rel = path.relative_to(ws)
            if not any(part in _SKIP for part in rel.parts):
                found.append(str(rel))
    return found


def _hashes(ws: Path) -> dict[str, str]:
    return {rel: _sha256((ws / rel).read_bytes()) for rel in _files(ws)}


def snapshot(ws: Path) -> Snapshot:
    """Record the workspace as it is now.

    Raises ``WorkspaceError`` unless ``ws`` is a built workspace: a directory that holds ``TOOLS.md`` and, if it has
    an ``svc/`` directory, all eight service files. A snapshot of anything less would pin nothing, and restoring it
    would empty a populated workspace.
    """
    ws = Path(ws)
    if not ws.is_dir():
        raise WorkspaceError(f"workspace {ws} does not exist")
    if not (ws / "TOOLS.md").is_file():
        raise WorkspaceError(f"workspace {ws} has no TOOLS.md, so it is not a built workspace")
    services = (ws / "svc").exists()
    if services:
        missing = [name for name in state.SERVICE_FILES if not (ws / "svc" / name).is_file()]
        if missing:
            raise WorkspaceError(
                f"the installed services of workspace {ws} are incomplete: "
                + "; ".join(f"svc/{name} is missing" for name in missing)
            )
    contents = {rel: (ws / rel).read_bytes() for rel in _files(ws)}
    modes = {rel: (ws / rel).stat().st_mode & 0o7777 for rel in contents}
    hashes = {rel: _sha256(data) for rel, data in contents.items()}
    pinned = ("TOOLS.md", *(f"svc/{name}" for name in state.SERVICE_FILES if services))
    return Snapshot(contents, modes, hashes, pinned)


def _put_back(ws: Path, snap: Snapshot, rel: str) -> None:
    path = ws / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        path.unlink()  # replace the link: writing through it would change what it points to
    path.write_bytes(snap.contents[rel])
    path.chmod(snap.modes[rel])


def _conflicts(ws: Path, snap: Snapshot) -> list[str]:
    """What stops the snapshot from being written back: a directory on the path of a file, and a link on the path of
    a directory (the write would go through it). A plain file on a directory's path is no conflict: it is removed
    first, as litter."""
    found: list[str] = []
    links: set[str] = set()
    for rel in sorted(snap.sha256):
        if (ws / rel).is_dir():
            found.append(f"{rel} is a directory, but the snapshot has a file there")
            continue
        for parent in PurePosixPath(rel).parents:
            name = str(parent)
            if name != "." and (ws / name).is_symlink():
                if name not in links:
                    links.add(name)
                    found.append(f"{name} is a link, but the snapshot has a directory there")
                break
    return found


def restore(ws: Path, snap: Snapshot) -> list[str]:
    """Put the workspace back to ``snap`` and report every change: ``added:``, ``deleted:`` or ``edited:`` a path,
    ``added:<dir>/`` for an empty directory that was removed, and ``added:<link>`` for a link to a directory, which is
    removed itself (what it points to is left alone).

    Raises ``WorkspaceError``, having changed nothing, if a pinned file (``TOOLS.md``, a service program) differs, or if
    something lies where the snapshot cannot be written back (a directory on a file's path, a link on a directory's).
    """
    ws = Path(ws)
    current = _hashes(ws)
    changed_pinned = [rel for rel in snap.pinned if current.get(rel) != snap.sha256[rel]]
    if changed_pinned:
        raise WorkspaceError(f"the installed services or TOOLS.md changed during the run: {', '.join(changed_pinned)}")
    conflicts = _conflicts(ws, snap)
    if conflicts:
        raise WorkspaceError(
            f"the workspace cannot be put back, and nothing was changed: {'; '.join(conflicts)}", conflicts
        )
    changed = []
    for rel in sorted(set(current) - set(snap.sha256)):  # the agent added it
        (ws / rel).unlink()
        changed.append("added:" + rel)
    for rel in sorted(set(snap.sha256) - set(current)):  # the agent deleted it
        _put_back(ws, snap, rel)
        changed.append("deleted:" + rel)
    for rel in sorted(set(current) & set(snap.sha256)):  # the agent edited it
        if current[rel] != snap.sha256[rel]:
            _put_back(ws, snap, rel)
            changed.append("edited:" + rel)
    for directory in sorted((p for p in ws.rglob("*") if p.is_dir()), key=lambda p: (-len(p.parts), str(p))):
        rel_dir = directory.relative_to(ws)
        if any(part in _SKIP for part in rel_dir.parts):
            continue
        if directory.is_symlink():  # an agent's link: nothing of the snapshot lies at or below it (_conflicts)
            directory.unlink()
            changed.append("added:" + str(rel_dir))
            continue
        if not any(directory.iterdir()):
            directory.rmdir()
            changed.append("added:" + str(rel_dir) + "/")
    return changed


# --- guard -----------------------------------------------------------------------------------


def _prompt_file_problems(ws: Path) -> list[str]:
    """A prompt file that is missing or changed, and skills the agent wrote.

    The six files are pasted into every system prompt, so one line added to any of them is read by every later
    episode without the agent calling a tool; that is why they are checked by hash.
    """
    problems = []
    for name in PROMPT_FILES:
        path = ws / name
        if not path.is_file():
            problems.append(f"{name} is missing; OpenClaw pastes it into every system prompt")
            continue
        got, want = _sha256(path.read_bytes()), FACTORY_PROMPT_SHA256[name]
        if got != want:
            problems.append(
                f"{name} is not the factory file (sha256 {got[:12]}, expected {want[:12]}); it is pasted into every "
                "system prompt, so whatever was added to it would be read by every episode"
            )
    for rel in _SKILL_DIRS:
        directory = ws / rel
        if directory.is_dir():
            found = sorted(str(p.relative_to(ws)) for p in directory.rglob("SKILL.md"))
            if found:
                problems.append(
                    f"the workspace carries agent-written skills {found}; their names and descriptions are pasted into "
                    "every system prompt under a mandatory-scan instruction"
                )
    return problems


def _memory_problems(ws: Path) -> list[str]:
    """Notes an earlier agent left: files under ``memory/`` and a ``MEMORY.md`` (OpenClaw also reads ``memory.md``).

    AGENTS.md tells the agent to read the notes at startup and ``MEMORY.md`` is pasted into the system prompt, so a
    note written in one run is read by the next.
    """
    found = []
    memory = ws / "memory"
    if memory.is_dir():
        found += sorted(str(p.relative_to(ws)) for p in memory.rglob("*") if p.is_file())
    found += sorted(p.name for p in ws.iterdir() if p.is_file() and p.name.lower() == "memory.md")
    if not found:
        return []
    return [
        f"the workspace carries agent-written memory {found}; an agent reads it, so episodes would not be independent"
    ]


def _tools_problems(ws: Path) -> list[str]:
    path = ws / "TOOLS.md"
    if not path.is_file():
        return ["TOOLS.md is missing"]
    data = path.read_bytes()
    suffix = b"\n" + _packaged("tools_section.md", TOOLS_SECTION_SHA256)
    body = data[: -len(suffix)] if data.endswith(suffix) else data
    if _sha256(body) != FACTORY_TOOLS_SHA256:
        return ["TOOLS.md is neither the factory file nor the factory file followed by the services section"]
    return []


def _service_problems(ws: Path) -> list[str]:
    """The installed service programs, when the workspace has an ``svc/``: each file is the pinned one and has the
    permission bits ``install`` gives it. An edited program changes what the judges see, and a program that lost its
    execute bit fails every call of the agent."""
    svc = ws / "svc"
    if not svc.exists():
        return []
    problems = []
    for name in state.SERVICE_FILES:
        path = svc / name
        if not path.is_file():
            problems.append(f"svc/{name} is missing")
            continue
        if _sha256(path.read_bytes()) != state.PROGRAM_SHA256[name]:
            problems.append(f"svc/{name} is not the pinned file")
        mode, want = stat.S_IMODE(path.stat().st_mode), state.installed_mode(name)
        if mode != want:
            problems.append(f"svc/{name} has mode {mode:04o}, but install gives it {want:04o}")
    return problems


# The agent can write the workspace's .git/config, so a command planted there (core.fsmonitor, a clean filter) would run
# with whatever environment git is started with. Git gets only what it needs to start, no credentials, and no
# configuration of the user or the system.
_GIT_ENV_KEPT = ("PATH", "LANG", "LC_ALL")


def _git_env() -> dict[str, str]:
    env = {name: os.environ[name] for name in _GIT_ENV_KEPT if name in os.environ}
    env.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
    return env


def _git(ws: Path, *args: str) -> str | None:
    """The output of ``git <args>`` in ``ws``, or None when git fails; it takes no optional lock, so it writes nothing
    to the repository. What the repository's own configuration could make it run is switched off: the file system
    monitor (``core.fsmonitor``) and the hooks; and whatever else it does run (a filter, say) gets the scrubbed
    environment of ``_git_env``, without credentials, and no configuration of the user or the system."""
    command = ["git", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}", *args]
    result = subprocess.run(command, cwd=ws, capture_output=True, text=True, env=_git_env(), timeout=120)
    return result.stdout.strip() if result.returncode == 0 else None


def _git_problems(ws: Path) -> list[str]:
    """Commits or tracked changes in the workspace repository.

    OpenClaw runs ``git init`` on a new workspace and never commits, so every commit was made by the agent, and
    ``git show HEAD:<path>`` reads back a note the restore removed. The repository itself is left untouched. A git
    that fails, or is missing, proves nothing, so it is a problem too.
    """
    if not (ws / ".git").exists():
        return []
    try:
        log = _git(ws, "log", "--oneline", "--all")
        if log is None:
            return [".git exists but `git log` failed; cannot prove the history is empty"]
        status = _git(ws, "status", "--porcelain")
    except FileNotFoundError:
        return ["the workspace has a .git directory but git is not installed, so its history cannot be checked"]
    except subprocess.TimeoutExpired:
        return ["git did not answer in time, so the history of the workspace repository cannot be checked"]
    problems = []
    if log:
        lines = log.splitlines()
        problems.append(
            f"{len(lines)} commit(s) in the workspace repository, none of which a factory workspace has "
            f"(OpenClaw runs `git init` and never commits): {'; '.join(lines[:5]) + (' ...' if len(lines) > 5 else '')}"
        )
    if status is None:
        problems.append(
            ".git exists but `git status` failed; cannot prove that no tracked file differs from the history"
        )
    else:
        # `??` (untracked) is the normal state of a factory workspace: nothing is tracked, so every file is untracked.
        tracked = [line for line in status.splitlines() if not line.startswith("??")]
        if tracked:
            problems.append(f"tracked files differ from the workspace repository's history: {'; '.join(tracked[:5])}")
    return problems


def guard(ws: Path) -> None:
    """Raise ``WorkspaceError`` listing everything an earlier agent left in the workspace, or that is not factory."""
    ws = Path(ws)
    if not ws.is_dir():
        raise WorkspaceError(f"workspace {ws} does not exist")
    problems = [
        *_prompt_file_problems(ws),
        *_memory_problems(ws),
        *_tools_problems(ws),
        *_service_problems(ws),
        *_git_problems(ws),
    ]
    if problems:
        raise WorkspaceError(f"the workspace {ws} is not clean:\n  - " + "\n  - ".join(problems), problems)
