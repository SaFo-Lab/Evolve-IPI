"""Where a run keeps its files: ``RuntimePaths`` and the ``RunDir`` result layout."""

from __future__ import annotations

import errno
import os
from itertools import product
from pathlib import Path

import pytest

from evolve_ipi import paths as paths_module
from evolve_ipi.config import ConfigError
from evolve_ipi.paths import AGENT_IDS, Kind, Line, RuntimePaths, agent_id
from evolve_ipi.records import RunDir, RunDirError, RunLocked

LINES = ("rl", "skills")
KINDS = ("file", "service")
RUNTIME_ENV = ("EVOLVE_IPI_RUNTIME", "EVOLVE_IPI_OUTPUTS", "EVOLVE_IPI_METACLAW_PYTHON")


@pytest.fixture(autouse=True)
def _clean_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)


def paths_in(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        root=tmp_path / "runtime",
        agent_home=tmp_path / "home",
        outputs=tmp_path / "outputs",
        metaclaw_python=tmp_path / "venv" / "bin" / "python",
    )


# --- defaults and environment ----------------------------------------------------------------


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    paths = RuntimePaths()
    assert paths.root == Path("/var/lib/evolve-ipi")
    assert paths.agent_home == Path.home()
    assert paths.outputs == Path.cwd() / "outputs"
    assert paths.metaclaw_python == Path("/opt/evolve-ipi/venv-metaclaw/bin/python")


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVOLVE_IPI_RUNTIME", str(tmp_path / "rt"))
    monkeypatch.setenv("EVOLVE_IPI_OUTPUTS", "results")
    monkeypatch.setenv("EVOLVE_IPI_METACLAW_PYTHON", ".venv-metaclaw/bin/python")
    paths = RuntimePaths()
    assert paths.root == tmp_path / "rt"
    assert paths.outputs == tmp_path / "results"
    assert paths.metaclaw_python == tmp_path / ".venv-metaclaw" / "bin" / "python"


@pytest.mark.parametrize("variable", RUNTIME_ENV)
@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_environment_value_is_refused_and_named(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    """Set but empty is a mistake, not a request for the default: OUTPUTS would silently write to ./outputs."""
    monkeypatch.setenv(variable, value)
    with pytest.raises(ConfigError, match=rf"^{variable} is set but empty"):
        RuntimePaths()
    monkeypatch.delenv(variable)
    RuntimePaths()  # unset, it takes the default


def test_agent_home_follows_the_home_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert RuntimePaths().agent_home == tmp_path


def test_a_virtualenv_python_is_not_resolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Resolving the symlink would leave the virtualenv and lose its packages."""
    real = tmp_path / "base" / "python3.12"
    real.parent.mkdir()
    real.write_text("")
    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(real)
    monkeypatch.setenv("EVOLVE_IPI_METACLAW_PYTHON", str(link))
    assert RuntimePaths().metaclaw_python == link


def test_runtime_paths_are_frozen(tmp_path: Path) -> None:
    with pytest.raises(AttributeError):
        paths_in(tmp_path).root = tmp_path  # type: ignore[misc]


# --- layout ----------------------------------------------------------------------------------


def test_layout_stays_inside_the_runtime_root_and_never_overlaps(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    seen: dict[Path, str] = {}
    for line, kind in product(LINES, KINDS):
        for what, path in {
            "profile": paths.profile(line, kind),
            "workspace": paths.workspace(line, kind),
            "service_state": paths.service_state(line, kind),
            "transcript_archive": paths.transcript_archive(line, kind),
        }.items():
            assert path.is_relative_to(paths.root), (what, line, kind)
            assert path not in seen, f"{what}({line}, {kind}) collides with {seen[path]}"
            seen[path] = f"{what}({line}, {kind})"
    for what, path in {"service_home": paths.service_home, "hf_home": paths.hf_home}.items():
        assert path.is_relative_to(paths.root)
        assert path not in seen
        seen[path] = what


def test_directories_that_must_stay_apart(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    for line, kind in product(LINES, KINDS):
        profile = paths.profile(line, kind)
        workspace = paths.workspace(line, kind)
        archive = paths.transcript_archive(line, kind)
        state = paths.service_state(line, kind)
        assert not workspace.is_relative_to(paths.root / "openclaw")
        assert not archive.is_relative_to(profile)
        assert not state.is_relative_to(workspace)
        assert not workspace.is_relative_to(state)


@pytest.mark.parametrize(
    ("line", "kind", "agent"),
    [("rl", "file", "rl"), ("rl", "service", "rl"), ("skills", "file", "mc"), ("skills", "service", "mc")],
)
def test_a_workspace_is_named_after_the_agent_of_its_line(tmp_path: Path, line: Line, kind: Kind, agent: str) -> None:
    paths = paths_in(tmp_path)
    assert agent_id(line) == agent
    expected = tmp_path / "runtime" / "workspaces" / f"{line}-{kind}" / f"workspace-{agent}"
    assert paths.workspace(line, kind) == expected


def test_there_is_one_agent_per_line(tmp_path: Path) -> None:
    assert dict(AGENT_IDS) == {"rl": "rl", "skills": "mc"}
    for bad in ("weights", "", "../x"):
        with pytest.raises(ValueError, match="line"):
            agent_id(bad)  # type: ignore[arg-type]


def test_the_two_directories_above_a_workspace_hold_nothing_but_workspaces(tmp_path: Path) -> None:
    """The agent can run ``grep -R`` on ``..`` and ``../..``. Neither may lead to a live transcript, the transcript
    archive, a profile or the service state of any run, whichever ones exist."""
    paths = paths_in(tmp_path)
    combos = list(product(LINES, KINDS))
    others = [
        *(paths.profile(line, kind) for line, kind in combos),
        *(paths.service_state(line, kind) for line, kind in combos),
        *(paths.transcript_archive(line, kind) for line, kind in combos),
        paths.service_home,
        paths.hf_home,
    ]
    for directory in (*others, *(paths.workspace(line, kind) for line, kind in combos)):
        directory.mkdir(parents=True)
    slots = sorted(f"{line}-{kind}" for line, kind in combos)
    for line, kind in combos:
        workspace = paths.workspace(line, kind)
        above, two_above = workspace.parent, workspace.parent.parent
        assert [entry.name for entry in above.iterdir()] == [workspace.name]  # `..`: this workspace and nothing else
        assert sorted(entry.name for entry in two_above.iterdir()) == slots  # `../..`: the slots of the workspaces
        for slot in two_above.iterdir():
            assert [entry.name for entry in slot.iterdir()] in (["workspace-rl"], ["workspace-mc"])
        for other in others:
            assert not other.is_relative_to(two_above), f"{other} lies below {two_above}"
            assert not two_above.is_relative_to(other), f"{two_above} lies below {other}"


def test_layout_is_deterministic_and_creates_nothing(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    assert paths.workspace("rl", "file") == paths.workspace("rl", "file")
    assert not paths.root.exists()


@pytest.mark.parametrize("method", ["profile", "workspace", "service_state", "transcript_archive"])
@pytest.mark.parametrize(
    ("line", "kind"), [("weights", "file"), ("rl", "files"), ("../x", "file"), ("rl", "")]
)
def test_unknown_line_or_kind_is_refused(tmp_path: Path, method: str, line: str, kind: str) -> None:
    with pytest.raises(ValueError, match=r"line|kind"):
        getattr(paths_in(tmp_path), method)(line, kind)


@pytest.mark.parametrize("git_entry", ["directory", "file"])
def test_workspace_refuses_a_git_checkout(tmp_path: Path, git_entry: str) -> None:
    """OpenClaw prints the git root into the system prompt, so a checkout around the workspace would
    change what the model sees."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    if git_entry == "directory":
        (checkout / ".git").mkdir()
    else:  # a worktree or submodule keeps a .git file
        (checkout / ".git").write_text("gitdir: elsewhere\n")
    paths = RuntimePaths(
        root=checkout / "run",
        agent_home=tmp_path / "home",
        outputs=tmp_path / "outputs",
        metaclaw_python=tmp_path / "python",
    )
    with pytest.raises(ConfigError) as err:
        paths.workspace("rl", "file")
    assert str(checkout) in str(err.value)
    assert "EVOLVE_IPI_RUNTIME" in str(err.value)
    # only the workspace has this constraint
    assert paths.profile("rl", "file").is_relative_to(paths.root)


def test_workspace_refuses_a_git_checkout_reached_through_a_symlink(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "run").mkdir()
    link = tmp_path / "link"
    link.symlink_to(checkout / "run")
    paths = RuntimePaths(
        root=link,
        agent_home=tmp_path / "home",
        outputs=tmp_path / "outputs",
        metaclaw_python=tmp_path / "python",
    )
    with pytest.raises(ConfigError):
        paths.workspace("skills", "service")


# --- RunDir ----------------------------------------------------------------------------------


def test_run_dir_layout(tmp_path: Path) -> None:
    run = RunDir.for_combo(tmp_path / "outputs", "qwen3.8-27b", "file-create", "rl")
    root = tmp_path / "outputs" / "qwen3.8-27b" / "file-create" / "rl"
    assert run.root == root
    assert run.config_path == root / "config.json"
    assert run.adapt_dir == root / "adapt"
    assert run.summary_path == root / "summary.json"
    assert run.eval_dir == root / "eval"
    assert not root.exists(), "RunDir computes paths and creates nothing"


def test_run_dir_is_frozen(tmp_path: Path) -> None:
    run = RunDir(tmp_path)
    with pytest.raises(AttributeError):
        run.root = tmp_path / "other"  # type: ignore[misc]


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "../x"])
@pytest.mark.parametrize("position", ["model", "task", "mechanism"])
def test_run_dir_rejects_names_that_are_not_one_path_component(
    tmp_path: Path, position: str, bad: str
) -> None:
    names = {"model": "qwen3.8-27b", "task": "file-create", "mechanism": "rl"}
    names[position] = bad
    with pytest.raises(ValueError, match=position):
        RunDir.for_combo(tmp_path, **names)


# --- the lock of the runtime directory and the home ------------------------------------------


NO_LOCKS = [errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL, errno.EISDIR]


@pytest.mark.parametrize("code", NO_LOCKS, ids=[errno.errorcode[code] for code in NO_LOCKS])
@pytest.mark.parametrize("which", ["runtime directory", "agent's home"])
def test_a_file_system_that_cannot_lock_a_directory_is_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: int, which: str
) -> None:
    """EINVAL and EISDIR: some network file systems answer a lock on a directory descriptor with them."""
    real = paths_module.fcntl.flock
    calls: list[int] = []

    def no_locks(descriptor: int, operation: int) -> None:
        if operation == paths_module.fcntl.LOCK_UN:
            real(descriptor, operation)
            return
        calls.append(descriptor)
        if which == "agent's home" and len(calls) == 1:
            real(descriptor, operation)  # the runtime directory locks; the home does not
            return
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(paths_module.fcntl, "flock", no_locks)
    paths = paths_in(tmp_path)
    with pytest.raises(RunDirError, match="the file system cannot lock files") as refused, paths.lock():
        pass
    assert not isinstance(refused.value, RunLocked)
    assert f"the {which} {paths.root if which == 'runtime directory' else paths.agent_home}" in str(refused.value)
    assert not paths.holds_lock()
    monkeypatch.setattr(paths_module.fcntl, "flock", real)
    with paths.lock():  # nothing was left held
        assert paths.holds_lock()


def test_another_failure_to_lock_a_directory_is_raised_as_it_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(descriptor: int, operation: int) -> None:
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(paths_module.fcntl, "flock", broken)
    with pytest.raises(OSError) as raised, paths_in(tmp_path).lock():
        pass
    assert raised.value.errno == errno.EIO and not isinstance(raised.value, RunDirError)
