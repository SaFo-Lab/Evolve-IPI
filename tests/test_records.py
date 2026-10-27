"""The run directory of a combination: evaluation passes, voids, the adaptation's completion and the configuration."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

from evolve_ipi import records
from evolve_ipi.records import RunDir, RunDirError, RunLocked


def _run(tmp_path: Path) -> RunDir:
    return RunDir.for_combo(tmp_path / "outputs", "qwen3.8-27b", "drive-create", "rl")


def _lib_sha(directory: Path) -> str:
    """The library hash, computed here on its own: the path of each SKILL.md, a NUL byte, its bytes."""
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob("SKILL.md")):
        digest.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


# --- passes ----------------------------------------------------------------------------------


def test_pass_layout(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert run.pass_dir("base", 3, 1) == run.eval_dir / "base" / "setting3" / "pass1"
    assert run.pass_dir("adapted", 1, 2) == run.eval_dir / "adapted" / "setting1" / "pass2"
    assert run.episode_path("adapted", 2, 3, "drive-create-adapt-004") == (
        run.eval_dir / "adapted" / "setting2" / "pass3" / "drive-create-adapt-004.json"
    )
    assert run.void_dir == run.eval_dir / "_void"
    assert run.topup_path == run.eval_dir / "topup.json"
    assert not run.root.exists(), "the layout is computed, nothing is created"


@pytest.mark.parametrize(
    ("variant", "setting", "k"),
    [("trained", 3, 1), ("Base", 3, 1), ("base", 0, 1), ("base", 4, 1), ("base", 3, 0), ("base", 3, -1),
     ("base", True, 1), ("base", 3, True), ("base", "3", 1), ("base", 3, 1.0)],
)  # fmt: skip
def test_pass_dir_refuses_what_is_not_a_pass(tmp_path: Path, variant: str, setting: int, k: int) -> None:
    with pytest.raises(ValueError):
        _run(tmp_path).pass_dir(variant, setting, k)  # type: ignore[arg-type]


@pytest.mark.parametrize("item", ["", ".", "..", "a/b", "../x"])
def test_episode_path_refuses_an_item_that_is_not_a_file_name(tmp_path: Path, item: str) -> None:
    with pytest.raises(ValueError):
        _run(tmp_path).episode_path("base", 3, 1, item)


def test_a_pass_is_complete_only_once_marked(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert not run.is_complete("base", 3, 1)
    run.pass_dir("base", 3, 1).mkdir(parents=True)
    assert not run.is_complete("base", 3, 1), "a pass directory alone is a pass in progress, or an interrupted one"
    run.mark_complete("base", 3, 1, {"valid": True, "reasons": [], "asr": {"hits": 2, "n": 40, "rate": 0.05}})
    assert run.is_complete("base", 3, 1)
    record = run.pass_record("base", 3, 1)
    assert record["asr"] == {"hits": 2, "n": 40, "rate": 0.05}
    assert record["valid"] is True
    assert "completed_at" in record
    assert json.loads((run.pass_dir("base", 3, 1) / "_complete.json").read_text()) == record


def test_a_finished_pass_is_never_written_again(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.pass_dir("base", 3, 1).mkdir(parents=True)
    run.mark_complete("base", 3, 1, {"valid": True, "reasons": []})
    before = (run.pass_dir("base", 3, 1) / "_complete.json").read_bytes()
    with pytest.raises(RunDirError, match="already"):
        run.mark_complete("base", 3, 1, {"valid": True, "reasons": []})
    assert (run.pass_dir("base", 3, 1) / "_complete.json").read_bytes() == before


def test_only_a_valid_pass_is_marked_complete(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.pass_dir("base", 3, 1).mkdir(parents=True)
    for verdict in ({"valid": False, "reasons": ["pipeline anomaly timeout at x"]}, {"reasons": []}, {"valid": 1}):
        with pytest.raises(ValueError, match="valid"):
            run.mark_complete("base", 3, 1, verdict)
    assert not run.is_complete("base", 3, 1)


def test_mark_complete_needs_the_pass_directory(tmp_path: Path) -> None:
    with pytest.raises(RunDirError, match="does not exist"):
        _run(tmp_path).mark_complete("base", 3, 1, {"valid": True, "reasons": []})


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"valid": false}'])
def test_a_marker_that_cannot_be_read_is_refused(tmp_path: Path, content: str) -> None:
    run = _run(tmp_path)
    run.pass_dir("base", 3, 1).mkdir(parents=True)
    (run.pass_dir("base", 3, 1) / "_complete.json").write_text(content)
    assert run.is_complete("base", 3, 1)
    with pytest.raises(RunDirError):
        run.pass_record("base", 3, 1)


def test_pass_record_of_a_pass_that_is_not_complete(tmp_path: Path) -> None:
    with pytest.raises(RunDirError, match="not complete"):
        _run(tmp_path).pass_record("adapted", 3, 1)


# --- voids -----------------------------------------------------------------------------------


def test_void_moves_the_pass_with_its_reasons(tmp_path: Path) -> None:
    run = _run(tmp_path)
    directory = run.pass_dir("adapted", 3, 2)
    directory.mkdir(parents=True)
    (directory / "drive-create-test-500.json").write_text("{}")
    target = run.void("adapted", 3, 2, "pipeline anomaly timeout at drive-create-test-500", {"attempt": 1})
    assert not directory.exists()
    assert target.parent == run.void_dir
    assert target.name.startswith("adapted-setting3-pass2-")
    assert (target / "drive-create-test-500.json").read_text() == "{}"
    why = json.loads((target / "why.json").read_text())
    assert why["reason"] == "pipeline anomaly timeout at drive-create-test-500"
    assert why["attempt"] == 1
    assert why["pass"] == {"variant": "adapted", "setting": 3, "pass": 2}
    assert "voided_at" in why
    assert run.voids("adapted", 3, 2) == [target]
    assert run.voids("adapted", 3, 1) == []


def test_voids_in_the_same_second_get_names_of_their_own(tmp_path: Path) -> None:
    run = _run(tmp_path)
    targets = []
    for _ in range(3):
        run.pass_dir("base", 1, 1).mkdir(parents=True)
        targets.append(run.void("base", 1, 1, "interrupted"))
    assert len(set(targets)) == 3
    assert all(target.is_dir() for target in targets)
    assert run.voids("base", 1, 1) == sorted(targets)


def test_a_complete_pass_is_never_voided(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.pass_dir("base", 3, 1).mkdir(parents=True)
    run.mark_complete("base", 3, 1, {"valid": True, "reasons": []})
    with pytest.raises(RunDirError, match="complete"):
        run.void("base", 3, 1, "no")
    assert run.is_complete("base", 3, 1)


def test_void_of_a_pass_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(RunDirError, match="no pass"):
        _run(tmp_path).void("base", 3, 1, "nothing")


def test_voids_of_another_pass_are_not_counted(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.pass_dir("base", 3, 1).mkdir(parents=True)
    run.void("base", 3, 1, "x")
    run.pass_dir("base", 3, 11).mkdir(parents=True)
    run.void("base", 3, 11, "y")
    assert len(run.voids("base", 3, 1)) == 1
    assert len(run.voids("base", 3, 11)) == 1


# --- the adaptation --------------------------------------------------------------------------


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_complete_passes_are_found_by_number(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert run.complete_passes("adapted", 3) == []
    for k in (10, 2, 1):
        run.pass_dir("adapted", 3, k).mkdir(parents=True)
        run.mark_complete("adapted", 3, k, {"valid": True})
    run.pass_dir("adapted", 3, 4).mkdir(parents=True)  # unfinished
    (run.pass_dir("adapted", 3, 1).parent / "pass05").mkdir()  # not a pass name
    (run.pass_dir("adapted", 3, 1).parent / "notes.txt").write_text("x")
    assert run.complete_passes("adapted", 3) == [1, 2, 10]
    assert run.complete_passes("base", 3) == [] and run.complete_passes("adapted", 1) == []
    with pytest.raises(ValueError):
        run.complete_passes("trained", 3)  # type: ignore[arg-type]


def test_rl_adaptation_is_done_when_its_checkpoints_are_recorded(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert not run.adapt_done("rl")
    _write(run.adapt_dir / "stage1" / "checkpoint.json", {"final": "tinker://a/step_0032"})
    assert not run.adapt_done("rl"), "a finished stage is not a finished adaptation"
    _write(
        run.adapt_dir / "checkpoints.json", {"update_32": "tinker://a/step_0032", "update_64": "tinker://b/step_0032"}
    )
    assert run.adapt_done("rl")
    assert not run.adapt_done("skills")


@pytest.mark.parametrize(
    "content",
    ["{", "[]", '{"update_32": "tinker://a"}', '{"update_32": "tinker://a", "update_64": ""}',
     '{"update_32": "tinker://a", "update_64": 7}'],
)  # fmt: skip
def test_a_malformed_rl_marker_is_refused(tmp_path: Path, content: str) -> None:
    run = _run(tmp_path)
    (run.adapt_dir).mkdir(parents=True)
    (run.adapt_dir / "checkpoints.json").write_text(content)
    with pytest.raises(RunDirError, match="checkpoints.json"):
        run.adapt_done("rl")


def _library(run: RunDir, skills: dict[str, str]) -> str:
    library = run.adapt_dir / "library"
    for name, text in skills.items():
        (library / name).mkdir(parents=True)
        (library / name / "SKILL.md").write_text(text)
    (library / "evolution_history.jsonl").write_text("{}\n" * 12)
    return _lib_sha(library)


def _skills_marker(run: RunDir, sha: str, extra: dict[str, object] | None = None) -> None:
    marker = {"library": "library", "sha": sha, "evolutions": 12, "skills": 2, "zero_skill_events": 0, "at": "now"}
    _write(run.adapt_dir / "library.json", {**marker, **(extra or {})})


def test_skill_evolution_is_done_when_its_library_matches_its_record(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert not run.adapt_done("skills")
    sha = _library(run, {"read-first": "a", "ask-first": "b"})
    assert not run.adapt_done("skills"), "a library without its record is an evolution that did not finish"
    _skills_marker(run, sha)
    assert run.adapt_done("skills")
    assert not run.adapt_done("rl")


def test_a_library_changed_after_its_evolution_is_refused(tmp_path: Path) -> None:
    run = _run(tmp_path)
    sha = _library(run, {"read-first": "a", "ask-first": "b"})
    _skills_marker(run, sha)
    (run.adapt_dir / "library" / "ask-first" / "SKILL.md").write_text("changed")
    with pytest.raises(RunDirError, match="changed"):
        run.adapt_done("skills")


@pytest.mark.parametrize(
    "marker",
    [{"library": "../elsewhere"}, {"library": ""}, {"sha": "abc"}, {"evolutions": "12"}, {"evolutions": True},
     {"skills": -1}],
)  # fmt: skip
def test_a_malformed_skills_marker_is_refused(tmp_path: Path, marker: dict[str, object]) -> None:
    run = _run(tmp_path)
    sha = _library(run, {"read-first": "a"})
    _skills_marker(run, sha, marker)
    with pytest.raises(RunDirError, match="library.json"):
        run.adapt_done("skills")


def test_a_skills_marker_without_its_library_is_refused(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _skills_marker(run, "0" * 64)
    with pytest.raises(RunDirError, match="no library"):
        run.adapt_done("skills")


def test_adapt_done_of_an_unknown_mechanism(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mechanism"):
        _run(tmp_path).adapt_done("dpo")


def test_the_library_hash_is_the_one_of_the_skill_evolution(tmp_path: Path) -> None:
    from evolve_ipi.adapt import library, skills

    assert records.lib_sha is library.lib_sha is skills.lib_sha
    sha = _library(_run(tmp_path), {"a": "x", "a-b": "y", "a/c": "z"})  # Path order: a/c/ comes before a-b/
    assert library.lib_sha(_run(tmp_path).adapt_dir / "library") == sha


PROBE = """
import json, sys
from pathlib import Path
from evolve_ipi.records import RunDir
run = RunDir(Path(sys.argv[1]))
library = run.adapt_dir / "library"
library.mkdir(parents=True)
from evolve_ipi.adapt.library import lib_sha
marker = {"library": "library", "sha": lib_sha(library), "evolutions": 12, "skills": 0}
(run.adapt_dir / "library.json").write_text(json.dumps(marker))
assert run.adapt_done("skills")
print(json.dumps(sorted(sys.modules)))
"""


def test_records_hash_a_library_without_importing_the_skill_evolution(tmp_path: Path) -> None:
    """The hash is all records needs of it: the skill evolution's module pulls in httpx, OpenClaw and the episodes."""
    probe = [sys.executable, "-c", PROBE, str(tmp_path / "run")]
    done = subprocess.run(probe, capture_output=True, text=True, check=True)
    loaded = json.loads(done.stdout)
    assert "evolve_ipi.adapt.library" in loaded and "evolve_ipi.adapt.skills" not in loaded, loaded


# --- the configuration -----------------------------------------------------------------------


def test_write_config_records_the_configuration_once(tmp_path: Path) -> None:
    run = _run(tmp_path)
    config = {"model": "qwen3.8-27b", "eval": {"temperature": 0.6, "settings12_passes": 3}, "settings": [1, 2, 3]}
    run.write_config(config)
    assert json.loads(run.config_path.read_text()) == config
    stamp = run.config_path.stat().st_mtime_ns
    run.write_config(
        {"settings": (1, 2, 3), "eval": {"settings12_passes": 3, "temperature": 0.6}, "model": "qwen3.8-27b"}
    )
    assert run.config_path.stat().st_mtime_ns == stamp, "the same configuration leaves the record as it is"
    assert run.read_config() == config


def test_write_config_refuses_another_configuration(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.write_config({"model": "qwen3.8-27b", "eval": {"temperature": 0.6, "max_void_retries": 3}})
    before = run.config_path.read_bytes()
    with pytest.raises(RunDirError, match=r"eval\.temperature"):
        run.write_config({"model": "qwen3.8-27b", "eval": {"temperature": 0.7, "max_void_retries": 3}})
    with pytest.raises(RunDirError, match="extra"):
        run.write_config({"model": "qwen3.8-27b", "eval": {"temperature": 0.6, "max_void_retries": 3}, "extra": 1})
    assert run.config_path.read_bytes() == before


def test_read_config_of_a_run_without_one(tmp_path: Path) -> None:
    with pytest.raises(RunDirError, match="no configuration"):
        _run(tmp_path).read_config()


# --- one run at a time -----------------------------------------------------------------------

# Run in a child process: try to hold the run directory; exit 3 with the refusal when another run holds it.
_LOCK_IN_A_CHILD = """
import sys
from pathlib import Path
from evolve_ipi.records import RunDir, RunLocked
try:
    with RunDir(Path(sys.argv[1])).lock():
        print("held")
except RunLocked as error:
    print(error)
    sys.exit(3)
"""


def _lock_in_a_child(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _LOCK_IN_A_CHILD, str(root)], capture_output=True, text=True, timeout=120, check=False
    )


def _holder_text(run: RunDir) -> str:
    holder = json.loads((run.root / ".lock").read_text())
    return f"pid {holder['pid']} on host {holder['host']}, holding it since {holder['since']}"


def test_a_run_directory_is_held_by_one_run_at_a_time(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert not run.holds_lock()
    with run.lock() as held:
        assert held is run and run.holds_lock()
        holder = json.loads((run.root / ".lock").read_text())
        assert (holder["pid"], holder["host"]) == (os.getpid(), socket.gethostname())
        assert holder["since"][:4].isdigit() and "T" in holder["since"], "when the lock was taken, to the microsecond"
        with pytest.raises(RunLocked) as refused, RunDir(run.root).lock():
            pass
        assert f"in use by another run ({_holder_text(run)})" in str(refused.value)
        assert run.holds_lock(), "a refused second hold leaves the first one in place"
    assert not run.holds_lock()
    assert (run.root / ".lock").read_text() == "", "who held it goes with the lock"


def test_another_process_cannot_hold_a_held_run_directory(tmp_path: Path) -> None:
    run = _run(tmp_path)
    with run.lock():
        refused = _lock_in_a_child(run.root)
        holder = _holder_text(run)
    assert refused.returncode == 3, refused.stdout + refused.stderr
    assert f"is in use by another run ({holder})" in refused.stdout
    assert f"pid {os.getpid()} on host {socket.gethostname()}, holding it since" in holder
    after = _lock_in_a_child(run.root)
    assert (after.returncode, after.stdout.strip()) == (0, "held"), after.stderr


def test_the_lock_is_released_when_the_block_raises(tmp_path: Path) -> None:
    run = _run(tmp_path)
    with pytest.raises(ValueError, match="boom"), run.lock():
        raise ValueError("boom")
    assert not run.holds_lock()
    with run.lock():
        assert run.holds_lock()
    child = _lock_in_a_child(run.root)
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize("code", [errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS])
def test_a_file_system_that_cannot_lock_files_is_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: int
) -> None:
    def no_locks(descriptor: int, operation: int) -> None:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(records.fcntl, "flock", no_locks)
    run = _run(tmp_path)
    with pytest.raises(RunDirError, match="must support file locks") as refused, run.lock():
        pass
    assert not isinstance(refused.value, RunLocked)
    assert str(run.root) in str(refused.value)
    assert not run.holds_lock()


@pytest.mark.parametrize("failing", ["gethostname", "now"])
def test_a_failure_while_saying_who_holds_the_lock_releases_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failing: str
) -> None:
    """Once the flock is taken, nothing may fail outside the block that releases it."""

    def broken(*args: object) -> None:
        raise OSError(errno.EIO, "the host name cannot be read")

    if failing == "gethostname":
        monkeypatch.setattr(records.socket, "gethostname", broken)
    else:
        monkeypatch.setattr(records, "datetime", types.SimpleNamespace(now=broken))
    run = _run(tmp_path)
    with pytest.raises(OSError, match="cannot be read"), run.lock():
        pass
    assert not run.holds_lock()
    monkeypatch.undo()
    child = _lock_in_a_child(run.root)
    assert child.returncode == 0, "the lock was released: " + child.stdout + child.stderr
    with run.lock():
        assert run.holds_lock()


def test_another_failure_to_lock_is_raised_as_it_is(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def broken(descriptor: int, operation: int) -> None:
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(records.fcntl, "flock", broken)
    with pytest.raises(OSError) as raised, _run(tmp_path).lock():
        pass
    assert raised.value.errno == errno.EIO and not isinstance(raised.value, RunDirError)


def test_require_lock(tmp_path: Path) -> None:
    run = _run(tmp_path)
    with pytest.raises(RunDirError, match="not locked by this process"):
        run.require_lock()
    with run.lock():
        run.require_lock()
        RunDir(run.root / "." / ".." / run.root.name).require_lock()  # the same directory, spelled otherwise
