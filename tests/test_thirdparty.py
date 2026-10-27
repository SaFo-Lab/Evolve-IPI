"""The pinned frameworks under third_party/ and the three patches applied to them."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from evolve_ipi import thirdparty
from evolve_ipi.thirdparty import (
    METACLAW_DIR,
    OPENCLAW_RL_DIR,
    OPENCLAW_TINKER_DIR,
    PATCHES,
    RL_HEADERS_PLUGIN_DIR,
    THIRD_PARTY,
    PatchError,
    apply_patches,
    submodule_commits,
    verify_patches,
)

REPO_ROOT = THIRD_PARTY.parent

OPENCLAW_RL_COMMIT = "f48ac358adf9873b5cb2210f1cb234a52ed8a8a3"
METACLAW_COMMIT = "922caf3a1cd093fb316e95183a8acc8aa47b3b21"

ROLLOUT = "OpenClaw-RL/openclaw-tinker/rollout.py"
SKILL_EVOLVER = "MetaClaw/metaclaw/skill_evolver.py"
API_SERVER = "MetaClaw/metaclaw/api_server.py"
PATCHED_FILES = (ROLLOUT, SKILL_EVOLVER, API_SERVER)

NO_SUBMODULE_HINT = "git submodule update --init"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sums(root: Path = THIRD_PARTY) -> dict[str, str]:
    sums: dict[str, str] = {}
    for line in (root / "patches" / "SHA256SUMS").read_text().splitlines():
        digest, _, name = line.partition("  ")
        sums[name] = digest
    return sums


def git(directory: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(directory), *args], check=True, capture_output=True, text=True)
    return done.stdout


def require_git_metadata(submodule: str, root: Path | None = None) -> None:
    """Skip the test when the submodule has no git metadata to read: a copy of the tree without ``.git``, or a build
    context whose ``.git`` files point nowhere. ``evolve-ipi check`` skips the same check, with the same reason."""
    directory = (root or THIRD_PARTY) / submodule
    done = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "--verify", "HEAD^{commit}"],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(directory.resolve().parent)},
        check=False,
    )
    if done.returncode != 0:
        pytest.skip(f"{directory} has no git metadata (a copy of the tree without .git), so it cannot be read")


def upstream_file(submodule: str, path: str) -> bytes:
    """A framework file as the pinned commit has it, whatever state the working tree is in."""
    require_git_metadata(submodule)
    return subprocess.run(
        ["git", "-C", str(THIRD_PARTY / submodule), "cat-file", "blob", f"HEAD:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def unpatched_tree(base: Path) -> Path:
    """A third_party directory of its own: the real patches around the unpatched upstream files."""
    root = base / "third_party"
    shutil.copytree(THIRD_PARTY / "patches", root / "patches")
    for patch in PATCHES:
        target = root / patch.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(upstream_file(patch.submodule, patch.target))
    return root


def patched_tree(base: Path) -> Path:
    root = unpatched_tree(base)
    apply_patches(root)
    return root


def append_byte(path: Path) -> None:
    with path.open("ab") as handle:
        handle.write(b" ")


# --- the submodules ----------------------------------------------------------------------------


def test_submodules_at_pinned_commits() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("no .git next to third_party/ (a copy of the tree), so the checked-out commits cannot be read")
    assert submodule_commits(THIRD_PARTY) == {
        "OpenClaw-RL": OPENCLAW_RL_COMMIT,
        "MetaClaw": METACLAW_COMMIT,
    }


def test_gitmodules_and_gitlinks_record_the_pins() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    urls = git(REPO_ROOT, "config", "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.url$")
    assert dict(line.split() for line in urls.splitlines()) == {
        "submodule.third_party/OpenClaw-RL.url": "https://github.com/Gen-Verse/OpenClaw-RL.git",
        "submodule.third_party/MetaClaw.url": "https://github.com/aiming-lab/MetaClaw.git",
    }
    gitlinks = {}
    for line in git(REPO_ROOT, "ls-files", "-s", "--", "third_party").splitlines():
        mode, commit, _, path = line.replace("\t", " ").split()
        if mode == "160000":
            gitlinks[path] = commit
    assert gitlinks == {
        "third_party/OpenClaw-RL": OPENCLAW_RL_COMMIT,
        "third_party/MetaClaw": METACLAW_COMMIT,
    }


def test_directory_constants_point_into_the_checkouts() -> None:
    assert OPENCLAW_RL_DIR == THIRD_PARTY / "OpenClaw-RL"
    assert OPENCLAW_TINKER_DIR == OPENCLAW_RL_DIR / "openclaw-tinker"
    assert METACLAW_DIR == THIRD_PARTY / "MetaClaw"
    assert RL_HEADERS_PLUGIN_DIR == OPENCLAW_RL_DIR / "extensions" / "rl-training-headers"
    assert (OPENCLAW_TINKER_DIR / "rollout.py").is_file()
    assert (RL_HEADERS_PLUGIN_DIR / "index.ts").is_file()
    assert (METACLAW_DIR / "metaclaw" / "api_server.py").is_file()


@pytest.mark.parametrize("layout", ["empty", "missing"])
def test_submodule_commits_refuses_a_directory_that_is_not_a_checkout(tmp_path: Path, layout: str) -> None:
    """Inside another repository an empty directory would report that repository's commit."""
    outer = tmp_path / "outer"
    outer.mkdir()
    git(outer, "init", "-q")
    git(outer, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "--allow-empty", "-q", "-m", "x")
    root = outer / "third_party"
    root.mkdir()
    if layout == "empty":
        for name in ("OpenClaw-RL", "MetaClaw"):
            (root / name).mkdir()
    with pytest.raises(PatchError, match=NO_SUBMODULE_HINT):
        submodule_commits(root)


# --- applying the patches ----------------------------------------------------------------------


def test_patched_files_match_sha256sums() -> None:
    apply_patches()
    verified = verify_patches(THIRD_PARTY)
    sums = read_sums()
    assert set(verified) == set(sums) == set(PATCHED_FILES)
    for name in PATCHED_FILES:
        assert verified[name] == sums[name] == sha256(THIRD_PARTY / name)


def test_apply_is_idempotent() -> None:
    apply_patches()
    before = {name: sha256(THIRD_PARTY / name) for name in PATCHED_FILES}
    apply_patches()
    assert {name: sha256(THIRD_PARTY / name) for name in PATCHED_FILES} == before


def test_patches_turn_the_pinned_upstream_files_into_the_recorded_ones(tmp_path: Path) -> None:
    root = unpatched_tree(tmp_path)
    with pytest.raises(PatchError, match="SHA256SUMS"):
        verify_patches(root)  # the upstream files are not what is recorded
    apply_patches(root)
    assert verify_patches(root) == read_sums()
    apply_patches(root)  # and a second run changes nothing
    assert verify_patches(root) == read_sums()


def test_apply_works_inside_an_enclosing_repository(tmp_path: Path) -> None:
    """`git apply` run in a directory that belongs to another repository skips, without any error,
    every file outside that directory. That is what an uninitialised submodule looks like."""
    outer = tmp_path / "outer"
    outer.mkdir()
    git(outer, "init", "-q")
    root = unpatched_tree(outer)
    apply_patches(root)
    assert verify_patches(root) == read_sums()


def test_apply_works_where_the_submodule_git_file_dangles(tmp_path: Path) -> None:
    """A build context without .git keeps the submodules' `.git` files, which then point nowhere."""
    root = unpatched_tree(tmp_path)
    for name in ("OpenClaw-RL", "MetaClaw"):
        (root / name / ".git").write_text(f"gitdir: ../../.git/modules/third_party/{name}\n")
    apply_patches(root)
    assert verify_patches(root) == read_sums()


@pytest.mark.parametrize("layout", ["empty", "missing"])
def test_apply_refuses_a_submodule_that_is_not_checked_out(tmp_path: Path, layout: str) -> None:
    root = unpatched_tree(tmp_path)
    shutil.rmtree(root / "OpenClaw-RL")
    if layout == "empty":
        (root / "OpenClaw-RL").mkdir()
    with pytest.raises(PatchError, match=NO_SUBMODULE_HINT) as err:
        apply_patches(root)
    assert "OpenClaw-RL" in str(err.value)


def test_apply_refuses_a_missing_patch_file(tmp_path: Path) -> None:
    root = unpatched_tree(tmp_path)
    (root / "patches" / "metaclaw-api-server.patch").unlink()
    with pytest.raises(PatchError, match="metaclaw-api-server.patch"):
        apply_patches(root)


@pytest.mark.parametrize("damage", ["edited context", "one hunk undone"])
def test_apply_refuses_a_file_that_is_neither_upstream_nor_patched(tmp_path: Path, damage: str) -> None:
    root = patched_tree(tmp_path)
    rollout = root / ROLLOUT
    text = rollout.read_text()
    if damage == "edited context":
        damaged = text.replace("last_log = start\n", "last_log = start + 1\n", 1)
    else:
        damaged = text.replace("{time.time() - start:.2f}s carry_over={len(pending)}", "{time.time() - start:.2f}s")
    assert damaged != text
    rollout.write_text(damaged)
    with pytest.raises(PatchError, match="openclaw-rl-rollout.patch") as err:
        apply_patches(root)
    assert "openclaw-tinker/rollout.py" in str(err.value)
    assert rollout.read_text() == damaged, "a refused patch must not be half applied"


def test_apply_changes_nothing_when_one_patch_cannot_apply(tmp_path: Path) -> None:
    root = unpatched_tree(tmp_path)
    (root / API_SERVER).write_text("not the upstream file\n")
    before = {name: sha256(root / name) for name in PATCHED_FILES}
    with pytest.raises(PatchError, match="metaclaw-api-server.patch"):
        apply_patches(root)
    assert {name: sha256(root / name) for name in PATCHED_FILES} == before


def test_apply_accepts_a_relative_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = unpatched_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    apply_patches(Path("third_party"))
    assert verify_patches(root) == read_sums()


def test_apply_names_a_missing_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = unpatched_tree(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "no-such-directory"))
    with pytest.raises(PatchError, match="git is not installed"):
        apply_patches(root)


def test_git_variables_of_the_caller_do_not_matter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A git hook that runs the tests leaves GIT_DIR and friends in the environment."""
    root = unpatched_tree(tmp_path)
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.setenv(name, str(tmp_path / "somewhere-else"))
    apply_patches(root)
    assert verify_patches(root) == read_sums()
    assert submodule_commits(THIRD_PARTY)["MetaClaw"] == METACLAW_COMMIT


# --- verifying the patched files ---------------------------------------------------------------


def test_verify_detects_tampering(tmp_path: Path) -> None:
    apply_patches()
    copy = tmp_path / "third_party"
    shutil.copytree(THIRD_PARTY / "patches", copy / "patches")
    for name in PATCHED_FILES:
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(THIRD_PARTY / name, copy / name)
    assert verify_patches(copy) == verify_patches(THIRD_PARTY)  # the copy is faithful

    append_byte(copy / ROLLOUT)
    with pytest.raises(PatchError, match="rollout.py"):
        verify_patches(copy)
    assert sha256(THIRD_PARTY / ROLLOUT) == read_sums()[ROLLOUT], "the real tree was touched"


def test_verify_names_every_file_that_differs(tmp_path: Path) -> None:
    root = patched_tree(tmp_path)
    append_byte(root / ROLLOUT)
    append_byte(root / API_SERVER)
    with pytest.raises(PatchError) as err:
        verify_patches(root)
    message = str(err.value)
    assert ROLLOUT in message and API_SERVER in message
    assert SKILL_EVOLVER not in message


def test_verify_reports_a_missing_patched_file(tmp_path: Path) -> None:
    root = patched_tree(tmp_path)
    (root / SKILL_EVOLVER).unlink()
    with pytest.raises(PatchError, match="skill_evolver.py"):
        verify_patches(root)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("drop the rollout line", "rollout.py"),
        ("add an unknown file", "OpenClaw-RL/README.md"),
        ("repeat a line", "twice"),
        ("garble a digest", "line 1"),
        ("delete the file", "SHA256SUMS"),
        ("not text", "SHA256SUMS"),
    ],
)
def test_verify_rejects_a_damaged_sha256sums(tmp_path: Path, damage: str, message: str) -> None:
    root = patched_tree(tmp_path)
    sums = root / "patches" / "SHA256SUMS"
    lines = sums.read_text().splitlines()
    if damage == "drop the rollout line":
        lines = [line for line in lines if not line.endswith("rollout.py")]
    elif damage == "add an unknown file":
        lines.append(f"{'0' * 64}  OpenClaw-RL/README.md")
    elif damage == "repeat a line":
        lines.append(lines[0])
    elif damage == "garble a digest":
        lines[0] = "xyz" + lines[0][3:]
    if damage == "delete the file":
        sums.unlink()
    elif damage == "not text":
        sums.write_bytes(b"\xff\xfe\x00 not utf-8\n")
    else:
        sums.write_text("\n".join(lines) + "\n")
    with pytest.raises(PatchError, match=message):
        verify_patches(root)


# --- the command line --------------------------------------------------------------------------


def test_cli_apply_then_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = unpatched_tree(tmp_path)
    assert thirdparty.main(["verify", "--root", str(root)]) == 1
    assert "SHA256SUMS" in capsys.readouterr().err

    assert thirdparty.main(["apply", "--root", str(root)]) == 0
    applied = capsys.readouterr().out
    assert all(patch.file in applied for patch in PATCHES)

    assert thirdparty.main(["verify", "--root", str(root)]) == 0
    verified = capsys.readouterr().out
    for name, digest in read_sums().items():
        assert f"{digest}  {name}" in verified


def test_cli_reports_a_failed_apply(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = unpatched_tree(tmp_path)
    shutil.rmtree(root / "MetaClaw")
    assert thirdparty.main(["apply", "--root", str(root)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert NO_SUBMODULE_HINT in err


def test_cli_runs_as_a_module(tmp_path: Path) -> None:
    root = unpatched_tree(tmp_path)

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "evolve_ipi.thirdparty", *args, "--root", str(root)],
            capture_output=True,
            text=True,
        )

    assert run("verify").returncode == 1
    done = run("apply")
    assert done.returncode == 0, done.stderr
    done = run("verify")
    assert done.returncode == 0, done.stderr
    append_byte(root / ROLLOUT)
    done = run("verify")
    assert done.returncode == 1
    assert "rollout.py" in done.stderr

    unknown = subprocess.run(
        [sys.executable, "-m", "evolve_ipi.thirdparty", "frobnicate"], capture_output=True, text=True
    )
    assert unknown.returncode == 2
    assert "frobnicate" in unknown.stderr


# --- what the patches directory holds ----------------------------------------------------------


def test_patches_directory_holds_exactly_the_shipped_files() -> None:
    names = sorted(path.name for path in (THIRD_PARTY / "patches").iterdir())
    assert names == [
        "NOTICE",
        "SHA256SUMS",
        "metaclaw-api-server.patch",
        "metaclaw-skill-evolver.patch",
        "openclaw-rl-rollout.patch",
    ]


def test_each_patch_touches_only_its_own_file() -> None:
    for patch in PATCHES:
        text = (THIRD_PARTY / "patches" / patch.file).read_text()
        assert re.findall(r"^diff --git a/(\S+) b/(\S+)$", text, re.MULTILINE) == [(patch.target, patch.target)]
        assert re.findall(r"^\+\+\+ b/(\S+)$", text, re.MULTILINE) == [patch.target]


DATE_OR_TALLY = re.compile(r"\d{4}-\d{2}-\d{2}|\b\d+ of \d+\b|\d{2,3},\d{3}")


def comment_problems(patch_name: str, patch_text: str) -> list[str]:
    """The comments a patch adds that say more than what the code does: a second one in a hunk, one longer than a
    line, or one that holds a date or a tally."""
    problems = []
    for hunk in re.split(r"^@@.*$", patch_text, flags=re.MULTILINE)[1:]:
        comments = [
            line[1:].strip()
            for line in hunk.splitlines()
            if line.startswith("+") and (line[1:].strip().startswith("#") or '"""' in line)
        ]
        if len(comments) > 1:
            problems.append(f"{patch_name}: {len(comments)} comments in one hunk")
        for comment in comments:
            if len(comment) > 100:
                problems.append(f"{patch_name}: a comment of {len(comment)} characters")
            if DATE_OR_TALLY.search(comment):
                problems.append(f"{patch_name}: a comment with a date or a tally: {comment}")
    return problems


def test_patches_add_at_most_one_short_comment_per_hunk() -> None:
    for patch in PATCHES:
        text = (THIRD_PARTY / "patches" / patch.file).read_text()
        assert comment_problems(patch.file, text) == []


def test_the_comment_check_sees_what_it_is_for() -> None:
    def hunk(*added: str) -> str:
        return "@@ -1,2 +1,4 @@\n x = 1\n" + "".join(f"+{line}\n" for line in added)

    assert comment_problems("p", hunk("# keeps the rest for the next step", "y = 2")) == []
    assert comment_problems("p", hunk("# one", "# two")) == ["p: 2 comments in one hunk"]
    assert comment_problems("p", hunk("# " + "x" * 99)) == ["p: a comment of 101 characters"]
    assert comment_problems("p", hunk("# changed on 2026-01-31")) != []
    assert comment_problems("p", hunk("# 7 of 24 samples were lost")) != []
    assert comment_problems("p", hunk("# 31,000 samples")) != []
    assert comment_problems("p", hunk('"""A docstring that is fine."""')) == []


def test_the_git_tests_skip_where_a_submodule_has_no_git_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy of the tree without `.git`: the submodule directories hold files and a `.git` file that points nowhere."""
    for name in ("OpenClaw-RL", "MetaClaw"):
        (tmp_path / name).mkdir()
        (tmp_path / name / ".git").write_text(f"gitdir: ../../.git/modules/third_party/{name}\n")
    monkeypatch.setitem(globals(), "THIRD_PARTY", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="no git metadata"):
        upstream_file("MetaClaw", "metaclaw/api_server.py")


def test_notice_names_the_licenses_the_patches_and_the_pins() -> None:
    notice = (THIRD_PARTY / "patches" / "NOTICE").read_text()
    for needle in ("OpenClaw-RL", "Apache License 2.0", "MetaClaw", "MIT License"):
        assert needle in notice
    for patch in PATCHES:
        assert patch.file in notice
    assert OPENCLAW_RL_COMMIT in notice and METACLAW_COMMIT in notice
