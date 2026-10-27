"""The two frameworks under ``third_party/`` and the three patches applied to them.

OpenClaw-RL and MetaClaw are git submodules, each pinned to one commit. Three of their source files
are patched. ``patches/SHA256SUMS`` holds the hash each of those files must have afterwards.

    python -m evolve_ipi.thirdparty apply     patch the checkouts in place (safe to repeat)
    python -m evolve_ipi.thirdparty verify    compare the patched files with SHA256SUMS
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
OPENCLAW_RL_DIR = THIRD_PARTY / "OpenClaw-RL"
OPENCLAW_TINKER_DIR = OPENCLAW_RL_DIR / "openclaw-tinker"
RL_HEADERS_PLUGIN_DIR = OPENCLAW_RL_DIR / "extensions" / "rl-training-headers"
METACLAW_DIR = THIRD_PARTY / "MetaClaw"

# The commit each submodule must be at; the gitlinks recorded in this repository are the same.
PINNED_COMMITS: Mapping[str, str] = MappingProxyType(
    {
        "OpenClaw-RL": "f48ac358adf9873b5cb2210f1cb234a52ed8a8a3",
        "MetaClaw": "922caf3a1cd093fb316e95183a8acc8aa47b3b21",
    }
)

_PATCHES_DIR = "patches"
_SUMS_FILE = "SHA256SUMS"
_SUMS_LINE = re.compile(r"([0-9a-f]{64})  (\S.*)")
_NOT_CHECKED_OUT = "run `git submodule update --init` in the repository root"

# Variables that would make git pick a repository other than the one asked for.
_REPOSITORY_VARIABLES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_PREFIX",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
)


class PatchError(Exception):
    """A patch cannot be applied, or a patched file is not what it should be."""


@dataclass(frozen=True)
class Patch:
    file: str  # the patch file in ``patches/``
    submodule: str  # the directory below ``third_party/`` it applies in
    target: str  # the file it changes, relative to the submodule

    @property
    def path(self) -> str:
        """The patched file relative to ``third_party/``, as ``SHA256SUMS`` names it."""
        return f"{self.submodule}/{self.target}"


PATCHES = (
    Patch("openclaw-rl-rollout.patch", "OpenClaw-RL", "openclaw-tinker/rollout.py"),
    Patch("metaclaw-skill-evolver.patch", "MetaClaw", "metaclaw/skill_evolver.py"),
    Patch("metaclaw-api-server.patch", "MetaClaw", "metaclaw/api_server.py"),
)


# --- git ---------------------------------------------------------------------------------------


def _run_git(directory: Path, args: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    if not directory.is_dir():
        raise PatchError(f"{directory} does not exist; {_NOT_CHECKED_OUT}")
    environment = {k: v for k, v in os.environ.items() if k not in _REPOSITORY_VARIABLES}
    environment.update(env)
    try:
        return subprocess.run(["git", *args], cwd=directory, env=environment, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise PatchError("git is not installed") from error


def _in_checkout(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in a submodule checkout. An empty directory must not be answered by the repository
    around it, so git may not look above the directory."""
    return _run_git(directory, args, {"GIT_CEILING_DIRECTORIES": str(directory.resolve().parent)})


def _git_apply(directory: Path, patch_file: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    """``git apply`` as a plain patch tool for the files below ``directory``.

    Left to find a repository, git breaks in two ways. A directory inside another repository has every
    file outside that repository's prefix skipped, without an error, so a patch that did nothing
    reports success. And a submodule whose ``.git`` file points nowhere, as in a build context without
    ``.git``, is a fatal error. With ``GIT_DIR`` naming no repository git does neither.
    """
    with tempfile.TemporaryDirectory() as no_repository:
        args = ["apply", "--whitespace=nowarn", *flags, str(patch_file)]
        return _run_git(directory, args, {"GIT_DIR": no_repository})


def submodule_commits(root: Path = THIRD_PARTY) -> dict[str, str]:
    """The commit each submodule under ``root`` is checked out at, by directory name.

    This needs the git metadata of a clone; a copy of the tree without ``.git`` has none.
    """
    commits = {}
    for name in PINNED_COMMITS:
        directory = Path(root) / name
        result = _in_checkout(directory, "rev-parse", "--verify", "HEAD^{commit}")
        if result.returncode != 0:
            raise PatchError(f"{directory} is not a git checkout ({result.stderr.strip()}); {_NOT_CHECKED_OUT}")
        commits[name] = result.stdout.strip()
    return commits


# --- applying ----------------------------------------------------------------------------------


def _apply(root: Path) -> list[tuple[Patch, bool]]:
    """Apply what is not applied yet; return each patch with whether this call applied it."""
    root = Path(root).resolve()
    plan: list[tuple[Patch, Path, bool]] = []  # each patch, its file, and whether it is still to be applied
    conflicts: list[str] = []
    for patch in PATCHES:
        directory = root / patch.submodule
        patch_file = root / _PATCHES_DIR / patch.file
        if not patch_file.is_file():
            raise PatchError(f"the patch file {patch_file} is missing")
        if not (directory / patch.target).is_file():
            raise PatchError(f"{directory / patch.target} does not exist; {_NOT_CHECKED_OUT}")
        if _git_apply(directory, patch_file, "--reverse", "--check").returncode == 0:
            plan.append((patch, patch_file, False))
            continue
        check = _git_apply(directory, patch_file, "--check")
        if check.returncode == 0:
            plan.append((patch, patch_file, True))
        else:
            conflicts.append(
                f"{patch.file} does not apply to {patch.path}, which is neither the pinned upstream "
                f"file nor the patched one:\n{check.stderr.strip()}"
            )
    if conflicts:  # nothing has been changed yet
        raise PatchError("\n".join(conflicts))
    for patch, patch_file, to_apply in plan:
        if not to_apply:
            continue
        result = _git_apply(root / patch.submodule, patch_file)
        if result.returncode != 0:
            raise PatchError(f"{patch.file} failed to apply to {patch.path}:\n{result.stderr.strip()}")
    return [(patch, to_apply) for patch, _, to_apply in plan]


def apply_patches(root: Path = THIRD_PARTY) -> None:
    """Patch the submodule checkouts under ``root``; patches that are already in are skipped.

    Nothing is changed unless every patch either is in already or applies cleanly. The checkouts
    stay modified, which is intended: only the pinned commits are recorded in this repository.
    """
    _apply(root)


# --- verifying ---------------------------------------------------------------------------------


def _read_sums(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PatchError(f"cannot read {path}: {error}") from error
    sums: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _SUMS_LINE.fullmatch(line)
        if match is None:
            raise PatchError(f"{path.name} line {number} is not '<sha256>  <path>': {line!r}")
        digest, name = match.groups()
        if name in sums:
            raise PatchError(f"{path.name} lists {name} twice (line {number})")
        sums[name] = digest
    patched = {patch.path for patch in PATCHES}
    for name in sorted(patched - sums.keys()):
        raise PatchError(f"{path.name} has no entry for {name}")
    for name in sorted(sums.keys() - patched):
        raise PatchError(f"{path.name} lists {name}, which no patch changes")
    return sums


def verify_patches(root: Path = THIRD_PARTY) -> dict[str, str]:
    """The sha256 of each patched file under ``root``, keyed by its path relative to ``root``.

    Raises ``PatchError`` naming every file that differs from ``patches/SHA256SUMS``.
    """
    root = Path(root)
    expected = _read_sums(root / _PATCHES_DIR / _SUMS_FILE)
    found: dict[str, str] = {}
    problems: list[str] = []
    for patch in PATCHES:
        path = root / patch.path
        if not path.is_file():
            problems.append(f"{patch.path} is missing")
            continue
        found[patch.path] = hashlib.sha256(path.read_bytes()).hexdigest()
        if found[patch.path] != expected[patch.path]:
            problems.append(
                f"{patch.path} has sha256 {found[patch.path]}, {_SUMS_FILE} says {expected[patch.path]}"
            )
    if problems:
        raise PatchError(
            "\n".join(problems) + "\nrun `python -m evolve_ipi.thirdparty apply`; a file that was edited by "
            "hand has to be restored with `git checkout` inside its submodule first"
        )
    return found


# --- command line ------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evolve_ipi.thirdparty",
        description="Apply the patches to the frameworks under third_party/, or check the patched files.",
    )
    parser.add_argument(
        "command",
        choices=("apply", "verify"),
        help="apply: patch the checkouts in place; verify: compare the patched files with SHA256SUMS",
    )
    parser.add_argument("--root", type=Path, default=THIRD_PARTY, help="the third_party directory to use")
    args = parser.parse_args(argv)
    try:
        if args.command == "apply":
            for patch, applied_now in _apply(args.root):
                print(f"{'applied' if applied_now else 'already applied'}: {patch.file}")
        else:
            for name, digest in verify_patches(args.root).items():
                print(f"{digest}  {name}")
    except PatchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
