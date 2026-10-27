"""The result directory of one (model, task, mechanism) combination, and the state of what has been done in it.

    outputs/<model>/<task>/<mechanism>/
        config.json                               the configuration the combination runs under
        adapt/                                    the adaptation; RL writes checkpoints.json last, skill evolution
                                                  library.json (next to the library) last
        eval/<variant>/setting<s>/pass<k>/        one evaluation pass of the base or the adapted agent
            <item>.json                           one episode
            _complete.json                        the pass is a valid measurement: written last, never again
        eval/_void/<variant>-setting<s>-pass<k>-<time>/   a pass that is not one, with why.json
        eval/topup.json                           the top-up decision of setting 3
        summary.json
        .lock                                     held by the run that works in the directory: its pid, host, start

A pass without ``_complete.json`` is unfinished: in progress, or left by a run that was interrupted. It is never read
as a result; the next run moves it to ``eval/_void/`` and runs the pass again. That is safe only while one run at a time
works in a directory, so a run holds the directory's lock (``RunDir.lock``) for as long as it works in it.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import socket
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_args

from evolve_ipi.adapt.library import lib_sha

__all__ = [
    "COMPLETE",
    "LOCK",
    "SETTINGS",
    "VARIANTS",
    "RunDir",
    "RunDirError",
    "RunLocked",
    "Variant",
    "now",
    "write_json",
]

Variant = Literal["base", "adapted"]
VARIANTS: tuple[Variant, ...] = get_args(Variant)
SETTINGS = (1, 2, 3)
COMPLETE = "_complete.json"
WHY = "why.json"
RL_MARKER = "checkpoints.json"
RL_UPDATES = ("update_32", "update_64")
SKILLS_MARKER = "library.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PASS = re.compile(r"pass([1-9][0-9]*)")
LOCK = ".lock"
# What a file system answers when it cannot lock files at all (some network and container volumes).
_NO_LOCKS = frozenset({errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS})
_HELD: dict[str, int] = {}  # the run directories whose lock this interpreter holds -> the pid that took it


class RunDirError(RuntimeError):
    """What the run directory holds cannot be trusted or does not fit what is asked of it; a person has to look."""


class RunLocked(RunDirError):
    """Another run holds the run directory."""


def _lock_key(root: Path) -> str:
    return os.path.realpath(root)


def _holder(path: Path) -> str:
    """Who holds the lock, as the lock file says. A pid alone is not enough: every container numbers its own processes,
    and every container of the image has the same host name, so the time the lock was taken tells two holders apart."""
    try:
        holder = json.loads(path.read_text(encoding="utf-8"))
        return f"pid {holder['pid']} on host {holder['host']}, holding it since {holder['since']}"
    except (OSError, ValueError, TypeError, KeyError):
        return "a run that has not written who it is yet"


def _component(label: str, name: str) -> str:
    if name in ("", ".", "..") or Path(name).name != name:
        raise ValueError(f"{label} must be a single path component, got {name!r}")
    return name


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_pass(variant: str, setting: int, k: int) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {', '.join(VARIANTS)}")
    if not _whole(setting) or setting not in SETTINGS:
        raise ValueError(f"setting must be one of {', '.join(map(str, SETTINGS))}, not {setting!r}")
    if not _whole(k) or k < 1:
        raise ValueError(f"a pass is numbered from 1, not {k!r}")


def now() -> str:
    """The local time, to the second, as every record of a run writes it."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` whole or not at all."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def _read_object(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RunDirError(f"{path} ({what}) cannot be read: {error}") from None
    if not isinstance(data, dict):
        raise RunDirError(f"{path} ({what}) is not a JSON object")
    return data


def _differences(recorded: Any, current: Any, where: str = "") -> list[str]:
    """The dotted paths at which two JSON values differ."""
    if isinstance(recorded, dict) and isinstance(current, dict):
        out = []
        for key in sorted(recorded.keys() | current.keys(), key=str):
            path = f"{where}.{key}" if where else str(key)
            if key not in recorded or key not in current:
                out.append(path)
            else:
                out += _differences(recorded[key], current[key], path)
        return out
    return [] if recorded == current else [where or "(the whole configuration)"]


@dataclass(frozen=True)
class RunDir:
    """``outputs/<model>/<task>/<mechanism>/``: its layout, and the state of the passes and the adaptation in it."""

    root: Path

    @classmethod
    def for_combo(cls, outputs: Path, model: str, task: str, mechanism: str) -> RunDir:
        return cls(
            Path(outputs) / _component("model", model) / _component("task", task) / _component("mechanism", mechanism)
        )

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def adapt_dir(self) -> Path:
        return self.root / "adapt"

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.json"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def void_dir(self) -> Path:
        return self.eval_dir / "_void"

    @property
    def topup_path(self) -> Path:
        return self.eval_dir / "topup.json"

    # --- evaluation passes ---------------------------------------------------------------------

    def pass_dir(self, variant: Variant, setting: int, k: int) -> Path:
        _check_pass(variant, setting, k)
        return self.eval_dir / variant / f"setting{setting}" / f"pass{k}"

    def episode_path(self, variant: Variant, setting: int, k: int, item_id: str) -> Path:
        return self.pass_dir(variant, setting, k) / f"{_component('item', item_id)}.json"

    def is_complete(self, variant: Variant, setting: int, k: int) -> bool:
        return (self.pass_dir(variant, setting, k) / COMPLETE).exists()

    def mark_complete(self, variant: Variant, setting: int, k: int, verdict: Mapping[str, Any]) -> Path:
        """Record that the pass is a valid measurement: ``verdict`` (whose ``valid`` must be True), with the time.

        Written last, and once: a finished pass is never written again.
        """
        directory = self.pass_dir(variant, setting, k)
        if verdict.get("valid") is not True:
            raise ValueError(f"only a valid pass is marked complete; {directory} has valid={verdict.get('valid')!r}")
        if not directory.is_dir():
            raise RunDirError(f"the pass {directory} does not exist")
        marker = directory / COMPLETE
        if marker.exists():
            raise RunDirError(f"{directory} is already complete; a finished pass is never written again")
        write_json(marker, {**verdict, "completed_at": now()})
        return marker

    def pass_record(self, variant: Variant, setting: int, k: int) -> dict[str, Any]:
        """What ``mark_complete`` recorded of a finished pass."""
        directory = self.pass_dir(variant, setting, k)
        marker = directory / COMPLETE
        if not marker.exists():
            raise RunDirError(f"the pass {directory} is not complete")
        record = _read_object(marker, "the record of a finished pass")
        if record.get("valid") is not True:
            raise RunDirError(f"{marker} does not record a valid pass")
        return record

    def complete_passes(self, variant: Variant, setting: int) -> list[int]:
        """The numbers of the complete passes of ``variant`` in ``setting``, in order."""
        _check_pass(variant, setting, 1)
        directory = self.eval_dir / variant / f"setting{setting}"
        if not directory.is_dir():
            return []
        found = [
            int(match[1])
            for path in directory.iterdir()
            if (match := _PASS.fullmatch(path.name)) is not None and (path / COMPLETE).is_file()
        ]
        return sorted(found)

    def void(
        self, variant: Variant, setting: int, k: int, reason: str, details: Mapping[str, Any] | None = None
    ) -> Path:
        """Move an unfinished pass to ``eval/_void/<variant>-setting<s>-pass<k>-<time>/``, with ``why.json``.

        A complete pass is never voided.
        """
        directory = self.pass_dir(variant, setting, k)
        if not directory.is_dir():
            raise RunDirError(f"there is no pass {directory} to void")
        if (directory / COMPLETE).exists():
            raise RunDirError(f"{directory} is complete; a finished pass is never voided")
        stamp = time.strftime("%Y%m%dT%H%M%S")
        name = f"{variant}-setting{setting}-pass{k}-{stamp}"
        target, number = self.void_dir / name, 2
        while target.exists():
            target, number = self.void_dir / f"{name}-{number}", number + 1
        self.void_dir.mkdir(parents=True, exist_ok=True)
        directory.rename(target)
        why = {
            "reason": reason,
            **(details or {}),
            "pass": {"variant": variant, "setting": setting, "pass": k},
            "voided_at": now(),
        }
        write_json(target / WHY, why)
        return target

    def voids(self, variant: Variant, setting: int, k: int) -> list[Path]:
        """The voided attempts at a pass, oldest first."""
        _check_pass(variant, setting, k)
        if not self.void_dir.is_dir():
            return []
        name = re.compile(rf"{variant}-setting{setting}-pass{k}-(?P<stamp>\d{{8}}T\d{{6}})(-(?P<number>\d+))?")
        found = []
        for path in self.void_dir.iterdir():
            match = name.fullmatch(path.name)
            if match:
                found.append((match["stamp"], int(match["number"] or 1), path))
        return [path for _, _, path in sorted(found)]

    # --- the adaptation ------------------------------------------------------------------------

    def adapt_done(self, mechanism: str) -> bool:
        """Whether the adaptation by ``mechanism`` finished: its completion record is there and fits what it records.

        False when there is no record. A record that is malformed, or a library that changed after its evolution
        finished, raises ``RunDirError``: such an adaptation is neither to be run again nor to be evaluated.
        """
        if mechanism == "rl":
            marker = self.adapt_dir / RL_MARKER
            if not marker.exists():
                return False
            data = _read_object(marker, "the checkpoints of the RL adaptation")
            bad = [key for key in RL_UPDATES if not isinstance(data.get(key), str) or not data[key]]
            if bad:
                raise RunDirError(f"{marker} has no checkpoint path for {', '.join(bad)}")
            return True
        if mechanism == "skills":
            marker = self.adapt_dir / SKILLS_MARKER
            if not marker.exists():
                return False
            data = _read_object(marker, "the record of the skill evolution")
            library, sha = data.get("library"), data.get("sha")
            problems = []
            if not isinstance(library, str) or library in ("", ".", "..") or Path(library).name != library:
                problems.append(f"library {library!r} is not a directory name")
            if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
                problems.append(f"sha {sha!r} is not a sha256")
            for key in ("evolutions", "skills"):
                if not _whole(data.get(key)) or data[key] < 0:
                    problems.append(f"{key} {data.get(key)!r} is not a count")
            if problems:
                raise RunDirError(f"{marker} is malformed: {'; '.join(problems)}")
            assert isinstance(library, str) and isinstance(sha, str)
            directory = self.adapt_dir / library
            if not directory.is_dir():
                raise RunDirError(f"{marker} records a finished library, but there is no library at {directory}")
            found = lib_sha(directory)
            if found != sha:
                raise RunDirError(
                    f"the library {directory} has changed since its evolution finished: its sha256 is {found}, "
                    f"{marker} records {sha}"
                )
            return True
        raise ValueError(f"unknown mechanism {mechanism!r}; expected rl or skills")

    # --- one run at a time ----------------------------------------------------------------------

    @contextmanager
    def lock(self) -> Iterator[RunDir]:
        """Hold the run directory while the block runs: an exclusive ``flock`` on ``<run>/.lock``, which says who holds
        it (pid, host and the time it was taken).

        Raises ``RunLocked`` naming the holder when another run has it, another process or another hold in this one;
        it never waits. A file system that cannot lock files is a ``RunDirError``. The lock is released when the block
        ends, however it ends, and by the system when the process dies.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / LOCK
        key = _lock_key(self.root)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            raise RunLocked(
                f"{self.root} is in use by another run ({_holder(path)}); a second run in the same directory would "
                "void the passes the first is running. Wait for it to end, or give this run another output directory."
            ) from None
        except OSError as error:
            os.close(descriptor)
            if error.errno in _NO_LOCKS:
                raise RunDirError(
                    f"{self.root}: the file system cannot lock files ({os.strerror(error.errno or 0)}). The outputs "
                    "volume must support file locks (flock): without them two runs could work in one directory and "
                    "void each other's passes."
                ) from error
            raise
        except BaseException:
            os.close(descriptor)
            raise
        try:  # from here on the flock is held: whatever fails, the finally releases it
            holder = {"pid": os.getpid(), "host": socket.gethostname(),
                      "since": datetime.now().astimezone().isoformat(timespec="microseconds")}  # fmt: skip
            os.ftruncate(descriptor, 0)
            os.write(descriptor, (json.dumps(holder) + "\n").encode())
            _HELD[key] = os.getpid()
            yield self
        finally:
            _HELD.pop(key, None)
            os.ftruncate(descriptor, 0)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def holds_lock(self) -> bool:
        """Whether this process holds the directory's lock (``lock``)."""
        return _HELD.get(_lock_key(self.root)) == os.getpid()

    def require_lock(self) -> None:
        """Raise ``RunDirError`` unless this process holds the directory's lock."""
        if not self.holds_lock():
            raise RunDirError(
                f"{self.root} is not locked by this process: a run holds RunDir.lock() while it works in the "
                "directory, so that no other run can void its passes"
            )

    # --- the configuration ---------------------------------------------------------------------

    def read_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise RunDirError(f"{self.root} has no configuration ({self.config_path.name})")
        return _read_object(self.config_path, "the configuration of the run")

    def write_config(self, config: Mapping[str, Any]) -> None:
        """Record the configuration the combination runs under; a run that has one already must run under the same.

        ``config`` is compared as JSON, so a tuple and a list with the same values are the same.
        """
        current = json.loads(json.dumps(config, ensure_ascii=False))
        if self.config_path.exists():
            changed = _differences(self.read_config(), current)
            if changed:
                raise RunDirError(
                    f"{self.config_path} records another configuration; it differs at {', '.join(changed[:10])}"
                    f"{' ...' if len(changed) > 10 else ''}. A run goes on only under the configuration it started "
                    "with: restore that configuration, or give this one another output directory."
                )
            return
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.config_path, current)
