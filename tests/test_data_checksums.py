"""``data/SHA256SUMS``: the sha256 of every file under ``data/``, checked before a command starts anything."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import pytest

from evolve_ipi import tasks
from evolve_ipi.config import ConfigError

LINE = re.compile(r"([0-9a-f]{64})  (\S+)")


@pytest.fixture
def data_copy(tmp_path: Path) -> Path:
    target = tmp_path / "data"
    shutil.copytree(tasks.DATA_DIR, target)
    return target


def test_the_sums_list_every_file_of_the_data_sorted_in_sha256sum_format() -> None:
    sums = tasks.DATA_DIR / tasks.DATA_SUMS
    lines = sums.read_text(encoding="utf-8").splitlines()
    names = [LINE.fullmatch(line).group(2) for line in lines]  # type: ignore[union-attr]
    files = sorted(path.relative_to(tasks.DATA_DIR).as_posix() for path in tasks.DATA_DIR.rglob("*") if path.is_file())
    assert names == sorted(names) == [name for name in files if name != tasks.DATA_SUMS]
    for line in lines:
        digest, name = LINE.fullmatch(line).groups()  # type: ignore[union-attr]
        assert hashlib.sha256((tasks.DATA_DIR / name).read_bytes()).hexdigest() == digest, name
    assert sums.read_bytes().endswith(b"\n") and b"\r" not in sums.read_bytes()


def test_the_data_as_published_passes_and_the_digest_is_that_of_the_sums() -> None:
    expected = hashlib.sha256((tasks.DATA_DIR / tasks.DATA_SUMS).read_bytes()).hexdigest()
    assert tasks.verify_data() == expected == tasks.verify_data(tasks.DATA_DIR)


def test_one_changed_byte_is_named(data_copy: Path) -> None:
    path = data_copy / "tasks" / "chat-invite" / "test.jsonl"
    data = bytearray(path.read_bytes())
    data[100] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ConfigError, match=r"tasks/chat-invite/test\.jsonl has changed") as raised:
        tasks.verify_data(data_copy)
    assert "SHA256SUMS" in str(raised.value) and "orders/" not in str(raised.value)


def test_a_missing_and_an_unlisted_file_are_named(data_copy: Path) -> None:
    (data_copy / "orders" / "skills" / "email-invite.json").unlink()
    (data_copy / "tasks" / "file-create" / "extra.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigError) as raised:
        tasks.verify_data(data_copy)
    message = str(raised.value)
    assert "orders/skills/email-invite.json is missing" in message
    assert "tasks/file-create/extra.jsonl is not listed" in message


def test_the_sums_themselves_must_be_there_and_well_formed(data_copy: Path) -> None:
    sums = data_copy / tasks.DATA_SUMS
    text = sums.read_text(encoding="utf-8")
    sums.write_text(text.replace("  orders/rl/chat-invite.json", " orders/rl/chat-invite.json"), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"SHA256SUMS line 1 is not '<sha256>  <path>'"):
        tasks.verify_data(data_copy)
    sums.write_text(text + text.splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="lists orders/rl/chat-invite.json twice"):
        tasks.verify_data(data_copy)
    sums.unlink()
    with pytest.raises(ConfigError, match="cannot read"):
        tasks.verify_data(data_copy)


@pytest.mark.parametrize("name", ["../outside.json", "/etc/passwd", "tasks/./x.json", "SHA256SUMS"])
def test_a_listed_path_must_be_a_file_below_the_data(data_copy: Path, name: str) -> None:
    sums = data_copy / tasks.DATA_SUMS
    sums.write_text(sums.read_text(encoding="utf-8") + f"{'0' * 64}  {name}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a file below"):
        tasks.verify_data(data_copy)


def test_the_data_directory_is_read_when_the_check_runs(monkeypatch: pytest.MonkeyPatch, data_copy: Path) -> None:
    (data_copy / "tasks" / "drive-create" / "task.yaml").write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(tasks, "DATA_DIR", data_copy)
    with pytest.raises(ConfigError, match="task.yaml has changed"):
        tasks.verify_data()
