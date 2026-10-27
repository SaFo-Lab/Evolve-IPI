"""Fixtures shared by the whole test suite."""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# A test must never reach a paid service with a developer's real credentials. A test that needs
# one of these sets a fake value itself.
SECRET_ENV_VARS = (
    "TINKER_API_KEY",
    "EVOLVER_BASE_URL",
    "EVOLVER_API_KEY",
    "EVOLVER_MODEL",
    "HF_TOKEN",
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    return repo_root / "configs"


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SECRET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def free_port() -> Callable[[], int]:
    """A function that returns a local port nothing listens on when it is asked."""
    return _free_port


@pytest.fixture
def listener() -> Iterator[int]:
    """A port this test process listens on until the test ends."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()
