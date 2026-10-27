"""The processes a run starts besides the agent: the model server, the RL trainer, MetaClaw and their checking tools.

    model_server_env      the model server's whole environment (and that of an evaluation's checkpoint check)
    trainer_env           the RL trainer's whole environment (and that of its checkpoint tools)
    metaclaw_env          the environment of a process of MetaClaw's Python: the proxy, the guards, the evolver check
    metaclaw_python       the Python MetaClaw is installed in, refused when it cannot be run
    starting, StartFailed a process that did not start
    stop_all              stop what was started, whatever is propagating

Each environment is built from nothing. It holds what its process needs, and of this process's environment only PATH,
LANG and TZ, the settings a host may need to reach Tinker, Hugging Face or the evolver (``NETWORK``: proxies and
certificate bundles) and, for the model server and the trainer, PYTHONPATH (where Python finds this package). The
OpenClaw gateway and the agent get none of this: ``agent.openclaw`` builds theirs from an allowlist of its own.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from evolve_ipi.agent.openclaw import OpenClawError
from evolve_ipi.config import ConfigError, Secrets
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import ProcessError

__all__ = [
    "NETWORK",
    "PASSED_ON",
    "StartFailed",
    "Stoppable",
    "metaclaw_env",
    "metaclaw_python",
    "model_server_env",
    "starting",
    "stop_all",
    "stop_failures",
    "trainer_env",
]

logger = logging.getLogger(__name__)

PASSED_ON = ("PATH", "LANG", "TZ")
NETWORK = tuple(
    name
    for upper in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
    for name in (upper, upper.lower())
)
_PROXIES = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")
_NO_PROXY = ("NO_PROXY", "no_proxy")
# The model server and MetaClaw's proxy talk to each other on the loopback interface, never through a proxy.
LOCAL_HOSTS = ("127.0.0.1", "localhost")
_STOP_FAILED = "stopping a process failed as well: "


# --- the environments -------------------------------------------------------------------------------------------------


def _inherited(*, pythonpath: bool) -> dict[str, str]:
    """What a child takes from this process's environment; a variable that is set but empty is not passed on.

    When a proxy is passed on, the local hosts are added to both spellings of NO_PROXY, which then hold the same list:
    Python's HTTP clients read the lower-case one first.
    """
    names = (*PASSED_ON, *(("PYTHONPATH",) if pythonpath else ()), *NETWORK)
    env = {name: value for name in names if (value := os.environ.get(name))}
    if any(name in env for name in _PROXIES):
        hosts: list[str] = []
        for entry in (*(env.get(name, "") for name in _NO_PROXY), ",".join(LOCAL_HOSTS)):
            for host in (part.strip() for part in entry.split(",")):
                if host and host not in hosts:
                    hosts.append(host)
        env.update(dict.fromkeys(_NO_PROXY, ",".join(hosts)))
    return env


def _on_tinker(paths: RuntimePaths, secrets: Secrets, what: str) -> dict[str, str]:
    if not secrets.tinker_api_key:
        raise ConfigError(
            f"TINKER_API_KEY is required: {what} on Tinker; set it in the environment or the .env file (see "
            ".env.example)"
        )
    env = _inherited(pythonpath=True)
    env.update(HOME=str(paths.service_home), HF_HOME=str(paths.hf_home), TINKER_API_KEY=secrets.tinker_api_key)
    if secrets.hf_token:
        env["HF_TOKEN"] = secrets.hf_token
    return env


def model_server_env(paths: RuntimePaths, secrets: Secrets) -> dict[str, str]:
    """The model server's whole environment: Tinker's key, a home of its own, Hugging Face's cache (and token, when one
    is set). Nothing of the evolver."""
    return _on_tinker(paths, secrets, "the model server samples every model")


def trainer_env(paths: RuntimePaths, secrets: Secrets) -> dict[str, str]:
    """The trainer's whole environment: the model server's, without Weights & Biases."""
    env = _on_tinker(paths, secrets, "the trainer trains the model")
    env["WANDB_DISABLED"] = "1"
    return env


def metaclaw_env(paths: RuntimePaths) -> dict[str, str]:
    """The environment of a process of MetaClaw's Python: HOME is the service home, never the agent's (``~/.metaclaw``
    is where the agent's File-Create documents point), and Hugging Face is never contacted. No key: an evolving proxy
    and the evolver check add the evolver's own."""
    env = _inherited(pythonpath=False)
    env.update(
        HOME=str(paths.service_home),
        PYTHONUNBUFFERED="1",
        PYTHONSAFEPATH="1",  # the script's own directory holds modules of this package, which must not shadow others
        HF_HUB_OFFLINE="1",  # the served model name is no Hugging Face id: the tokenizer load fails at once
        TRANSFORMERS_OFFLINE="1",
    )
    return env


def metaclaw_python(paths: RuntimePaths) -> Path:
    """The Python of the environment MetaClaw is installed in; ``ConfigError`` when it is not a program that can run."""
    python = paths.metaclaw_python
    if not (python.is_file() and os.access(python, os.X_OK)):
        raise ConfigError(
            f"MetaClaw's Python {python} does not exist or cannot be run; set EVOLVE_IPI_METACLAW_PYTHON to the "
            "python of the environment MetaClaw is installed in (docker/requirements-metaclaw.txt)"
        )
    return python


# --- starting and stopping --------------------------------------------------------------------------------------------


class StartFailed(RuntimeError):
    """A process did not start: a gated tokenizer without HF_TOKEN, a checkpoint that does not load, another OpenClaw
    version. Starting it again would fail the same way, so what needed it stops at once instead of starting over."""


@contextmanager
def starting(what: str) -> Iterator[None]:
    """Turn a failure to start ``what`` into ``StartFailed``, with the failure as its cause."""
    try:
        yield
    except (ProcessError, OpenClawError) as error:
        raise StartFailed(f"the {what} did not start: {type(error).__name__}: {error}") from error


class Stoppable(Protocol):
    def stop(self) -> None: ...


def stop_all(started: Sequence[Stoppable], propagating: BaseException | None = None) -> None:
    """Stop everything ``started``, the last started first, trying each one whatever the others do.

    With nothing ``propagating``, the first failure to stop is raised. While an exception propagates it stays the one
    that does: a failure to stop is logged and attached to it as a note, so that a Ctrl-C is never turned into a
    failure that would start the work again. A Ctrl-C during the stopping itself is raised in any case.
    """
    errors: list[BaseException] = []
    for thing in reversed(started):
        try:
            thing.stop()
        except BaseException as error:  # noqa: BLE001 -- every process is stopped before anything is raised
            errors.append(error)
    interrupts = [failure for failure in errors if not isinstance(failure, Exception)]
    if propagating is None or (interrupts and isinstance(propagating, Exception)):
        if errors:
            raise (interrupts or errors)[0]
        return
    for failure in errors:
        logger.error("stopping a process failed while %s propagated: %s", type(propagating).__name__, failure)
        propagating.add_note(f"{_STOP_FAILED}{type(failure).__name__}: {failure}")


def stop_failures(error: BaseException) -> dict[str, list[str]]:
    """The failures to stop that ``stop_all`` attached to ``error``, for a void's ``why.json``."""
    notes = getattr(error, "__notes__", ())
    failed = [note.removeprefix(_STOP_FAILED) for note in notes if note.startswith(_STOP_FAILED)]
    return {"stop_failures": failed} if failed else {}
