"""The processes a run starts besides the agent: their environments, the MetaClaw Python, failed starts and stops.

The agent's own environment (the OpenClaw gateway and every ``openclaw`` call) is pinned here too, from the other side:
nothing that a child process of the run is given may reach it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from evolve_ipi import children
from evolve_ipi.agent import openclaw
from evolve_ipi.agent.openclaw import OpenClawError, gateway_env, profile_for, write_profile
from evolve_ipi.config import ConfigError, Secrets
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import ProcessError

TINKER = Secrets(tinker_api_key="tinker-test-key")
NETWORK = {
    "HTTP_PROXY": "http://proxy.example:3128",
    "https_proxy": "http://proxy.example:3129",
    "SSL_CERT_FILE": "/etc/ssl/corporate.pem",
    "REQUESTS_CA_BUNDLE": "/etc/ssl/corporate-bundle.pem",
}
# What the parent of a run may hold that no child is given: keys, the image's own settings, uv's and Hugging Face's.
FOREIGN = {
    "TINKER_API_KEY": "tinker-secret",
    "EVOLVER_API_KEY": "evolver-secret",
    "OPENAI_API_KEY": "openai-secret",
    "OPENCLAW_GATEWAY_TOKEN": "gateway-secret",
    "EVOLVE_IPI_IMAGE": "1",
    "HF_HOME": "/somewhere/hf",
    "UV_CACHE_DIR": "/somewhere/uv",
    "UV_PYTHON": "3.12",
    "VIRTUAL_ENV": "/somewhere/venv",
}


@pytest.fixture
def paths(tmp_path: Path) -> RuntimePaths:
    home = tmp_path / "home"
    home.mkdir()
    return RuntimePaths(
        root=tmp_path / "runtime", agent_home=home, outputs=tmp_path / "outputs", metaclaw_python=Path(sys.executable)
    )


@pytest.fixture
def parent(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """The environment of the run itself: PATH, LANG, TZ and PYTHONPATH, and everything a child must not get."""
    for name in (*children.NETWORK, "PYTHONPATH"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {**FOREIGN, "LANG": "C.UTF-8", "TZ": "America/New_York", "PYTHONPATH": "/src"}.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


# --- the environments -------------------------------------------------------------------------------------------------


def test_the_model_server_gets_tinker_its_homes_and_this_package(
    paths: RuntimePaths, parent: pytest.MonkeyPatch
) -> None:
    env = children.model_server_env(paths, TINKER)
    assert env == {
        "PATH": children.os.environ["PATH"],
        "LANG": "C.UTF-8",
        "TZ": "America/New_York",
        "PYTHONPATH": "/src",
        "HOME": str(paths.service_home),
        "HF_HOME": str(paths.hf_home),
        "TINKER_API_KEY": "tinker-test-key",
    }
    with_token = children.model_server_env(paths, Secrets(tinker_api_key="tinker-test-key", hf_token="hf-test-token"))
    assert set(with_token) - set(env) == {"HF_TOKEN"} and with_token["HF_TOKEN"] == "hf-test-token"


def test_the_trainer_gets_what_the_model_server_gets_and_no_wandb(
    paths: RuntimePaths, parent: pytest.MonkeyPatch
) -> None:
    env = children.trainer_env(paths, TINKER)
    assert env == {**children.model_server_env(paths, TINKER), "WANDB_DISABLED": "1"}


def test_metaclaw_gets_neither_a_key_nor_this_package(paths: RuntimePaths, parent: pytest.MonkeyPatch) -> None:
    env = children.metaclaw_env(paths)
    assert env == {
        "PATH": children.os.environ["PATH"],
        "LANG": "C.UTF-8",
        "TZ": "America/New_York",
        "HOME": str(paths.service_home),
        "PYTHONUNBUFFERED": "1",
        "PYTHONSAFEPATH": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


@pytest.mark.parametrize("builder", ["model_server_env", "trainer_env"])
def test_a_child_on_tinker_needs_the_key(paths: RuntimePaths, builder: str) -> None:
    with pytest.raises(ConfigError, match="TINKER_API_KEY"):
        getattr(children, builder)(paths, Secrets(hf_token="hf-test-token"))


def test_every_child_reaches_the_network_as_the_run_does(paths: RuntimePaths, parent: pytest.MonkeyPatch) -> None:
    for name, value in {**NETWORK, "NO_PROXY": "corp.example", "no_proxy": "intranet.example,localhost"}.items():
        parent.setenv(name, value)
    built = [
        children.model_server_env(paths, TINKER),
        children.trainer_env(paths, TINKER),
        children.metaclaw_env(paths),
    ]
    for env in built:
        assert {name: env[name] for name in NETWORK} == NETWORK
        # The model server is local: MetaClaw reaches it through httpx, which honours the proxy variables.
        local = "corp.example,intranet.example,localhost,127.0.0.1"
        assert (env["NO_PROXY"], env["no_proxy"]) == (local, local)


def test_without_a_proxy_nothing_is_added(paths: RuntimePaths, parent: pytest.MonkeyPatch) -> None:
    parent.setenv("SSL_CERT_FILE", "/etc/ssl/corporate.pem")
    env = children.metaclaw_env(paths)
    assert env["SSL_CERT_FILE"] == "/etc/ssl/corporate.pem"
    assert "NO_PROXY" not in env and "no_proxy" not in env


def test_a_proxy_alone_gets_the_local_hosts_in_both_spellings(paths: RuntimePaths, parent: pytest.MonkeyPatch) -> None:
    parent.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    env = children.model_server_env(paths, TINKER)
    assert (env["NO_PROXY"], env["no_proxy"]) == ("127.0.0.1,localhost", "127.0.0.1,localhost")


def test_an_empty_variable_is_not_passed_on(paths: RuntimePaths, parent: pytest.MonkeyPatch) -> None:
    parent.setenv("HTTP_PROXY", "")
    parent.setenv("PYTHONPATH", "")
    env = children.model_server_env(paths, TINKER)
    assert "HTTP_PROXY" not in env and "PYTHONPATH" not in env and "NO_PROXY" not in env


def test_the_agent_gets_nothing_a_child_of_the_run_gets(
    tmp_path: Path, paths: RuntimePaths, parent: pytest.MonkeyPatch
) -> None:
    """The gateway's environment is an allowlist: the agent's tools run in it."""
    for name, value in {**NETWORK, "NO_PROXY": "corp.example"}.items():
        parent.setenv(name, value)
    profile = profile_for("rl", "file", paths, 20000, 20010)
    write_profile(profile)
    parent.setattr(openclaw, "_find", lambda program, search_path=None: tmp_path / "bin" / program)
    env = gateway_env(profile)
    assert set(env) == {
        "PATH", "HOME", "LANG", "TZ", "SHELL", "NODE_OPTIONS", "OPENCLAW_CONFIG_PATH", "OPENCLAW_STATE_DIR",
        "OPENCLAW_GATEWAY_PORT", "OPENCLAW_GATEWAY_URL", "OPENCLAW_GATEWAY_TOKEN",
    }  # fmt: skip
    given = {*FOREIGN, *NETWORK, "NO_PROXY", "no_proxy", "PYTHONPATH", "HF_TOKEN", "WANDB_DISABLED"}
    given.discard("OPENCLAW_GATEWAY_TOKEN")  # the gateway's own, read from the profile: checked by value below
    assert not given & set(env) and not any(name.startswith(("UV_", "EVOLVE_IPI_", "HF_")) for name in env)
    assert env["HOME"] == str(paths.agent_home) and env["LANG"] == "C.UTF-8"
    assert env["OPENCLAW_GATEWAY_TOKEN"] == profile.token_path.read_text(encoding="utf-8").strip()
    leaked = [value for value in (*FOREIGN.values(), *NETWORK.values()) if len(value) > 1 and value in env.values()]
    assert leaked == []


# --- the MetaClaw Python ----------------------------------------------------------------------------------------------


def test_the_metaclaw_python_must_be_a_program(tmp_path: Path, paths: RuntimePaths) -> None:
    assert children.metaclaw_python(paths) == Path(sys.executable)
    for python in (tmp_path / "missing" / "python", tmp_path):
        absent = RuntimePaths(root=paths.root, agent_home=paths.agent_home, metaclaw_python=python)
        with pytest.raises(ConfigError, match=rf"MetaClaw's Python {python} .*EVOLVE_IPI_METACLAW_PYTHON"):
            children.metaclaw_python(absent)
    script = tmp_path / "python"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o644)
    with pytest.raises(ConfigError, match="cannot be run"):
        children.metaclaw_python(RuntimePaths(root=paths.root, agent_home=paths.agent_home, metaclaw_python=script))


# --- starting and stopping --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("error", [ProcessError("exited with status 1"), OpenClawError("2026.4.0 is installed")])
def test_a_process_that_does_not_start(error: Exception) -> None:
    match = f"^the gateway did not start: {type(error).__name__}: "
    with pytest.raises(children.StartFailed, match=match) as raised, children.starting("gateway"):
        raise error
    assert raised.value.__cause__ is error
    assert not isinstance(raised.value, (ProcessError, OpenClawError)), "a start failure is never restarted"


def test_starting_lets_anything_else_through() -> None:
    with pytest.raises(KeyError), children.starting("gateway"):
        raise KeyError("not a start failure")


class Stub:
    def __init__(self, name: str, log: list[str], error: BaseException | None = None) -> None:
        self.name, self.log, self.error = name, log, error

    def stop(self) -> None:
        self.log.append(self.name)
        if self.error is not None:
            raise self.error


def test_stop_all_stops_the_last_started_first_and_raises_the_first_failure() -> None:
    log: list[str] = []
    started = [Stub("gateway", log), Stub("server", log, ProcessError("could not stop server")), Stub("proxy", log)]
    with pytest.raises(ProcessError, match="could not stop server"):
        children.stop_all(started)
    assert log == ["proxy", "server", "gateway"]


def test_while_an_exception_propagates_a_failure_to_stop_is_a_note(caplog: pytest.LogCaptureFixture) -> None:
    log: list[str] = []
    interrupt = KeyboardInterrupt()
    started = [Stub("server", log, ProcessError("could not stop server")), Stub("proxy", log)]
    with caplog.at_level(logging.ERROR, logger="evolve_ipi.children"):
        children.stop_all(started, interrupt)
    assert log == ["proxy", "server"]
    assert children.stop_failures(interrupt) == {"stop_failures": ["ProcessError: could not stop server"]}
    assert "KeyboardInterrupt" in caplog.text and "could not stop server" in caplog.text
    assert children.stop_failures(KeyboardInterrupt()) == {}


def test_a_ctrl_c_while_stopping_wins_over_an_error() -> None:
    log: list[str] = []
    started = [Stub("server", log, KeyboardInterrupt()), Stub("proxy", log)]
    with pytest.raises(KeyboardInterrupt):
        children.stop_all(started, ProcessError("the pass failed"))
    assert log == ["proxy", "server"]
