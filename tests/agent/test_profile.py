"""The OpenClaw profile: the config built for each line, the files on disk, and the environment of its processes."""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any

import pytest
from fake_cli import GATEWAY_PORT, POLICY_PORT, PROXY_PORT, FakeCli

from evolve_ipi.agent import openclaw
from evolve_ipi.agent.openclaw import (
    GATEWAY_SPAN,
    OpenClawError,
    Profile,
    build_config,
    check_profile,
    gateway_env,
    profile_for,
    write_profile,
)
from evolve_ipi.config import ConfigError, load_configs
from evolve_ipi.paths import Line, RuntimePaths, agent_id
from evolve_ipi.thirdparty import RL_HEADERS_PLUGIN_DIR

ZERO_COST = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
SECRETS = ("TINKER_API_KEY", "EVOLVER_API_KEY", "EVOLVER_MODEL", "OPENAI_API_KEY", "HF_TOKEN", "AZURE_OPENAI_API_KEY")


def golden(
    profile_dir: Path,
    workspace: Path,
    *,
    agent: str,
    provider: str,
    model: str,
    label: str,
    provider_port: int,
    gateway_port: int,
) -> dict[str, Any]:
    """The configuration a profile must have, written out in full: the JSON OpenClaw is started with."""
    return {
        "models": {
            "providers": {
                provider: {
                    "baseUrl": f"http://127.0.0.1:{provider_port}/v1",
                    "apiKey": "dummy",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": model,
                            "name": label,
                            "api": "openai-completions",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": ZERO_COST,
                            "contextWindow": 32768,
                            "maxTokens": 4096,
                        }
                    ],
                }
            }
        },
        "agents": {
            "defaults": {"compaction": {"mode": "safeguard"}, "sandbox": {"mode": "off"}},
            "list": [
                {"id": "main", "workspace": f"{profile_dir}/workspace-main"},
                {
                    "id": agent,
                    "name": agent,
                    "workspace": str(workspace),
                    "agentDir": f"{profile_dir}/state/agents/{agent}",
                    "model": f"{provider}/{model}",
                },
            ],
        },
        "commands": {"native": "auto", "nativeSkills": "auto", "restart": True, "ownerDisplay": "raw"},
        "gateway": {
            "mode": "local",
            "bind": "loopback",
            "controlUi": {"allowedOrigins": [f"http://127.0.0.1:{gateway_port}"]},
        },
        "plugins": {
            "entries": {"rl-training-headers": {"enabled": True}},
            "installs": {
                "rl-training-headers": {
                    "source": "path",
                    "sourcePath": str(RL_HEADERS_PLUGIN_DIR),
                    "installPath": f"{profile_dir}/state/extensions/rl-training-headers",
                    "version": "1.0.0",
                }
            },
        },
    }


def same_json(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Equal as values and as text: the keys come in the order OpenClaw's config has always had here."""
    assert actual == expected
    assert json.dumps(actual, indent=2) == json.dumps(expected, indent=2)


# --- the config ------------------------------------------------------------------------------


def test_build_config_rl_exact(paths: RuntimePaths, make_profile: Callable[..., Profile]) -> None:
    p = make_profile("rl", "file", gateway_port=GATEWAY_PORT, provider_port=POLICY_PORT)
    same_json(
        build_config(p),
        golden(
            paths.root / "openclaw" / "rl-file",
            paths.root / "workspaces" / "rl-file" / "workspace-rl",
            agent="rl",
            provider="rl",
            model="policy",
            label="RL policy model",
            provider_port=POLICY_PORT,
            gateway_port=GATEWAY_PORT,
        ),
    )
    assert (p.agent_id, p.provider, p.model_id, p.model_ref) == ("rl", "rl", "policy", "rl/policy")


def test_build_config_skills_exact(paths: RuntimePaths, make_profile: Callable[..., Profile]) -> None:
    p = make_profile("skills", "service", gateway_port=GATEWAY_PORT + 1, provider_port=PROXY_PORT)
    same_json(
        build_config(p),
        golden(
            paths.root / "openclaw" / "skills-service",
            paths.root / "workspaces" / "skills-service" / "workspace-mc",
            agent="mc",
            provider="metaclaw",
            model="qwen3.8-27b",
            label="MetaClaw proxy model",
            provider_port=PROXY_PORT,
            gateway_port=GATEWAY_PORT + 1,
        ),
    )
    assert (p.agent_id, p.provider, p.model_id, p.model_ref) == (
        "mc",
        "metaclaw",
        "qwen3.8-27b",
        "metaclaw/qwen3.8-27b",
    )


@pytest.mark.parametrize("line", ["rl", "skills"])
def test_the_config_follows_the_ports_and_the_paths(
    line: str, paths: RuntimePaths, make_profile: Callable[..., Profile]
) -> None:
    config = build_config(make_profile(line, "service", gateway_port=19000, provider_port=31000))
    (provider,) = config["models"]["providers"].values()
    assert provider["baseUrl"] == "http://127.0.0.1:31000/v1"
    assert config["gateway"]["controlUi"]["allowedOrigins"] == ["http://127.0.0.1:19000"]
    agent = config["agents"]["list"][1]
    assert Path(agent["workspace"]) == paths.workspace(line, "service")  # type: ignore[arg-type]
    assert Path(agent["agentDir"]).is_relative_to(paths.profile(line, "service"))  # type: ignore[arg-type]


def test_every_line_and_kind_has_directories_of_its_own(make_profile: Callable[..., Profile]) -> None:
    profiles = [make_profile(line, kind) for line in ("rl", "skills") for kind in ("file", "service")]
    assert len({p.dir for p in profiles}) == 4
    assert len({p.workspace for p in profiles}) == 4
    for p in profiles:
        assert not p.workspace.is_relative_to(p.dir), "the agent's workspace must not hold the OpenClaw state"
        assert p.home == profiles[0].home


@pytest.mark.parametrize(("line", "agent"), [("rl", "rl"), ("skills", "mc")])
def test_the_agent_of_a_profile_is_the_one_that_names_its_workspace(
    line: Line, agent: str, make_profile: Callable[..., Profile]
) -> None:
    """One mapping from line to agent serves the directory and the config."""
    p = make_profile(line, "service")
    assert p.agent_id == agent == agent_id(line)
    assert p.workspace.name == f"workspace-{agent}"
    assert p.workspace.parent.name == f"{line}-service"


def test_the_shipped_ports_make_valid_profiles(paths: RuntimePaths, configs_dir: Path) -> None:
    """The gateway takes its port and the five above it; the policy server, the proxy and the model server of
    configs/runtime.yaml are not among them."""
    ports = load_configs(configs_dir).ports
    rl = profile_for("rl", "file", paths, ports.gateway, ports.policy)
    skills = profile_for("skills", "service", paths, ports.gateway, ports.proxy)
    assert (rl.gateway_port, rl.provider_port) == (ports.gateway, ports.policy)
    assert (skills.gateway_port, skills.provider_port) == (ports.gateway, ports.proxy)
    assert not {ports.policy, ports.proxy, ports.model} & set(range(ports.gateway, ports.gateway + GATEWAY_SPAN))


def test_a_search_upward_from_the_workspace_finds_nothing_of_any_run(
    paths: RuntimePaths, make_profile: Callable[..., Profile]
) -> None:
    """The agent can run ``grep -R`` on ``..`` and on ``../..``. Everything a run writes is there by then: the profile
    with its transcripts, the gateway's log and token, the archived transcripts and the state of the services. None of
    it may be below either directory."""
    combos = list(product(("rl", "skills"), ("file", "service")))
    for line, kind in combos:
        p = make_profile(line, kind)
        p.workspace.mkdir(parents=True)
        write_profile(p)
        p.sessions_dir.mkdir(parents=True)
        (p.sessions_dir / "an-earlier-episode.jsonl").write_text("what the agent was asked before\n")
        (p.dir / "gateway.log").write_text("the log of the gateway\n")
        for state in (paths.service_state(line, kind), paths.transcript_archive(line, kind)):
            state.mkdir(parents=True)
            (state / "access.log").write_text("what the services were asked before\n")
    slots = {f"{line}-{kind}" for line, kind in combos}
    for line, kind in combos:
        workspace = make_profile(line, kind).workspace
        for above in (workspace.parent, workspace.parent.parent):
            below = {path.name for path in above.rglob("*")}
            assert below <= {*slots, "workspace-rl", "workspace-mc"}, sorted(below)
            assert not any(path.is_file() for path in above.rglob("*"))


def test_profile_for_refuses_a_workspace_inside_a_git_checkout(tmp_path: Path) -> None:
    (tmp_path / "checkout" / ".git").mkdir(parents=True)
    paths = RuntimePaths(
        root=tmp_path / "checkout" / "rt", agent_home=tmp_path, outputs=tmp_path, metaclaw_python=tmp_path
    )
    with pytest.raises(ConfigError, match="git checkout"):
        profile_for("rl", "file", paths, GATEWAY_PORT, POLICY_PORT)


@pytest.mark.parametrize(
    ("gateway_port", "provider_port"),
    [
        (0, POLICY_PORT),
        (GATEWAY_PORT, 0),
        (70000, POLICY_PORT),
        (GATEWAY_PORT, 70000),
        (65531, POLICY_PORT),
        (GATEWAY_PORT, GATEWAY_PORT + 2),
    ],
)
def test_profile_for_rejects_ports_that_cannot_work(gateway_port: int, provider_port: int, paths: RuntimePaths) -> None:
    """A gateway takes its port and the five above it, and the provider cannot sit among them."""
    with pytest.raises(ValueError, match="port"):
        profile_for("rl", "file", paths, gateway_port, provider_port)


def test_profile_for_needs_absolute_paths(tmp_path: Path) -> None:
    """The CLI runs in the agent's workspace, so a relative path in its environment would point somewhere else."""
    relative_root = RuntimePaths(root=Path("rt"), agent_home=tmp_path, outputs=tmp_path, metaclaw_python=tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        profile_for("rl", "file", relative_root, GATEWAY_PORT, POLICY_PORT)
    relative_home = RuntimePaths(
        root=tmp_path / "rt", agent_home=Path("home"), outputs=tmp_path, metaclaw_python=tmp_path
    )
    with pytest.raises(ValueError, match="absolute"):
        profile_for("rl", "file", relative_home, GATEWAY_PORT, POLICY_PORT)


def test_profile_for_rejects_an_unknown_line_or_kind(paths: RuntimePaths) -> None:
    with pytest.raises(ValueError, match="line"):
        profile_for("weights", "file", paths, GATEWAY_PORT, POLICY_PORT)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind"):
        profile_for("rl", "files", paths, GATEWAY_PORT, POLICY_PORT)  # type: ignore[arg-type]


# --- the profile on disk ---------------------------------------------------------------------


def plugin_files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_write_profile_writes_config_token_and_plugin(profile: Profile) -> None:
    write_profile(profile)
    text = (profile.dir / "openclaw.json").read_text(encoding="utf-8")
    assert text == json.dumps(build_config(profile), indent=2) + "\n"
    token = profile.token_path
    assert token == profile.dir / "gateway.token"
    assert len(token.read_text()) == 65 and token.read_text().endswith("\n")
    assert int(token.read_text(), 16) >= 0  # 64 hex digits
    assert stat.S_IMODE(token.stat().st_mode) == 0o600
    installed = profile.state_dir / "extensions" / "rl-training-headers"
    assert plugin_files(installed) == plugin_files(RL_HEADERS_PLUGIN_DIR)
    assert (profile.state_dir / "agents" / "rl").is_dir()
    assert (profile.dir / "workspace-main").is_dir()


def test_write_profile_leaves_the_agents_workspace_to_whoever_fills_it(
    make_profile: Callable[..., Profile],
) -> None:
    """OpenClaw sets up a workspace of its own, with a git repository, when it first sees an empty one."""
    p = make_profile()
    write_profile(p)
    assert not p.workspace.exists()


def test_write_profile_is_repeatable_and_keeps_the_token(profile: Profile) -> None:
    write_profile(profile)
    token = profile.token_path.read_text()
    stale = profile.state_dir / "extensions" / "rl-training-headers" / "stale.txt"
    stale.write_text("left over")
    profile.config_path.write_text("{}")
    write_profile(profile)
    assert profile.token_path.read_text() == token
    assert not stale.exists(), "the plugin directory is replaced, not added to"
    assert profile.config_path.read_text() == json.dumps(build_config(profile), indent=2) + "\n"


def test_write_profile_tightens_the_mode_of_an_existing_token(profile: Profile) -> None:
    write_profile(profile)
    profile.token_path.chmod(0o644)
    write_profile(profile)
    assert stat.S_IMODE(profile.token_path.stat().st_mode) == 0o600


@contextmanager
def umask(value: int) -> Iterator[None]:
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


def test_the_token_is_created_with_its_final_mode(profile: Profile, monkeypatch: pytest.MonkeyPatch) -> None:
    """Not made private afterwards: it is private from the moment it exists, whatever the umask."""
    monkeypatch.setattr(Path, "chmod", lambda self, mode, **kwargs: None)
    with umask(0):
        write_profile(profile)
    assert stat.S_IMODE(profile.token_path.stat().st_mode) == 0o600


def test_write_profile_names_a_missing_plugin_checkout(
    profile: Profile, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(openclaw, "RL_HEADERS_PLUGIN_DIR", tmp_path / "no-such-checkout")
    with pytest.raises(OpenClawError, match="submodule"):
        write_profile(profile)


def test_check_profile_passes_a_profile_that_was_just_written(profile: Profile) -> None:
    write_profile(profile)
    assert check_profile(profile) == []


def test_check_profile_finds_what_is_missing_or_edited(profile: Profile) -> None:
    assert any("openclaw.json" in problem for problem in check_profile(profile))  # nothing written yet
    write_profile(profile)

    config = json.loads(profile.config_path.read_text())
    config["agents"]["defaults"]["sandbox"]["mode"] = "all"
    profile.config_path.write_text(json.dumps(config, indent=2) + "\n")
    (problem,) = [p for p in check_profile(profile) if "openclaw.json" in p]
    assert "differs" in problem
    write_profile(profile)

    profile.token_path.chmod(0o640)
    assert any("gateway.token" in p and "readable" in p for p in check_profile(profile))
    profile.token_path.chmod(0o600)
    profile.token_path.write_text("\n")
    assert any("gateway.token" in p and "empty" in p for p in check_profile(profile))
    profile.token_path.unlink()
    assert any("gateway.token" in p and "missing" in p for p in check_profile(profile))
    write_profile(profile)

    installed = profile.state_dir / "extensions" / "rl-training-headers"
    (installed / "index.ts").write_text("// edited")
    assert any("rl-training-headers" in p and "differs" in p for p in check_profile(profile))
    shutil.rmtree(installed)
    assert any("rl-training-headers" in p and "missing" in p for p in check_profile(profile))
    write_profile(profile)
    assert check_profile(profile) == []

    shutil.rmtree(profile.home)
    assert any("home" in p for p in check_profile(profile))


# --- the environment of the gateway and of every CLI call ---------------------------------------


def test_gateway_env_has_no_secrets(profile: Profile, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch) -> None:
    write_profile(profile)
    for name in SECRETS:
        monkeypatch.setenv(name, f"secret-of-{name}")
    monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "the-parents-token")
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    env = gateway_env(profile)
    assert not [name for name in env if name in SECRETS]
    assert not [value for value in env.values() if "secret-of-" in value or "the-parents-token" in value]
    assert env["OPENCLAW_GATEWAY_TOKEN"] == profile.token_path.read_text().strip()


def test_gateway_env_is_exactly_the_allowlist(profile: Profile, fake_cli: FakeCli) -> None:
    write_profile(profile)
    assert gateway_env(profile) == {
        "PATH": f"{fake_cli.bin_dir}:/usr/bin:/bin",
        "HOME": str(profile.home),
        "LANG": "C.UTF-8",
        "TZ": "America/New_York",
        "SHELL": "/bin/bash",
        "NODE_OPTIONS": "--max-old-space-size=8192",
        "OPENCLAW_CONFIG_PATH": str(profile.dir / "openclaw.json"),
        "OPENCLAW_STATE_DIR": str(profile.dir / "state"),
        "OPENCLAW_GATEWAY_PORT": str(GATEWAY_PORT),
        "OPENCLAW_GATEWAY_URL": f"ws://127.0.0.1:{GATEWAY_PORT}",
        "OPENCLAW_GATEWAY_TOKEN": profile.token_path.read_text().strip(),
    }


def test_gateway_env_path_starts_with_the_node_that_runs_openclaw(
    profile: Profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_dir, tool_dir, other_node = tmp_path / "node-bin", tmp_path / "tool-bin", tmp_path / "other-node"
    for directory, names in ((node_dir, ["node"]), (tool_dir, ["openclaw"]), (other_node, ["node"])):
        directory.mkdir()
        for name in names:
            (directory / name).write_text("#!/bin/sh\n")
            (directory / name).chmod(0o755)
    write_profile(profile)
    monkeypatch.setenv("PATH", f"{node_dir}:{tool_dir}:{other_node}:/usr/bin:/opt/else")
    assert gateway_env(profile)["PATH"] == f"{node_dir}:{tool_dir}:/usr/bin:/bin"
    # OpenClaw and node in one directory: listed once.
    (node_dir / "openclaw").write_text("#!/bin/sh\n")
    (node_dir / "openclaw").chmod(0o755)
    assert gateway_env(profile)["PATH"] == f"{node_dir}:/usr/bin:/bin"


def test_gateway_env_needs_the_token_file(profile: Profile, fake_cli: FakeCli) -> None:
    with pytest.raises(OpenClawError, match="gateway.token"):
        gateway_env(profile)


def test_gateway_env_refuses_an_empty_token(profile: Profile, fake_cli: FakeCli) -> None:
    """An empty token could be read as no token, and the gateway would answer whoever asks."""
    write_profile(profile)
    profile.token_path.write_text("\n")
    with pytest.raises(OpenClawError, match=r"gateway\.token is empty"):
        gateway_env(profile)


def test_gateway_env_needs_openclaw_and_node_on_path(
    profile: Profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_profile(profile)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(OpenClawError, match="openclaw.*PATH"):
        gateway_env(profile)
    (empty / "openclaw").write_text("#!/bin/sh\n")
    (empty / "openclaw").chmod(0o755)
    with pytest.raises(OpenClawError, match="node.*PATH"):
        gateway_env(profile)
