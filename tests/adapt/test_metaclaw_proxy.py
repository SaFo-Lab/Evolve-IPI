"""The MetaClaw proxy script: the environment it takes, the check of the MetaClaw it imported, and (where MetaClaw is
installed) the objects it builds and a real start."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from metaclaw_helpers import ADAPT_DIR, PROXY, metaclaw_env, metaclaw_python

from evolve_ipi import thirdparty
from evolve_ipi.adapt import metaclaw_proxy, skills
from evolve_ipi.config import Configs, Secrets, load_configs
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import port_in_use

DUMP = Path(__file__).with_name("metaclaw_dump.py")
SECRETS = Secrets(
    tinker_api_key="tinker-test-key",
    evolver_base_url="https://evolver.invalid/v1",
    evolver_api_key="evolver-test-key",
    evolver_model="evolver-test-model",
)


@pytest.fixture(scope="module")
def cfgs() -> Configs:
    return load_configs(Path(__file__).resolve().parents[2] / "configs")


def base_env(tmp_path: Path, evolve: bool = False) -> dict[str, str]:
    env = {
        "SKILLS_DIR": str(tmp_path / "library"),
        "AUTO_EVOLVE": "1" if evolve else "0",
        "REQUIRE_EMPTY": "1",
        "LLM_BASE": "http://127.0.0.1:30013/v1",
        "LLM_MODEL": "qwen3.8-27b",
        "PROXY_PORT": "30011",
        "RECORD_DIR": str(tmp_path / "records"),
        "EXPECT_SHA256_SKILL_EVOLVER": "a" * 64,
        "EXPECT_SHA256_API_SERVER": "b" * 64,
        "SKILL_TOP_K": "6",
        "TASK_SPECIFIC_TOP_K": "10",
        "EVOLVE_EVERY": "1000",
        "MAX_NEW_SKILLS": "3",
        "EVOLVER_MAX_COMPLETION_TOKENS": "3000",
    }
    if evolve:
        env.update(
            OPENAI_BASE_URL="https://evolver.invalid/v1", OPENAI_API_KEY="k", SKILL_EVOLVER_MODEL="evolver-test-model"
        )
    return env


# --- the environment -----------------------------------------------------------------------------


def test_settings_of_an_evaluation_proxy(tmp_path: Path) -> None:
    settings = metaclaw_proxy.read_settings(base_env(tmp_path))
    assert settings.skills_dir == str(tmp_path / "library")
    assert (settings.evolve, settings.require_empty, settings.port) == (False, True, 30011)
    assert (settings.llm_base, settings.llm_model) == ("http://127.0.0.1:30013/v1", "qwen3.8-27b")
    assert settings.expect_sha256 == {"skill_evolver.py": "a" * 64, "api_server.py": "b" * 64}
    assert (settings.skill_top_k, settings.task_specific_top_k, settings.evolve_every) == (6, 10, 1000)
    assert (settings.max_new_skills, settings.evolver_max_completion_tokens) == (3, 3000)
    assert (settings.evolver_base_url, settings.evolver_api_key, settings.evolver_model) == ("", "", "")


def test_settings_of_an_evolving_proxy_keep_the_key_out_of_repr(tmp_path: Path) -> None:
    settings = metaclaw_proxy.read_settings(base_env(tmp_path, evolve=True))
    assert settings.evolve
    assert (settings.evolver_base_url, settings.evolver_api_key, settings.evolver_model) == (
        "https://evolver.invalid/v1",
        "k",
        "evolver-test-model",
    )
    assert "evolver_api_key" not in repr(settings)


@pytest.mark.parametrize("name", sorted(base_env(Path("/x"))))
def test_every_variable_is_required(tmp_path: Path, name: str) -> None:
    env = base_env(tmp_path)
    del env[name]
    with pytest.raises(metaclaw_proxy.SettingsError, match=f"^{name} is not set"):
        metaclaw_proxy.read_settings(env)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("AUTO_EVOLVE", "yes", "AUTO_EVOLVE must be 1 or 0, not 'yes'"),
        ("REQUIRE_EMPTY", "true", "REQUIRE_EMPTY must be 1 or 0, not 'true'"),
        ("PROXY_PORT", "30011x", "PROXY_PORT must be a whole number"),
        ("PROXY_PORT", "70000", "PROXY_PORT must be a port number"),
        ("EVOLVE_EVERY", "0", "EVOLVE_EVERY must be at least 1"),
        ("SKILL_TOP_K", "-1", "SKILL_TOP_K must be at least 1"),
        ("EXPECT_SHA256_API_SERVER", "abc", "EXPECT_SHA256_API_SERVER must be a sha256 in hex"),
        ("SKILLS_DIR", "relative/library", "SKILLS_DIR must be an absolute path"),
        ("RECORD_DIR", "records", "RECORD_DIR must be an absolute path"),
    ],
)
def test_malformed_variables_are_refused(tmp_path: Path, name: str, value: str, message: str) -> None:
    env = base_env(tmp_path)
    env[name] = value
    with pytest.raises(metaclaw_proxy.SettingsError, match=f"^{message}"):
        metaclaw_proxy.read_settings(env)


@pytest.mark.parametrize("name", ["OPENAI_BASE_URL", "OPENAI_API_KEY", "SKILL_EVOLVER_MODEL"])
def test_evolution_needs_every_evolver_variable(tmp_path: Path, name: str) -> None:
    env = base_env(tmp_path, evolve=True)
    del env[name]
    with pytest.raises(metaclaw_proxy.SettingsError, match=f"AUTO_EVOLVE=1 needs {name}"):
        metaclaw_proxy.read_settings(env)


@pytest.mark.parametrize("name", ["OPENAI_BASE_URL", "OPENAI_API_KEY", "SKILL_EVOLVER_MODEL"])
def test_an_evaluation_proxy_refuses_evolver_variables(tmp_path: Path, name: str) -> None:
    env = base_env(tmp_path)
    env[name] = "x"
    with pytest.raises(metaclaw_proxy.SettingsError, match=f"AUTO_EVOLVE=0 but {name} is set"):
        metaclaw_proxy.read_settings(env)


def test_the_scripts_and_the_module_are_plain_ascii() -> None:
    for path in (PROXY, ADAPT_DIR / "metaclaw_guards.py", ADAPT_DIR / "skills.py"):
        path.read_bytes().decode("ascii")  # raises on any byte that is not ASCII


def test_the_proxy_script_imports_only_the_standard_library_and_metaclaw() -> None:
    imported = set()
    for line in PROXY.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(("import ", "from ")):
            imported.add(line.split()[1].split(".")[0])
    assert imported - sys.stdlib_module_names == {"metaclaw"}
    assert not hasattr(metaclaw_proxy, "metaclaw")  # metaclaw is imported only when the script runs


# --- the check of the imported MetaClaw, against a stand-in package -----------------------------------


def fake_tree(root: Path, api_server: str = "API = 1\n", skill_evolver: str = "EVOLVER = 1\n") -> Path:
    """A copy of the proxy script in the repository layout, next to a stand-in ``metaclaw`` package."""
    script = root / "src" / "evolve_ipi" / "adapt" / "metaclaw_proxy.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(PROXY, script)
    package = root / "third_party" / "MetaClaw" / "metaclaw"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "api_server.py").write_text(api_server, encoding="utf-8")
    (package / "skill_evolver.py").write_text(skill_evolver, encoding="utf-8")
    return script


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run_script(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=60, check=False
    )


def test_the_proxy_refuses_files_other_than_the_expected_ones(tmp_path: Path) -> None:
    script = fake_tree(tmp_path / "repo")
    env = metaclaw_env(tmp_path, PYTHONPATH=str(tmp_path / "repo" / "third_party" / "MetaClaw"), **base_env(tmp_path))
    env["EXPECT_SHA256_API_SERVER"] = sha("API = 1\n")
    env["EXPECT_SHA256_SKILL_EVOLVER"] = "c" * 64
    done = run_script(script, env)
    assert done.returncode == 1
    assert "METACLAW patched" not in done.stdout
    assert "the imported MetaClaw files are not the expected ones" in done.stderr
    assert f"skill_evolver.py has sha256 {sha('EVOLVER = 1' + chr(10))}, expected {'c' * 64}" in done.stderr


def test_the_proxy_refuses_a_metaclaw_from_elsewhere(tmp_path: Path) -> None:
    script = fake_tree(tmp_path / "repo")
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(tmp_path / "repo" / "third_party" / "MetaClaw" / "metaclaw", elsewhere / "metaclaw")
    env = metaclaw_env(tmp_path, PYTHONPATH=str(elsewhere), **base_env(tmp_path))
    env["EXPECT_SHA256_API_SERVER"] = sha("API = 1\n")
    env["EXPECT_SHA256_SKILL_EVOLVER"] = sha("EVOLVER = 1\n")
    done = run_script(script, env)
    assert done.returncode == 1
    assert "METACLAW patched" not in done.stdout
    expected = tmp_path / "repo" / "third_party" / "MetaClaw" / "metaclaw"
    assert f"metaclaw was imported from {(elsewhere / 'metaclaw').resolve()}, not from {expected.resolve()}" in (
        done.stderr
    )


def test_the_proxy_names_what_it_imported_once_the_files_match(tmp_path: Path) -> None:
    script = fake_tree(tmp_path / "repo")
    env = metaclaw_env(tmp_path, PYTHONPATH=str(tmp_path / "repo" / "third_party" / "MetaClaw"), **base_env(tmp_path))
    env["EXPECT_SHA256_API_SERVER"] = sha("API = 1\n")
    env["EXPECT_SHA256_SKILL_EVOLVER"] = sha("EVOLVER = 1\n")
    done = run_script(script, env)
    package = (tmp_path / "repo" / "third_party" / "MetaClaw" / "metaclaw").resolve()
    assert done.stdout.splitlines()[0] == (
        f"METACLAW patched {package} skill_evolver={sha('EVOLVER = 1' + chr(10))} api_server={sha('API = 1' + chr(10))}"
    )
    assert done.returncode == 1  # the stand-in package has nothing else to import


def test_a_missing_variable_stops_the_script_before_any_import(tmp_path: Path) -> None:
    script = fake_tree(tmp_path / "repo")
    env = metaclaw_env(tmp_path, **base_env(tmp_path))
    del env["RECORD_DIR"]
    done = run_script(script, env)
    assert (done.returncode, done.stdout) == (1, "")
    assert done.stderr.strip() == "metaclaw_proxy: RECORD_DIR is not set"


# --- with MetaClaw installed ------------------------------------------------------------------------


def dump(tmp_path: Path, env: dict[str, str]) -> dict[str, Any]:
    out = tmp_path / "built.json"
    done = subprocess.run(
        [str(metaclaw_python()), str(DUMP), str(out), str(PROXY)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    built: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return built


@pytest.fixture
def proxy_env(tmp_path: Path, cfgs: Configs) -> Callable[[bool], dict[str, str]]:
    def make(evolve: bool) -> dict[str, str]:
        paths = RuntimePaths(root=tmp_path / "runtime", agent_home=tmp_path / "home", metaclaw_python=metaclaw_python())
        (tmp_path / "runtime").mkdir(exist_ok=True)
        model = cfgs.models["gpt-oss-120b"]
        return skills.proxy_env(
            model,
            tmp_path / "library",
            evolve,
            True,
            tmp_path / "records",
            cfgs.ports,
            SECRETS,
            skills=cfgs.skills,
            paths=paths,
        )

    return make


@pytest.mark.metaclaw
def test_the_evolving_proxy_builds_exactly_the_experiments_objects(
    tmp_path: Path, proxy_env: Callable[[bool], dict[str, str]], cfgs: Configs
) -> None:
    built = dump(tmp_path, proxy_env(True))
    library, records = tmp_path / "library", tmp_path / "records"
    history = str(library / "evolution_history.jsonl")
    expected = {
        "mode": "skills_only",
        "model_name": "openai-gpt-oss-120b",
        "served_model_name": "openai-gpt-oss-120b",
        "llm_model_id": "openai-gpt-oss-120b",
        "llm_provider": "custom",
        "llm_auth_method": "api_key",
        "llm_api_base": f"http://127.0.0.1:{cfgs.ports.model}/v1",
        "llm_api_key": "dummy",
        "use_skills": True,
        "skills_dir": str(library),
        "retrieval_mode": "template",
        "skill_top_k": 6,
        "task_specific_top_k": 10,
        "enable_skill_evolution": True,
        "skill_evolution_every_n_turns": 1000,
        "max_new_skills": 3,
        "evolver_provider": "openai",
        "evolver_api_base": "https://evolver.invalid/v1",
        "evolver_api_key": "evolver-test-key",
        "evolver_model_id": "evolver-test-model",
        "skill_evolution_history_path": history,
        "use_prm": False,
        "memory_enabled": False,
        "proxy_host": "127.0.0.1",
        "proxy_port": cfgs.ports.proxy,
        "max_context_tokens": 0,
        "record_enabled": True,
        "record_dir": str(records) + "/",
    }
    config, defaults = built["config"], built["config_defaults"]
    assert {key: config[key] for key in expected} == expected
    assert {key: value for key, value in config.items() if key not in expected} == {
        key: value for key, value in defaults.items() if key not in expected
    }
    manager = built["skill_manager"]
    assert manager["type"] == "SkillManager"
    assert (manager["_skills_dir"], manager["retrieval_mode"], manager["task_specific_top_k"]) == (
        str(library),
        "template",
        10,
    )
    assert manager["_embedding_model"] is None
    evolver = built["skill_evolver"]
    assert evolver["type"] == "SkillEvolver"
    assert (evolver["max_new_skills"], evolver["max_completion_tokens"], evolver["history_path"]) == (3, 3000, history)
    assert evolver["_custom_client"] is None
    assert evolver["_openai_model"] == "evolver-test-model"
    assert evolver["_openai_client"]["base_url"].rstrip("/") == "https://evolver.invalid/v1"
    assert evolver["_openai_client"]["api_key"] == "evolver-test-key"
    server = built["worker"]["_server"]
    assert (server["_sampling_client"], server["prm_scorer"], server["memory_manager"]) == (None, None, None)
    assert built["tokenizer_is_none"] is True
    assert built["same_objects"] == {"server_config": True, "evolver_history_is_config_history": True}
    assert (records / "conversations.jsonl").is_file()


@pytest.mark.metaclaw
def test_the_evaluation_proxy_has_no_evolver(tmp_path: Path, proxy_env: Callable[[bool], dict[str, str]]) -> None:
    env = proxy_env(False)
    assert not {"OPENAI_BASE_URL", "OPENAI_API_KEY", "SKILL_EVOLVER_MODEL"} & env.keys()
    built = dump(tmp_path, env)
    config = built["config"]
    assert config["enable_skill_evolution"] is False
    assert (config["evolver_api_base"], config["evolver_api_key"], config["evolver_model_id"]) == ("", "", "")
    assert built["skill_evolver"] is None
    assert built["tokenizer_is_none"] is True


@pytest.mark.metaclaw
def test_the_real_proxy_refuses_a_wrong_sha(tmp_path: Path, proxy_env: Callable[[bool], dict[str, str]]) -> None:
    env = proxy_env(False)
    env["EXPECT_SHA256_API_SERVER"] = "0" * 64
    done = subprocess.run(
        [str(metaclaw_python()), str(PROXY)], capture_output=True, text=True, env=env, timeout=300, check=False
    )
    assert done.returncode == 1
    assert "METACLAW patched" not in done.stdout
    real = thirdparty.verify_patches()["MetaClaw/metaclaw/api_server.py"]
    assert f"api_server.py has sha256 {real}, expected {'0' * 64}" in done.stderr


@pytest.mark.metaclaw
def test_the_real_proxy_refuses_a_library_that_is_not_empty(
    tmp_path: Path, proxy_env: Callable[[bool], dict[str, str]]
) -> None:
    env = proxy_env(False)
    (tmp_path / "library" / "a-skill").mkdir(parents=True)
    (tmp_path / "library" / "a-skill" / "SKILL.md").write_text("x", encoding="utf-8")
    done = subprocess.run(
        [str(metaclaw_python()), str(PROXY)], capture_output=True, text=True, env=env, timeout=300, check=False
    )
    assert done.returncode == 1
    assert "METACLAW patched" in done.stdout
    assert "REQUIRE_EMPTY=1 but the library is not empty (1 SKILL.md, history empty)" in done.stderr


@pytest.mark.metaclaw
def test_start_proxy_starts_the_real_proxy_and_stops_it(
    tmp_path: Path, cfgs: Configs, free_port: Callable[[], int]
) -> None:
    paths = RuntimePaths(root=tmp_path / "runtime", agent_home=tmp_path / "home", metaclaw_python=metaclaw_python())
    ports = cfgs.ports.model_copy(update={"proxy": free_port(), "model": free_port()})
    library = tmp_path / "library"
    proxy = skills.start_proxy(
        cfgs.models["qwen3.8-27b"],
        library,
        False,
        True,
        tmp_path / "records",
        ports,
        Secrets(),
        skills=cfgs.skills,
        paths=paths,
    )
    try:
        assert proxy.alive()
        log = proxy.log_text()
        assert "METACLAW patched " in log
        assert f"skills_only proxy READY on 127.0.0.1:{ports.proxy} " in log
        assert port_in_use(ports.proxy)
    finally:
        proxy.stop()
    assert not port_in_use(ports.proxy)
    assert library.is_dir() and not any(library.iterdir())
    assert (tmp_path / "records" / "conversations.jsonl").is_file()
    assert proxy.log_path == tmp_path / "records" / "proxy.log"
    assert ADAPT_DIR == skills.PROXY_SCRIPT.parent
