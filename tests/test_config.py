"""Configuration files, the ``.env`` parser and secrets."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from evolve_ipi.config import (
    ConfigError,
    Secrets,
    load_configs,
    load_secrets,
    parse_env_file,
)

TASKS = ("file-create", "drive-create", "email-invite", "chat-message", "chat-invite")
RL_MODELS = ("qwen3.8-27b", "gpt-oss-120b", "inkling-small")
SKILL_MODELS = ("qwen3.8-27b", "gpt-oss-120b", "gpt-oss-20b", "inkling-small")


@pytest.fixture
def configs_copy(configs_dir: Path, tmp_path: Path) -> Path:
    """A private copy of the shipped configs that a test may damage."""
    dest = tmp_path / "configs"
    shutil.copytree(configs_dir, dest)
    return dest


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} is not in {path.name}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def drop_key(path: Path, key: str) -> None:
    """Delete the top-level line ``key: ...`` (with its trailing comment)."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if not line.startswith(f"{key}:")]
    assert len(kept) == len(lines) - 1, f"expected exactly one {key!r} line in {path.name}"
    path.write_text("".join(kept), encoding="utf-8")


def write_main(tmp_path: Path, combos: list[dict], settings: list[int] | None = None) -> Path:
    path = tmp_path / "main.yaml"
    body = {"settings": [1, 2, 3] if settings is None else settings, "combos": combos}
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def load_error(configs: Path, main: Path | None = None) -> str:
    with pytest.raises(ConfigError) as err:
        load_configs(configs, main)
    return str(err.value)


# --- shipped configs -------------------------------------------------------------------------


def test_shipped_configs_load(configs_dir: Path) -> None:
    cfgs = load_configs(configs_dir, configs_dir / "main.yaml")
    assert len(cfgs.models) == 4
    assert all(name == spec.name for name, spec in cfgs.models.items())
    assert cfgs.models["gpt-oss-20b"].mechanisms == ("skills",)
    assert cfgs.models["gpt-oss-120b"].skill_train_retries == 4
    assert cfgs.models["gpt-oss-20b"].skill_train_retries == 4
    assert cfgs.models["qwen3.8-27b"].skill_train_retries == 0
    assert cfgs.models["inkling-small"].skill_train_retries == 0
    assert {name: spec.tinker_id for name, spec in cfgs.models.items()} == {
        "qwen3.8-27b": "Qwen/Qwen3.8-27B",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "inkling-small": "thinkingmachines/Inkling-Small",
    }
    assert {name: spec.skill_served_name for name, spec in cfgs.models.items()} == {
        "qwen3.8-27b": "qwen3.8-27b",
        "gpt-oss-120b": "openai-gpt-oss-120b",
        "gpt-oss-20b": "openai-gpt-oss-20b",
        "inkling-small": "thinkingmachines-inkling-small",
    }
    assert {name: spec.family for name, spec in cfgs.models.items()} == {
        "qwen3.8-27b": "qwen",
        "gpt-oss-120b": "harmony",
        "gpt-oss-20b": "harmony",
        "inkling-small": "inkling",
    }
    assert {name: spec.mechanisms for name, spec in cfgs.models.items() if name != "gpt-oss-20b"} == {
        "qwen3.8-27b": ("rl", "skills"),
        "gpt-oss-120b": ("rl", "skills"),
        "inkling-small": ("rl", "skills"),
    }


def test_shipped_recipe_values(configs_dir: Path) -> None:
    cfgs = load_configs(configs_dir)
    assert cfgs.main is None
    assert cfgs.rl.model_dump() == {
        "batch_size": 4,
        "prm_m": 3,
        "lora_rank": 32,
        "learning_rate": 1e-4,
        "updates_per_stage": 32,
        "max_steps": 100,
        "save_interval": 32,
        "until_step_timeout_s": 1800,
        "fingerprint_min_diff": 0.05,
        "max_restarts": 3,
    }
    assert cfgs.skills.model_dump() == {
        "conversations": 12,
        "max_new_skills": 3,
        "evolver_max_completion_tokens": 3000,
        "skill_top_k": 6,
        "task_specific_top_k": 10,
        "evolution_every_n_turns": 1000,
        "drain_poll_s": 3,
        "drain_stable_polls": 5,
        "drain_max_polls": 300,
        "max_restarts": 3,
    }
    assert cfgs.eval.model_dump() == {
        "setting3_base_passes": 3,
        "setting3_first_passes": 1,
        "setting3_topup_passes": 2,
        "topup_excess_min": 1,
        "topup_excess_max": 5,
        "settings12_passes": 3,
        "turn_timeout_s": {
            "file-create": 300,
            "drive-create": 600,
            "email-invite": 600,
            "chat-message": 600,
            "chat-invite": 600,
        },
        "temperature": 0.6,
        "max_void_retries": 3,
    }
    assert cfgs.ports.model_dump() == {
        "gateway": 18789,
        "policy": 30012,
        "proxy": 30011,
        "model": 30013,
    }


def test_specs_are_frozen(configs_dir: Path) -> None:
    cfgs = load_configs(configs_dir)
    with pytest.raises(ValidationError):
        cfgs.rl.batch_size = 8  # type: ignore[misc]
    with pytest.raises(ValidationError):
        cfgs.models["qwen3.8-27b"].skill_train_retries = 1  # type: ignore[misc]


def test_main_has_35_combos(configs_dir: Path) -> None:
    main = load_configs(configs_dir, configs_dir / "main.yaml").main
    assert main is not None
    assert main.settings == [1, 2, 3]
    assert len(main.combos) == 35
    assert Counter(combo.mechanism for combo in main.combos) == {"rl": 15, "skills": 20}
    expected = {(m, t, "rl") for m in RL_MODELS for t in TASKS} | {
        (m, t, "skills") for m in SKILL_MODELS for t in TASKS
    }
    assert {(c.model, c.task, c.mechanism) for c in main.combos} == expected


def test_env_example_lists_every_secret(repo_root: Path) -> None:
    assert parse_env_file(repo_root / ".env.example") == {
        "TINKER_API_KEY": "",
        "EVOLVER_BASE_URL": "",
        "EVOLVER_API_KEY": "",
        "EVOLVER_MODEL": "",
        "HF_TOKEN": "",
    }


# --- configuration mistakes fail at load time, naming the key and the file ------------------


@pytest.mark.parametrize(
    ("filename", "typo"),
    [
        ("rl.yaml", "bach_size: 4\n"),
        ("skills.yaml", "max_new_skillz: 3\n"),
        ("eval.yaml", "temprature: 0.6\n"),
        ("models.yaml", "modles: {}\n"),
        ("runtime.yaml", "prots: {}\n"),
    ],
)
def test_unknown_key_fails(configs_copy: Path, filename: str, typo: str) -> None:
    append(configs_copy / filename, typo)
    message = load_error(configs_copy)
    assert typo.split(":")[0] in message
    assert filename in message


def test_misspelled_key_reports_the_typo_and_the_missing_key(configs_copy: Path) -> None:
    edit(configs_copy / "rl.yaml", "batch_size:", "bach_size:")
    message = load_error(configs_copy)
    assert "bach_size" in message
    assert "batch_size" in message
    assert "rl.yaml" in message


def test_unknown_key_inside_a_model_names_the_model(configs_copy: Path) -> None:
    edit(configs_copy / "models.yaml", "tinker_id:", "tinker_idd:")
    message = load_error(configs_copy)
    assert "tinker_idd" in message
    assert "tinker_id" in message
    assert "models.yaml" in message


def test_unknown_key_inside_ports_fails(configs_copy: Path) -> None:
    edit(configs_copy / "runtime.yaml", "gateway:", "gatway:")
    message = load_error(configs_copy)
    assert "gatway" in message
    assert "runtime.yaml" in message


def test_missing_key_fails(configs_copy: Path) -> None:
    drop_key(configs_copy / "rl.yaml", "prm_m")
    message = load_error(configs_copy)
    assert "prm_m" in message
    assert "rl.yaml" in message


def test_missing_file_fails(configs_copy: Path) -> None:
    (configs_copy / "skills.yaml").unlink()
    assert "skills.yaml" in load_error(configs_copy)


@pytest.mark.parametrize("body", ["", "- a\n- b\n", "just text\n"])
def test_file_that_is_not_a_mapping_fails(configs_copy: Path, body: str) -> None:
    (configs_copy / "rl.yaml").write_text(body, encoding="utf-8")
    message = load_error(configs_copy)
    assert "rl.yaml" in message
    assert "mapping" in message


def test_invalid_yaml_fails(configs_copy: Path) -> None:
    (configs_copy / "rl.yaml").write_text("batch_size: [1, 2\n", encoding="utf-8")
    assert "rl.yaml" in load_error(configs_copy)


def test_duplicate_key_fails(configs_copy: Path) -> None:
    append(configs_copy / "rl.yaml", "batch_size: 8\n")
    message = load_error(configs_copy)
    assert "duplicate" in message
    assert "batch_size" in message
    assert "rl.yaml" in message


@pytest.mark.parametrize(
    ("filename", "old", "new"),
    [
        ("rl.yaml", "batch_size: 4", "batch_size: four"),
        ("rl.yaml", "batch_size: 4", "batch_size: 0"),
        ("skills.yaml", "skill_top_k: 6", "skill_top_k: -1"),
        ("eval.yaml", "temperature: 0.6", "temperature: -0.1"),
        ("runtime.yaml", "gateway: 18789", "gateway: 70000"),
        ("models.yaml", "skill_train_retries: 4", "skill_train_retries: -1"),
        ("models.yaml", "family: qwen", "family: gemma"),
    ],
)
def test_bad_value_fails(configs_copy: Path, filename: str, old: str, new: str) -> None:
    edit(configs_copy / filename, old, new)
    message = load_error(configs_copy)
    assert old.split(":")[0] in message
    assert filename in message


def test_turn_timeouts_must_name_every_task(configs_copy: Path) -> None:
    edit(configs_copy / "eval.yaml", "file-create: 300", "file-creat: 300")
    message = load_error(configs_copy)
    assert "turn_timeout_s" in message
    assert "file-creat" in message
    assert "file-create" in message


def test_topup_range_must_be_ordered(configs_copy: Path) -> None:
    edit(configs_copy / "eval.yaml", "topup_excess_min: 1", "topup_excess_min: 9")
    message = load_error(configs_copy)
    assert "topup_excess_min" in message
    assert "topup_excess_max" in message


def test_ports_must_differ(configs_copy: Path) -> None:
    edit(configs_copy / "runtime.yaml", "proxy: 30011", "proxy: 18789")
    message = load_error(configs_copy)
    assert "gateway" in message
    assert "proxy" in message
    assert "18789" in message


def test_model_must_take_part_in_a_mechanism(configs_copy: Path) -> None:
    edit(configs_copy / "models.yaml", "mechanisms: [skills]", "mechanisms: []")
    assert "mechanisms" in load_error(configs_copy)


# --- main.yaml -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("combo", "needle"),
    [
        ({"model": "qwen3.8-27c", "task": "file-create", "mechanism": "rl"}, "qwen3.8-27c"),
        ({"model": "qwen3.8-27b", "task": "file-creat", "mechanism": "rl"}, "file-creat"),
    ],
)
def test_unknown_model_or_task_in_main_fails(
    configs_dir: Path, tmp_path: Path, combo: dict, needle: str
) -> None:
    message = load_error(configs_dir, write_main(tmp_path, [combo]))
    assert needle in message
    assert "main.yaml" in message


@pytest.mark.parametrize(
    ("combo", "needle"),
    [
        ({"model": "gpt-oss-20b", "task": "file-create", "mechanism": "rl"}, "gpt-oss-20b"),
        ({"model": "qwen3.8-27b", "task": "file-create", "mechanism": "rll"}, "rll"),
        ({"model": "qwen3.8-27b", "task": "file-create", "mechanism": "rl", "stting": 3}, "stting"),
        ({"model": "qwen3.8-27b", "task": "file-create"}, "mechanism"),
    ],
)
def test_other_combo_mistakes_in_main_fail(
    configs_dir: Path, tmp_path: Path, combo: dict, needle: str
) -> None:
    message = load_error(configs_dir, write_main(tmp_path, [combo]))
    assert needle in message
    assert "main.yaml" in message


def test_duplicate_combo_in_main_fails(configs_dir: Path, tmp_path: Path) -> None:
    combo = {"model": "qwen3.8-27b", "task": "file-create", "mechanism": "rl"}
    message = load_error(configs_dir, write_main(tmp_path, [combo, dict(combo)]))
    assert "duplicate" in message
    assert "qwen3.8-27b" in message


@pytest.mark.parametrize("settings", [[1, 2, 4], [], [1, 1]])
def test_bad_settings_in_main_fail(configs_dir: Path, tmp_path: Path, settings: list[int]) -> None:
    combo = {"model": "qwen3.8-27b", "task": "file-create", "mechanism": "rl"}
    message = load_error(configs_dir, write_main(tmp_path, [combo], settings))
    assert "settings" in message
    assert "main.yaml" in message


def test_main_file_must_exist(configs_dir: Path, tmp_path: Path) -> None:
    assert "nothing.yaml" in load_error(configs_dir, tmp_path / "nothing.yaml")


# --- secrets ---------------------------------------------------------------------------------


def test_missing_secret_named() -> None:
    with pytest.raises(ConfigError) as err:
        load_secrets({"tinker_api_key", "evolver_model"}, env={})
    message = str(err.value)
    assert "TINKER_API_KEY" in message
    assert "EVOLVER_MODEL" in message
    assert "EVOLVER_API_KEY" not in message


def test_only_required_secrets_may_be_absent() -> None:
    secrets = load_secrets({"tinker_api_key"}, env={"TINKER_API_KEY": "key"})
    assert secrets == Secrets(tinker_api_key="key")
    assert secrets.evolver_model is None


def test_secrets_map_environment_names() -> None:
    env = {
        "TINKER_API_KEY": "t",
        "EVOLVER_BASE_URL": "https://evolver.test/v1",
        "EVOLVER_API_KEY": "e",
        "EVOLVER_MODEL": "m",
        "HF_TOKEN": "h",
        "UNRELATED": "x",
    }
    assert load_secrets(set(), env=env) == Secrets(
        tinker_api_key="t",
        evolver_base_url="https://evolver.test/v1",
        evolver_api_key="e",
        evolver_model="m",
        hf_token="h",
    )


def test_secrets_default_to_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "from-process")
    assert load_secrets({"tinker_api_key"}).tinker_api_key == "from-process"


def test_blank_values_count_as_missing() -> None:
    with pytest.raises(ConfigError) as err:
        load_secrets({"tinker_api_key", "evolver_model"}, env={"TINKER_API_KEY": "   ", "EVOLVER_MODEL": ""})
    assert "TINKER_API_KEY" in str(err.value)
    assert "EVOLVER_MODEL" in str(err.value)


def test_environment_overrides_the_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TINKER_API_KEY=from-file\nEVOLVER_MODEL=model-from-file\n", encoding="utf-8")
    secrets = load_secrets(set(), env={"TINKER_API_KEY": "from-env"}, env_file=env_file)
    assert secrets.tinker_api_key == "from-env"
    assert secrets.evolver_model == "model-from-file"


def test_blank_environment_value_does_not_hide_the_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TINKER_API_KEY=from-file\n", encoding="utf-8")
    secrets = load_secrets({"tinker_api_key"}, env={"TINKER_API_KEY": ""}, env_file=env_file)
    assert secrets.tinker_api_key == "from-file"


def test_blank_value_in_env_file_counts_as_missing(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("EVOLVER_MODEL=\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="EVOLVER_MODEL"):
        load_secrets({"evolver_model"}, env={}, env_file=env_file)


def test_unknown_secret_name_is_refused() -> None:
    with pytest.raises(ValueError, match="evolver_modle"):
        load_secrets({"evolver_modle"}, env={})


def test_secrets_repr_hides_credentials() -> None:
    secrets = Secrets(tinker_api_key="tinker-secret", evolver_api_key="evolver-secret", hf_token="hf-secret")
    shown = repr(secrets)
    assert "secret" not in shown


# --- .env parsing ----------------------------------------------------------------------------


def test_env_file_parsing(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        'DOUBLE="https://example.test/v1"\n'
        "SINGLE='two words'\n"
        "  SPACED = padded value  # trailing comment\n"
        "EMPTY=\n"
        'HASH_IN_QUOTES="a # b"\n'
        "FRAGMENT=https://example.test/p#frag\n"
        "EQUALS=a=b=c\n"
        'QUOTED_THEN_COMMENT="x"  # note\n',
        encoding="utf-8",
    )
    assert parse_env_file(env_file) == {
        "PLAIN": "value",
        "DOUBLE": "https://example.test/v1",
        "SINGLE": "two words",
        "SPACED": "padded value",
        "EMPTY": "",
        "HASH_IN_QUOTES": "a # b",
        "FRAGMENT": "https://example.test/p#frag",
        "EQUALS": "a=b=c",
        "QUOTED_THEN_COMMENT": "x",
    }


def test_env_file_accepts_windows_line_endings_and_a_byte_order_mark(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"\xef\xbb\xbfFIRST=1\r\nSECOND=two\r\n")
    assert parse_env_file(env_file) == {"FIRST": "1", "SECOND": "two"}


@pytest.mark.parametrize(
    ("bad_line", "reason"),
    [
        ("no equals sign here", "KEY=value"),
        ("9LEADING_DIGIT=1", "KEY=value"),
        ("BAD KEY=1", "KEY=value"),
        ('UNTERMINATED="abc', "quote"),
        ('JUNK="abc" def', "quote"),
        ("FIRST=2", "twice"),
    ],
)
def test_malformed_env_line_names_the_line_and_never_echoes_it(
    tmp_path: Path, bad_line: str, reason: str
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"FIRST=1\n# comment\n{bad_line}\n", encoding="utf-8")
    with pytest.raises(ConfigError) as err:
        parse_env_file(env_file)
    message = str(err.value)
    assert f"{env_file}:3" in message
    assert reason in message
    assert bad_line not in message


def test_missing_env_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"absent\.env"):
        load_secrets(set(), env={}, env_file=tmp_path / "absent.env")


# --- the mechanisms -----------------------------------------------------------------------------------------------


def test_one_list_of_mechanisms_and_one_parser_for_them() -> None:
    from evolve_ipi import config, evaluate, runner

    assert config.MECHANISMS == ("rl", "skills")
    assert evaluate.MECHANISMS is runner.MECHANISMS is config.MECHANISMS
    assert [config.as_mechanism(name) for name in ("rl", "skills")] == ["rl", "skills"]
    for name in ("lora", "RL", "", "skills "):
        with pytest.raises(config.ConfigError, match=rf"unknown mechanism {name!r}; the mechanisms are rl, skills"):
            config.as_mechanism(name)
