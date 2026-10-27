"""Typed access to ``configs/*.yaml`` and to secrets from the environment or a ``.env`` file.

Every mistake fails when the configuration is loaded, with the file and the key named: an unknown,
missing or malformed key, a missing file, a missing secret. Nothing falls back to a default; the
values that define the experiment live in the YAML files and nowhere else.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)

TASK_NAMES = ("file-create", "drive-create", "email-invite", "chat-message", "chat-invite")
_SETTINGS = (1, 2, 3)

Mechanism = Literal["rl", "skills"]
MECHANISMS: tuple[Mechanism, ...] = get_args(Mechanism)
Port = Annotated[int, Field(ge=1, le=65535)]


class ConfigError(Exception):
    """A configuration file, a secret or a runtime path is missing, unknown or malformed."""


def as_mechanism(name: str) -> Mechanism:
    """``name`` as one of the two mechanisms; ``ConfigError`` naming both for anything else."""
    for mechanism in MECHANISMS:
        if name == mechanism:
            return mechanism
    raise ConfigError(f"unknown mechanism {name!r}; the mechanisms are {', '.join(MECHANISMS)}")


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- the specs that configs/*.yaml are read into ---------------------------------------------


class _ModelEntry(_Spec):
    """One model as written in ``models.yaml``; its name is the key it is filed under."""

    tinker_id: Annotated[str, Field(min_length=1)]
    family: Literal["qwen", "harmony", "inkling"]
    skill_served_name: Annotated[str, Field(min_length=1)]
    mechanisms: Annotated[tuple[Mechanism, ...], Field(min_length=1)]
    skill_train_retries: NonNegativeInt


class ModelSpec(_ModelEntry):
    name: str


class RLSpec(_Spec):
    batch_size: PositiveInt
    prm_m: PositiveInt
    lora_rank: PositiveInt
    learning_rate: PositiveFloat
    updates_per_stage: PositiveInt
    max_steps: PositiveInt
    save_interval: PositiveInt
    until_step_timeout_s: PositiveInt
    fingerprint_min_diff: PositiveFloat
    max_restarts: NonNegativeInt


class SkillsSpec(_Spec):
    conversations: PositiveInt
    max_new_skills: PositiveInt
    evolver_max_completion_tokens: PositiveInt
    skill_top_k: PositiveInt
    task_specific_top_k: PositiveInt
    evolution_every_n_turns: PositiveInt
    drain_poll_s: PositiveFloat
    drain_stable_polls: PositiveInt
    drain_max_polls: PositiveInt
    max_restarts: NonNegativeInt


class EvalSpec(_Spec):
    setting3_base_passes: PositiveInt
    setting3_first_passes: PositiveInt
    setting3_topup_passes: NonNegativeInt
    topup_excess_min: NonNegativeInt
    topup_excess_max: NonNegativeInt
    settings12_passes: PositiveInt
    turn_timeout_s: dict[str, PositiveInt]
    temperature: NonNegativeFloat
    max_void_retries: NonNegativeInt

    @field_validator("turn_timeout_s")
    @classmethod
    def _one_timeout_per_task(cls, timeouts: dict[str, int]) -> dict[str, int]:
        unknown = sorted(set(timeouts) - set(TASK_NAMES))
        missing = [task for task in TASK_NAMES if task not in timeouts]
        if unknown or missing:
            raise ValueError(
                f"needs exactly one timeout per task ({', '.join(TASK_NAMES)}); "
                f"unknown: {unknown or 'none'}; missing: {missing or 'none'}"
            )
        return timeouts

    @model_validator(mode="after")
    def _topup_range_is_ordered(self) -> EvalSpec:
        if self.topup_excess_min > self.topup_excess_max:
            raise ValueError(
                f"topup_excess_min ({self.topup_excess_min}) must not exceed "
                f"topup_excess_max ({self.topup_excess_max})"
            )
        return self


class PortsSpec(_Spec):
    gateway: Port
    policy: Port
    proxy: Port
    model: Port

    @model_validator(mode="after")
    def _ports_differ(self) -> PortsSpec:
        owners: dict[int, str] = {}
        for name in type(self).model_fields:
            port = getattr(self, name)
            if port in owners:
                raise ValueError(f"{owners[port]} and {name} both use port {port}")
            owners[port] = name
        return self


class _RuntimeFile(_Spec):
    ports: PortsSpec


class _ModelsFile(_Spec):
    models: dict[str, _ModelEntry]

    @field_validator("models")
    @classmethod
    def _at_least_one(cls, models: dict[str, _ModelEntry]) -> dict[str, _ModelEntry]:
        if not models:
            raise ValueError("needs at least one model")
        return models


class Combo(_Spec):
    model: str
    task: str
    mechanism: str

    @field_validator("task")
    @classmethod
    def _known_task(cls, task: str) -> str:
        if task not in TASK_NAMES:
            raise ValueError(f"unknown task {task!r}; known tasks: {', '.join(TASK_NAMES)}")
        return task


class MainSpec(_Spec):
    combos: list[Combo]
    settings: list[int]

    @field_validator("combos")
    @classmethod
    def _combos_are_distinct(cls, combos: list[Combo]) -> list[Combo]:
        if not combos:
            raise ValueError("needs at least one combination")
        seen: set[tuple[str, str, str]] = set()
        for combo in combos:
            key = (combo.model, combo.task, combo.mechanism)
            if key in seen:
                raise ValueError(f"duplicate combination {' / '.join(key)}")
            seen.add(key)
        return combos

    @field_validator("settings")
    @classmethod
    def _settings_are_valid(cls, settings: list[int]) -> list[int]:
        if not settings:
            raise ValueError("needs at least one setting")
        unknown = sorted(set(settings) - set(_SETTINGS))
        if unknown:
            raise ValueError(f"unknown setting(s) {unknown}; the settings are 1, 2 and 3")
        if len(set(settings)) != len(settings):
            raise ValueError("lists a setting twice")
        return settings


@dataclass(frozen=True)
class Configs:
    models: dict[str, ModelSpec]
    rl: RLSpec
    skills: SkillsSpec
    eval: EvalSpec
    ports: PortsSpec
    main: MainSpec | None = None


# --- reading the files -----------------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """A SafeLoader that refuses a key written twice; PyYAML would silently keep the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        mapping = super().construct_mapping(node, deep=deep)
        if len(mapping) != len(node.value):
            seen: set[Any] = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        None, None, f"found duplicate key {key!r}", key_node.start_mark
                    )
                seen.add(key)
        return mapping


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=_UniqueKeyLoader)
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the file ({exc.strerror or exc})") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping of keys to values at the top level")
    return data


def _location(loc: tuple[int | str, ...]) -> str:
    where = ""
    for part in loc:
        if isinstance(part, int):
            where += f"[{part}]"
        else:
            where += f".{part}" if where else str(part)
    return where


def _describe(path: Path, error: ValidationError) -> str:
    lines = [f"{path}: invalid configuration"]
    for item in error.errors(include_url=False):
        if item["type"] == "extra_forbidden":
            text = "unknown key"
        elif item["type"] == "missing":
            text = "missing key"
        else:
            text = item["msg"].removeprefix("Value error, ")
        where = _location(item["loc"])
        lines.append(f"  {where}: {text}" if where else f"  {text}")
    return "\n".join(lines)


def _validate[Model: BaseModel](model: type[Model], data: Any, path: Path) -> Model:
    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise ConfigError(_describe(path, error)) from None


def _load_models(path: Path) -> dict[str, ModelSpec]:
    entries = _validate(_ModelsFile, _read_yaml(path), path).models
    return {name: ModelSpec(name=name, **entry.model_dump()) for name, entry in entries.items()}


def _load_main(path: Path, models: Mapping[str, ModelSpec]) -> MainSpec:
    main = _validate(MainSpec, _read_yaml(path), path)
    problems = []
    for index, combo in enumerate(main.combos):
        model = models.get(combo.model)
        if model is None:
            problems.append(
                f"  combos[{index}].model: unknown model {combo.model!r}; known models: {', '.join(models)}"
            )
        elif combo.mechanism not in model.mechanisms:
            problems.append(
                f"  combos[{index}].mechanism: model {combo.model!r} does not take part in "
                f"{combo.mechanism!r}; it takes part in: {', '.join(model.mechanisms)}"
            )
    if problems:
        raise ConfigError("\n".join([f"{path}: invalid configuration", *problems]))
    return main


def load_configs(configs_dir: Path, main: Path | None = None) -> Configs:
    """Read ``models``, ``rl``, ``skills``, ``eval`` and ``runtime`` ``.yaml`` from ``configs_dir``.

    ``main`` is the experiment file (``main.yaml``); it is read only when given, and checked against
    the models and tasks that the other files define.
    """
    directory = Path(configs_dir)
    models = _load_models(directory / "models.yaml")
    rl_path, skills_path, eval_path, runtime_path = (
        directory / name for name in ("rl.yaml", "skills.yaml", "eval.yaml", "runtime.yaml")
    )
    return Configs(
        models=models,
        rl=_validate(RLSpec, _read_yaml(rl_path), rl_path),
        skills=_validate(SkillsSpec, _read_yaml(skills_path), skills_path),
        eval=_validate(EvalSpec, _read_yaml(eval_path), eval_path),
        ports=_validate(_RuntimeFile, _read_yaml(runtime_path), runtime_path).ports,
        main=None if main is None else _load_main(Path(main), models),
    )


# --- secrets ---------------------------------------------------------------------------------

# Secrets field -> name of the environment variable (and of the key in a .env file)
_SECRET_ENV = {
    "tinker_api_key": "TINKER_API_KEY",
    "evolver_base_url": "EVOLVER_BASE_URL",
    "evolver_api_key": "EVOLVER_API_KEY",
    "evolver_model": "EVOLVER_MODEL",
    "hf_token": "HF_TOKEN",
}


@dataclass(frozen=True)
class Secrets:
    """Credentials and endpoints; ``None`` means not set. The credentials stay out of ``repr``."""

    tinker_api_key: str | None = field(default=None, repr=False)
    evolver_base_url: str | None = None
    evolver_api_key: str | None = field(default=None, repr=False)
    evolver_model: str | None = None
    hf_token: str | None = field(default=None, repr=False)


_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INLINE_COMMENT = re.compile(r"\s+#.*$")


def parse_env_file(path: Path) -> dict[str, str]:
    """Read ``KEY=value`` lines from a ``.env`` file.

    Blank lines and ``#`` comment lines are skipped. A value may be wrapped in single or double
    quotes, which are removed; an unquoted value ends at a ``#`` that follows whitespace. A line that
    cannot be read raises ``ConfigError`` with its number, but never shows its text, because the text
    may be a credential.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the env file ({exc.strerror or exc})") from None
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, equals, rest = line.partition("=")
        key = key.strip()
        if not equals or not _ENV_KEY.fullmatch(key):
            raise ConfigError(f"{path}:{number}: expected a line of the form KEY=value")
        rest = rest.strip()
        if rest[:1] in ("'", '"'):
            end = rest.find(rest[0], 1)
            if end == -1:
                raise ConfigError(f"{path}:{number}: the value starts with a quote that is never closed")
            after = rest[end + 1 :].strip()
            if after and not after.startswith("#"):
                raise ConfigError(f"{path}:{number}: unexpected text after the closing quote")
            value = rest[1:end]
        else:
            value = _INLINE_COMMENT.sub("", rest).strip()
        if key in values:
            raise ConfigError(f"{path}:{number}: {key} is set twice")
        values[key] = value
    return values


def load_secrets(
    require: Iterable[str],
    env: Mapping[str, str] | None = None,
    env_file: Path | None = None,
) -> Secrets:
    """Collect the secrets and raise ``ConfigError`` naming every required one that is missing.

    ``require`` holds ``Secrets`` field names. A value comes from ``env`` (the process environment
    when ``None``) or, below it, from ``env_file`` when one is given. A blank value counts as not set
    in both places, so an unfilled ``KEY=`` line of ``.env.example`` is reported as missing.
    """
    required = set(require)
    unknown = sorted(required - _SECRET_ENV.keys())
    if unknown:
        raise ValueError(f"unknown secret name(s) {unknown}; known names: {', '.join(_SECRET_ENV)}")
    sources: list[Mapping[str, str]] = []
    if env_file is not None:
        sources.append(parse_env_file(env_file))
    sources.append(os.environ if env is None else env)
    found: dict[str, str | None] = {}
    for secret, variable in _SECRET_ENV.items():
        found[secret] = None
        for source in sources:  # a later source overrides an earlier one, but only with a real value
            value = source.get(variable, "").strip()
            if value:
                found[secret] = value
    missing = [_SECRET_ENV[secret] for secret in _SECRET_ENV if secret in required and found[secret] is None]
    if missing:
        where = "the environment" if env_file is None else f"the environment or {env_file}"
        raise ConfigError(
            f"missing required secret(s): {', '.join(missing)}. Set each one in {where} "
            "(see .env.example); an empty value counts as missing."
        )
    return Secrets(**found)
