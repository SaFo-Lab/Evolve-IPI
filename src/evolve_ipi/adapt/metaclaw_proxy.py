"""MetaClaw's skills_only proxy, as the experiment runs it.

A standalone script for the MetaClaw environment's Python: it imports the standard library and ``metaclaw``, nothing
else. ``evolve_ipi.adapt.skills.start_proxy`` starts it with an environment of its own (below).

First it proves which MetaClaw it imported: the package must be this repository's ``third_party/MetaClaw/metaclaw``,
and its two patched files must have the sha256 the caller expects. It prints

    METACLAW patched <package dir> skill_evolver=<sha256> api_server=<sha256>

Then it runs MetaClaw's proxy in skills_only mode. The proxy forwards each model request of the agent to the model
server (``LLM_BASE``), adds the skills it retrieves for the request to the system message, and, with ``AUTO_EVOLVE=1``,
runs MetaClaw's skill evolver on every conversation it is told has ended (``X-Session-Done``), which writes new skills
into ``SKILLS_DIR``. The model name is the model server's served name, not a Hugging Face id, so MetaClaw cannot load
a tokenizer for it and takes its tokenizer-free path: it records and buffers the turns and builds no RL samples. The
script stops if a tokenizer did load. Once the proxy serves, it logs

    skills_only proxy READY on 127.0.0.1:<port> -> <LLM_BASE> (...)

The environment (every variable is required; nothing has a default):

    SKILLS_DIR, RECORD_DIR       the skill library, and a fresh directory for MetaClaw's records (absolute paths)
    AUTO_EVOLVE, REQUIRE_EMPTY   1 or 0: evolve the library; refuse to start unless the library is empty
    LLM_BASE, LLM_MODEL          the model server's endpoint and served model name
    PROXY_PORT                   where the proxy listens, on 127.0.0.1
    EXPECT_SHA256_SKILL_EVOLVER, EXPECT_SHA256_API_SERVER   the sha256 of the two patched MetaClaw files
    SKILL_TOP_K, TASK_SPECIFIC_TOP_K, EVOLVE_EVERY, MAX_NEW_SKILLS, EVOLVER_MAX_COMPLETION_TOKENS
                                 the values of configs/skills.yaml
    OPENAI_BASE_URL, OPENAI_API_KEY, SKILL_EVOLVER_MODEL
                                 the evolver's endpoint, key and model, read by MetaClaw's evolver: required with
                                 AUTO_EVOLVE=1 and refused with AUTO_EVOLVE=0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# This file lies at src/evolve_ipi/adapt/ of the repository; MetaClaw is its submodule.
METACLAW_PACKAGE = Path(__file__).resolve().parents[3] / "third_party" / "MetaClaw" / "metaclaw"
PATCHED_FILES = ("skill_evolver.py", "api_server.py")
EVOLVER_VARIABLES = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "SKILL_EVOLVER_MODEL")
HISTORY_FILE = "evolution_history.jsonl"
HOST = "127.0.0.1"
LLM_API_KEY = "dummy"  # the model server checks no key
_SHA256 = re.compile(r"[0-9a-f]{64}")
_INTEGER = re.compile(r"-?[0-9]+")

logger = logging.getLogger("metaclaw_proxy")


class SettingsError(ValueError):
    """A variable of the environment is missing or malformed."""


@dataclass(frozen=True)
class Settings:
    skills_dir: str
    record_dir: str
    evolve: bool
    require_empty: bool
    llm_base: str
    llm_model: str
    port: int
    expect_sha256: dict[str, str]
    skill_top_k: int
    task_specific_top_k: int
    evolve_every: int
    max_new_skills: int
    evolver_max_completion_tokens: int
    evolver_base_url: str  # "" for a proxy that does not evolve
    evolver_api_key: str = field(repr=False)
    evolver_model: str


def _get(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "")
    if not value:
        raise SettingsError(f"{name} is not set")
    return value


def _flag(env: Mapping[str, str], name: str) -> bool:
    value = _get(env, name)
    if value not in ("1", "0"):
        raise SettingsError(f"{name} must be 1 or 0, not {value!r}")
    return value == "1"


def _count(env: Mapping[str, str], name: str) -> int:
    value = _get(env, name)
    if not _INTEGER.fullmatch(value):
        raise SettingsError(f"{name} must be a whole number, not {value!r}")
    if int(value) < 1:
        raise SettingsError(f"{name} must be at least 1, not {value!r}")
    return int(value)


def _absolute(env: Mapping[str, str], name: str) -> str:
    value = _get(env, name)
    if not os.path.isabs(value):
        raise SettingsError(f"{name} must be an absolute path, not {value!r}")
    return value


def _sha256(env: Mapping[str, str], name: str) -> str:
    value = _get(env, name)
    if not _SHA256.fullmatch(value):
        raise SettingsError(f"{name} must be a sha256 in hex, not {value!r}")
    return value


def read_settings(env: Mapping[str, str]) -> Settings:
    """The proxy's settings from its environment; raises ``SettingsError`` naming the variable that is wrong."""
    port = _count(env, "PROXY_PORT")
    if port > 65535:
        raise SettingsError(f"PROXY_PORT must be a port number (1 to 65535), not {port}")
    evolve = _flag(env, "AUTO_EVOLVE")
    present = [name for name in EVOLVER_VARIABLES if env.get(name)]
    if evolve:
        for name in EVOLVER_VARIABLES:
            if name not in present:
                raise SettingsError(f"AUTO_EVOLVE=1 needs {name} (the evolver's endpoint, key and model)")
    elif present:
        raise SettingsError(f"AUTO_EVOLVE=0 but {present[0]} is set: a proxy that does not evolve gets no evolver")
    return Settings(
        skills_dir=_absolute(env, "SKILLS_DIR"),
        record_dir=_absolute(env, "RECORD_DIR"),
        evolve=evolve,
        require_empty=_flag(env, "REQUIRE_EMPTY"),
        llm_base=_get(env, "LLM_BASE"),
        llm_model=_get(env, "LLM_MODEL"),
        port=port,
        expect_sha256={
            "skill_evolver.py": _sha256(env, "EXPECT_SHA256_SKILL_EVOLVER"),
            "api_server.py": _sha256(env, "EXPECT_SHA256_API_SERVER"),
        },
        skill_top_k=_count(env, "SKILL_TOP_K"),
        task_specific_top_k=_count(env, "TASK_SPECIFIC_TOP_K"),
        evolve_every=_count(env, "EVOLVE_EVERY"),  # at 0 MetaClaw would buffer no turn, so nothing could evolve
        max_new_skills=_count(env, "MAX_NEW_SKILLS"),
        evolver_max_completion_tokens=_count(env, "EVOLVER_MAX_COMPLETION_TOKENS"),
        evolver_base_url=env.get("OPENAI_BASE_URL", "") if evolve else "",
        evolver_api_key=env.get("OPENAI_API_KEY", "") if evolve else "",
        evolver_model=env.get("SKILL_EVOLVER_MODEL", "") if evolve else "",
    )


def check_metaclaw(package_dir: Path, expected: Mapping[str, str]) -> tuple[Path, dict[str, str]]:
    """Import MetaClaw and require it to be ``package_dir`` with the expected files; returns where it is and the
    sha256 of the two patched files. Exits naming the difference otherwise."""
    import metaclaw  # type: ignore[import-not-found]
    import metaclaw.api_server  # type: ignore[import-not-found]
    import metaclaw.skill_evolver  # type: ignore[import-not-found]

    root = Path(_file_of(metaclaw)).resolve().parent
    if root != package_dir.resolve():
        sys.exit(f"metaclaw_proxy: metaclaw was imported from {root}, not from {package_dir.resolve()}")
    modules = {"skill_evolver.py": metaclaw.skill_evolver, "api_server.py": metaclaw.api_server}
    got = {}
    for name, module in modules.items():
        path = Path(_file_of(module)).resolve()
        if path != root / name:
            sys.exit(f"metaclaw_proxy: {module.__name__} was imported from {path}, not from {root / name}")
        got[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    wrong = [
        f"{name} has sha256 {got[name]}, expected {expected[name]}"
        for name in PATCHED_FILES
        if got[name] != expected[name]
    ]
    if wrong:
        sys.exit(f"metaclaw_proxy: the imported MetaClaw files are not the expected ones: {'; '.join(wrong)}")
    return root, got


def _file_of(module: Any) -> str:
    path = getattr(module, "__file__", None)
    if not path:
        sys.exit(f"metaclaw_proxy: the module {module.__name__} has no file")
    return str(path)


def build(settings: Settings) -> Any:
    """MetaClaw's config, skill manager, skill evolver and rollout worker, as the experiment ran them."""
    from metaclaw.config import MetaClawConfig  # type: ignore[import-not-found]
    from metaclaw.rollout import AsyncRolloutWorker  # type: ignore[import-not-found]
    from metaclaw.skill_evolver import SkillEvolver
    from metaclaw.skill_manager import SkillManager  # type: ignore[import-not-found]

    skills_dir = settings.skills_dir
    Path(skills_dir).mkdir(parents=True, exist_ok=True)
    if settings.require_empty:
        populated = sorted(Path(skills_dir).glob("*/SKILL.md"))
        history = Path(skills_dir) / HISTORY_FILE
        stale = history.exists() and history.stat().st_size > 0
        if populated or stale:
            sys.exit(
                f"metaclaw_proxy: REQUIRE_EMPTY=1 but the library is not empty ({len(populated)} SKILL.md, "
                f"history {'not empty' if stale else 'empty'}): {skills_dir}"
            )

    served = settings.llm_model
    config = MetaClawConfig(
        mode="skills_only",
        # The served name, not a Hugging Face id: the tokenizer load fails, and the proxy takes the tokenizer-free path.
        model_name=served,
        served_model_name=served,
        llm_provider="custom",
        llm_auth_method="api_key",
        llm_api_base=settings.llm_base,
        llm_api_key=LLM_API_KEY,
        llm_model_id=served,
        use_skills=True,
        skills_dir=skills_dir,
        retrieval_mode="template",
        skill_top_k=settings.skill_top_k,
        task_specific_top_k=settings.task_specific_top_k,
        enable_skill_evolution=settings.evolve,
        skill_evolution_every_n_turns=settings.evolve_every,
        max_new_skills=settings.max_new_skills,
        evolver_provider="openai",
        evolver_api_base=settings.evolver_base_url,
        evolver_api_key=settings.evolver_api_key,
        evolver_model_id=settings.evolver_model,
        skill_evolution_history_path=str(Path(skills_dir) / HISTORY_FILE),
        use_prm=False,
        memory_enabled=False,
        proxy_host=HOST,
        proxy_port=settings.port,
        max_context_tokens=0,
        record_enabled=True,
        record_dir=settings.record_dir.rstrip("/") + "/",
    )
    manager = SkillManager(
        skills_dir=skills_dir, retrieval_mode="template", task_specific_top_k=settings.task_specific_top_k
    )
    logger.info("SkillManager loaded %s from %s", manager.get_skill_count(), skills_dir)
    evolver = None
    if settings.evolve:
        # MetaClaw's own evolver client: it reads OPENAI_BASE_URL, OPENAI_API_KEY and SKILL_EVOLVER_MODEL, sends
        # max_completion_tokens and nothing else (no temperature).
        evolver = SkillEvolver(
            max_new_skills=settings.max_new_skills,
            max_completion_tokens=settings.evolver_max_completion_tokens,
            history_path=config.skill_evolution_history_path,
        )
        logger.info(
            "SkillEvolver ON - evolver=%s every %d turns", settings.evolver_model, config.skill_evolution_every_n_turns
        )
    else:
        logger.info("SkillEvolver OFF - frozen library (eval phase)")
    worker = AsyncRolloutWorker(
        config=config,
        sampling_client=None,
        skill_manager=manager,
        prm_scorer=None,
        skill_evolver=evolver,
        memory_manager=None,
    )
    # With a tokenizer the proxy would build RL training samples into a queue nobody drains: the wrong path entirely.
    if worker._server._tokenizer is not None:
        sys.exit(
            f"metaclaw_proxy: ABORT: tokenizer for model_name={served!r} unexpectedly loaded; skills_only requires "
            "the tokenizer-None path."
        )
    return worker


async def serve(settings: Settings) -> None:
    worker = build(settings)
    worker.resume_submission()
    worker.start()
    logger.info(
        "skills_only proxy READY on %s:%d -> %s (auto_evolve=%s, skills_dir=%s)",
        HOST,
        settings.port,
        settings.llm_base,
        settings.evolve,
        settings.skills_dir,
    )
    try:
        while True:
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        worker.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    try:
        settings = read_settings(os.environ)
    except SettingsError as error:
        sys.exit(f"metaclaw_proxy: {error}")
    root, got = check_metaclaw(METACLAW_PACKAGE, settings.expect_sha256)
    print(
        f"METACLAW patched {root} skill_evolver={got['skill_evolver.py']} api_server={got['api_server.py']}",
        flush=True,
    )
    asyncio.run(serve(settings))


if __name__ == "__main__":
    main()
