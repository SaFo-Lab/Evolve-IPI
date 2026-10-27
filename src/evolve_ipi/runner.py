"""The sequence behind the commands: startup checks, the run directory of a combination, and ``run CONFIG``.

Every command checks what it can before it starts a process or makes a paid call (``startup``): the configuration, the
names it was given, the secrets it needs, the data under ``data/`` (against ``data/SHA256SUMS``) and the task data,
the runtime directories, the OpenClaw version, the patched frameworks and the ports; for an adaptation that is still to
run, also what that adaptation checks before it starts, for all combinations before the first one starts. It then holds
the runtime directory and the agent's home (``RuntimePaths.lock``) and works in the run directory of a combination
(``outputs/<model>/<task>/<mechanism>``) while holding that directory's lock, after recording the configuration it
runs under there. A directory that was started under another configuration is refused, naming the keys that differ.
Only then, with every lock it needs held, does a command that evolves a skill library ask the evolver once whether it
answers: one request, whatever the number of libraries, and none when a lock is refused. In the image, the host name,
the volume under the results and an init process, which only ``docker run`` decides, are checked first.

``run`` walks the combinations of an experiment file in order: the adaptation unless it is finished, then the
evaluation. Both resume by themselves, so running the same command again after an interruption goes on where it
stopped and repeats nothing that finished.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evolve_ipi.adapt.rl import adapt_rl, check_setup
from evolve_ipi.adapt.skills import adapt_skills, check_inputs, evolver_preflight
from evolve_ipi.agent.openclaw import GATEWAY_SPAN, OPENCLAW_VERSION, check_version
from evolve_ipi.children import metaclaw_python
from evolve_ipi.config import (
    MECHANISMS,
    TASK_NAMES,
    Combo,
    ConfigError,
    Configs,
    ModelSpec,
    PortsSpec,
    Secrets,
    as_mechanism,
    load_configs,
    load_secrets,
)
from evolve_ipi.evaluate import MEASURE, Base, adapted_agent, evaluate_combo, items_for_setting, pool, run_pass
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import require_free
from evolve_ipi.records import RunDir, Variant
from evolve_ipi.tasks import Task, load_rl_order, load_skill_order, load_task, verify_data
from evolve_ipi.thirdparty import PINNED_COMMITS, verify_patches

__all__ = [
    "IMAGE_HOSTNAME",
    "AdaptOutcome",
    "ComboResult",
    "EvalOutcome",
    "Progress",
    "Startup",
    "adapt",
    "adapt_combo",
    "check_image_hostname",
    "check_openclaw",
    "choose_mechanism",
    "combo_of",
    "eval_agent",
    "forget_secrets",
    "format_table",
    "in_image",
    "loaded_secrets",
    "ports_needed",
    "resolved_config",
    "run",
    "run_dir_of",
    "startup",
]

logger = logging.getLogger(__name__)

# The image sets this. What it pins about the machine (the host name OpenClaw writes into the agent's system prompt, the
# outputs volume) depends on the flags of `docker run`, so these are checked only there.
IMAGE_VARIABLE = "EVOLVE_IPI_IMAGE"
IMAGE_HOSTNAME = "evolve-ipi"
MOUNTINFO = Path("/proc/self/mountinfo")
_EVOLVER_SECRETS = ("evolver_base_url", "evolver_api_key", "evolver_model")
_LOADED: list[Secrets] = []  # every Secrets this process has read, for the CLI to keep out of what it prints


# --- progress ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComboResult:
    """One finished combination: where it lives, whether its adaptation ran in this invocation, its summary."""

    model: str
    task: str
    mechanism: str
    run_dir: Path
    adapted_now: bool
    summary: Path


@dataclass
class Progress:
    """Where a command is, for the CLI to say what to resume after an interruption or a failure, and what finished."""

    combo: str | None = None
    run_dir: Path | None = None
    finished: list[ComboResult] = field(default_factory=list)

    def begin(self, combo: str, run_dir: Path) -> None:
        self.combo, self.run_dir = combo, run_dir


# --- the names a command was given ------------------------------------------------------------------------------------


def _label(combo: Combo) -> str:
    return f"{combo.model} / {combo.task} / {combo.mechanism}"


def _model(cfgs: Configs, name: str) -> ModelSpec:
    if name not in cfgs.models:
        raise ConfigError(f"unknown model {name!r}; known models: {', '.join(cfgs.models)}")
    return cfgs.models[name]


def combo_of(cfgs: Configs, model: str, task: str, mechanism: str) -> Combo:
    """One combination named on the command line, checked against ``models.yaml`` and the five tasks."""
    spec = _model(cfgs, model)
    if task not in TASK_NAMES:
        raise ConfigError(f"unknown task {task!r}; known tasks: {', '.join(TASK_NAMES)}")
    as_mechanism(mechanism)
    if mechanism not in spec.mechanisms:
        raise ConfigError(
            f"model {model} does not take part in {mechanism!r}; models.yaml gives it: {', '.join(spec.mechanisms)}"
        )
    return Combo(model=model, task=task, mechanism=mechanism)


def choose_mechanism(cfgs: Configs, model: str, requested: str | None, adapted: Path | None) -> str:
    """The mechanism an evaluation is about.

    An adapted agent has the mechanism whose adaptation its run directory holds; a base agent has the one asked for, or
    the first the model takes part in.
    """
    spec = _model(cfgs, model)
    if adapted is None:
        return requested or spec.mechanisms[0]
    source = RunDir(Path(adapted))
    if not source.root.is_dir():
        raise ConfigError(f"--adapted {adapted} is not a directory")
    done = [mechanism for mechanism in MECHANISMS if source.adapt_done(mechanism)]
    if not done:
        raise ConfigError(
            f"--adapted {adapted} holds no finished adaptation: it needs adapt/checkpoints.json (RL) or "
            "adapt/library.json (skill evolution)"
        )
    if len(done) > 1:
        raise ConfigError(f"--adapted {adapted} holds a finished adaptation of both mechanisms; it must hold one")
    if requested is not None and requested != done[0]:
        raise ConfigError(f"--adapted {adapted} was adapted with {done[0]}, not with {requested}")
    return done[0]


# --- startup ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Startup:
    """What a command has once its checks passed: the configuration, the secrets it needs, the paths and the tasks."""

    cfgs: Configs
    secrets: Secrets
    paths: RuntimePaths
    patched: dict[str, str]  # sha256 of each patched framework file
    tasks: dict[str, Task]
    data_sums: str  # sha256 of data/SHA256SUMS, whose files were checked against it


def in_image() -> bool:
    return os.environ.get(IMAGE_VARIABLE) == "1"


def _hostname() -> str:
    return socket.gethostname()


def _is_first_process() -> bool:
    return os.getpid() == 1


def check_image_hostname() -> None:
    """Raise ``ConfigError`` unless the host name is the one the image pins. OpenClaw writes it into the agent's system
    prompt (``host=...``); without ``--hostname`` a container's name is its id, and the model would see another input."""
    name = _hostname()
    if name != IMAGE_HOSTNAME:
        raise ConfigError(
            f"the host name of this container is {name!r}, but the image runs with --hostname {IMAGE_HOSTNAME}: OpenClaw "
            "writes the host name into the agent's system prompt, so another one changes what the model sees. "
            f"Start the container with `docker run --hostname {IMAGE_HOSTNAME} ...`"
        )


def _mount_points(mountinfo: Path) -> set[str]:
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ConfigError(
            f"cannot tell what is mounted: {mountinfo} cannot be read ({error.strerror or error})"
        ) from None
    # field 5 is the mount point; a space, tab, newline or backslash in it is written as an octal escape
    return {
        re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), line.split()[4]) for line in lines if len(line.split()) > 4
    }


def check_mounted(directory: Path, *, mountinfo: Path | None = None) -> None:
    """Raise ``ConfigError`` unless ``directory``, or a directory above it other than ``/``, is a mount point: what a
    container writes anywhere else is lost with it (``docker run --rm``)."""
    points = _mount_points(mountinfo if mountinfo is not None else MOUNTINFO)
    path = Path(os.path.realpath(directory))
    if not any(str(candidate) in points for candidate in (path, *path.parents) if str(candidate) != "/"):
        raise ConfigError(
            f"the output directory {directory} is not on a mounted volume, so the results would disappear with the "
            'container (`docker run --rm`). Mount one: -v "$PWD/outputs:/work/outputs" (or point --outputs at a mounted '
            "directory)"
        )


def _image_guards(results: Path) -> None:
    """In the image, the three things the flags of ``docker run`` decide: the host name, the volume the results are
    written to, and an init process that reaps what a stopped run leaves."""
    if not in_image():
        return
    check_image_hostname()
    check_mounted(results)
    if _is_first_process():
        logger.warning(
            "this run is the first process of the container: run with --init so stopped processes are reaped"
        )


def loaded_secrets() -> list[Secrets]:
    """The secrets this process has read since ``forget_secrets``."""
    return list(_LOADED)


def forget_secrets() -> None:
    _LOADED.clear()


def _check_port_config(ports: PortsSpec) -> None:
    first, last = ports.gateway, ports.gateway + GATEWAY_SPAN - 1
    problems = [
        f"ports.{name} ({getattr(ports, name)}) lies among the gateway's {GATEWAY_SPAN} ports {first} to {last}"
        for name in ("policy", "proxy", "model")
        if first <= getattr(ports, name) <= last
    ]
    if last > 65535:
        problems.append(f"ports.gateway ({first}) leaves no room for the {GATEWAY_SPAN - 1} ports above it")
    if problems:
        raise ConfigError("runtime.yaml: " + "; ".join(problems) + ". The gateway listens on all of its ports.")


def _writable_location(directory: Path, what: str, variable: str) -> None:
    """``directory`` exists and may be written to, or can be made below a directory that may. Nothing is made here."""
    existing = directory
    while not existing.exists():
        existing = existing.parent
    advice = f"set {variable} to a directory that you may write to"
    if not existing.is_dir():
        raise ConfigError(f"the {what} {directory} cannot be made: {existing} is not a directory; {advice}")
    if not os.access(existing, os.W_OK | os.X_OK):
        raise ConfigError(f"the {what} {directory} cannot be written: {existing} is not writable; {advice}")


def _check_paths(paths: RuntimePaths, combos: Sequence[Combo], tasks: dict[str, Task]) -> None:
    _writable_location(paths.root, "runtime directory", "EVOLVE_IPI_RUNTIME")
    _writable_location(paths.outputs, "output directory", "EVOLVE_IPI_OUTPUTS (or --outputs)")
    if not paths.agent_home.is_dir():
        raise ConfigError(f"the agent's home {paths.agent_home} is not a directory")
    if not os.access(paths.agent_home, os.W_OK | os.X_OK):
        raise ConfigError(f"the agent's home {paths.agent_home} is not writable")
    for line, kind in sorted({(as_mechanism(combo.mechanism), tasks[combo.task].kind) for combo in combos}):
        paths.workspace(line, kind)  # refuses a workspace inside a git checkout


def _probe_env(directory: Path) -> dict[str, str]:
    """The environment to ask ``openclaw --version`` in: a home of its own, so that no profile is made in a real one."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(directory),
        "LANG": "C.UTF-8",
        "OPENCLAW_STATE_DIR": str(directory / "state"),
        "OPENCLAW_CONFIG_PATH": str(directory / "openclaw.json"),
    }


def check_openclaw() -> str:
    """Raise ``OpenClawError`` unless the OpenClaw on PATH is the pinned version; return that version."""
    with tempfile.TemporaryDirectory(prefix="evolve-ipi-probe-") as probe:
        check_version(_probe_env(Path(probe)))
    return OPENCLAW_VERSION


def ports_needed(ports: PortsSpec, combos: Sequence[Combo]) -> list[int]:
    """The local ports the combinations use: the gateway's, the RL line's endpoint, the skill line's proxy and model."""
    needed = list(range(ports.gateway, ports.gateway + GATEWAY_SPAN))
    if any(combo.mechanism == "rl" for combo in combos):
        needed.append(ports.policy)
    if any(combo.mechanism == "skills" for combo in combos):
        needed += [ports.proxy, ports.model]
    return needed


def _secret_fields(combos: Sequence[Combo], adapting: bool) -> tuple[str, ...]:
    fields = ["tinker_api_key"]
    if adapting and any(combo.mechanism == "skills" for combo in combos):
        fields += _EVOLVER_SECRETS  # the evolver writes the skills; evaluating a finished library never calls it
    return tuple(fields)


def startup(
    cfgs: Configs,
    combos: Sequence[Combo],
    *,
    adapting: bool,
    env_file: Path | None = None,
    paths: RuntimePaths | None = None,
    results: Path | None = None,
) -> Startup:
    """Check everything a command about ``combos`` needs, before it starts a process or makes a paid call.

    ``adapting`` is whether it runs an adaptation, which for skill evolution needs the evolver: the checks of each
    adaptation that is still to run (``rl.check_setup``, ``skills.check_inputs``) are made here, last, for all
    combinations. The one request to the evolver is not: it costs money, so it waits until the command holds its locks
    (``adapt``, ``run``). Raises ``ConfigError`` (the configuration, the secrets, the data under ``data/``, the runtime
    directories, the task orders), ``TaskDataError``, ``OpenClawError`` (not the pinned version), ``PatchError``
    (frameworks not patched) or ``PortInUse``, each naming what is wrong. In the image (``EVOLVE_IPI_IMAGE=1``) the host
    name and the volume ``results`` (default: the outputs) is written to are checked first.
    """
    paths = paths if paths is not None else RuntimePaths()
    _image_guards(results if results is not None else paths.outputs)
    _check_port_config(cfgs.ports)
    secrets = load_secrets(_secret_fields(combos, adapting), env_file=env_file)
    _LOADED.append(secrets)
    data_sums = verify_data()
    tasks = {name: load_task(name) for name in dict.fromkeys(combo.task for combo in combos)}
    for combo in combos:
        if combo.mechanism == "rl":
            load_rl_order(combo.task)
        else:
            order = load_skill_order(combo.task)
            if len(order) != cfgs.skills.conversations:
                raise ConfigError(
                    f"the skill order of {combo.task} holds {len(order)} conversations, but skills.yaml says "
                    f"{cfgs.skills.conversations}"
                )
    _check_paths(paths, combos, tasks)
    if any(combo.mechanism == "skills" for combo in combos):
        metaclaw_python(paths)
    check_openclaw()
    patched = verify_patches()
    require_free(ports_needed(cfgs.ports, combos))
    if adapting:
        _check_adaptations(cfgs, combos, tasks, secrets, paths)
    return Startup(cfgs, secrets, paths, patched, tasks, data_sums)


def _pending(paths: RuntimePaths, combos: Sequence[Combo]) -> list[Combo]:
    return [combo for combo in combos if not run_dir_of(paths, combo).adapt_done(combo.mechanism)]


def _check_adaptations(
    cfgs: Configs, combos: Sequence[Combo], tasks: dict[str, Task], secrets: Secrets, paths: RuntimePaths
) -> None:
    """What each adaptation that is still to be run checks before it starts, for all of them now, so that a mistake of
    the third combination is found before the first one is paid for. A finished adaptation checks nothing."""
    for combo in _pending(paths, combos):
        model = cfgs.models[combo.model]
        if combo.mechanism == "rl":
            check_setup(model, tasks[combo.task], cfgs, secrets)
        else:
            check_inputs(model, cfgs, secrets, paths, load_skill_order(combo.task))


def _ask_the_evolver(start: Startup, combos: Sequence[Combo]) -> None:
    """One request to the evolver when a skill library of ``combos`` is still to be evolved, for all of them; the
    caller holds the locks its work needs, so that a run refused for a lock it shares costs nothing."""
    assert start.paths.holds_lock(), "the evolver is asked only once the runtime directory and the home are held"
    if any(combo.mechanism == "skills" for combo in _pending(start.paths, combos)):
        evolver_preflight(start.secrets, start.cfgs.skills, paths=start.paths)


# --- the run directory of a combination -------------------------------------------------------------------------------


def run_dir_of(paths: RuntimePaths, combo: Combo) -> RunDir:
    return RunDir.for_combo(paths.outputs, combo.model, combo.task, combo.mechanism)


def resolved_config(
    cfgs: Configs, model: ModelSpec, task: str, mechanism: str, patched: dict[str, str], data_sums: str
) -> dict[str, Any]:
    """Everything that decides what a combination measures; recorded as ``config.json``, and a run directory is used
    only under the same configuration again. ``data_sums`` is the sha256 of ``data/SHA256SUMS``: a run directory is
    used only with the data it was started on.

    The ports, the paths and the secrets are not part of it: they are where and by whom the run is done. Neither is the
    list of settings an experiment file asks for, which only selects what is run.
    """
    recipe = cfgs.rl if mechanism == "rl" else cfgs.skills
    return {
        "model": model.model_dump(mode="json"),
        "task": task,
        "mechanism": mechanism,
        mechanism: recipe.model_dump(mode="json"),
        "eval": cfgs.eval.model_dump(mode="json"),
        "openclaw": OPENCLAW_VERSION,
        "frameworks": {"commits": dict(PINNED_COMMITS), "patched_files": dict(patched)},
        "data_sha256sums": data_sums,
    }


def _verify_recorded(run: RunDir, config: dict[str, Any]) -> None:
    """Refuse a run directory recorded under another configuration, without writing anything: ``write_config`` of the
    configuration that is recorded already changes nothing, and of another one raises, naming the keys that differ."""
    if run.config_path.exists():
        run.write_config(config)


def adapt_combo(start: Startup, run: RunDir, model: ModelSpec, task: Task, mechanism: str) -> bool:
    """Adapt the agent in ``run`` unless that is finished; returns whether the adaptation ran now. The caller holds the
    lock of ``run`` and has asked the evolver already (``_ask_the_evolver``), which skill evolution then does not."""
    if run.adapt_done(mechanism):
        logger.info("%s on %s with %s: the adaptation is finished, kept", model.name, task.name, mechanism)
        return False
    if mechanism == "rl":
        adapt_rl(model, task, run, start.cfgs, start.secrets, paths=start.paths)
    else:
        adapt_skills(model, task, run, start.cfgs, start.secrets, paths=start.paths, preflight=False)
    return True


# --- adapt ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptOutcome:
    run_dir: Path
    adapted_now: bool
    result: dict[str, str]  # what the adaptation left: the checkpoints of RL, or the library and its hash


def _describe_adaptation(run: RunDir, mechanism: str) -> dict[str, str]:
    if mechanism == "rl":
        marker = json.loads((run.adapt_dir / "checkpoints.json").read_text(encoding="utf-8"))
        return {"update 32": marker["update_32"], "update 64": marker["update_64"]}
    marker = json.loads((run.adapt_dir / "library.json").read_text(encoding="utf-8"))
    return {
        "library": str(run.adapt_dir / marker["library"]),
        "skills": str(marker["skills"]),
        "sha256": marker["sha"],
    }


def adapt(start: Startup, combo: Combo, *, progress: Progress | None = None) -> AdaptOutcome:
    """Adapt one combination in its run directory (``adapt rl`` and ``adapt skills``)."""
    progress = progress if progress is not None else Progress()
    model, task = start.cfgs.models[combo.model], start.tasks[combo.task]
    run = run_dir_of(start.paths, combo)
    config = resolved_config(start.cfgs, model, combo.task, combo.mechanism, start.patched, start.data_sums)
    with start.paths.lock(), run.lock():
        progress.begin(_label(combo), run.root)
        run.write_config(config)
        _ask_the_evolver(start, [combo])
        adapted_now = adapt_combo(start, run, model, task, combo.mechanism)
        result = _describe_adaptation(run, combo.mechanism)
    return AdaptOutcome(run.root, adapted_now, result)


# --- eval -------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalOutcome:
    run_dir: Path
    variant: Variant
    mechanism: str
    setting: int
    measure: str
    passes: list[dict[str, Any]]
    pooled: dict[str, Any]


def _check_adapted_matches(source: RunDir, combo: Combo) -> None:
    """A run directory that records what it was made for must have been made for this model, task and mechanism."""
    if not source.config_path.exists():
        return
    recorded = source.read_config()
    found = (recorded.get("model", {}).get("name"), recorded.get("task"), recorded.get("mechanism"))
    wanted = (combo.model, combo.task, combo.mechanism)
    if found != wanted:
        raise ConfigError(
            f"--adapted {source.root} was adapted for {' / '.join(map(str, found))}, not for {' / '.join(wanted)}"
        )


def eval_agent(
    start: Startup,
    combo: Combo,
    *,
    setting: int,
    passes: int,
    limit: int | None = None,
    adapted: Path | None = None,
    progress: Progress | None = None,
) -> EvalOutcome:
    """Evaluate one agent in one setting: the base agent, or the agent the adaptation in ``adapted`` left.

    The passes are written to the combination's run directory, or to ``adapted`` when that is given, where
    ``run`` finds them as well. With ``limit`` only the first items of the setting are run, in ``limit<N>/`` inside that
    directory, so that a pass of a few items never takes the place of a pass of all. Settings 1 and 2 use the items the
    agent was adapted on, which the run directory of the adaptation says.
    """
    progress = progress if progress is not None else Progress()
    model, task = start.cfgs.models[combo.model], start.tasks[combo.task]
    source = RunDir(Path(adapted)) if adapted is not None else run_dir_of(start.paths, combo)
    out = RunDir(source.root / f"limit{limit}") if limit else source
    variant: Variant = "adapted" if adapted is not None else "base"
    mechanism = as_mechanism(combo.mechanism)
    config = resolved_config(start.cfgs, model, combo.task, combo.mechanism, start.patched, start.data_sums)
    if limit:
        config["limit"] = limit
    if adapted is not None:
        _check_adapted_matches(source, combo)
    if setting != 3 and not source.adapt_done(mechanism):  # before anything is made: a refusal leaves no directory
        raise ConfigError(
            f"setting {setting} uses the items the agent was adapted on, and {source.adapt_dir} holds no finished "
            f"adaptation: run `evolve-ipi adapt {mechanism}` first, or evaluate setting 3"
        )
    with start.paths.lock(), out.lock():
        progress.begin(_label(combo), out.root)
        out.write_config(config)
        agent = adapted_agent(source, mechanism, model, start.cfgs) if adapted is not None else Base(mechanism)
        items = items_for_setting(task, setting, source)
        if limit:
            items = items[:limit]
        records = [
            run_pass(
                out, variant, setting, k, agent, model, task, start.cfgs, start.secrets, paths=start.paths, items=items
            )
            for k in range(1, passes + 1)
        ]
    return EvalOutcome(out.root, variant, mechanism, setting, MEASURE[setting], records, pool(task, records))


# --- run CONFIG -------------------------------------------------------------------------------------------------------


def _plan(start: Startup, cfgs: Configs) -> list[tuple[Combo, ModelSpec, Task, RunDir, dict[str, Any]]]:
    """Each combination with its run directory and configuration, after the checks that need no process: no run
    directory was started under another configuration, and a finished adaptation is still what it recorded and what the
    evaluation will read (a library that changed, a checkpoint that is gone)."""
    assert cfgs.main is not None
    plan = []
    for combo in cfgs.main.combos:
        model = cfgs.models[combo.model]
        config = resolved_config(cfgs, model, combo.task, combo.mechanism, start.patched, start.data_sums)
        run_dir = run_dir_of(start.paths, combo)
        _verify_recorded(run_dir, config)
        if run_dir.adapt_done(combo.mechanism):
            adapted_agent(run_dir, combo.mechanism, model, cfgs)
        plan.append((combo, model, start.tasks[combo.task], run_dir, config))
    return plan


def run(
    config_path: Path,
    *,
    configs_dir: Path | None = None,
    env_file: Path | None = None,
    paths: RuntimePaths | None = None,
    progress: Progress | None = None,
) -> list[ComboResult]:
    """Adapt and evaluate every combination of the experiment file ``config_path``, in order.

    The other configuration files are read from ``configs_dir``, by default the directory of ``config_path``. All
    startup checks, the check that no combination was recorded under another configuration and the one request to the
    evolver come before the first combination starts; the request is made while the run holds every run directory of
    the experiment, so that one in use is refused first. A combination whose adaptation is finished is not adapted
    again, and its evaluation goes on from the passes that are complete. Stops at the first failure, with the
    combinations that finished in ``progress``.
    """
    progress = progress if progress is not None else Progress()
    config_path = Path(config_path)
    cfgs = load_configs(configs_dir if configs_dir is not None else config_path.parent, config_path)
    assert cfgs.main is not None
    combos = cfgs.main.combos
    start = startup(cfgs, combos, adapting=True, env_file=env_file, paths=paths)
    results: list[ComboResult] = []
    with start.paths.lock():
        plan = _plan(start, cfgs)
        with ExitStack() as held:
            for run_dir in {run_dir.root: run_dir for _, _, _, run_dir, _ in plan}.values():
                held.enter_context(run_dir.lock())
            _ask_the_evolver(start, combos)
        for number, (combo, model, task, run_dir, config) in enumerate(plan, start=1):
            progress.begin(_label(combo), run_dir.root)
            logger.info("[%d/%d] %s", number, len(plan), _label(combo))
            with run_dir.lock():
                run_dir.write_config(config)
                adapted_now = adapt_combo(start, run_dir, model, task, combo.mechanism)
                evaluate_combo(
                    run_dir,
                    model,
                    task,
                    combo.mechanism,
                    cfgs,
                    start.secrets,
                    paths=start.paths,
                    settings=cfgs.main.settings,
                )
            result = ComboResult(
                combo.model, combo.task, combo.mechanism, run_dir.root, adapted_now, run_dir.summary_path
            )
            results.append(result)
            progress.finished.append(result)
    logger.info("all %d combinations finished", len(results))
    return results


def format_table(results: Sequence[ComboResult]) -> str:
    """The combinations that finished and where their summaries are. ``ran`` is an adaptation that ran in this
    invocation, ``kept`` one that an earlier invocation finished."""
    if not results:
        return "no combination finished"
    rows = [("model", "task", "mechanism", "adaptation", "summary")]
    rows += [(r.model, r.task, r.mechanism, "ran" if r.adapted_now else "kept", str(r.summary)) for r in results]
    widths = [max(len(row[column]) for row in rows) for column in range(5)]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in rows
    )
