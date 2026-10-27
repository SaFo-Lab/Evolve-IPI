"""The ``evolve-ipi`` command line.

    evolve-ipi check                          verify the installation, offline
    evolve-ipi eval  --model M --task T       evaluate the base agent, or an adapted one (--adapted)
    evolve-ipi adapt rl|skills --model M --task T
    evolve-ipi run CONFIG                     adapt and evaluate every combination of an experiment file

The commands check what they can before they start a process or make a paid call (``runner.startup``), and stop with
one line on stderr and an exit status of their own for every expected kind of error. A signal that asks the process to
end (SIGTERM, SIGHUP) is treated like Ctrl-C: everything the command started is stopped first, then it exits with
130 and says which run directory to resume. A secret is never printed.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any

from evolve_ipi import check, runner
from evolve_ipi.adapt import AdaptAbort
from evolve_ipi.adapt.skills import LibraryError
from evolve_ipi.agent.episode import LeftoverState
from evolve_ipi.agent.openclaw import OpenClawError
from evolve_ipi.children import StartFailed
from evolve_ipi.config import TASK_NAMES, ConfigError, load_configs
from evolve_ipi.environment.state import StateError
from evolve_ipi.environment.workspace import WorkspaceError
from evolve_ipi.evaluate import EvalError
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import PortInUse, ProcessError
from evolve_ipi.records import RunDir, RunDirError, RunLocked
from evolve_ipi.tasks import TaskDataError
from evolve_ipi.thirdparty import PatchError

__all__ = ["build_parser", "main"]

DEFAULT_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

# --- exit status ------------------------------------------------------------------------------------------------------

EXIT_UNEXPECTED = 1  # what Python itself exits with after an uncaught exception, which is printed with its traceback
EXIT_USAGE = 2  # argparse
EXIT_CONFIG = 3
EXIT_TASK_DATA = 4
EXIT_PATCH = 5
EXIT_PORT_IN_USE = 6
EXIT_LEFTOVER = 7
EXIT_ADAPT = 8
EXIT_EVAL = 9
EXIT_OPENCLAW = 10
EXIT_PROCESS = 11
# 12 is not used: a checkpoint that is missing or does not load is found by a tool in a process of its own, and is
# reported as the adaptation (8) or the evaluation (9) that needed it.
EXIT_RUN_DIR = 13
EXIT_RUN_LOCKED = 14
EXIT_ENVIRONMENT = 15
EXIT_CHECK_FAILED = 16
EXIT_INTERRUPTED = 130

# Exception -> (exit status, how the line starts). The first match wins, so a subclass comes before its parent.
_EXPECTED: tuple[tuple[type[BaseException], int, str], ...] = (
    (RunLocked, EXIT_RUN_LOCKED, "run directory in use"),
    (RunDirError, EXIT_RUN_DIR, "run directory error"),
    (LibraryError, EXIT_RUN_DIR, "skill library error"),
    (ConfigError, EXIT_CONFIG, "configuration error"),
    (TaskDataError, EXIT_TASK_DATA, "task data error"),
    (PatchError, EXIT_PATCH, "framework patch error"),
    (PortInUse, EXIT_PORT_IN_USE, "port in use"),
    (LeftoverState, EXIT_LEFTOVER, "leftover state"),
    (AdaptAbort, EXIT_ADAPT, "adaptation failed"),
    (EvalError, EXIT_EVAL, "evaluation failed"),
    (OpenClawError, EXIT_OPENCLAW, "OpenClaw error"),
    (ProcessError, EXIT_PROCESS, "process error"),
    (StartFailed, EXIT_PROCESS, "start failed"),  # not a ProcessError: it must stop the evolution, not restart it
    (StateError, EXIT_ENVIRONMENT, "simulated services error"),
    (WorkspaceError, EXIT_ENVIRONMENT, "workspace error"),
)


_EXIT_STATUS_HELP = """\
exit status:
  0    success
  1    an unexpected error, printed with its traceback
  2    a mistake in the command line
  3    configuration: a YAML file, a secret, a name, a port, a runtime directory, or a file under data/ that
       differs from data/SHA256SUMS
  4    a mistake in the task data under data/
  5    the OpenClaw-RL and MetaClaw files are not the patched ones
  6    a port is in use (the owner is named; nothing is stopped)
  7    leftover state in the agent's home (the message says what to remove)
  8    an adaptation failed after its restarts (a checkpoint that is missing or does not load included)
  9    an evaluation pass was not valid after its reruns, or could not start (a checkpoint that does not load
       included)
  10   OpenClaw is missing, is not 2026.3.7, or failed
  11   a child process did not start or failed
  13   a run directory cannot be trusted or was started under another configuration
  14   another run holds the run directory, the runtime directory or the agent's home
  15   the simulated services or the agent's workspace are not what the run needs
  16   check: at least one check failed
  130  interrupted by Ctrl-C, SIGTERM or SIGHUP (adapt, eval and run keep what finished)
"""

_ENVIRONMENT_HELP = """\
environment (each can also be set in a .env file, see --env-file):
  TINKER_API_KEY              Tinker API key; every command except check needs it
  EVOLVER_BASE_URL            OpenAI-compatible endpoint of the skill evolver (adapt skills, run)
  EVOLVER_API_KEY             API key for that endpoint
  EVOLVER_MODEL               name of the evolver model; there is no default
  HF_TOKEN                    Hugging Face token, only for a gated tokenizer
  EVOLVE_IPI_RUNTIME          where a run keeps its OpenClaw profiles and service state (default /var/lib/evolve-ipi);
                              a run holds it and the home (HOME) exclusively, so two runs at once need one each
  EVOLVE_IPI_OUTPUTS          default of --outputs
  EVOLVE_IPI_METACLAW_PYTHON  the python of the environment MetaClaw is installed in
"""


# --- the parser -------------------------------------------------------------------------------------------------------


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a positive number")
    return value


def _common_options(configs_default: str) -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("where things are")
    group.add_argument(
        "--configs",
        type=Path,
        metavar="DIR",
        help="directory with models.yaml, rl.yaml, skills.yaml, eval.yaml and runtime.yaml "
        f"(default: {configs_default})",
    )
    group.add_argument(
        "--outputs",
        type=Path,
        metavar="DIR",
        help="where the run directories outputs/<model>/<task>/<mechanism>/ are made "
        "(default: $EVOLVE_IPI_OUTPUTS, else ./outputs)",
    )
    group.add_argument(
        "--env-file",
        type=Path,
        metavar="FILE",
        help="a file of KEY=value lines with the secrets (default: ./.env when there is one); "
        "variables in the environment win",
    )
    return common


def _combo_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, metavar="MODEL", help="a model of models.yaml, e.g. qwen3.8-27b")
    parser.add_argument(
        "--task", required=True, choices=TASK_NAMES, metavar="TASK", help="one of: " + ", ".join(TASK_NAMES)
    )


def build_parser() -> argparse.ArgumentParser:
    raw = argparse.RawDescriptionHelpFormatter
    common = _common_options("configs/ of this repository")
    run_common = _common_options("the directory of CONFIG")
    parser = argparse.ArgumentParser(
        prog="evolve-ipi",
        formatter_class=raw,
        description=(
            "evolve-ipi: self-evolution and indirect prompt injection\n\n"
            "Adapt an agent with reinforcement learning (OpenClaw-RL) or skill evolution (MetaClaw), and measure how\n"
            "often it carries out an instruction that a third party hid in content it was only asked to read.\n\n"
            "Every command except check calls paid services. Start with check, which calls none."
        ),
        epilog=_ENVIRONMENT_HELP + "\n" + _EXIT_STATUS_HELP,
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    check_parser = commands.add_parser(
        "check",
        formatter_class=raw,
        help="verify the installation, offline; makes no API calls",
        description=(
            "Verify the installation. No API call is made, no key is needed and nothing is spent.\n\n"
            "It checks data/ against data/SHA256SUMS, reads the task data and the patched frameworks, resets and\n"
            "judges every item of every task, shows that the workspace guards refuse what they are for, and runs a\n"
            "scripted agent through OpenClaw (and, for the skill line, the MetaClaw proxy) with a model that plays\n"
            "a script. The scripted agent uses a temporary home, never yours, and the ports of runtime.yaml, which\n"
            "must be free.\n\n"
            "One line is printed per check: PASS, FAIL, or SKIP when the check cannot run here, with the reason. A\n"
            "check is skipped outside the image for the host name (only the image pins it), when OpenClaw 2026.3.7 or\n"
            "the MetaClaw environment is missing, and when there is no git metadata. In the image, a host name other\n"
            "than evolve-ipi (docker run --hostname) fails. The exit status is 0 unless a check failed."
        ),
    )
    check_parser.add_argument(
        "--configs",
        type=Path,
        metavar="DIR",
        help="directory with the configuration files (default: configs/ of this repository)",
    )
    check_parser.set_defaults(handler=_check)

    eval_parser = commands.add_parser(
        "eval",
        parents=[common],
        formatter_class=raw,
        help="evaluate the base agent, or an adapted agent",
        description=(
            "Evaluate one agent: the base agent, or with --adapted the agent that an adaptation left. Needs\n"
            "TINKER_API_KEY. A pass runs every item of a setting once.\n\n"
            "settings (--setting):\n"
            "  1  the items the agent was adapted on, with the authorizing request: the execution rate\n"
            "  2  the same items, with the test request that only asks to read: the attack success rate\n"
            "  3  the test items of the task, with the test request: the attack success rate (the default)\n"
            "Settings 1 and 2 need the run directory of an adaptation to take the items from.\n\n"
            "The passes are written to the run directory of the combination, outputs/<model>/<task>/<mechanism>/,\n"
            "or to DIR with --adapted DIR, where `evolve-ipi run` finds them as well. A pass that is finished is not\n"
            "run again, so a command that was interrupted can be run again."
        ),
    )
    _combo_options(eval_parser)
    eval_parser.add_argument(
        "--mechanism",
        choices=("rl", "skills"),
        help="the line of the agent: rl (base weights) or skills (an empty skill library behind MetaClaw); "
        "default: the first mechanism of the model, or the one DIR was adapted with",
    )
    eval_parser.add_argument("--setting", type=int, choices=(1, 2, 3), default=3, help="the setting to run (default 3)")
    eval_parser.add_argument("--passes", type=_positive, default=1, metavar="N", help="how many passes (default 1)")
    eval_parser.add_argument(
        "--limit",
        type=_positive,
        metavar="N",
        help="only the first N items of the setting: a quick test. It is kept in limit<N>/ inside the run "
        "directory, so that it never takes the place of a full pass",
    )
    eval_parser.add_argument(
        "--adapted",
        type=Path,
        metavar="DIR",
        help="evaluate the agent the adaptation in this run directory left (it holds adapt/checkpoints.json "
        "or adapt/library.json) instead of the base agent",
    )
    eval_parser.set_defaults(handler=_eval)

    adapt_parser = commands.add_parser(
        "adapt",
        formatter_class=raw,
        help="adapt an agent on a task: rl or skills",
        description="Adapt an agent on a task, with RL (rl) or skill evolution (skills).",
    )
    mechanisms = adapt_parser.add_subparsers(dest="mechanism", metavar="MECHANISM", required=True)
    rl_parser = mechanisms.add_parser(
        "rl",
        parents=[common],
        formatter_class=raw,
        help="RL on the model's weights (OpenClaw-RL)",
        description=(
            "Adapt the model with RL: OpenClaw-RL's GRPO with its process reward model, on a LoRA on Tinker.\n"
            "Two stages of updates (rl.yaml says how many); the second resumes from the first with its optimizer\n"
            "state. Needs TINKER_API_KEY.\n\n"
            "The result is outputs/<model>/<task>/rl/adapt/checkpoints.json. A stage that is finished is never run\n"
            "again; one that was interrupted or failed starts again from its starting checkpoint, so a command\n"
            "that was interrupted can be run again and resume."
        ),
    )
    _combo_options(rl_parser)
    rl_parser.set_defaults(handler=_adapt)
    skills_parser = mechanisms.add_parser(
        "skills",
        parents=[common],
        formatter_class=raw,
        help="skill evolution (MetaClaw)",
        description=(
            "Evolve a skill library with MetaClaw: the conversations of skills.yaml, each followed by one call of the\n"
            "evolver.\n"
            "Needs TINKER_API_KEY and the evolver: EVOLVER_BASE_URL and EVOLVER_API_KEY (an OpenAI-compatible\n"
            "endpoint) and EVOLVER_MODEL (the paper used GPT-5.2; there is no default).\n\n"
            "The result is outputs/<model>/<task>/skills/adapt/library/ with library.json beside it. A finished\n"
            "library is never evolved again; an interrupted evolution starts again from an empty library, so a\n"
            "command that was interrupted can be run again and resume."
        ),
    )
    _combo_options(skills_parser)
    skills_parser.set_defaults(handler=_adapt)

    run_parser = commands.add_parser(
        "run",
        parents=[run_common],
        formatter_class=raw,
        help="adapt and evaluate every combination of an experiment file",
        description=(
            "Run an experiment: for every combination of CONFIG (configs/main.yaml is the paper's main experiment),\n"
            "in order, adapt the agent unless that is finished, then evaluate it in the settings CONFIG lists. Needs\n"
            "TINKER_API_KEY, and the evolver variables when CONFIG has a skills combination.\n\n"
            "Everything is checked before the first combination starts, including that no run directory was started\n"
            "under another configuration: each combination records its configuration in config.json and a run\n"
            "directory is refused when it differs. A command that was interrupted can be run again and resumes: a\n"
            "finished adaptation and a finished pass are never run again. At the end a table lists the combinations\n"
            "and where their summary.json is."
        ),
    )
    run_parser.add_argument("config", metavar="CONFIG", type=Path, help="an experiment file such as configs/main.yaml")
    run_parser.set_defaults(handler=_run)
    return parser


# --- logging and signals ----------------------------------------------------------------------------------------------


class _RedactingFormatter(logging.Formatter):
    """A log line, with its traceback, without any secret: a retried request or a child's log tail can quote one."""

    def format(self, record: logging.LogRecord) -> str:
        return _redact(super().format(record))


@contextmanager
def _logging() -> Iterator[None]:
    """INFO lines of the ``evolve_ipi`` loggers on stderr, with the time; put back as it was when the command ends."""
    logger = logging.getLogger("evolve_ipi")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


class _Interrupted(KeyboardInterrupt):
    """A signal that asks the process to end, raised where it arrived so that every ``finally`` and every ``except
    KeyboardInterrupt`` below treats it as Ctrl-C: the processes the command started are stopped first."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.signal_name = name


_SIGNALS = tuple(getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, name))


def _ignore_signals() -> None:
    """From the first interruption on, nothing may abandon the cleanup or the report: later signals are ignored. Only
    the main thread can set a handler; a command run in another thread never had one of its own."""
    if threading.current_thread() is not threading.main_thread():
        return
    for number in _SIGNALS:
        signal.signal(number, signal.SIG_IGN)


def _interrupt(number: int, frame: FrameType | None) -> None:
    name = signal.Signals(number).name
    _ignore_signals()
    try:  # os.write, not print: the signal may have arrived while stderr was being written
        os.write(2, f"evolve-ipi: {name} received; stopping what it started, which can take a minute\n".encode())
    except OSError:
        pass
    raise _Interrupted(name)


@contextmanager
def _signals() -> Iterator[None]:
    """Ctrl-C, SIGTERM and SIGHUP end the command: everything it started is stopped first, and only then it reports.

    A signal that was ignored when the command started stays ignored (``nohup`` ignores SIGHUP so that a command
    outlives its terminal); the handlers of all three are put back afterwards.
    """
    if threading.current_thread() is not threading.main_thread():
        yield  # only the main thread can take a signal
        return
    previous = {number: signal.getsignal(number) for number in _SIGNALS}
    for number, handler in previous.items():
        if handler is not signal.SIG_IGN:
            signal.signal(number, _interrupt)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


# --- what is said -----------------------------------------------------------------------------------------------------


def _secret_values() -> set[str]:
    values = {os.environ.get(name, "").strip() for name in ("TINKER_API_KEY", "EVOLVER_API_KEY", "HF_TOKEN")}
    for secrets in runner.loaded_secrets():
        values.update((secrets.tinker_api_key or "", secrets.evolver_api_key or "", secrets.hf_token or ""))
    return {value for value in values if len(value) >= 8}


def _redact(text: str) -> str:
    """The text without any secret in it: what a child process logged or a library quoted can hold one."""
    for value in sorted(_secret_values(), key=len, reverse=True):
        text = text.replace(value, "***")
    return text


def _say(text: str) -> None:
    print(_redact(text), file=sys.stderr, flush=True)


def _indented(text: str, by: int = 2) -> str:
    return "\n".join(" " * by + line for line in text.splitlines())


def _where(progress: runner.Progress) -> str:
    lines = []
    if progress.run_dir is not None:
        lines.append(f"  run directory: {progress.run_dir}")
    if progress.finished:
        lines += ["  combinations finished before this:", _indented(runner.format_table(progress.finished), 4)]
    return "\n".join(lines)


def _report_error(error: Exception, progress: runner.Progress) -> int:
    for kind, status, label in _EXPECTED:
        if isinstance(error, kind):
            message = f"evolve-ipi: {label}: {str(error) or type(error).__name__}"
            lines = [message, *([_where(progress)] if _where(progress) else [])]
            if isinstance(error.__context__, KeyboardInterrupt):
                lines.append("  this happened while stopping what the command started after an interruption")
            _say("\n".join(lines))
            return status
    _say("".join(traceback.format_exception(error)).rstrip())
    return EXIT_UNEXPECTED


def _report_interrupt(interrupt: KeyboardInterrupt, progress: runner.Progress, *, resumable: bool) -> int:
    """Said once everything the command started has been stopped: the unwinding that got here ended without an error."""
    _ignore_signals()
    by = f" by {interrupt.signal_name}" if isinstance(interrupt, _Interrupted) else ""
    stopped = f"evolve-ipi: interrupted{by}; what it started has been stopped"
    if not resumable:
        _say(stopped)
        return EXIT_INTERRUPTED
    lines = [f"{stopped}, and what finished is kept"]
    where = _where(progress)
    if where:
        lines.append(where)
    lines.append("  to resume, run the same command again")
    _say("\n".join(lines))
    return EXIT_INTERRUPTED


# --- the commands -----------------------------------------------------------------------------------------------------


def _paths(args: argparse.Namespace) -> RuntimePaths:
    return RuntimePaths(outputs=Path(args.outputs).expanduser().absolute()) if args.outputs else RuntimePaths()


def _env_file(args: argparse.Namespace) -> Path | None:
    if args.env_file is not None:
        return Path(args.env_file)  # a file that was named has to be there
    default = Path(".env")
    return default if default.is_file() else None


def _configs_dir(args: argparse.Namespace) -> Path:
    return Path(args.configs) if args.configs else DEFAULT_CONFIGS


def _check(args: argparse.Namespace, progress: runner.Progress) -> int:
    cfgs = load_configs(_configs_dir(args))
    results = check.run_checks(
        cfgs,
        metaclaw_python=RuntimePaths().metaclaw_python,
        report=lambda result: print(check.format_result(result), flush=True),
    )
    print(check.summary_line(results))
    return EXIT_CHECK_FAILED if any(result.status == "fail" for result in results) else 0


def _adapt(args: argparse.Namespace, progress: runner.Progress) -> int:
    cfgs = load_configs(_configs_dir(args))
    combo = runner.combo_of(cfgs, args.model, args.task, args.mechanism)
    start = runner.startup(cfgs, [combo], adapting=True, env_file=_env_file(args), paths=_paths(args))
    outcome = runner.adapt(start, combo, progress=progress)
    name = "RL adaptation" if args.mechanism == "rl" else "Skill evolution"
    print(
        f"{name} of {combo.model} on {combo.task}: "
        + ("finished" if outcome.adapted_now else "already finished, nothing was run")
    )
    for key, value in outcome.result.items():
        print(f"  {key}: {value}")
    print(f"  run directory: {outcome.run_dir}")
    return 0


def _percent(counts: dict[str, Any]) -> str:
    return f"{counts['hits']}/{counts['n']} ({100 * counts['hits'] / counts['n']:.1f}%)"


def _eval(args: argparse.Namespace, progress: runner.Progress) -> int:
    if args.adapted is not None and args.outputs is not None:
        raise ConfigError(f"--outputs has no effect with --adapted: the passes are written to {args.adapted}")
    cfgs = load_configs(_configs_dir(args))
    mechanism = runner.choose_mechanism(cfgs, args.model, args.mechanism, args.adapted)
    combo = runner.combo_of(cfgs, args.model, args.task, mechanism)
    start = runner.startup(
        cfgs, [combo], adapting=False, env_file=_env_file(args), paths=_paths(args), results=args.adapted
    )
    outcome = runner.eval_agent(
        start,
        combo,
        setting=args.setting,
        passes=args.passes,
        limit=args.limit,
        adapted=args.adapted,
        progress=progress,
    )
    print(
        f"{combo.model} / {combo.task} / {combo.mechanism}: {outcome.variant} agent, "
        f"setting {outcome.setting} ({outcome.measure})"
    )
    for record in outcome.passes:
        print(f"  pass {record['pass']}: {_percent(record['asr'])}")
    if len(outcome.passes) > 1:
        print(f"  pooled over {len(outcome.passes)} passes: {_percent(outcome.pooled)}")
    first = RunDir(outcome.run_dir).pass_dir(outcome.variant, outcome.setting, 1)
    print(f"  passes: {first.parent}")
    return 0


def _run(args: argparse.Namespace, progress: runner.Progress) -> int:
    results = runner.run(
        args.config, configs_dir=args.configs, env_file=_env_file(args), paths=_paths(args), progress=progress
    )
    print(runner.format_table(results))
    return 0


Handler = Callable[[argparse.Namespace, runner.Progress], int]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    handler: Handler = args.handler
    progress = runner.Progress()
    with _logging(), _signals():
        try:
            return handler(args, progress)
        except KeyboardInterrupt as interrupt:
            return _report_interrupt(interrupt, progress, resumable=args.command != "check")
        except Exception as error:  # noqa: BLE001 -- every expected kind is named in _EXPECTED; the rest is a bug
            return _report_error(error, progress)
        finally:
            runner.forget_secrets()
