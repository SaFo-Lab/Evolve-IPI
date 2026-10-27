"""The ``evolve-ipi`` command line: its options, what it checks before it spends anything, what it says when something
is wrong, and what it does when it is stopped.

``adapt_rl``, ``adapt_skills``, ``evaluate_combo`` and ``run_pass`` are stand-ins (``cli_helpers.Work``), so no test
here starts a process or reaches a paid service.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from cli_helpers import (
    PATCHED,
    SECRET_VALUES,
    Work,
    combo,
    edit_yaml,
    free_ports,
    in_image,
    install_world,
    write_configs,
)

from evolve_ipi import check, cli, procs, runner, tasks
from evolve_ipi.adapt import AdaptAbort, rl, skills
from evolve_ipi.adapt.library import lib_sha
from evolve_ipi.adapt.skills import LibraryError, StartFailed
from evolve_ipi.agent.episode import LeftoverState
from evolve_ipi.agent.openclaw import OpenClawError
from evolve_ipi.config import ConfigError, PortsSpec
from evolve_ipi.environment.state import StateError
from evolve_ipi.environment.workspace import WorkspaceError
from evolve_ipi.evaluate import EvalError
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import PortInUse, ProcessError
from evolve_ipi.records import RunDir, RunDirError, RunLocked, write_json
from evolve_ipi.tasks import TaskDataError
from evolve_ipi.thirdparty import PatchError

MODEL = "qwen3.8-27b"
TASK = "file-create"
ADAPT_RL = ["adapt", "rl", "--model", MODEL, "--task", TASK]
ADAPT_SKILLS = ["adapt", "skills", "--model", MODEL, "--task", TASK]


@dataclass
class Cli:
    """The command line run in process, with ``--configs`` pointing at a copy of ``configs/`` that has free ports."""

    tmp_path: Path
    work: Work
    configs: Path
    monkeypatch: pytest.MonkeyPatch
    capsys: pytest.CaptureFixture[str]

    def __call__(self, *argv: str) -> tuple[int, str, str]:
        extra = [] if argv[0] in ("run",) or "--configs" in argv else ["--configs", str(self.configs)]
        code = cli.main([*argv, *extra])
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err

    def run_dir(self, mechanism: str = "rl", model: str = MODEL, task: str = TASK) -> RunDir:
        return RunDir.for_combo(self.tmp_path / "outputs", model, task, mechanism)


@pytest.fixture
def c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Iterator[Cli]:
    work = install_world(monkeypatch, tmp_path)
    handlers_before = list(logging.getLogger("evolve_ipi").handlers)
    signals_before = {name: signal.getsignal(getattr(signal, name)) for name in ("SIGINT", "SIGTERM", "SIGHUP")}
    yield Cli(tmp_path, work, write_configs(tmp_path), monkeypatch, capsys)
    assert logging.getLogger("evolve_ipi").handlers == handlers_before, "the command line leaves the logging as it was"
    assert {name: signal.getsignal(getattr(signal, name)) for name in signals_before} == signals_before


check_plan = check._plan()


def first_line(err: str) -> str:
    """The first line the command says about itself; the progress lines of the log come before it."""
    return next((line for line in err.splitlines() if line.startswith("evolve-ipi:")), "")


# --- the options ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "phrases"),
    [
        (["--help"], ["check", "eval", "adapt", "run", "TINKER_API_KEY", "EVOLVER_MODEL", "exit status"]),
        (["check", "--help"], ["No API call is made", "--configs", "SKIP", "host name", "--hostname", "SHA256SUMS"]),
        (
            ["eval", "--help"],
            ["--model", "--task", "--mechanism", "--setting", "--passes", "--limit", "--adapted", "--outputs", "limit"],
        ),
        (["adapt", "--help"], ["rl", "skills"]),
        (["adapt", "rl", "--help"], ["--model", "--task", "TINKER_API_KEY", "resume"]),
        (["adapt", "skills", "--help"], ["--model", "--task", "EVOLVER_BASE_URL", "EVOLVER_API_KEY", "EVOLVER_MODEL"]),
        (["run", "--help"], ["CONFIG", "--configs", "--env-file", "resume", "config.json"]),
    ],
    ids=["evolve-ipi", "check", "eval", "adapt", "adapt-rl", "adapt-skills", "run"],
)
def test_every_command_has_help(capsys: pytest.CaptureFixture[str], argv: list[str], phrases: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)
    out = capsys.readouterr().out
    assert raised.value.code == 0
    assert out.startswith("usage: evolve-ipi")
    for phrase in phrases:
        assert phrase in out, phrase


def test_the_command_line_starts_without_loading_the_model_server() -> None:
    """`--help` must work on a checkout without submodules and must not wait for torch to load."""
    probe = (
        "import sys; import evolve_ipi.cli; "
        "assert 'torch' not in sys.modules and 'evolve_ipi.models.server' not in sys.modules, sorted(sys.modules)"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


def test_the_help_does_not_state_numbers_that_the_configuration_owns(capsys: pytest.CaptureFixture[str]) -> None:
    """How many conversations and updates there are is configs/skills.yaml and configs/rl.yaml; a help text that said
    so would be wrong as soon as one of them is edited."""
    for argv in (
        ["--help"],
        ["eval", "--help"],
        ["adapt", "rl", "--help"],
        ["adapt", "skills", "--help"],
        ["run", "--help"],
    ):
        with pytest.raises(SystemExit):
            cli.main(argv)
        out = capsys.readouterr().out
        assert not re.search(r"\b(12 conversations|64 updates|stages of 32|40 test)", out), argv


def test_no_command_prints_the_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    assert "evolve-ipi: self-evolution and indirect prompt injection" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["eval", "--model", MODEL, "--task", "email"], "invalid choice: 'email'"),
        (["eval", "--task", TASK], "--model"),
        (["eval", "--model", MODEL], "--task"),
        (["eval", "--model", MODEL, "--task", TASK, "--setting", "4"], r"invalid choice: '?4'?"),
        (["eval", "--model", MODEL, "--task", TASK, "--passes", "0"], "--passes"),
        (["eval", "--model", MODEL, "--task", TASK, "--limit", "-2"], "--limit"),
        (["eval", "--model", MODEL, "--task", TASK, "--mechanism", "lora"], "invalid choice: 'lora'"),
        (["adapt"], "required"),
        (["adapt", "lora", "--model", MODEL, "--task", TASK], "invalid choice: 'lora'"),
        (["adapt", "rl", "--task", TASK], "--model"),
        (["run"], "CONFIG"),
        (["frobnicate"], "frobnicate"),
    ],
)
def test_a_usage_mistake_is_refused_by_the_parser(
    capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)
    assert raised.value.code == cli.EXIT_USAGE == 2
    assert re.search(message, capsys.readouterr().err), message


# --- mistakes are refused before anything is started or spent ---------------------------------------------------------


def test_an_unknown_model_is_refused_before_any_process_starts(c: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    def started(*args: object, **kwargs: object) -> None:
        raise AssertionError("a process was started")

    monkeypatch.setattr(subprocess, "run", started)
    monkeypatch.setattr(subprocess, "Popen", started)
    monkeypatch.setattr(runner, "check_openclaw", started)
    for argv in (
        ["eval", "--model", "gpt-9", "--task", TASK],
        ["adapt", "rl", "--model", "gpt-9", "--task", TASK],
        ["adapt", "skills", "--model", "gpt-9", "--task", TASK],
    ):
        code, out, err = c(*argv)
        assert code == cli.EXIT_CONFIG and out == ""
        assert first_line(err).startswith("evolve-ipi: configuration error: unknown model 'gpt-9'")
        assert "known models: qwen3.8-27b, gpt-oss-120b, gpt-oss-20b, inkling-small" in err
    assert c.work.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["adapt", "rl", "--model", "gpt-oss-20b", "--task", TASK],
        ["eval", "--model", "gpt-oss-20b", "--task", TASK, "--mechanism", "rl"],
    ],
)
def test_a_mechanism_the_model_does_not_take_part_in_is_refused(c: Cli, argv: list[str]) -> None:
    code, _, err = c(*argv)
    assert code == cli.EXIT_CONFIG
    assert "gpt-oss-20b does not take part in 'rl'" in err and "skills" in err
    assert c.work.calls == []


@pytest.mark.parametrize(
    ("argv", "missing"),
    [
        (["eval", "--model", MODEL, "--task", TASK], ["TINKER_API_KEY"]),
        (ADAPT_RL, ["TINKER_API_KEY"]),
        (ADAPT_SKILLS, ["TINKER_API_KEY"]),
    ],
)
def test_a_missing_tinker_key_is_named(c: Cli, argv: list[str], missing: list[str]) -> None:
    c.monkeypatch.delenv("TINKER_API_KEY")
    code, out, err = c(*argv)
    assert code == cli.EXIT_CONFIG and out == ""
    assert first_line(err).startswith("evolve-ipi: configuration error: missing required secret(s): TINKER_API_KEY")
    assert c.work.calls == []


def test_skill_evolution_names_every_missing_evolver_variable(c: Cli) -> None:
    c.monkeypatch.delenv("EVOLVER_MODEL")
    code, _, err = c(*ADAPT_SKILLS)
    assert code == cli.EXIT_CONFIG
    assert "missing required secret(s): EVOLVER_MODEL." in err
    c.monkeypatch.delenv("EVOLVER_BASE_URL")
    c.monkeypatch.delenv("EVOLVER_API_KEY")
    _, _, err = c(*ADAPT_SKILLS)
    assert "EVOLVER_BASE_URL, EVOLVER_API_KEY, EVOLVER_MODEL" in err and "TINKER_API_KEY" not in err
    assert c.work.calls == []


def test_rl_adaptation_and_evaluation_do_not_ask_for_the_evolver(c: Cli) -> None:
    for name in ("EVOLVER_BASE_URL", "EVOLVER_API_KEY", "EVOLVER_MODEL"):
        c.monkeypatch.delenv(name)
    assert c(*ADAPT_RL)[0] == 0
    assert c("eval", "--model", MODEL, "--task", TASK, "--mechanism", "skills")[0] == 0


def test_a_run_needs_the_evolver_only_when_it_has_a_skills_combination(c: Cli) -> None:
    for name in ("EVOLVER_BASE_URL", "EVOLVER_API_KEY", "EVOLVER_MODEL"):
        c.monkeypatch.delenv(name)
    only_rl = write_configs(c.tmp_path, combos=[combo(MODEL, TASK, "rl")], name="rl")
    assert c("run", str(only_rl / "main.yaml"))[0] == 0
    with_skills = write_configs(c.tmp_path, combos=[combo(MODEL, TASK, "rl"), combo(MODEL, TASK, "skills")], name="s")
    code, _, err = c("run", str(with_skills / "main.yaml"))
    assert code == cli.EXIT_CONFIG and "EVOLVER_BASE_URL, EVOLVER_API_KEY, EVOLVER_MODEL" in err


@pytest.mark.parametrize(
    ("file", "mutate", "key"),
    [
        ("models.yaml", lambda d: d["models"][MODEL].update(tinker_idd="x"), "tinker_idd"),
        ("rl.yaml", lambda d: d.update(bach_size=d.pop("batch_size")), "bach_size"),
        ("skills.yaml", lambda d: d.update(max_new_skill=d.pop("max_new_skills")), "max_new_skill"),
        ("eval.yaml", lambda d: d.update(temprature=d.pop("temperature")), "temprature"),
        ("runtime.yaml", lambda d: d["ports"].update(gatewy=d["ports"].pop("gateway")), "gatewy"),
    ],
    ids=["models", "rl", "skills", "eval", "runtime"],
)
def test_a_typo_in_a_configuration_key_is_named_with_its_file(
    c: Cli, file: str, mutate: Callable[[dict], None], key: str
) -> None:
    edit_yaml(c.configs / file, mutate)
    for argv in (["eval", "--model", MODEL, "--task", TASK], ADAPT_RL, ["check"]):
        code, out, err = c(*argv)
        assert code == cli.EXIT_CONFIG and out == ""
        assert str(c.configs / file) in err and f"{key}: unknown key" in err
    assert c.work.calls == []


def test_a_typo_in_the_experiment_file_is_named(c: Cli) -> None:
    configs = write_configs(c.tmp_path, combos=[combo(MODEL, TASK, "rl")], name="exp")
    edit_yaml(configs / "main.yaml", lambda d: d.update(setings=d.pop("settings")))
    code, _, err = c("run", str(configs / "main.yaml"))
    assert code == cli.EXIT_CONFIG and "setings: unknown key" in err and "main.yaml" in err


def test_an_unknown_model_in_the_experiment_file_is_named(c: Cli) -> None:
    configs = write_configs(c.tmp_path, combos=[combo("gpt-9", TASK, "rl")], name="exp")
    code, _, err = c("run", str(configs / "main.yaml"))
    assert code == cli.EXIT_CONFIG and "combos[0].model: unknown model 'gpt-9'" in err


def test_a_port_in_use_is_refused_with_its_owner_and_nothing_is_stopped(c: Cli, listener: int) -> None:
    c.monkeypatch.setattr(runner, "require_free", procs.require_free)
    ports = free_ports()
    held = PortsSpec(gateway=ports.gateway, policy=listener, proxy=ports.proxy, model=ports.model)
    configs = write_configs(c.tmp_path, ports=held, name="busy")
    code, _, err = c(*ADAPT_RL, "--configs", str(configs))
    assert code == cli.EXIT_PORT_IN_USE
    assert first_line(err).startswith(f"evolve-ipi: port in use: port {listener} is in use by pid {os.getpid()}")
    assert c.work.calls == []


def test_a_wrong_openclaw_or_unpatched_frameworks_stop_the_command_at_startup(c: Cli) -> None:
    def wrong() -> str:
        raise OpenClawError("OpenClaw '2026.4.1' is installed, but the experiment runs on OpenClaw 2026.3.7")

    def unpatched() -> dict[str, str]:
        raise PatchError("OpenClaw-RL/openclaw-tinker/rollout.py has sha256 aaa, SHA256SUMS says bbb")

    c.monkeypatch.setattr(runner, "check_openclaw", wrong)
    code, _, err = c(*ADAPT_RL)
    assert code == cli.EXIT_OPENCLAW and "OpenClaw error: OpenClaw '2026.4.1' is installed" in err
    c.monkeypatch.setattr(runner, "check_openclaw", lambda: "2026.3.7")
    c.monkeypatch.setattr(runner, "verify_patches", unpatched)
    code, _, err = c(*ADAPT_RL)
    assert code == cli.EXIT_PATCH and "framework patch error: OpenClaw-RL/openclaw-tinker/rollout.py" in err
    assert c.work.calls == []


def test_the_secrets_can_come_from_an_env_file(c: Cli) -> None:
    c.monkeypatch.delenv("TINKER_API_KEY")
    keys = c.tmp_path / "keys.env"
    keys.write_text("TINKER_API_KEY=key-from-the-file-0004\n", encoding="utf-8")
    assert c(*ADAPT_RL, "--env-file", str(keys))[0] == 0
    code, _, err = c(*ADAPT_RL, "--env-file", str(c.tmp_path / "missing.env"))
    assert code == cli.EXIT_CONFIG and "missing.env" in err and "cannot read the env file" in err


def test_a_dot_env_file_in_the_working_directory_is_read(c: Cli) -> None:
    c.monkeypatch.delenv("TINKER_API_KEY")
    c.monkeypatch.chdir(c.tmp_path)
    assert c(*ADAPT_RL)[0] == cli.EXIT_CONFIG
    (c.tmp_path / ".env").write_text("TINKER_API_KEY=key-in-dot-env-0005\n", encoding="utf-8")
    assert c(*ADAPT_RL)[0] == 0


def test_in_the_image_a_command_without_the_hostname_flag_is_refused_and_help_still_works(
    c: Cli, capsys: pytest.CaptureFixture[str]
) -> None:
    in_image(c.monkeypatch, c.tmp_path, hostname="3f9a1c0d2b7e", mounted=[c.tmp_path / "outputs"])
    for argv in (
        ADAPT_RL,
        ADAPT_SKILLS,
        ["eval", "--model", MODEL, "--task", TASK],
        ["run", str(write_configs(c.tmp_path, combos=[combo(MODEL, TASK, "rl")], name="exp") / "main.yaml")],
    ):
        code, out, err = c(*argv)
        assert code == cli.EXIT_CONFIG and out == ""
        assert first_line(err).startswith("evolve-ipi: configuration error: the host name of this container is")
        assert "--hostname evolve-ipi" in err
    assert c.work.calls == [] and c.work.ports_checked == []
    for argv in (["--help"], ["eval", "--help"], ["run", "--help"]):
        with pytest.raises(SystemExit) as raised:
            cli.main(argv)
        assert raised.value.code == 0
    assert "usage: evolve-ipi" in capsys.readouterr().out


def test_in_the_image_check_fails_without_the_hostname_flag_so_the_quick_start_cannot_pass(c: Cli) -> None:
    in_image(c.monkeypatch, c.tmp_path, hostname="3f9a1c0d2b7e")
    c.monkeypatch.setattr(check, "_plan", lambda: [(name, fn) for name, fn in check_plan if name == "host name"])
    code, out, _ = c("check")
    assert code == cli.EXIT_CHECK_FAILED
    assert out.splitlines()[0].startswith("FAIL  host name") and "--hostname evolve-ipi" in out
    assert out.splitlines()[-1] == "checks: 0 passed, 1 failed, 0 skipped"
    in_image(c.monkeypatch, c.tmp_path, hostname="evolve-ipi")
    code, out, _ = c("check")
    assert code == 0 and re.match(r"PASS  host name\s+evolve-ipi, ", out.splitlines()[0])


def test_in_the_image_check_needs_no_outputs_volume(c: Cli) -> None:
    in_image(c.monkeypatch, c.tmp_path, mounted=[])
    c.monkeypatch.setattr(check, "_plan", lambda: [(name, fn) for name, fn in check_plan if name == "host name"])
    assert c("check")[0] == 0


def test_in_the_image_results_on_a_volume_that_is_not_mounted_are_refused_with_the_flag(c: Cli) -> None:
    in_image(c.monkeypatch, c.tmp_path, mounted=[])
    code, out, err = c(*ADAPT_RL)
    assert code == cli.EXIT_CONFIG and out == ""
    assert "is not on a mounted volume" in err and '-v "$PWD/outputs:/work/outputs"' in err
    assert c(*ADAPT_RL, "--outputs", str(c.tmp_path / "mounted-elsewhere"))[0] == cli.EXIT_CONFIG
    in_image(c.monkeypatch, c.tmp_path, mounted=[c.tmp_path / "mounted-elsewhere"])
    assert c(*ADAPT_RL, "--outputs", str(c.tmp_path / "mounted-elsewhere" / "deeper"))[0] == 0


def test_in_the_image_an_adapted_agent_is_evaluated_where_its_directory_is_mounted(c: Cli) -> None:
    assert c(*ADAPT_RL)[0] == 0
    directory = c.run_dir().root
    in_image(c.monkeypatch, c.tmp_path, mounted=[])
    code, _, err = c("eval", "--model", MODEL, "--task", TASK, "--adapted", str(directory))
    assert code == cli.EXIT_CONFIG and f"the output directory {directory} is not on a mounted volume" in err
    in_image(c.monkeypatch, c.tmp_path, mounted=[c.tmp_path / "outputs"])
    assert c("eval", "--model", MODEL, "--task", TASK, "--adapted", str(directory))[0] == 0


# --- what the errors look like ----------------------------------------------------------------------------------------

EXPECTED = [
    (ConfigError("the runtime directory /x cannot be made"), cli.EXIT_CONFIG, "configuration error"),
    (TaskDataError("data/tasks/x/test.jsonl:3: not valid JSON"), cli.EXIT_TASK_DATA, "task data error"),
    (PatchError("rollout.py has sha256 aaa"), cli.EXIT_PATCH, "framework patch error"),
    (PortInUse("port 30012 is in use by pid 12"), cli.EXIT_PORT_IN_USE, "port in use"),
    (LeftoverState("/home/a/.openclaw-rl holds x.txt: remove it"), cli.EXIT_LEFTOVER, "leftover state"),
    (AdaptAbort("stage 1 failed 4 times"), cli.EXIT_ADAPT, "adaptation failed"),
    (EvalError("pass 1 was not a valid measurement 4 times"), cli.EXIT_EVAL, "evaluation failed"),
    (OpenClawError("openclaw was not found on PATH"), cli.EXIT_OPENCLAW, "OpenClaw error"),
    (ProcessError("trainer exited with status 1"), cli.EXIT_PROCESS, "process error"),
    (
        StartFailed("the model server did not start: ProcessError: model server exited with status 1"),
        cli.EXIT_PROCESS,
        "start failed",
    ),
    (RunDirError("the library changed since its evolution finished"), cli.EXIT_RUN_DIR, "run directory error"),
    (LibraryError("the finished library changed"), cli.EXIT_RUN_DIR, "skill library error"),
    (RunLocked("outputs/x is in use by another run (pid 7)"), cli.EXIT_RUN_LOCKED, "run directory in use"),
    (StateError("access.log line 3 is not JSON"), cli.EXIT_ENVIRONMENT, "simulated services error"),
    (WorkspaceError("the workspace is not clean"), cli.EXIT_ENVIRONMENT, "workspace error"),
]


def test_every_kind_of_error_has_an_exit_status_of_its_own() -> None:
    by_kind = {type(error): code for error, code, _ in EXPECTED}
    assert set(by_kind) == {kind for kind, _, _ in cli._EXPECTED}, "every expected kind is tested here"
    assert by_kind[WorkspaceError] == by_kind[StateError], "both say the simulated environment is not as needed"
    assert by_kind[LibraryError] == by_kind[RunDirError], "both say that a person has to look at a run directory"
    assert by_kind[StartFailed] == by_kind[ProcessError], "both say that a child process did not start or failed"
    assert len(set(by_kind.values())) == len(by_kind) - 3, "every other kind has a status of its own"
    reserved = {0, cli.EXIT_UNEXPECTED, cli.EXIT_USAGE, cli.EXIT_CHECK_FAILED, cli.EXIT_INTERRUPTED}
    assert len(reserved) == 5 and not reserved & set(by_kind.values())


def test_every_exit_status_the_help_lists_is_one_an_error_gives() -> None:
    """The table of --help against the mapping: nothing listed that no error gives, nothing given that is not listed."""
    listed = [int(status) for status in re.findall(r"^  (\d+) ", cli._EXIT_STATUS_HELP, re.MULTILINE)]
    given = {status for _, status, _ in cli._EXPECTED}
    reserved = {0, cli.EXIT_UNEXPECTED, cli.EXIT_USAGE, cli.EXIT_CHECK_FAILED, cli.EXIT_INTERRUPTED}
    assert sorted(listed) == listed and len(set(listed)) == len(listed)
    assert set(listed) == given | reserved
    assert 12 not in listed, "a checkpoint that does not load is the adaptation's (8) or the evaluation's (9)"


@pytest.mark.parametrize(("error", "code", "label"), EXPECTED, ids=[type(e).__name__ for e, _, _ in EXPECTED])
def test_an_expected_error_is_one_line_and_no_traceback(c: Cli, error: Exception, code: int, label: str) -> None:
    c.work.raise_on("adapt-rl", 1, error)
    status, out, err = c(*ADAPT_RL)
    assert status == code and out == ""
    assert first_line(err) == f"evolve-ipi: {label}: {str(error).splitlines()[0]}"
    assert "Traceback" not in err and 'File "' not in err
    assert f"run directory: {c.run_dir().root}" in err, "an error during a combination says where it was"


def test_a_process_that_did_not_start_names_its_cause_in_one_line(c: Cli) -> None:
    """`StartFailed` is not a `ProcessError` (the evolution would restart on one); the command line says it as one."""
    assert not issubclass(StartFailed, (ProcessError, AdaptAbort))
    cause = "the gateway did not start: OpenClawError: OpenClaw '2026.4.1' is installed, but 2026.3.7 is pinned"
    c.work.raise_on("adapt-skills", 1, StartFailed(cause))
    code, out, err = c(*ADAPT_SKILLS)
    assert code == cli.EXIT_PROCESS and out == "" and "Traceback" not in err
    assert first_line(err) == f"evolve-ipi: start failed: {cause}"


def test_an_evaluation_whose_process_did_not_start_is_an_evaluation_error_with_the_cause(c: Cli) -> None:
    cause = (
        "the model server did not start: ProcessError: model server exited with status 1 before logging its ready line"
    )
    c.work.raise_on("pass", 1, EvalError(cause))
    code, out, err = c("eval", "--model", MODEL, "--task", TASK)
    assert code == cli.EXIT_EVAL and out == "" and "Traceback" not in err
    assert first_line(err) == f"evolve-ipi: evaluation failed: {cause}"


def test_a_multi_line_message_keeps_its_lines_under_the_first(c: Cli) -> None:
    c.work.raise_on("adapt-rl", 1, EvalError("pass 1 was not valid 4 times:\n  attempt 1: loop\n  attempt 2: loop"))
    _, _, err = c(*ADAPT_RL)
    lines = err.splitlines()
    start = lines.index("evolve-ipi: evaluation failed: pass 1 was not valid 4 times:")
    assert lines[start : start + 3] == [
        "evolve-ipi: evaluation failed: pass 1 was not valid 4 times:",
        "  attempt 1: loop",
        "  attempt 2: loop",
    ]


def test_an_error_before_the_combination_starts_names_no_run_directory(c: Cli) -> None:
    c.monkeypatch.delenv("TINKER_API_KEY")
    _, _, err = c(*ADAPT_RL)
    assert "run directory" not in err


def test_an_unexpected_exception_keeps_its_traceback(c: Cli) -> None:
    c.work.raise_on("adapt-rl", 1, RuntimeError("a bug"))
    code, out, err = c(*ADAPT_RL)
    assert code == cli.EXIT_UNEXPECTED == 1 and out == ""
    assert "Traceback (most recent call last)" in err and "RuntimeError: a bug" in err
    assert not RunDir.for_combo(c.tmp_path / "outputs", MODEL, TASK, "rl").holds_lock()


def test_no_secret_is_ever_printed(c: Cli) -> None:
    secret = SECRET_VALUES["TINKER_API_KEY"]
    other = SECRET_VALUES["EVOLVER_API_KEY"]
    c.work.raise_on("adapt-rl", 1, ProcessError(f"trainer exited; its log: TINKER_API_KEY={secret} and {other}"))
    code, _, err = c(*ADAPT_RL)
    assert code == cli.EXIT_PROCESS and secret not in err and other not in err and "***" in err
    c.work.raise_on("adapt-rl", 2, RuntimeError(f"unexpected, with {secret}"))
    code, _, err = c(*ADAPT_RL)
    assert code == cli.EXIT_UNEXPECTED and secret not in err and "RuntimeError: unexpected, with ***" in err


def test_a_secret_from_an_env_file_is_kept_out_as_well(c: Cli) -> None:
    c.monkeypatch.delenv("TINKER_API_KEY")
    keys = c.tmp_path / "keys.env"
    keys.write_text("TINKER_API_KEY=file-only-secret-0006\n", encoding="utf-8")
    c.work.raise_on("adapt-rl", 1, ProcessError("log: file-only-secret-0006"))
    code, _, err = c(*ADAPT_RL, "--env-file", str(keys))
    assert code == cli.EXIT_PROCESS and "file-only-secret-0006" not in err


# --- being stopped ----------------------------------------------------------------------------------------------------


def test_ctrl_c_stops_the_command_with_130_and_names_the_directory_to_resume(c: Cli) -> None:
    c.work.raise_on("adapt-rl", 1, KeyboardInterrupt())
    code, out, err = c(*ADAPT_RL)
    assert code == cli.EXIT_INTERRUPTED == 130 and out == ""
    assert first_line(err).startswith("evolve-ipi: interrupted")
    assert f"run directory: {c.run_dir().root}" in err and "run the same command again" in err
    assert "Traceback" not in err and not c.run_dir().holds_lock()


@pytest.fixture
def sighup_default() -> Iterator[None]:
    """SIGHUP as a terminal leaves it. A test suite run under nohup starts with SIGHUP ignored, which the command keeps
    (and a child process inherits): a test that sends SIGHUP needs the default back. Request it before ``c``."""
    previous = signal.signal(signal.SIGHUP, signal.SIG_DFL)
    try:
        yield
    finally:
        signal.signal(signal.SIGHUP, previous)


@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_a_signal_is_treated_like_ctrl_c_and_the_cleanup_runs(sighup_default: None, c: Cli, name: str) -> None:
    cleaned: list[str] = []
    stopped = c.work

    def adapt(*args: object, **kwargs: object) -> dict[str, str]:
        try:
            os.kill(os.getpid(), getattr(signal, name))
            for _ in range(100):  # the handler runs between two bytecodes; this only gives it room
                pass
            raise AssertionError("the signal did not interrupt the adaptation")
        finally:
            cleaned.append("the trainer, the model server and the gateway are stopped here")

    c.monkeypatch.setattr(runner, "adapt_rl", adapt)
    code, _, err = c(*ADAPT_RL)
    assert code == cli.EXIT_INTERRUPTED
    assert f"interrupted by {name}" in first_line(err) and f"run directory: {c.run_dir().root}" in err
    assert cleaned and stopped.calls == []
    assert not c.run_dir().holds_lock()


def test_an_interrupted_run_says_which_combination_and_how_many_finished(c: Cli) -> None:
    configs = write_configs(
        c.tmp_path, combos=[combo(MODEL, TASK, "rl"), combo(MODEL, "drive-create", "rl")], name="exp"
    )
    c.work.raise_on("evaluate", 2, KeyboardInterrupt())
    code, _, err = c("run", str(configs / "main.yaml"))
    assert code == cli.EXIT_INTERRUPTED
    assert f"run directory: {c.run_dir(task='drive-create').root}" in err
    assert "combinations finished before this" in err and TASK in err
    code, out, _ = c("run", str(configs / "main.yaml"))
    assert code == 0 and out.count("kept") == 2, "both adaptations finished before the interruption"


def test_check_does_not_claim_a_resume_that_it_does_not_have(c: Cli) -> None:
    def interrupted(cfgs: object, *, report: object, **options: object) -> list[check.Result]:
        raise KeyboardInterrupt

    c.monkeypatch.setattr(check, "run_checks", interrupted)
    code, _, err = c("check")
    assert code == cli.EXIT_INTERRUPTED
    assert first_line(err) == "evolve-ipi: interrupted; what it started has been stopped"
    assert "finished work" not in err and "same command" not in err and "run directory" not in err


def test_nothing_interrupts_the_report_of_an_interruption(c: Cli) -> None:
    """A signal that arrives while the stop is being reported is ignored, like the ones during the cleanup."""
    c.work.raise_on("adapt-rl", 1, KeyboardInterrupt())
    said: list[str] = []
    real = cli._say

    def say(text: str) -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(100):  # a handler would run between two of these
            pass
        said.append(text)
        real(text)

    c.monkeypatch.setattr(cli, "_say", say)
    try:
        code, _, err = c(*ADAPT_RL)
    except KeyboardInterrupt:  # (a plain failure, not an interrupt of pytest itself)
        pytest.fail("a signal interrupted the report of the interruption")
    assert code == cli.EXIT_INTERRUPTED and len(said) == 1
    assert "interrupted; what it started has been stopped, and what finished is kept" in err


def test_a_cleanup_that_fails_after_an_interruption_is_not_reported_as_a_clean_stop(c: Cli) -> None:
    def adapt(*args: object, **kwargs: object) -> None:
        try:
            raise KeyboardInterrupt
        finally:
            raise ProcessError("the trainer (pid 4242) could not be stopped")

    c.monkeypatch.setattr(runner, "adapt_rl", adapt)
    code, _, err = c(*ADAPT_RL)
    assert code == cli.EXIT_PROCESS
    assert first_line(err) == "evolve-ipi: process error: the trainer (pid 4242) could not be stopped"
    assert "this happened while stopping what the command started after an interruption" in err
    assert "has been stopped" not in err


def test_the_secrets_of_an_env_file_are_forgotten_when_the_command_ends(c: Cli) -> None:
    c.monkeypatch.delenv("TINKER_API_KEY")
    keys = c.tmp_path / "keys.env"
    keys.write_text("TINKER_API_KEY=file-only-secret-0007\n", encoding="utf-8")
    assert c(*ADAPT_RL, "--env-file", str(keys))[0] == 0
    assert runner.loaded_secrets() == []


def test_the_logging_is_put_back_as_it_was(c: Cli) -> None:
    logger = logging.getLogger("evolve_ipi")
    logger.setLevel(logging.ERROR)
    try:
        assert c(*ADAPT_RL)[0] == 0
        assert logger.level == logging.ERROR
    finally:
        logger.setLevel(logging.NOTSET)


def test_a_logged_secret_never_reaches_stderr(c: Cli) -> None:
    secret, other = SECRET_VALUES["TINKER_API_KEY"], SECRET_VALUES["EVOLVER_API_KEY"]

    def adapt(*args: object, **kwargs: object) -> dict[str, str]:
        log = logging.getLogger("evolve_ipi.adapt.rl")
        log.warning("stage 1 failed (%s); starting it again", ProcessError(f"trainer exited; log tail: key={secret}"))
        try:
            raise ProcessError(f"its log: Authorization: Bearer {other}")
        except ProcessError:
            log.exception("a restart")  # the traceback is formatted by the same handler
        return c.work.adapt_rl(*args, **kwargs)  # type: ignore[arg-type]

    c.monkeypatch.setattr(runner, "adapt_rl", adapt)
    code, out, err = c(*ADAPT_RL)
    assert code == 0 and secret not in err and other not in err and secret not in out
    assert "key=***" in err and "Bearer ***" in err and "stage 1 failed" in err


SIGNAL_PARENT = Path(__file__).with_name("cli_signal_parent.py")


@contextmanager
def signalled_command(
    tmp_path: Path, mode: str, grace: float = 3.0, *, ignoring: str = ""
) -> Iterator[tuple[subprocess.Popen[str], Path]]:
    """`adapt rl` whose adaptation runs a child process, in a process of its own that a test can send signals to."""
    work = tmp_path / "signalled"
    work.mkdir()
    parent = subprocess.Popen(
        [sys.executable, str(SIGNAL_PARENT), str(work), mode, str(grace)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "IGNORE_SIGNAL": ignoring},
    )
    try:
        assert parent.stdout is not None and parent.stdout.readline().strip() == "ready"
        yield parent, work / "child.pid"
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.communicate()
        pid_file = work / "child.pid"
        if pid_file.exists():
            try:
                os.killpg(int(pid_file.read_text()), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_command_started_under_nohup_outlives_its_terminal(tmp_path: Path) -> None:
    """`nohup` ignores SIGHUP before it runs the command; the command must not install a handler over that."""
    with signalled_command(tmp_path, "polite", ignoring="SIGHUP") as (parent, pid_file):
        parent.send_signal(signal.SIGHUP)
        with pytest.raises(subprocess.TimeoutExpired):
            parent.wait(timeout=3)
        assert parent.poll() is None and alive(int(pid_file.read_text()))
        parent.send_signal(signal.SIGTERM)  # SIGTERM was not ignored, and still ends it
        _, err = parent.communicate(timeout=60)
    assert parent.returncode == cli.EXIT_INTERRUPTED and "interrupted by SIGTERM" in err


def test_a_signal_that_is_not_ignored_still_ends_the_command_with_sighup(sighup_default: None, tmp_path: Path) -> None:
    with signalled_command(tmp_path, "polite") as (parent, pid_file):
        parent.send_signal(signal.SIGHUP)
        _, err = parent.communicate(timeout=60)
        assert parent.returncode == cli.EXIT_INTERRUPTED and "interrupted by SIGHUP" in err
        assert not alive(int(pid_file.read_text()))


def test_more_signals_during_the_cleanup_do_not_abandon_it(tmp_path: Path) -> None:
    """The child ignores SIGTERM, so stopping it takes the whole grace period and ends with SIGKILL. A second SIGTERM
    and a Ctrl-C meanwhile (an impatient user, or an orchestrator) must neither stop the stopping nor leave the child."""
    started = time.monotonic()
    with signalled_command(tmp_path, "stubborn", grace=4.0) as (parent, pid_file):
        child = int(pid_file.read_text())
        parent.send_signal(signal.SIGTERM)
        time.sleep(1)
        parent.send_signal(signal.SIGTERM)
        parent.send_signal(signal.SIGINT)
        parent.send_signal(signal.SIGHUP)
        _, err = parent.communicate(timeout=60)
        assert time.monotonic() - started >= 4, "the child was given its whole grace period"
        assert parent.returncode == cli.EXIT_INTERRUPTED
        assert not alive(child), "and then it was killed, although more signals came"
    lines = err.splitlines()
    assert lines[0] == "evolve-ipi: SIGTERM received; stopping what it started, which can take a minute"
    assert sum("received" in line for line in lines) == 1, "the first signal is the only one that is announced"
    assert "evolve-ipi: interrupted by SIGTERM; what it started has been stopped, and what finished is kept" in lines
    assert "Traceback" not in err


# --- the lock ---------------------------------------------------------------------------------------------------------


HOLD_THE_LOCK = """
import sys, time
from pathlib import Path
from evolve_ipi.records import RunDir

with RunDir(Path(sys.argv[1])).lock():
    print("held", flush=True)
    time.sleep(60)
"""

HOLD_THE_RUNTIME = """
import sys, time
from pathlib import Path
from evolve_ipi.paths import RuntimePaths

with RuntimePaths(root=Path(sys.argv[1]), agent_home=Path(sys.argv[2])).lock():
    print("held", flush=True)
    time.sleep(60)
"""


@contextmanager
def held_by_another_process(script: str, *arguments: Path) -> Iterator[subprocess.Popen[str]]:
    child = subprocess.Popen([sys.executable, "-c", script, *map(str, arguments)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "held"
        yield child
    finally:
        child.kill()
        child.wait()
        if child.stdout is not None:
            child.stdout.close()


@pytest.mark.parametrize(
    "argv",
    [
        ADAPT_RL,
        ADAPT_SKILLS,
        ["eval", "--model", MODEL, "--task", TASK],
        ["eval", "--model", MODEL, "--task", TASK, "--limit", "2"],
    ],
    ids=["adapt-rl", "adapt-skills", "eval", "eval-limit"],
)
def test_a_command_in_a_directory_another_run_holds_is_refused(c: Cli, argv: list[str]) -> None:
    directory = c.run_dir("skills" if "skills" in argv else "rl")
    if "--limit" in argv:
        directory = RunDir(directory.root / "limit2")
    with held_by_another_process(HOLD_THE_LOCK, directory.root) as child:
        code, out, err = c(*argv)
    assert code == cli.EXIT_RUN_LOCKED and out == ""
    assert first_line(err).startswith("evolve-ipi: run directory in use: ")
    assert f"pid {child.pid}" in err
    assert c.work.calls == []


@pytest.mark.parametrize("held", ["runtime", "home"])
@pytest.mark.parametrize(
    "argv",
    [ADAPT_RL, ADAPT_SKILLS, ["eval", "--model", MODEL, "--task", TASK]],
    ids=["adapt-rl", "adapt-skills", "eval"],
)
def test_a_command_is_refused_while_another_run_holds_its_runtime_directory_or_its_home(
    c: Cli, argv: list[str], held: str
) -> None:
    """Another run on other ports, with the same runtime directory or the same HOME, would clobber the profile, the
    workspace and File-Create's target directory this run rebuilds."""
    other = c.tmp_path / "somebody-elses"
    root, home = (c.tmp_path / "runtime", other) if held == "runtime" else (other, c.tmp_path / "home")
    with held_by_another_process(HOLD_THE_RUNTIME, root, home):
        code, out, err = c(*argv)
    assert code == cli.EXIT_RUN_LOCKED and out == ""
    what = "the runtime directory" if held == "runtime" else "the agent's home"
    assert first_line(err).startswith(
        f"evolve-ipi: run directory in use: {what} {c.tmp_path / held if held == 'runtime' else c.tmp_path / 'home'}"
    )
    assert "EVOLVE_IPI_RUNTIME" in err and "HOME" in err and "its own container" in err
    assert "another output directory" not in err
    assert c.work.calls == []


def test_the_runtime_directory_and_the_home_are_held_for_the_whole_command(c: Cli) -> None:
    """The stand-ins assert it at every adaptation and every pass; afterwards both are free again."""
    assert c(*ADAPT_RL)[0] == 0 and c("eval", "--model", MODEL, "--task", TASK)[0] == 0
    paths = RuntimePaths()
    assert not paths.holds_lock()
    with paths.lock():
        assert paths.holds_lock()


# --- adapt ------------------------------------------------------------------------------------------------------------


def test_adapt_rl_adapts_in_the_run_directory_and_prints_the_checkpoints(c: Cli) -> None:
    code, out, err = c(*ADAPT_RL)
    assert code == 0 and err.count("Traceback") == 0
    assert c.work.calls == [("adapt-rl", MODEL, TASK)]
    run = c.run_dir()
    assert run.adapt_done("rl") and run.config_path.is_file()
    assert json.loads(run.config_path.read_text())["mechanism"] == "rl"
    assert out.startswith(f"RL adaptation of {MODEL} on {TASK}: finished\n")
    assert f"update 32: tinker://{MODEL}/{TASK}/32" in out and f"update 64: tinker://{MODEL}/64" in out
    assert f"run directory: {run.root}" in out


def test_adapt_skills_adapts_in_the_run_directory_and_prints_the_library(c: Cli) -> None:
    code, out, _ = c(*ADAPT_SKILLS)
    assert code == 0 and c.work.calls == [("adapt-skills", MODEL, TASK)]
    assert out.startswith(f"Skill evolution of {MODEL} on {TASK}: finished\n")
    run = c.run_dir("skills")
    marker = json.loads((run.adapt_dir / "library.json").read_text())
    assert f"library: {run.adapt_dir / 'library'}" in out and f"sha256: {marker['sha']}" in out


def test_a_finished_adaptation_is_not_run_again(c: Cli) -> None:
    assert c(*ADAPT_RL)[0] == 0
    code, out, _ = c(*ADAPT_RL)
    assert code == 0 and c.work.calls == [("adapt-rl", MODEL, TASK)], "the second call adapted nothing"
    assert "already finished" in out
    assert "update 64" in out


def test_adapting_again_under_another_configuration_is_refused(c: Cli) -> None:
    assert c(*ADAPT_RL)[0] == 0
    edit_yaml(c.configs / "rl.yaml", lambda d: d.update(lora_rank=16))
    code, _, err = c(*ADAPT_RL)
    assert code == cli.EXIT_RUN_DIR and "rl.lora_rank" in err and "records another configuration" in err
    assert len(c.work.calls) == 1


def test_the_ports_are_free_to_change_between_invocations(c: Cli) -> None:
    assert c(*ADAPT_RL)[0] == 0
    other = write_configs(c.tmp_path, name="other-ports")
    assert c(*ADAPT_RL, "--configs", str(other))[0] == 0


def test_a_relative_outputs_directory_is_made_in_the_working_directory(c: Cli) -> None:
    c.monkeypatch.chdir(c.tmp_path)
    assert c(*ADAPT_RL, "--outputs", "results")[0] == 0
    assert RunDir.for_combo(c.tmp_path / "results", MODEL, TASK, "rl").adapt_done("rl")


def test_outputs_can_be_moved(c: Cli) -> None:
    elsewhere = c.tmp_path / "elsewhere"
    assert c(*ADAPT_RL, "--outputs", str(elsewhere))[0] == 0
    assert RunDir.for_combo(elsewhere, MODEL, TASK, "rl").adapt_done("rl")
    assert not c.run_dir().root.exists()


def test_adapt_skills_asks_the_evolver_once_at_startup_and_eval_never(c: Cli) -> None:
    assert c(*ADAPT_SKILLS)[0] == 0
    assert c.work.preflights == ["evolver-model-test"]
    assert c("eval", "--model", MODEL, "--task", TASK, "--mechanism", "skills")[0] == 0
    assert c.work.preflights == ["evolver-model-test"], "an evaluation never calls the evolver"
    assert c(*ADAPT_SKILLS)[0] == 0 and len(c.work.preflights) == 1, "and a finished library needs no question either"


def test_a_recipe_that_would_stop_the_adaptation_is_refused_at_startup_with_the_key_named(c: Cli) -> None:
    c.monkeypatch.setattr(runner, "check_setup", rl.check_setup)  # the real checks of the RL line
    c.monkeypatch.setattr(rl, "verify_patches", lambda: dict(PATCHED))
    edit_yaml(c.configs / "rl.yaml", lambda data: data.update(max_steps=40))
    code, out, err = c(*ADAPT_RL)
    assert code == cli.EXIT_CONFIG and out == ""
    assert first_line(err).startswith("evolve-ipi: configuration error: rl.yaml: max_steps is 40; it must be at least")
    assert c.work.calls == []


# --- eval -------------------------------------------------------------------------------------------------------------


def passes(c: Cli) -> list[tuple[object, ...]]:
    return [call for call in c.work.calls if call[0] == "pass"]


def test_eval_without_options_runs_one_pass_of_the_base_agent_on_the_test_set(c: Cli) -> None:
    code, out, _ = c("eval", "--model", MODEL, "--task", TASK)
    assert code == 0
    ((kind, model, task, variant, setting, k, agent, items),) = passes(c)
    assert (kind, model, task, variant, setting, k, agent) == ("pass", MODEL, TASK, "base", 3, 1, "Base")
    assert len(items) == 40 and list(items) == sorted(items)  # type: ignore[arg-type]
    assert all("-test-" in item for item in items)  # type: ignore[attr-defined]
    run = c.run_dir()
    assert run.config_path.is_file(), "the base agent's passes go where `run` finds them"
    assert "setting 3" in out and "attack success rate" in out and "base agent" in out
    assert "1/40 (2.5%)" in out
    assert f"passes: {run.eval_dir / 'base' / 'setting3'}" in out


def test_eval_takes_the_mechanism_of_the_first_it_knows(c: Cli) -> None:
    c("eval", "--model", "gpt-oss-20b", "--task", TASK)
    assert c.run_dir("skills", "gpt-oss-20b").config_path.is_file()
    c("eval", "--model", MODEL, "--task", TASK, "--mechanism", "skills")
    assert c.run_dir("skills").config_path.is_file()


def test_limit_runs_the_first_items_apart_from_the_passes_of_a_full_run(c: Cli) -> None:
    code, out, _ = c("eval", "--model", MODEL, "--task", TASK, "--limit", "2", "--passes", "3")
    assert code == 0
    test_items = (tasks.DATA_DIR / "tasks" / TASK / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ids = sorted(json.loads(line)["id"] for line in test_items)[:2]
    assert [(call[5], call[7]) for call in passes(c)] == [(1, tuple(ids)), (2, tuple(ids)), (3, tuple(ids))]
    limited = RunDir(c.run_dir().root / "limit2")
    assert json.loads(limited.config_path.read_text())["limit"] == 2
    assert not c.run_dir().config_path.exists(), "a limited evaluation leaves the combination's own directory alone"
    assert "pooled over 3 passes: 3/6 (50.0%)" in out
    assert f"passes: {limited.eval_dir / 'base' / 'setting3'}" in out


def test_passes_are_numbered_from_one(c: Cli) -> None:
    c("eval", "--model", MODEL, "--task", TASK, "--passes", "3")
    assert [call[5] for call in passes(c)] == [1, 2, 3]


@pytest.mark.parametrize("setting", ["1", "2"])
def test_the_settings_one_and_two_need_an_adaptation_to_take_the_items_from(c: Cli, setting: str) -> None:
    code, out, err = c("eval", "--model", MODEL, "--task", TASK, "--setting", setting)
    assert code == cli.EXIT_CONFIG and out == ""
    assert first_line(err).startswith(f"evolve-ipi: configuration error: setting {setting} uses the items the agent")
    assert "holds no finished adaptation" in err and "evolve-ipi adapt rl" in err and "or evaluate setting 3" in err
    assert passes(c) == []
    assert not (c.tmp_path / "outputs").exists(), "a refused command leaves no run directory behind"
    assert not (c.tmp_path / "runtime").exists()


@pytest.mark.parametrize("setting", [1, 2])
def test_the_settings_one_and_two_use_the_items_the_agent_was_adapted_on(c: Cli, setting: int) -> None:
    assert c(*ADAPT_RL)[0] == 0
    code, out, _ = c("eval", "--model", MODEL, "--task", TASK, "--setting", str(setting))
    assert code == 0
    ((_, _, _, variant, number, _, agent, items),) = passes(c)
    assert (variant, number, agent) == ("base", setting, "Base")
    assert len(items) == 6 and all("-adapt-" in item for item in items)  # type: ignore[arg-type]
    assert ("execution rate" if setting == 1 else "attack success rate") in out


@pytest.mark.parametrize(
    ("adapt", "mechanism", "agent"), [(ADAPT_RL, "rl", "Checkpoint"), (ADAPT_SKILLS, "skills", "Library")]
)
def test_an_adapted_agent_is_evaluated_from_the_directory_of_its_adaptation(
    c: Cli, adapt: list[str], mechanism: str, agent: str
) -> None:
    assert c(*adapt)[0] == 0
    directory = c.run_dir(mechanism)
    code, out, _ = c("eval", "--model", MODEL, "--task", TASK, "--adapted", str(directory.root), "--passes", "2")
    assert code == 0
    assert [(call[3], call[4], call[5], call[6]) for call in passes(c)] == [
        ("adapted", 3, 1, agent),
        ("adapted", 3, 2, agent),
    ]
    assert "adapted agent" in out and "pooled over 2 passes: 4/80 (5.0%)" in out
    assert f"passes: {directory.eval_dir / 'adapted' / 'setting3'}" in out


def test_an_adapted_agent_in_setting_one_uses_the_items_it_was_adapted_on(c: Cli) -> None:
    assert c(*ADAPT_SKILLS)[0] == 0
    directory = c.run_dir("skills").root
    code, _, _ = c(
        "eval", "--model", MODEL, "--task", TASK, "--adapted", str(directory), "--setting", "1", "--limit", "5"
    )
    assert code == 0
    ((_, _, _, variant, setting, _, agent, items),) = passes(c)
    assert (variant, setting, agent, len(items)) == ("adapted", 1, "Library", 5)  # type: ignore[arg-type]
    assert RunDir(directory / "limit5").config_path.is_file()


def test_outputs_has_no_effect_on_an_adapted_agent_and_is_refused(c: Cli) -> None:
    assert c(*ADAPT_RL)[0] == 0
    directory = str(c.run_dir().root)
    code, _, err = c("eval", "--model", MODEL, "--task", TASK, "--adapted", directory, "--outputs", "elsewhere")
    assert code == cli.EXIT_CONFIG and "--outputs has no effect with --adapted" in err and directory in err
    assert passes(c) == []


def test_the_run_directory_of_an_adaptation_must_hold_one(c: Cli) -> None:
    empty = c.tmp_path / "empty"
    empty.mkdir()
    for directory in (empty, c.tmp_path / "missing"):
        code, _, err = c("eval", "--model", MODEL, "--task", TASK, "--adapted", str(directory))
        assert code == cli.EXIT_CONFIG and f"--adapted {directory}" in err
    assert "holds no finished adaptation" in err or "is not a directory" in err
    assert passes(c) == []


def test_an_adaptation_of_another_model_task_or_mechanism_is_refused(c: Cli) -> None:
    assert c(*ADAPT_RL)[0] == 0
    directory = str(c.run_dir().root)
    code, _, err = c("eval", "--model", "gpt-oss-120b", "--task", TASK, "--adapted", directory)
    assert code == cli.EXIT_CONFIG and "was adapted for qwen3.8-27b / file-create / rl, not for gpt-oss-120b" in err
    code, _, err = c("eval", "--model", MODEL, "--task", "drive-create", "--adapted", directory)
    assert code == cli.EXIT_CONFIG and "not for qwen3.8-27b / drive-create / rl" in err
    code, _, err = c("eval", "--model", MODEL, "--task", TASK, "--adapted", directory, "--mechanism", "skills")
    assert code == cli.EXIT_CONFIG and "was adapted with rl, not with skills" in err
    assert passes(c) == []


def test_eval_needs_only_the_tinker_key_and_records_nothing_secret(c: Cli) -> None:
    c.monkeypatch.delenv("EVOLVER_MODEL")
    assert c("eval", "--model", MODEL, "--task", TASK, "--mechanism", "skills")[0] == 0
    for path in c.tmp_path.joinpath("outputs").rglob("*.json"):
        assert not any(value in path.read_text() for value in SECRET_VALUES.values() if "secret" in value)


def test_a_failed_pass_is_reported_by_its_error(c: Cli) -> None:
    c.work.raise_on("pass", 2, EvalError("pass 2 of the base agent in setting 3 was not a valid measurement"))
    code, _, err = c("eval", "--model", MODEL, "--task", TASK, "--passes", "3")
    assert code == cli.EXIT_EVAL and "pass 2 of the base agent" in err
    assert [call[5] for call in passes(c)] == [1, 2]


# --- run --------------------------------------------------------------------------------------------------------------


def test_run_prints_the_table_of_combinations_and_summaries(c: Cli) -> None:
    configs = write_configs(
        c.tmp_path, combos=[combo(MODEL, TASK, "rl"), combo(MODEL, "drive-create", "skills")], name="exp"
    )
    code, out, err = c("run", str(configs / "main.yaml"))
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == ["model", "task", "mechanism", "adaptation", "summary"]
    assert lines[1].split() == [MODEL, TASK, "rl", "ran", str(c.run_dir().summary_path)]
    assert lines[2].split() == [
        MODEL,
        "drive-create",
        "skills",
        "ran",
        str(c.run_dir("skills", task="drive-create").summary_path),
    ]
    assert "[1/2]" in err and "[2/2]" in err


def test_a_failed_run_reports_the_error_and_what_finished_before_it(c: Cli) -> None:
    configs = write_configs(
        c.tmp_path, combos=[combo(MODEL, TASK, "rl"), combo(MODEL, "drive-create", "rl")], name="exp"
    )
    c.work.raise_on("evaluate", 2, EvalError("pass 1 was not valid"))
    code, out, err = c("run", str(configs / "main.yaml"))
    assert code == cli.EXIT_EVAL and out == ""
    assert first_line(err) == "evolve-ipi: evaluation failed: pass 1 was not valid"
    assert f"run directory: {c.run_dir(task='drive-create').root}" in err
    assert "combinations finished before this:" in err and f"{MODEL}  {TASK}" in err.replace("   ", "  ")


def test_run_reads_the_other_configuration_files_next_to_the_experiment_file(c: Cli) -> None:
    configs = write_configs(c.tmp_path, combos=[combo(MODEL, TASK, "rl")], name="exp")
    edit_yaml(configs / "rl.yaml", lambda d: d.update(lora_rank=8))
    assert c("run", str(configs / "main.yaml"))[0] == 0
    assert json.loads(c.run_dir().config_path.read_text())["rl"]["lora_rank"] == 8


def test_configs_can_be_given_for_run(c: Cli) -> None:
    experiment = write_configs(c.tmp_path, combos=[combo(MODEL, TASK, "rl")], name="exp")
    mine = write_configs(c.tmp_path, name="mine")
    edit_yaml(mine / "rl.yaml", lambda d: d.update(lora_rank=4))
    assert c("run", str(experiment / "main.yaml"), "--configs", str(mine))[0] == 0
    assert json.loads(c.run_dir().config_path.read_text())["rl"]["lora_rank"] == 4


# --- check ------------------------------------------------------------------------------------------------------------


def test_check_prints_a_line_per_check_and_exits_with_zero_when_none_failed(c: Cli) -> None:
    def fake_checks(cfgs: object, *, report: Callable[[check.Result], None], **options: object) -> list[check.Result]:
        results = [check.Result("data", "pass", "5 tasks"), check.Result("agent", "skip", "no OpenClaw")]
        for result in results:
            report(result)
        return results

    c.monkeypatch.setattr(check, "run_checks", fake_checks)
    code, out, err = c("check")
    assert code == 0 and err == ""
    assert out.splitlines()[:2] == [
        check.format_result(check.Result("data", "pass", "5 tasks")),
        check.format_result(check.Result("agent", "skip", "no OpenClaw")),
    ]
    assert out.splitlines()[-1] == "checks: 1 passed, 0 failed, 1 skipped"


def test_check_exits_with_its_own_status_when_a_check_failed(c: Cli) -> None:
    def fake_checks(cfgs: object, *, report: Callable[[check.Result], None], **options: object) -> list[check.Result]:
        results = [check.Result("data", "pass", "ok"), check.Result("judges", "fail", "a judge never fires")]
        for result in results:
            report(result)
        return results

    c.monkeypatch.setattr(check, "run_checks", fake_checks)
    code, out, _ = c("check")
    assert code == cli.EXIT_CHECK_FAILED
    assert check.format_result(check.Result("judges", "fail", "a judge never fires")) in out
    assert out.splitlines()[-1] == "checks: 1 passed, 1 failed, 0 skipped"


def test_check_needs_no_key(c: Cli) -> None:
    for name in SECRET_VALUES:
        c.monkeypatch.delenv(name)
    c.monkeypatch.setattr(check, "run_checks", lambda cfgs, *, report, **options: [])
    assert c("check")[0] == 0


# --- the log ----------------------------------------------------------------------------------------------------------


def test_progress_is_logged_to_stderr_with_timestamps(c: Cli) -> None:
    def adapt(*args: object, **kwargs: object) -> dict[str, str]:
        logging.getLogger("evolve_ipi.adapt.rl").info("episode 1 of 144: file-create-adapt-000")
        logging.getLogger("other.library").info("not ours")
        return c.work.adapt_rl(*args, **kwargs)  # type: ignore[arg-type]

    c.monkeypatch.setattr(runner, "adapt_rl", adapt)
    code, out, err = c(*ADAPT_RL)
    assert code == 0
    import re

    assert re.search(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d INFO evolve_ipi\.adapt\.rl: episode 1 of 144", err, re.MULTILINE)
    assert "not ours" not in err and "not ours" not in out


# --- one request to the evolver, and only once the locks are held ---------------------------------------------------


def test_a_command_run_in_another_thread_is_interrupted_like_any_other(c: Cli) -> None:
    """Only the main thread can set signal handlers; a command in another thread reports its interruption alike."""

    def interrupted(*args: object, **options: object) -> None:
        raise KeyboardInterrupt

    c.monkeypatch.setattr(runner, "startup", interrupted)
    result: dict[str, object] = {}

    def command() -> None:
        try:
            result["code"] = c(*ADAPT_RL)[0]
        except BaseException as error:  # noqa: BLE001 -- the test reports whatever escaped
            result["escaped"] = error

    thread = threading.Thread(target=command)
    thread.start()
    thread.join(timeout=60)
    assert result == {"code": cli.EXIT_INTERRUPTED}


@pytest.mark.parametrize("command", [ADAPT_SKILLS, ["eval", "--model", MODEL, "--task", TASK]])
def test_a_runtime_directory_in_use_is_refused_before_any_cost_and_names_no_run_directory(
    c: Cli, command: list[str]
) -> None:
    with RuntimePaths().lock():
        code, out, err = c(*command)
    assert code == cli.EXIT_RUN_LOCKED and out == ""
    assert first_line(err).startswith("evolve-ipi: run directory in use: the runtime directory ")
    assert "run directory:" not in err, "no combination's run directory was entered"
    assert c.work.preflights == [] and c.work.calls == []


def stand_in_conversations(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ``adapt_skills``, with its preflight; only its conversations and evolutions are replaced."""

    def evolve(model: object, task: tasks.Task, run: RunDir, *args: object) -> dict[str, object]:
        library = run.adapt_dir / "library"
        library.mkdir(parents=True)
        rows = "".join(f'{{"id": "{item}"}}\n' for item in sorted(task.adaptation)[:12])
        (run.adapt_dir / "episodes.jsonl").write_text(rows, encoding="utf-8")
        marker = {"library": "library", "sha": lib_sha(library), "evolutions": 12, "skills": 0}
        write_json(run.adapt_dir / "library.json", marker)
        return {"library": library, "sha": marker["sha"], "evolutions": 12, "skills": 0}

    monkeypatch.setattr(runner, "adapt_skills", skills.adapt_skills)
    monkeypatch.setattr(skills, "check_inputs", lambda *args, **options: None)
    monkeypatch.setattr(skills, "_evolve", evolve)


def test_every_command_asks_the_evolver_exactly_once_when_it_evolves_and_never_otherwise(c: Cli) -> None:
    asked: list[str] = []

    def counting(secrets: object, spec: object, *, paths: object) -> None:
        asked.append("request")

    stand_in_conversations(c.monkeypatch)
    c.monkeypatch.setattr(runner, "evolver_preflight", counting)
    c.monkeypatch.setattr(skills, "evolver_preflight", counting)
    assert c(*ADAPT_SKILLS)[0] == 0 and len(asked) == 1, "adapt skills"
    assert c(*ADAPT_RL)[0] == 0 and len(asked) == 1, "adapt rl"
    assert c("eval", "--model", MODEL, "--task", TASK, "--mechanism", "skills")[0] == 0 and len(asked) == 1, "eval"
    experiment = write_configs(
        c.tmp_path,
        combos=[combo(MODEL, "drive-create", "skills"), combo("gpt-oss-120b", "chat-invite", "skills")],
        name="two-libraries",
    )
    assert c("run", str(experiment / "main.yaml"))[0] == 0
    assert len(asked) == 2, "run: one request for its two libraries"
    assert c("run", str(experiment / "main.yaml"))[0] == 0 and len(asked) == 2, "both finished: no request"
