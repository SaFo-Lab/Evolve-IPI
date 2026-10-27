"""The sequence behind the commands: the startup checks, the configuration of a run directory and ``run CONFIG``.

``adapt_rl``, ``adapt_skills`` and ``evaluate_combo`` are stand-ins that leave the files the real ones leave, so these
tests run nothing paid and start no process.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import eval_world
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
    set_secrets,
    write_configs,
)

from evolve_ipi import paths as paths_module
from evolve_ipi import procs, runner, tasks
from evolve_ipi.adapt import rl
from evolve_ipi.agent.openclaw import OPENCLAW_VERSION, OpenClawError
from evolve_ipi.config import ConfigError, PortsSpec, Secrets, load_configs
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import PortInUse
from evolve_ipi.records import RunDir, RunDirError, RunLocked
from evolve_ipi.tasks import TaskDataError, load_task
from evolve_ipi.thirdparty import PatchError

RL_A = combo("qwen3.8-27b", "file-create", "rl")
SKILLS_B = combo("qwen3.8-27b", "drive-create", "skills")
RL_C = combo("gpt-oss-120b", "chat-invite", "rl")
RL_DRIVE = combo("qwen3.8-27b", "drive-create", "rl")
SKILLS_C = combo("gpt-oss-120b", "chat-invite", "skills")
DATA = tasks.verify_data()  # the digest of data/SHA256SUMS that config.json records


@dataclass
class World:
    """A runtime in ``tmp_path``, every secret set, and the paid work replaced."""

    tmp_path: Path
    work: Work
    monkeypatch: pytest.MonkeyPatch

    @property
    def paths(self) -> RuntimePaths:
        """The runtime as the environment describes it now: a test may change the environment."""
        return RuntimePaths()

    def configs(self, *combos: dict[str, str], settings: tuple[int, ...] = (1, 2, 3), name: str = "configs") -> Path:
        return write_configs(self.tmp_path, combos=combos or None, settings=settings, name=name)

    def run(self, directory: Path, **options: object) -> list[runner.ComboResult]:
        return runner.run(directory / "main.yaml", paths=self.paths, **options)  # type: ignore[arg-type]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return World(tmp_path, install_world(monkeypatch, tmp_path), monkeypatch)


def run_dir(world: World, spec: dict[str, str]) -> RunDir:
    return RunDir.for_combo(world.paths.outputs, spec["model"], spec["task"], spec["mechanism"])


# --- run CONFIG -------------------------------------------------------------------------------------------------------


def test_run_walks_the_combinations_in_order(world: World) -> None:
    directory = world.configs(RL_A, SKILLS_B, RL_C, settings=(3, 1))
    results = world.run(directory)

    assert world.work.calls == [
        ("adapt-rl", "qwen3.8-27b", "file-create"),
        ("evaluate", "qwen3.8-27b", "file-create", "rl"),
        ("adapt-skills", "qwen3.8-27b", "drive-create"),
        ("evaluate", "qwen3.8-27b", "drive-create", "skills"),
        ("adapt-rl", "gpt-oss-120b", "chat-invite"),
        ("evaluate", "gpt-oss-120b", "chat-invite", "rl"),
    ]
    assert world.work.settings_seen == [[3, 1]] * 3, "the settings of the experiment file, in its order"
    assert [(r.model, r.task, r.mechanism, r.adapted_now) for r in results] == [
        ("qwen3.8-27b", "file-create", "rl", True),
        ("qwen3.8-27b", "drive-create", "skills", True),
        ("gpt-oss-120b", "chat-invite", "rl", True),
    ]
    for result, spec in zip(results, (RL_A, SKILLS_B, RL_C), strict=True):
        assert result.run_dir == run_dir(world, spec).root
        assert result.summary == run_dir(world, spec).summary_path and result.summary.is_file()


def test_run_uses_the_configuration_beside_the_experiment_file_unless_told_otherwise(world: World) -> None:
    directory = world.configs(RL_A)
    elsewhere = write_configs(world.tmp_path, name="other")
    edit_yaml(elsewhere / "rl.yaml", lambda data: data.update(lora_rank=16))

    world.run(directory)
    assert json.loads(run_dir(world, RL_A).config_path.read_text())["rl"]["lora_rank"] == 32

    other = world.tmp_path / "second-outputs"
    world.monkeypatch.setenv("EVOLVE_IPI_OUTPUTS", str(other))
    runner.run(directory / "main.yaml", configs_dir=elsewhere, paths=RuntimePaths())
    recorded = json.loads(RunDir.for_combo(other, "qwen3.8-27b", "file-create", "rl").config_path.read_text())
    assert recorded["rl"]["lora_rank"] == 16


def test_a_finished_adaptation_is_not_run_again_and_its_evaluation_goes_on(world: World) -> None:
    directory = world.configs(RL_A, SKILLS_B)
    world.run(directory)
    world.work.calls.clear()

    results = world.run(directory)

    assert world.work.calls == [
        ("evaluate", "qwen3.8-27b", "file-create", "rl"),
        ("evaluate", "qwen3.8-27b", "drive-create", "skills"),
    ], "evaluate_combo is the resume point of the passes: it never runs a finished pass again"
    assert [r.adapted_now for r in results] == [False, False]


def test_an_interrupted_run_resumes_where_it_stopped(world: World) -> None:
    directory = world.configs(RL_A, SKILLS_B, RL_C)
    world.work.raise_on("evaluate", 2, KeyboardInterrupt())
    progress = runner.Progress()

    with pytest.raises(KeyboardInterrupt):
        world.run(directory, progress=progress)

    assert progress.run_dir == run_dir(world, SKILLS_B).root, "the CLI names this directory to resume"
    assert [(r.model, r.task) for r in progress.finished] == [("qwen3.8-27b", "file-create")]
    assert not run_dir(world, SKILLS_B).holds_lock(), "the lock goes with the interruption"
    world.work.calls.clear()

    results = world.run(directory)

    assert world.work.calls == [
        ("evaluate", "qwen3.8-27b", "file-create", "rl"),
        ("evaluate", "qwen3.8-27b", "drive-create", "skills"),
        ("adapt-rl", "gpt-oss-120b", "chat-invite"),
        ("evaluate", "gpt-oss-120b", "chat-invite", "rl"),
    ], "nothing is adapted twice; the third combination starts from its beginning"
    assert [r.adapted_now for r in results] == [False, False, True]


def test_a_failure_stops_the_run_and_keeps_what_finished(world: World) -> None:
    from evolve_ipi.evaluate import EvalError

    directory = world.configs(RL_A, SKILLS_B, RL_C)
    world.work.raise_on("evaluate", 2, EvalError("pass 1 was not a valid measurement"))
    progress = runner.Progress()

    with pytest.raises(EvalError, match="pass 1 was not a valid measurement"):
        world.run(directory, progress=progress)

    assert [call[0] for call in world.work.calls] == ["adapt-rl", "evaluate", "adapt-skills", "evaluate"]
    assert [(r.task, r.mechanism) for r in progress.finished] == [("file-create", "rl")]
    assert (progress.combo, progress.run_dir) == ("qwen3.8-27b / drive-create / skills", run_dir(world, SKILLS_B).root)
    assert run_dir(world, SKILLS_B).adapt_done("skills"), "the finished adaptation stays for the next run"


def test_the_configuration_is_recorded_on_first_use(world: World) -> None:
    directory = world.configs(RL_A, SKILLS_B)
    world.run(directory)
    cfgs = load_configs(directory)

    rl = json.loads(run_dir(world, RL_A).config_path.read_text())
    skills = json.loads(run_dir(world, SKILLS_B).config_path.read_text())

    assert rl["model"] == {**cfgs.models["qwen3.8-27b"].model_dump(mode="json")}
    assert (rl["task"], rl["mechanism"]) == ("file-create", "rl")
    assert rl["rl"] == cfgs.rl.model_dump(mode="json") and "skills" not in rl
    assert skills["skills"] == cfgs.skills.model_dump(mode="json") and "rl" not in skills
    assert rl["eval"] == skills["eval"] == cfgs.eval.model_dump(mode="json")
    assert rl["openclaw"] == OPENCLAW_VERSION
    assert rl["frameworks"]["patched_files"] == PATCHED
    assert set(rl["frameworks"]["commits"]) == {"OpenClaw-RL", "MetaClaw"}
    assert rl["data_sha256sums"] == skills["data_sha256sums"] == DATA
    assert set(rl) == {"model", "task", "mechanism", "rl", "eval", "openclaw", "frameworks", "data_sha256sums"}, (
        "no ports, paths, secrets"
    )
    assert not any(value in json.dumps(rl) for value in SECRET_VALUES.values())


def test_a_combination_recorded_under_another_configuration_is_refused_before_anything_runs(world: World) -> None:
    directory = world.configs(RL_A, SKILLS_B)
    world.run(directory)
    world.work.calls.clear()
    edit_yaml(directory / "skills.yaml", lambda data: data.update(max_new_skills=5))
    edit_yaml(directory / "eval.yaml", lambda data: data.update(temperature=0.7))

    with pytest.raises(RunDirError) as raised:
        world.run(directory)

    message = str(raised.value)
    assert str(run_dir(world, RL_A).config_path) in message, (
        "the first combination already differs, by eval.temperature"
    )
    assert "eval.temperature" in message
    assert world.work.calls == []


def test_a_later_combination_that_differs_is_found_before_the_first_one_runs(world: World) -> None:
    directory = world.configs(RL_A, SKILLS_B)
    world.run(directory)
    shutil.rmtree(run_dir(world, RL_A).root)  # the first combination starts from nothing again
    world.work.calls.clear()
    edit_yaml(directory / "skills.yaml", lambda data: data.update(skill_top_k=3))

    with pytest.raises(RunDirError, match=r"skills\.skill_top_k"):
        world.run(directory)

    assert world.work.calls == [], "no money is spent on the first combination before the second one is refused"
    assert not run_dir(world, RL_A).config_path.exists()


def test_a_second_run_in_the_same_directory_is_refused_while_the_first_holds_it(world: World) -> None:
    directory = world.configs(RL_A)
    with run_dir(world, RL_A).lock(), pytest.raises(RunLocked, match="in use by another run"):
        world.run(directory)
    assert world.work.calls == []


def test_every_combination_is_locked_while_it_works_and_not_after(world: World) -> None:
    directory = world.configs(RL_A, SKILLS_B)
    world.run(directory)  # the stand-ins fail unless the combination's run directory is locked
    assert not run_dir(world, RL_A).holds_lock() and not run_dir(world, SKILLS_B).holds_lock()


def test_the_final_table_lists_every_combination_with_its_summary(world: World) -> None:
    directory = world.configs(RL_A, SKILLS_B)
    first = world.run(directory)
    again = world.run(directory)

    table = runner.format_table(again).splitlines()
    assert table[0].split() == ["model", "task", "mechanism", "adaptation", "summary"]
    assert table[1].split() == ["qwen3.8-27b", "file-create", "rl", "kept", str(run_dir(world, RL_A).summary_path)]
    assert table[2].split()[:4] == ["qwen3.8-27b", "drive-create", "skills", "kept"]
    assert runner.format_table(first).splitlines()[1].split()[3] == "ran"
    assert runner.format_table([]) == "no combination finished"


def test_progress_lines_are_logged(world: World, caplog: pytest.LogCaptureFixture) -> None:
    directory = world.configs(RL_A, SKILLS_B)
    with caplog.at_level("INFO", logger="evolve_ipi.runner"):
        world.run(directory)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "1/2" in text and "2/2" in text
    assert "qwen3.8-27b / file-create / rl" in text


def test_an_interrupted_adaptation_is_resumed_by_the_next_invocation(world: World) -> None:
    """It left no marker, so it is not finished: the next invocation adapts again (the adapters restart the stage or the
    library from its beginning) and then goes on to the evaluation."""
    directory = world.configs(RL_A, SKILLS_B)
    world.work.raise_on("adapt-rl", 1, KeyboardInterrupt())
    progress = runner.Progress()

    with pytest.raises(KeyboardInterrupt):
        world.run(directory, progress=progress)

    assert world.work.calls == [("adapt-rl", "qwen3.8-27b", "file-create")]
    assert progress.run_dir == run_dir(world, RL_A).root and progress.finished == []
    assert not run_dir(world, RL_A).adapt_done("rl") and not run_dir(world, RL_A).holds_lock()
    world.work.calls.clear()

    results = world.run(directory)

    assert world.work.calls == [
        ("adapt-rl", "qwen3.8-27b", "file-create"),
        ("evaluate", "qwen3.8-27b", "file-create", "rl"),
        ("adapt-skills", "qwen3.8-27b", "drive-create"),
        ("evaluate", "qwen3.8-27b", "drive-create", "skills"),
    ]
    assert [r.adapted_now for r in results] == [True, True]


def test_the_checks_of_every_adaptation_come_before_the_first_combination(world: World) -> None:
    """A mistake in the third combination must not be found after the first two were paid for."""
    directory = world.configs(SKILLS_B, RL_A, RL_C)
    world.run(directory)
    assert sorted(world.work.checked) == [
        ("rl", "gpt-oss-120b", "chat-invite"),
        ("rl", "qwen3.8-27b", "file-create"),
        ("skills", "qwen3.8-27b", ""),
    ]
    assert world.work.preflights == ["evolver-model-test"], "one request for the evolver, however many combinations"
    assert [call[0] for call in world.work.calls] == [
        "adapt-skills",
        "evaluate",
        "adapt-rl",
        "evaluate",
        "adapt-rl",
        "evaluate",
    ]


def test_a_bad_recipe_of_a_later_combination_is_found_before_any_combination_runs(world: World) -> None:
    world.monkeypatch.setattr(runner, "check_setup", rl.check_setup)  # the real checks of the RL line
    world.monkeypatch.setattr(rl, "verify_patches", lambda: dict(PATCHED))
    directory = world.configs(SKILLS_B, RL_A)
    edit_yaml(directory / "rl.yaml", lambda data: data.update(save_interval=16))

    with pytest.raises(ConfigError, match=r"rl\.yaml: save_interval is 16; it must equal updates_per_stage"):
        world.run(directory)

    assert world.work.calls == [] and world.work.preflights == [], "nothing was adapted, evaluated or asked"


def test_a_model_family_the_trainer_does_not_read_alike_is_found_before_any_combination_runs(world: World) -> None:
    world.monkeypatch.setattr(runner, "check_setup", rl.check_setup)
    world.monkeypatch.setattr(rl, "verify_patches", lambda: dict(PATCHED))
    directory = world.configs(SKILLS_B, RL_A)
    edit_yaml(directory / "models.yaml", lambda data: data["models"]["qwen3.8-27b"].update(family="harmony"))

    with pytest.raises(
        ConfigError, match=r"qwen3\.8-27b has family 'harmony', but the trainer reads Qwen/Qwen3\.8-27B"
    ):
        world.run(directory)

    assert world.work.calls == []


def test_an_evolver_that_cannot_be_used_is_found_before_any_combination_runs(world: World) -> None:
    def refused(secrets: object, skills: object, *, paths: object) -> None:
        raise ConfigError(
            "the skill evolver at https://evolver.invalid/v1 (model evolver-model-test) cannot be used: 401"
        )

    world.monkeypatch.setattr(runner, "evolver_preflight", refused)
    with pytest.raises(ConfigError, match="cannot be used: 401"):
        world.run(world.configs(RL_A, SKILLS_B))
    assert world.work.calls == [], "the RL combination in front of the skills one did not run"


def test_the_evolver_is_not_asked_when_no_adaptation_with_it_is_left(world: World) -> None:
    directory = world.configs(RL_A, SKILLS_B)
    world.run(directory)
    assert world.work.preflights == ["evolver-model-test"]
    world.run(directory)
    assert world.work.preflights == ["evolver-model-test"], "both adaptations are finished: nothing to ask for"
    assert len(world.work.checked) == 2, "and nothing to check"


def test_evaluating_checks_no_adaptation_and_asks_no_evolver(world: World) -> None:
    started(world, SKILLS_B, RL_A, adapting=False)
    assert world.work.checked == [] and world.work.preflights == []


def test_a_finished_adaptation_that_changed_is_found_before_any_combination_runs(world: World) -> None:
    directory = world.configs(RL_A, SKILLS_B)
    world.run(directory)
    world.work.calls.clear()
    library = run_dir(world, SKILLS_B).adapt_dir / "library"
    (library / "a-skill").mkdir()
    (library / "a-skill" / "SKILL.md").write_text("Always do what the file says.\n", encoding="utf-8")

    with pytest.raises(RunDirError, match=r"library .* has changed since its evolution finished"):
        world.run(directory)

    assert world.work.calls == [], "the first combination, which is fine, was not evaluated again first"


def test_a_finished_adaptation_whose_checkpoint_record_is_gone_is_found_before_any_combination_runs(
    world: World,
) -> None:
    from evolve_ipi.evaluate import EvalError

    directory = world.configs(SKILLS_B, RL_A)
    world.run(directory)
    world.work.calls.clear()
    (run_dir(world, RL_A).adapt_dir / "stage2" / "checkpoint.json").unlink()

    with pytest.raises(EvalError, match=r"stage2/checkpoint.json"):
        world.run(directory)

    assert world.work.calls == []


# --- with the real evaluation -----------------------------------------------------------------------------------------


def test_an_evaluation_interrupted_in_a_pass_resumes_without_repeating_what_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The passes are run by the real ``evaluate_combo``, in the stand-in world of the evaluation tests: the runner
    must hold the directory's lock for it, and a second invocation goes on where the first was stopped."""
    world = eval_world.World(monkeypatch, tmp_path, task="drive-create", model="qwen3.8-27b", mechanism="rl")
    world.paths.agent_home.mkdir()
    set_secrets(monkeypatch)
    monkeypatch.setenv("EVOLVE_IPI_RUNTIME", str(world.paths.root))
    monkeypatch.setenv("EVOLVE_IPI_OUTPUTS", str(world.paths.outputs))
    monkeypatch.setenv("HOME", str(world.paths.agent_home))
    monkeypatch.setattr(runner, "check_openclaw", lambda: OPENCLAW_VERSION)
    monkeypatch.setattr(runner, "verify_patches", lambda: dict(PATCHED))
    monkeypatch.setattr(runner, "require_free", lambda ports: None)
    monkeypatch.setattr(runner, "check_setup", lambda *args, **kwargs: None)
    adapted: list[str] = []

    def adapt(model: object, task: object, run: RunDir, cfgs: object, secrets: object, *, paths: object) -> None:
        assert run.holds_lock()
        adapted.append(run.root.name)
        world.finish_rl()

    monkeypatch.setattr(runner, "adapt_rl", adapt)
    configs = write_configs(tmp_path, combos=[RL_DRIVE], settings=(3,))
    run = world.run
    first_item = min(world.task.test)
    world.acts[(4, first_item)] = eval_world.Act(raise_=KeyboardInterrupt())  # the fourth server: the adapted agent

    with pytest.raises(KeyboardInterrupt):
        runner.run(configs / "main.yaml", paths=world.paths)

    assert [run.is_complete("base", 3, k) for k in (1, 2, 3)] == [True, True, True]
    assert run.pass_dir("adapted", 3, 1).is_dir() and not run.is_complete("adapted", 3, 1)
    assert not run.holds_lock() and not run.summary_path.exists() and world.all_stopped()
    world.calls.clear()

    results = runner.run(configs / "main.yaml", paths=world.paths)

    assert adapted == ["rl"], "the adaptation finished in the first invocation and is not run again"
    assert [r.adapted_now for r in results] == [False]
    assert len(world.episodes()) == 40, "only the adapted agent's pass ran again: three passes of the base agent stay"
    (void,) = run.voids("adapted", 3, 1)
    assert json.loads((void / "why.json").read_text())["reason"].startswith("interrupted")
    assert run.is_complete("adapted", 3, 1) and run.summary_path.is_file()
    assert json.loads(run.summary_path.read_text())["settings"]["3"]["base"]["n"] == 120
    assert not run.holds_lock() and world.all_stopped()


HOLD_THE_RUNTIME = """
import sys, time
from pathlib import Path
from evolve_ipi.paths import RuntimePaths

with RuntimePaths(root=Path(sys.argv[1]), agent_home=Path(sys.argv[2])).lock():
    print("held", flush=True)
    time.sleep(60)
"""


def hold_in_another_process(root: Path, home: Path) -> subprocess.Popen[str]:
    child = subprocess.Popen(
        [sys.executable, "-c", HOLD_THE_RUNTIME, str(root), str(home)], stdout=subprocess.PIPE, text=True
    )
    assert child.stdout is not None and child.stdout.readline().strip() == "held"
    return child


def release(child: subprocess.Popen[str]) -> None:
    child.kill()
    child.wait()
    if child.stdout is not None:
        child.stdout.close()


@pytest.mark.parametrize("held", ["runtime directory", "agent's home"])
def test_a_second_run_with_the_same_runtime_directory_or_home_is_refused(world: World, held: str) -> None:
    """Other ports do not help: the profile, the workspace and File-Create's target directory are shared."""
    paths = world.paths
    other = world.tmp_path / "other"
    child = hold_in_another_process(
        *((paths.root, other) if held == "runtime directory" else (other, paths.agent_home))
    )
    try:
        with pytest.raises(RunLocked) as raised:
            world.run(world.configs(RL_A))
    finally:
        release(child)
    message = str(raised.value)
    assert message.startswith(f"the {held} ")
    assert str(paths.root if held == "runtime directory" else paths.agent_home) in message
    assert "EVOLVE_IPI_RUNTIME" in message and "HOME" in message and "its own container" in message
    assert "output directory" not in message, "not the advice for a run directory"
    assert world.work.calls == []
    assert world.run(world.configs(RL_A, name="again"))[0].adapted_now is True, "free again once the other run ended"


@pytest.mark.parametrize("held", ["runtime directory", "agent's home"])
def test_a_refused_lock_costs_no_request_to_the_evolver(world: World, held: str) -> None:
    """The evolver is asked once the locks are held: a second run that shares one is refused before it spends."""
    paths = world.paths
    other = world.tmp_path / "other"
    child = hold_in_another_process(
        *((paths.root, other) if held == "runtime directory" else (other, paths.agent_home))
    )
    try:
        with pytest.raises(RunLocked):
            world.run(world.configs(SKILLS_B, RL_A))
        start = started(world, SKILLS_B, name="adapt")
        combo_b = runner.combo_of(start.cfgs, SKILLS_B["model"], SKILLS_B["task"], "skills")
        with pytest.raises(RunLocked):
            runner.adapt(start, combo_b)
    finally:
        release(child)
    assert world.work.preflights == [] and world.work.calls == []
    assert world.work.checked, "the checks that cost nothing were made before the lock"


HOLD_A_RUN_DIRECTORY = """
import sys, time
from pathlib import Path
from evolve_ipi.records import RunDir

with RunDir(Path(sys.argv[1])).lock():
    print("held", flush=True)
    time.sleep(60)
"""


def test_a_run_directory_in_use_is_refused_before_the_evolver_is_asked_and_before_the_first_combination(
    world: World,
) -> None:
    directory = world.configs(RL_A, SKILLS_B)
    second = run_dir(world, SKILLS_B)
    second.root.mkdir(parents=True)
    child = subprocess.Popen(
        [sys.executable, "-c", HOLD_A_RUN_DIRECTORY, str(second.root)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "held"
        with pytest.raises(RunLocked, match="is in use by another run"):
            world.run(directory)
    finally:
        release(child)
    assert world.work.preflights == [] and world.work.calls == [], "nothing was asked, adapted or evaluated"
    world.run(directory)
    assert world.work.preflights == ["evolver-model-test"]


def test_run_asks_the_evolver_once_for_every_pending_library_and_the_evolutions_not_again(world: World) -> None:
    world.run(world.configs(SKILLS_B, RL_A, SKILLS_C))
    assert world.work.preflights == ["evolver-model-test"]
    assert world.work.skills_preflight == [False, False]


def test_the_runtime_directory_and_the_home_are_held_around_the_adaptation_and_the_evaluation(world: World) -> None:
    """``Work`` asserts it in ``adapt_*`` and ``evaluate_combo``; a run that does not hold them fails there."""
    world.run(world.configs(RL_A, SKILLS_B))
    assert [call[0] for call in world.work.calls] == ["adapt-rl", "evaluate", "adapt-skills", "evaluate"]
    assert not world.paths.holds_lock()


def test_a_lock_the_same_process_already_holds_is_refused_too(world: World) -> None:
    with world.paths.lock(), pytest.raises(RunLocked, match="is in use by another run"):
        world.run(world.configs(RL_A))


def test_the_lock_writes_nothing_where_the_agent_can_look(world: World) -> None:
    before = sorted(path.name for path in world.paths.agent_home.iterdir())
    with world.paths.lock():
        assert sorted(path.name for path in world.paths.agent_home.iterdir()) == before
        assert not list(world.paths.root.iterdir())
    assert sorted(path.name for path in world.paths.agent_home.iterdir()) == before


def test_the_home_may_be_the_runtime_directory(world: World) -> None:
    paths = RuntimePaths(root=world.tmp_path / "same", agent_home=world.tmp_path / "same")
    with paths.lock():
        assert paths.holds_lock()


def test_the_lock_is_close_on_exec(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trainer or a gateway that outlived the run must not keep either directory locked."""
    inheritable: list[bool] = []
    real = fcntl.flock

    def spy(descriptor: int, operation: int) -> None:
        inheritable.append(os.get_inheritable(descriptor))
        real(descriptor, operation)

    monkeypatch.setattr(paths_module.fcntl, "flock", spy)
    with world.paths.lock():
        pass
    assert inheritable == [False] * 4, "locked and unlocked: one descriptor for the runtime directory, one for the home"


# --- the startup checks -----------------------------------------------------------------------------------------------


def started(world: World, *specs: dict[str, str], adapting: bool = True, name: str = "configs") -> runner.Startup:
    cfgs = load_configs(world.configs(*specs, name=name))
    combos = [runner.combo_of(cfgs, s["model"], s["task"], s["mechanism"]) for s in specs]
    return runner.startup(cfgs, combos, adapting=adapting, paths=world.paths)


def test_startup_returns_the_configuration_the_secrets_and_the_paths(world: World) -> None:
    start = started(world, RL_A)
    assert start.cfgs.models["qwen3.8-27b"].tinker_id == "Qwen/Qwen3.8-27B"
    assert start.secrets.tinker_api_key == SECRET_VALUES["TINKER_API_KEY"]
    assert start.paths == world.paths
    assert start.patched == PATCHED
    assert start.data_sums == DATA


def test_data_that_differs_from_its_checksums_is_refused_before_anything_starts(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = world.tmp_path / "data"
    shutil.copytree(tasks.DATA_DIR, copy)
    path = copy / "tasks" / "drive-create" / "test.jsonl"
    path.write_bytes(path.read_bytes().replace(b"drive", b"Drive", 1))
    monkeypatch.setattr(tasks, "DATA_DIR", copy)
    for adapting in (True, False):
        with pytest.raises(ConfigError, match=r"tasks/drive-create/test\.jsonl has changed"):
            started(world, RL_A, SKILLS_B, adapting=adapting, name=f"configs-{adapting}")
    assert world.work.ports_checked == [] and world.work.checked == [] and world.work.preflights == []


def test_a_run_directory_started_on_other_data_is_refused(world: World) -> None:
    directory = world.configs(RL_A)
    world.run(directory)
    run = run_dir(world, RL_A)
    assert json.loads(run.config_path.read_text())["data_sha256sums"] == DATA
    config = json.loads(run.config_path.read_text())
    run.config_path.write_text(json.dumps({**config, "data_sha256sums": "0" * 64}))
    with pytest.raises(RunDirError, match="data_sha256sums"):
        world.run(directory)


def test_an_unknown_model_task_or_mechanism_is_refused_naming_it(world: World) -> None:
    cfgs = load_configs(world.configs())
    with pytest.raises(ConfigError, match=r"unknown model 'gpt-9'.*known models: qwen3\.8-27b, gpt-oss-120b"):
        runner.combo_of(cfgs, "gpt-9", "file-create", "rl")
    with pytest.raises(ConfigError, match=r"unknown task 'email'.*known tasks: file-create, drive-create"):
        runner.combo_of(cfgs, "qwen3.8-27b", "email", "rl")
    with pytest.raises(ConfigError, match="unknown mechanism 'lora'.*rl, skills"):
        runner.combo_of(cfgs, "qwen3.8-27b", "file-create", "lora")
    with pytest.raises(ConfigError, match=r"gpt-oss-20b does not take part in 'rl'.*skills"):
        runner.combo_of(cfgs, "gpt-oss-20b", "file-create", "rl")
    assert runner.combo_of(cfgs, "gpt-oss-20b", "file-create", "skills").mechanism == "skills"


def test_a_missing_tinker_key_is_named_before_anything_runs(world: World) -> None:
    world.monkeypatch.delenv("TINKER_API_KEY")
    with pytest.raises(ConfigError, match="TINKER_API_KEY"):
        world.run(world.configs(RL_A))
    assert world.work.calls == []


def test_skill_evolution_needs_every_evolver_variable_and_names_each_missing_one(world: World) -> None:
    for name in ("EVOLVER_MODEL", "EVOLVER_API_KEY"):
        world.monkeypatch.delenv(name)
    with pytest.raises(ConfigError) as raised:
        started(world, SKILLS_B)
    assert "EVOLVER_MODEL" in str(raised.value) and "EVOLVER_API_KEY" in str(raised.value)
    assert "EVOLVER_BASE_URL" not in str(raised.value) and "TINKER_API_KEY" not in str(raised.value)


def test_an_experiment_with_one_skills_combination_needs_the_evolver_and_one_without_does_not(world: World) -> None:
    for name in ("EVOLVER_MODEL", "EVOLVER_API_KEY", "EVOLVER_BASE_URL"):
        world.monkeypatch.delenv(name)
    assert len(world.run(world.configs(RL_A, RL_C, name="rl-only"))) == 2

    with pytest.raises(ConfigError, match="EVOLVER_MODEL"):
        world.run(world.configs(RL_A, SKILLS_B, name="with-skills"))
    assert [call for call in world.work.calls if call[0] == "adapt-skills"] == []


def test_evaluating_a_skills_agent_needs_no_evolver(world: World) -> None:
    for name in ("EVOLVER_MODEL", "EVOLVER_API_KEY", "EVOLVER_BASE_URL"):
        world.monkeypatch.delenv(name)
    start = started(world, SKILLS_B, adapting=False)
    assert start.secrets.evolver_model is None and start.secrets.tinker_api_key
    world.monkeypatch.delenv("TINKER_API_KEY")
    with pytest.raises(ConfigError, match="TINKER_API_KEY"):
        started(world, SKILLS_B, adapting=False, name="again")


def test_the_secrets_can_come_from_an_env_file(world: World, tmp_path: Path) -> None:
    for name in SECRET_VALUES:
        world.monkeypatch.delenv(name)
    env_file = tmp_path / "keys.env"
    env_file.write_text("TINKER_API_KEY=from-the-file-0003\n", encoding="utf-8")
    cfgs = load_configs(world.configs())
    combos = [runner.combo_of(cfgs, "qwen3.8-27b", "file-create", "rl")]
    start = runner.startup(cfgs, combos, adapting=True, env_file=env_file, paths=world.paths)
    assert start.secrets.tinker_api_key == "from-the-file-0003"
    with pytest.raises(ConfigError, match=r"keys2\.env.*cannot read"):
        runner.startup(cfgs, combos, adapting=True, env_file=tmp_path / "keys2.env", paths=world.paths)


def test_the_metaclaw_python_must_exist_for_skills_and_is_not_needed_for_rl(world: World) -> None:
    gone = world.tmp_path / "gone" / "python"
    world.monkeypatch.setenv("EVOLVE_IPI_METACLAW_PYTHON", str(gone))
    paths = RuntimePaths()
    cfgs = load_configs(world.configs())
    skills = [runner.combo_of(cfgs, "qwen3.8-27b", "file-create", "skills")]
    with pytest.raises(ConfigError, match=rf"{gone}.*EVOLVE_IPI_METACLAW_PYTHON"):
        runner.startup(cfgs, skills, adapting=True, paths=paths)
    rl = [runner.combo_of(cfgs, "qwen3.8-27b", "file-create", "rl")]
    assert runner.startup(cfgs, rl, adapting=True, paths=paths).paths == paths


def test_the_data_of_every_task_is_read_at_startup(world: World) -> None:
    def broken(name: str) -> None:
        raise TaskDataError(f"data/tasks/{name}/test.jsonl:3: not valid JSON")

    world.monkeypatch.setattr(runner, "load_task", broken)
    with pytest.raises(TaskDataError, match=r"test\.jsonl:3"):
        started(world, RL_A)


def test_a_skill_order_of_the_wrong_length_is_refused_at_startup(world: World) -> None:
    world.monkeypatch.setattr(runner, "load_skill_order", lambda task: ["a", "b"])
    with pytest.raises(ConfigError, match=r"holds 2 conversations.*skills\.yaml says 12"):
        started(world, SKILLS_B)


def test_the_installed_openclaw_must_be_the_pinned_one(world: World) -> None:
    def wrong() -> str:
        raise OpenClawError("OpenClaw '2026.4.1' is installed, but the experiment runs on OpenClaw 2026.3.7")

    world.monkeypatch.setattr(runner, "check_openclaw", wrong)
    with pytest.raises(OpenClawError, match="2026.3.7"):
        world.run(world.configs(RL_A))
    assert world.work.calls == []


def test_unpatched_frameworks_are_refused(world: World) -> None:
    def unpatched() -> dict[str, str]:
        raise PatchError("OpenClaw-RL/openclaw-tinker/rollout.py has sha256 aaa, SHA256SUMS says bbb")

    world.monkeypatch.setattr(runner, "verify_patches", unpatched)
    with pytest.raises(PatchError, match="rollout.py"):
        world.run(world.configs(RL_A))
    assert world.work.calls == []


@pytest.mark.parametrize(
    ("specs", "uses"),
    [((RL_A, RL_C), ("policy",)), ((SKILLS_B,), ("proxy", "model")), ((RL_A, SKILLS_B), ("policy", "proxy", "model"))],
    ids=["rl", "skills", "both"],
)
def test_the_ports_a_run_uses_are_the_ones_that_are_checked(
    world: World, specs: tuple[dict[str, str], ...], uses: tuple[str, ...]
) -> None:
    directory = world.configs(*specs)
    ports = load_configs(directory).ports
    world.run(directory)
    gateway = list(range(ports.gateway, ports.gateway + 6))
    assert world.work.ports_checked == [sorted(gateway + [getattr(ports, name) for name in uses])]


def test_a_port_in_use_is_refused_with_its_owner(world: World, listener: int) -> None:
    world.monkeypatch.setattr(runner, "require_free", procs.require_free)
    ports = free_ports()
    busy = PortsSpec(gateway=ports.gateway, policy=listener, proxy=ports.proxy, model=ports.model)
    directory = write_configs(world.tmp_path, combos=[RL_A], ports=busy, name="busy")
    with pytest.raises(PortInUse, match=rf"port {listener} is in use by pid {os.getpid()}"):
        world.run(directory)
    assert world.work.calls == []


def test_a_gateway_port_in_use_is_refused_wherever_in_its_span(world: World, listener: int) -> None:
    """The gateway takes its port and the five above it, so one held port among them stops the run."""
    world.monkeypatch.setattr(runner, "require_free", procs.require_free)
    first = listener - 3
    ports = PortsSpec(gateway=first, policy=first - 10, proxy=first - 11, model=first - 12)
    directory = write_configs(world.tmp_path, combos=[RL_A], ports=ports, name="span")
    with pytest.raises(PortInUse, match=rf"port {listener} is in use"):
        world.run(directory)
    assert world.work.calls == []


def test_only_the_ports_a_run_uses_are_checked(world: World, listener: int) -> None:
    """An RL run never uses the proxy or the model port, so a process there is none of its business."""
    world.monkeypatch.setattr(runner, "require_free", procs.require_free)
    ports = free_ports()
    held = PortsSpec(gateway=ports.gateway, policy=ports.policy, proxy=listener, model=ports.model)
    assert len(world.run(write_configs(world.tmp_path, combos=[RL_A], ports=held, name="rl"))) == 1
    with pytest.raises(PortInUse, match=f"port {listener}"):
        world.run(write_configs(world.tmp_path, combos=[SKILLS_B], ports=held, name="skills"))


def test_a_provider_port_inside_the_gateways_ports_is_a_configuration_error(world: World) -> None:
    ports = free_ports()
    clash = PortsSpec(gateway=ports.gateway, policy=ports.gateway + 2, proxy=ports.proxy, model=ports.model)
    directory = write_configs(world.tmp_path, combos=[RL_A], ports=clash, name="clash")
    with pytest.raises(ConfigError, match=rf"ports\.policy .*{ports.gateway + 2}.*gateway"):
        world.run(directory)


@pytest.mark.parametrize(("offset", "refused"), [(5, True), (6, False)])
def test_the_last_of_the_gateways_six_ports_is_the_last_that_is_refused(
    world: World, offset: int, refused: bool
) -> None:
    ports = free_ports()
    spec = PortsSpec(gateway=ports.gateway, policy=ports.gateway + offset, proxy=ports.proxy, model=ports.model)
    directory = write_configs(world.tmp_path, combos=[RL_A], ports=spec, name="edge")
    if refused:
        with pytest.raises(
            ConfigError, match=rf"ports\.policy \({ports.gateway + 5}\) lies among the gateway's 6 ports"
        ):
            world.run(directory)
    else:
        assert len(world.run(directory)) == 1


def test_a_directory_with_two_finished_adaptations_is_refused_for_an_evaluation(world: World) -> None:
    cfgs = load_configs(world.configs())
    model, task = cfgs.models["qwen3.8-27b"], load_task("file-create")
    secrets = Secrets(tinker_api_key="k", evolver_base_url="https://e.invalid", evolver_api_key="k", evolver_model="m")
    run = RunDir(world.tmp_path / "both")
    with world.paths.lock(), run.lock():
        world.work.adapt_rl(model, task, run, cfgs, secrets, paths=world.paths)
        world.work.adapt_skills(model, task, run, cfgs, secrets, paths=world.paths)
    with pytest.raises(ConfigError, match="holds a finished adaptation of both mechanisms; it must hold one"):
        runner.choose_mechanism(cfgs, "qwen3.8-27b", None, run.root)


def test_a_runtime_directory_that_cannot_be_made_is_a_configuration_error(world: World) -> None:
    blocker = world.tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    world.monkeypatch.setenv("EVOLVE_IPI_RUNTIME", str(blocker / "runtime"))
    with pytest.raises(ConfigError, match=rf"runtime directory {blocker}/runtime cannot be made.*EVOLVE_IPI_RUNTIME"):
        world.run(world.configs(RL_A))


@pytest.mark.skipif(os.geteuid() == 0, reason="root may write anywhere")
def test_a_directory_that_may_not_be_written_is_a_configuration_error(world: World) -> None:
    read_only = world.tmp_path / "read-only"
    read_only.mkdir()
    read_only.chmod(0o555)
    try:
        world.monkeypatch.setenv("EVOLVE_IPI_OUTPUTS", str(read_only / "outputs"))
        with pytest.raises(ConfigError, match=r"output directory .*cannot be written.*--outputs"):
            world.run(world.configs(RL_A))
    finally:
        read_only.chmod(0o755)


def test_the_startup_checks_make_no_directory(world: World) -> None:
    started(world, RL_A)
    paths = world.paths
    assert not paths.root.exists() and not paths.outputs.exists()


def test_the_agents_home_must_exist(world: World) -> None:
    world.monkeypatch.setenv("HOME", str(world.tmp_path / "no-such-home"))
    with pytest.raises(ConfigError, match=r"home .*no-such-home"):
        world.run(world.configs(RL_A))


def test_the_workspace_may_not_lie_inside_a_git_checkout(world: World) -> None:
    (world.tmp_path / "checkout" / ".git").mkdir(parents=True)
    world.monkeypatch.setenv("EVOLVE_IPI_RUNTIME", str(world.tmp_path / "checkout" / "runtime"))
    with pytest.raises(ConfigError, match="git checkout"):
        world.run(world.configs(RL_A))


def test_the_typo_of_a_key_is_named_with_its_file(world: World) -> None:
    directory = world.configs(RL_A)
    edit_yaml(directory / "rl.yaml", lambda data: data.__setitem__("bach_size", data.pop("batch_size")))
    with pytest.raises(ConfigError, match=r"rl\.yaml: invalid configuration") as raised:
        world.run(directory)
    assert "  bach_size: unknown key" in str(raised.value) and "  batch_size: missing key" in str(raised.value)
    assert world.work.calls == []


# --- the image ------------------------------------------------------------------------------------------------------
# What `docker run` decides and the image cannot: the host name, the volume under the results, an init process.
# The tests make the process look like the one in the image; none needs a container.


def test_outside_the_image_nothing_about_the_container_is_checked(world: World) -> None:
    world.monkeypatch.setattr(runner, "_hostname", lambda: "3f9a1c0d2b7e")
    world.monkeypatch.setattr(runner, "_is_first_process", lambda: True)
    world.monkeypatch.setattr(runner, "MOUNTINFO", world.tmp_path / "no-such-mountinfo")
    for value in (None, "0", "true", ""):
        if value is None:
            world.monkeypatch.delenv("EVOLVE_IPI_IMAGE", raising=False)
        else:
            world.monkeypatch.setenv("EVOLVE_IPI_IMAGE", value)
        assert len(world.run(world.configs(RL_A, name=f"c{value}"))) == 1


def test_in_the_image_with_the_right_flags_a_run_starts_quietly(world: World, caplog: pytest.LogCaptureFixture) -> None:
    in_image(world.monkeypatch, world.tmp_path, mounted=[world.paths.outputs])
    with caplog.at_level(logging.WARNING, logger="evolve_ipi.runner"):
        assert len(world.run(world.configs(RL_A))) == 1
    assert caplog.records == []


def test_in_the_image_another_host_name_is_refused_before_anything_runs_naming_the_flag(world: World) -> None:
    in_image(world.monkeypatch, world.tmp_path, hostname="3f9a1c0d2b7e", mounted=[world.paths.outputs])
    with pytest.raises(ConfigError) as raised:
        world.run(world.configs(RL_A, SKILLS_B))
    message = str(raised.value)
    assert "'3f9a1c0d2b7e'" in message and "--hostname evolve-ipi" in message and "system prompt" in message
    assert world.work.calls == [] and world.work.checked == [] and world.work.ports_checked == []
    assert not world.paths.outputs.exists() and not world.paths.root.exists()


@pytest.mark.parametrize("adapting", [True, False])
def test_the_host_name_is_checked_for_every_command_that_starts_an_agent(world: World, adapting: bool) -> None:
    in_image(world.monkeypatch, world.tmp_path, hostname="abc", mounted=[world.paths.outputs])
    with pytest.raises(ConfigError, match="--hostname evolve-ipi"):
        started(world, RL_A, adapting=adapting)


def test_the_first_process_of_the_container_is_warned_once_and_not_refused(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    in_image(world.monkeypatch, world.tmp_path, mounted=[world.paths.outputs], first=True)
    with caplog.at_level(logging.WARNING, logger="evolve_ipi.runner"):
        assert len(world.run(world.configs(RL_A, SKILLS_B))) == 2
    (warning,) = caplog.records
    assert warning.levelname == "WARNING" and "run with --init so stopped processes are reaped" in warning.getMessage()


def test_in_the_image_results_that_would_vanish_with_the_container_are_refused(world: World) -> None:
    in_image(world.monkeypatch, world.tmp_path, mounted=[])
    with pytest.raises(ConfigError) as raised:
        world.run(world.configs(RL_A))
    message = str(raised.value)
    assert f"the output directory {world.paths.outputs} is not on a mounted volume" in message
    assert '-v "$PWD/outputs:/work/outputs"' in message and "--rm" in message
    assert world.work.calls == []


@pytest.mark.parametrize(
    ("mounted", "accepted"),
    [
        (lambda out: [out], True),
        (lambda out: [out.parent], True),  # a volume at /work holds /work/outputs
        (lambda out: [out.parent.parent], True),
        (lambda out: [out / "inside"], False),  # below the outputs does not hold them
        (lambda out: [out.parent / "elsewhere"], False),
        (lambda out: [], False),  # "/" alone is the container's own file system
    ],
    ids=["outputs", "parent", "grandparent", "below", "beside", "only-root"],
)
def test_the_outputs_must_be_a_mount_point_or_below_one(world: World, mounted: object, accepted: bool) -> None:
    outputs = world.paths.outputs
    in_image(world.monkeypatch, world.tmp_path, mounted=mounted(outputs))  # type: ignore[operator]
    if accepted:
        started(world, RL_A)
    else:
        with pytest.raises(ConfigError, match="not on a mounted volume"):
            started(world, RL_A)


def test_a_mount_point_with_a_space_in_its_path_is_read_as_the_kernel_escapes_it(world: World) -> None:
    outputs = world.tmp_path / "my results" / "outputs"
    world.monkeypatch.setenv("EVOLVE_IPI_OUTPUTS", str(outputs))
    in_image(world.monkeypatch, world.tmp_path, mounted=[world.tmp_path / "my results"])
    assert "\\040" in runner.MOUNTINFO.read_text()
    started(world, RL_A)


def test_the_directory_the_results_go_to_is_the_one_that_must_be_mounted(world: World) -> None:
    """`eval --adapted DIR` writes into DIR, not into the outputs."""
    elsewhere = world.tmp_path / "volume" / "run"
    in_image(world.monkeypatch, world.tmp_path, mounted=[world.paths.outputs])
    cfgs = load_configs(world.configs())
    combos = [runner.combo_of(cfgs, "qwen3.8-27b", "file-create", "rl")]
    with pytest.raises(ConfigError, match=rf"the output directory {elsewhere} is not on a mounted volume"):
        runner.startup(cfgs, combos, adapting=False, paths=world.paths, results=elsewhere)
    in_image(world.monkeypatch, world.tmp_path, mounted=[world.tmp_path / "volume"])
    runner.startup(cfgs, combos, adapting=False, paths=world.paths, results=elsewhere)


def test_what_is_mounted_must_be_readable_in_the_image(world: World) -> None:
    in_image(world.monkeypatch, world.tmp_path)
    world.monkeypatch.setattr(runner, "MOUNTINFO", world.tmp_path / "gone")
    with pytest.raises(ConfigError, match=r"cannot tell what is mounted: .*gone cannot be read"):
        started(world, RL_A)


# --- the resolved configuration ---------------------------------------------------------------------------------------


def test_the_resolved_configuration_is_json_and_follows_the_mechanism(world: World) -> None:
    cfgs = load_configs(world.configs())
    model = cfgs.models["gpt-oss-120b"]
    config = runner.resolved_config(cfgs, model, "email-invite", "skills", PATCHED, DATA)
    assert json.loads(json.dumps(config)) == config
    assert list(config) == [
        "model", "task", "mechanism", "skills", "eval", "openclaw", "frameworks", "data_sha256sums",
    ]  # fmt: skip
    assert config["model"]["skill_train_retries"] == 4 and config["data_sha256sums"] == DATA
    assert runner.resolved_config(cfgs, model, "email-invite", "rl", PATCHED, DATA)["rl"]["lora_rank"] == 32


def test_the_ports_are_no_part_of_the_resolved_configuration(world: World) -> None:
    first = load_configs(write_configs(world.tmp_path, name="a"))
    second = load_configs(write_configs(world.tmp_path, name="b"))
    assert first.ports != second.ports
    model = first.models["qwen3.8-27b"]
    assert runner.resolved_config(first, model, "file-create", "rl", PATCHED, DATA) == runner.resolved_config(
        second, second.models["qwen3.8-27b"], "file-create", "rl", PATCHED, DATA
    )


def test_the_experiment_file_is_read_with_its_own_checks(world: World) -> None:
    directory = world.configs()
    edit_yaml(directory / "main.yaml", lambda data: data.update(combos=[combo("qwen3.8-27b", "file-create", "lora")]))
    with pytest.raises(ConfigError, match=r"combos\[0\]\.mechanism"):
        world.run(directory)


def test_run_needs_the_experiment_file(world: World) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        runner.run(world.tmp_path / "missing" / "main.yaml", paths=world.paths)


def test_the_helpers_leave_no_secret_in_the_recorded_files(world: World) -> None:
    world.run(world.configs(RL_A, SKILLS_B))
    for path in world.paths.outputs.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert not any(value in text for value in SECRET_VALUES.values() if "secret" in value), path
