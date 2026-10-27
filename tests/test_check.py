"""``evolve-ipi check``: what it verifies, that each check can fail, and that the scripted agent runs through the real
stack in a temporary home.

The checks that read the data, the pins and the services run everywhere. The scripted agent needs OpenClaw 2026.3.7
(marker ``openclaw``) and, on the skill line, the MetaClaw environment (marker ``metaclaw``); both use ports found free
when the test runs, a temporary runtime root and a temporary home.
"""

from __future__ import annotations

import functools
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from cli_helpers import (
    free_ports,
    in_image,
    install_world,
    metaclaw_problem,
    openclaw_problem,
    runtime_env,
    write_configs,
)

from evolve_ipi import check, cli, procs, runner, tasks, thirdparty
from evolve_ipi.agent.openclaw import OPENCLAW_VERSION, OpenClawError
from evolve_ipi.config import Configs, PortsSpec, load_configs
from evolve_ipi.environment import state, verifiers, workspace
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import port_in_use

REPO = Path(__file__).resolve().parents[1]
NO_METACLAW = Path("/nonexistent/venv-metaclaw/bin/python")
CHECK_NAMES = [
    "host name",
    "data checksums",
    "task data",
    "orders",
    "patches",
    "submodules",
    "service programs",
    "workspace files",
    "workspace guards",
    "service reset",
    "judges file-create",
    "judges drive-create",
    "judges email-invite",
    "judges chat-message",
    "judges chat-invite",
    "openclaw",
    "ports",
    "agent rl file-create",
    "agent rl drive-create",
    "agent skills file-create",
    "agent skills drive-create",
]
AGENT_CHECKS = [name for name in CHECK_NAMES if name.startswith("agent ")]


_PLAN = check._plan()


CLI_MAIN = "import sys; from evolve_ipi.cli import main; sys.exit(main())"


def not_installed() -> str:
    raise OpenClawError("openclaw was not found on PATH; install openclaw@2026.3.7 with npm")


@pytest.fixture(scope="module")
def offline() -> list[check.Result]:
    """Every check on the shipped tree, with OpenClaw taken to be missing, so that nothing is started."""
    thirdparty.apply_patches()  # as the tests of the patches do: the checkouts are meant to be patched
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("EVOLVE_IPI_IMAGE", raising=False)  # also when the suite runs in the image
        patch.setattr(check, "check_openclaw", not_installed)
        return check.run_checks(load_configs(REPO / "configs"), metaclaw_python=NO_METACLAW)


@pytest.fixture(autouse=True)
def _outside_the_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite also runs in the image, where the variable is set; a test that needs it sets it itself."""
    monkeypatch.delenv("EVOLVE_IPI_IMAGE", raising=False)


@pytest.fixture
def ctx(tmp_path: Path) -> check.Context:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cfgs = load_configs(write_configs(tmp_path))
    return check.Context(cfgs, NO_METACLAW, tasks.DATA_DIR, thirdparty.THIRD_PARTY, scratch)


def run_one(ctx: check.Context, name: str) -> check.Result:
    """One check of the plan, with the mapping from what it raises to a result that ``run_checks`` makes."""
    return check._run_one(name, functools.partial(dict(check._plan())[name], ctx))


# --- the lines --------------------------------------------------------------------------------------------------------


def test_a_result_is_one_line_with_a_status_a_name_and_a_detail() -> None:
    line = check.format_result(check.Result("patches", "pass", "3 patched framework files match"))
    assert line.startswith("PASS  patches")
    assert line.endswith("  3 patched framework files match") and "\n" not in line
    assert line.index("3 patched") == 4 + 2 + check.NAME_WIDTH + 2, "the details line up"
    assert check.format_result(check.Result("ports", "fail", "busy")).startswith("FAIL  ports")
    assert check.format_result(check.Result("agent", "skip", "no")).startswith("SKIP  agent")


def test_a_long_detail_keeps_its_further_lines_under_the_first() -> None:
    lines = check.format_result(check.Result("workspace", "fail", "not clean:\n  - a\n  - b")).splitlines()
    assert len(lines) == 3 and lines[0].endswith("not clean:")
    assert lines[1].startswith(" " * (4 + 2 + check.NAME_WIDTH + 2 + 2) + "- a")


def test_the_summary_counts_each_status() -> None:
    results = [check.Result("a", "pass", ""), check.Result("b", "pass", ""), check.Result("c", "skip", "")]
    assert check.summary_line(results) == "checks: 2 passed, 0 failed, 1 skipped"
    assert check.summary_line([]) == "checks: 0 passed, 0 failed, 0 skipped"


def test_every_name_fits_the_column() -> None:
    assert all(len(name) <= check.NAME_WIDTH for name in CHECK_NAMES)
    assert [name for name, _ in check._plan()] == CHECK_NAMES


# --- the shipped tree -------------------------------------------------------------------------------------------------


def test_every_offline_check_passes_on_the_shipped_tree(offline: list[check.Result]) -> None:
    assert [result.name for result in offline] == CHECK_NAMES
    by_name = {result.name: result for result in offline}
    needs_openclaw = {"openclaw", "ports", "host name", *AGENT_CHECKS}
    for name in CHECK_NAMES:
        if name in needs_openclaw:
            continue
        expected = "pass" if name != "submodules" or (REPO / ".git").exists() else "skip"
        assert by_name[name].status == expected, f"{name}: {by_name[name].detail}"


def test_what_needs_openclaw_is_skipped_with_the_reason(offline: list[check.Result]) -> None:
    skipped = [result for result in offline if result.name in {"openclaw", "ports", *AGENT_CHECKS}]
    assert len(skipped) == 6
    for result in skipped:
        assert result.status == "skip", result
        assert "OpenClaw 2026.3.7 is not available here: openclaw was not found on PATH" in result.detail, result


def test_the_details_say_what_was_verified(offline: list[check.Result]) -> None:
    by_name = {result.name: result.detail for result in offline}
    assert by_name["host name"].startswith("not in the image")
    assert by_name["data checksums"] == "25 files under data/ match data/SHA256SUMS"
    assert by_name["task data"] == "5 tasks, each with 48 adaptation items and 40 test items"
    assert by_name["orders"] == (
        "RL: stage 1 on 12 items, stage 2 on 36, in each of the 5 tasks, disjoint; skills: 12 different conversations"
    )
    assert by_name["patches"].startswith("3 patched framework files")
    assert by_name["service programs"].startswith("7 service programs")
    assert "88 documents" in by_name["workspace files"]
    assert by_name["service reset"].startswith("352 items")
    assert by_name["judges file-create"].startswith("88 items") and by_name["judges chat-invite"].startswith("88 items")


def test_outside_the_image_the_host_name_is_not_checked(ctx: check.Context) -> None:
    result = run_one(ctx, "host name")
    assert result.status == "skip" and "not in the image (EVOLVE_IPI_IMAGE is not set)" in result.detail


def test_in_the_image_the_host_name_must_be_the_one_the_image_pins(
    ctx: check.Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_image(monkeypatch, tmp_path, hostname="evolve-ipi")
    passed = run_one(ctx, "host name")
    assert passed.status == "pass" and passed.detail.startswith("evolve-ipi, the host name OpenClaw writes")
    in_image(monkeypatch, tmp_path, hostname="3f9a1c0d2b7e")
    failed = run_one(ctx, "host name")
    assert failed.status == "fail" and "'3f9a1c0d2b7e'" in failed.detail
    assert "--hostname evolve-ipi" in failed.detail and "ConfigError" in failed.detail


# --- each check can fail ----------------------------------------------------------------------------------------------


def copy_data(tmp_path: Path) -> Path:
    destination = tmp_path / "data"
    shutil.copytree(tasks.DATA_DIR, destination)
    return destination


def test_a_task_with_the_wrong_number_of_items_fails(ctx: check.Context, tmp_path: Path) -> None:
    ctx.data_dir = copy_data(tmp_path)
    path = ctx.data_dir / "tasks" / "chat-message" / "test.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]), encoding="utf-8")
    result = run_one(ctx, "task data")
    assert (
        result.status == "fail" and "chat-message has 48 adaptation and 39 test items, not 48 and 40" in result.detail
    )


def test_a_data_file_that_cannot_be_read_fails_once_and_the_checks_that_need_it_are_skipped(
    ctx: check.Context, tmp_path: Path
) -> None:
    ctx.data_dir = copy_data(tmp_path)
    (ctx.data_dir / "tasks" / "drive-create" / "adaptation.jsonl").write_text("{broken\n", encoding="utf-8")
    first = run_one(ctx, "task data")
    assert first.status == "fail" and "TaskDataError" in first.detail and "adaptation.jsonl:1" in first.detail
    for name in ("orders", "workspace files", "service reset", "judges file-create", "judges drive-create"):
        later = run_one(ctx, name)
        assert (later.status, later.detail) == ("skip", "the task data failed its check above"), name


def test_an_order_that_names_no_item_fails(ctx: check.Context, tmp_path: Path) -> None:
    ctx.data_dir = copy_data(tmp_path)
    path = ctx.data_dir / "orders" / "skills" / "email-invite.json"
    path.write_text(path.read_text().replace("email-invite-adapt-", "email-invite-adapt-9", 1), encoding="utf-8")
    result = run_one(ctx, "orders")
    assert result.status == "fail" and "not an adaptation item of email-invite" in result.detail


def edit_order(ctx: check.Context, tmp_path: Path, task: str, change: Callable[[dict[str, list[str]]], None]) -> None:
    ctx.data_dir = copy_data(tmp_path)
    path = ctx.data_dir / "orders" / "rl" / f"{task}.json"
    orders = json.loads(path.read_text())
    change(orders)
    path.write_text(json.dumps(orders), encoding="utf-8")


def test_an_rl_order_that_is_not_passes_over_its_items_fails(ctx: check.Context, tmp_path: Path) -> None:
    edit_order(ctx, tmp_path, "file-create", lambda orders: orders["stage2"].__setitem__(0, orders["stage1"][0]))
    result = run_one(ctx, "orders")
    assert result.status == "fail" and "TaskDataError: positions 0.. of a File-Create RL order" in result.detail


def test_two_rl_stages_that_share_an_item_fail(ctx: check.Context, tmp_path: Path) -> None:
    def share(orders: dict[str, list[str]]) -> None:
        shared, replaced = orders["stage1"][0], orders["stage2"][0]
        orders["stage2"] = [shared if item == replaced else item for item in orders["stage2"]]

    edit_order(ctx, tmp_path, "drive-create", share)
    result = run_one(ctx, "orders")
    assert result.status == "fail" and "drive-create: the two RL stages share 1 items" in result.detail


def test_stage_sizes_that_differ_between_tasks_are_said_per_task(ctx: check.Context, tmp_path: Path) -> None:
    """The line describes every task, not the last one read."""

    def shorter_stage_1(orders: dict[str, list[str]]) -> None:  # File-Create: passes over 11 items instead of 12
        items = list(dict.fromkeys(orders["stage1"]))[:11]
        orders["stage1"] = items * (len(orders["stage1"]) // 12)

    edit_order(ctx, tmp_path, "file-create", shorter_stage_1)
    result = run_one(ctx, "orders")
    assert result.status == "pass", result.detail
    assert result.detail.startswith("RL: file-create 11+36 items, drive-create 12+36 items, email-invite 12+36 items")
    assert "chat-invite 12+36 items, disjoint" in result.detail


def test_a_changed_data_file_fails_the_checksums(ctx: check.Context, tmp_path: Path) -> None:
    ctx.data_dir = copy_data(tmp_path)
    path = ctx.data_dir / "tasks" / "email-invite" / "test.jsonl"
    path.write_bytes(path.read_bytes().replace(b"a", b"b", 1))
    result = run_one(ctx, "data checksums")
    assert result.status == "fail" and "ConfigError" in result.detail
    assert "tasks/email-invite/test.jsonl has changed" in result.detail


def mini_third_party(tmp_path: Path) -> Path:
    """The checks of the patches read three files and ``SHA256SUMS``: a copy of those, patched as the real ones."""
    thirdparty.apply_patches()
    root = tmp_path / "repo" / "third_party"
    shutil.copytree(thirdparty.THIRD_PARTY / "patches", root / "patches")
    for patch in thirdparty.PATCHES:
        (root / patch.path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(thirdparty.THIRD_PARTY / patch.path, root / patch.path)
    return root


def test_a_patched_file_that_changed_fails_and_is_named(ctx: check.Context, tmp_path: Path) -> None:
    ctx.third_party = mini_third_party(tmp_path)
    assert run_one(ctx, "patches").status == "pass"
    with (ctx.third_party / "MetaClaw" / "metaclaw" / "api_server.py").open("ab") as handle:
        handle.write(b"\n# edited\n")
    result = run_one(ctx, "patches")
    assert result.status == "fail" and "MetaClaw/metaclaw/api_server.py" in result.detail
    assert "PatchError" in result.detail and "skill_evolver.py" not in result.detail


def test_without_git_metadata_the_commits_are_skipped_and_the_patches_are_still_checked(
    ctx: check.Context, tmp_path: Path
) -> None:
    ctx.third_party = mini_third_party(tmp_path)  # tmp/repo has no .git, as a copy of the tree in a container
    skipped = run_one(ctx, "submodules")
    assert skipped.status == "skip" and "no .git" in skipped.detail
    assert run_one(ctx, "patches").status == "pass"


def test_a_checkout_that_is_not_a_checkout_fails(ctx: check.Context, tmp_path: Path) -> None:
    ctx.third_party = mini_third_party(tmp_path)
    (tmp_path / "repo" / ".git").mkdir()  # the tree claims to be a clone, but its submodules are not checkouts
    result = run_one(ctx, "submodules")
    assert result.status == "fail" and "is not a git checkout" in result.detail


def test_a_submodule_at_another_commit_fails(ctx: check.Context, tmp_path: Path) -> None:
    root = tmp_path / "repo" / "third_party"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q", str(tmp_path / "repo")], check=True)
    for name in thirdparty.PINNED_COMMITS:
        (root / name).mkdir(parents=True)
        subprocess.run([*git, "-C", str(root / name), "init", "-q"], check=True)
        subprocess.run([*git, "-C", str(root / name), "commit", "--allow-empty", "-q", "-m", "x"], check=True)
    ctx.third_party = root
    result = run_one(ctx, "submodules")
    assert result.status == "fail"
    assert "OpenClaw-RL is at" in result.detail and "not f48ac35" in result.detail and "not 922caf3" in result.detail


def test_a_service_program_that_changed_by_one_byte_fails_and_the_judges_wait(
    ctx: check.Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = tmp_path / "services"
    shutil.copytree(state.SERVICES_DIR, services)
    with (services / "mail").open("ab") as handle:
        handle.write(b" ")
    monkeypatch.setattr(state, "SERVICES_DIR", services)
    result = run_one(ctx, "service programs")
    assert result.status == "fail" and "mail is not the pinned file" in result.detail
    assert str(services) in result.detail and "StateError" in result.detail
    waiting = run_one(ctx, "judges chat-message")
    assert (waiting.status, waiting.detail) == ("skip", "the service programs failed their check above")


def test_a_missing_service_program_fails(ctx: check.Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    services = tmp_path / "services"
    shutil.copytree(state.SERVICES_DIR, services)
    (services / "_lib.py").unlink()
    monkeypatch.setattr(state, "SERVICES_DIR", services)
    result = run_one(ctx, "service programs")
    assert result.status == "fail" and "_lib.py is missing" in result.detail


def test_a_workspace_guard_that_accepts_everything_fails(ctx: check.Context, monkeypatch: pytest.MonkeyPatch) -> None:
    assert run_one(ctx, "workspace guards").status == "pass"
    monkeypatch.setattr(workspace, "guard", lambda ws: None)
    result = run_one(ctx, "workspace guards")
    assert result.status == "fail" and "accepted an edited prompt file, agent-written memory" in result.detail
    assert "a program without its execute bit" in result.detail


def test_a_guard_that_misses_one_kind_of_change_names_it(ctx: check.Context, monkeypatch: pytest.MonkeyPatch) -> None:
    real = workspace.guard

    def blind_to_skills(ws: Path) -> None:
        if (ws / "skills").exists():
            return
        real(ws)

    monkeypatch.setattr(workspace, "guard", blind_to_skills)
    result = run_one(ctx, "workspace guards")
    assert result.status == "fail" and result.detail == "the workspace guard accepted an agent-written skill"


def test_a_reset_that_does_not_restore_the_world_fails(ctx: check.Context, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = itertools.count()
    real = state.initial_state

    def with_noise(episode: dict) -> dict:
        return {**real(episode), "noise": next(counter)}

    monkeypatch.setattr(state, "initial_state", with_noise)
    result = run_one(ctx, "service reset")
    assert result.status == "fail" and "does not give the same world every time" in result.detail


def test_a_judge_that_never_fires_fails_and_names_the_item(ctx: check.Context, monkeypatch: pytest.MonkeyPatch) -> None:
    miss = {"hit": False, "params_exact": None, "matches": [], "new_records": 0}
    monkeypatch.setattr(verifiers, "judge_asr", lambda episode, state_dir: dict(miss))
    result = run_one(ctx, "judges drive-create")
    assert result.status == "fail" and "SelfCheckError" in result.detail
    assert "drive-create-adapt-000" in result.detail and "judge_asr is False" in result.detail


def test_a_file_judge_that_counts_every_target_fails(ctx: check.Context, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verifiers, "file_attack_success", lambda target: True)
    result = run_one(ctx, "judges file-create")
    assert result.status == "fail" and "absent target counted as a hit" in result.detail


def test_a_bug_in_a_check_is_a_failed_line_with_a_traceback_and_the_others_still_run(
    ctx: check.Context, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(ctx: check.Context) -> str:
        raise RuntimeError("a bug")

    monkeypatch.setattr(check, "_programs", broken)
    result = run_one(ctx, "service programs")
    assert result.status == "fail" and result.detail == "RuntimeError: a bug (traceback on stderr)"
    assert "Traceback (most recent call last)" in capsys.readouterr().err
    assert run_one(ctx, "patches").status == "pass"


def test_an_interrupt_is_not_a_result(ctx: check.Context, monkeypatch: pytest.MonkeyPatch) -> None:
    def stopped(ctx: check.Context) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(check, "_patches", stopped)
    with pytest.raises(KeyboardInterrupt):
        run_one(ctx, "patches")


# --- the scripted agent: what is skipped, and why ---------------------------------------------------------------------


def test_without_openclaw_every_scripted_check_is_skipped_with_the_reason(
    ctx: check.Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "check_openclaw", not_installed)
    results = [run_one(ctx, name) for name in ("openclaw", "ports", *AGENT_CHECKS)]
    assert [r.status for r in results] == ["skip"] * 6
    assert "OpenClaw 2026.3.7 is not available here: openclaw was not found on PATH" in results[0].detail
    assert all(OPENCLAW_VERSION in r.detail for r in results)


def test_openclaw_of_another_version_is_skipped_too(ctx: check.Context, monkeypatch: pytest.MonkeyPatch) -> None:
    def other() -> str:
        raise OpenClawError("OpenClaw '2026.4.1' is installed, but the experiment runs on OpenClaw 2026.3.7")

    monkeypatch.setattr(check, "check_openclaw", other)
    result = run_one(ctx, "openclaw")
    assert result.status == "skip" and "'2026.4.1' is installed" in result.detail


def test_the_ports_are_checked_and_a_busy_one_is_refused_with_its_owner(
    ctx: check.Context, listener: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "check_openclaw", lambda: OPENCLAW_VERSION)
    assert run_one(ctx, "openclaw").status == "pass"
    free = ctx.cfgs.ports
    ctx.cfgs = Configs(
        ctx.cfgs.models, ctx.cfgs.rl, ctx.cfgs.skills, ctx.cfgs.eval,
        PortsSpec(gateway=free.gateway, policy=listener, proxy=free.proxy, model=free.model),
    )  # fmt: skip
    busy = run_one(ctx, "ports")
    assert busy.status == "fail" and f"port {listener} is in use by pid {os.getpid()}" in busy.detail
    assert "nothing was stopped" in busy.detail.lower() or "Nothing was started" in busy.detail
    for name in AGENT_CHECKS:
        waiting = run_one(ctx, name)
        assert waiting.status == "skip" and "ports are in use" in waiting.detail, name


def test_the_ports_of_the_skill_line_are_only_checked_where_it_can_run(
    ctx: check.Context, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(check, "check_openclaw", lambda: OPENCLAW_VERSION)
    asked: list[list[int]] = []
    monkeypatch.setattr(check, "require_free", lambda ports: asked.append(sorted(ports)))
    run_one(ctx, "openclaw")
    ports = ctx.cfgs.ports
    gateway = list(range(ports.gateway, ports.gateway + 6))
    assert run_one(ctx, "ports").detail.startswith(f"free: gateway {ports.gateway}-{ports.gateway + 5}, policy")
    ctx.metaclaw_python = Path(sys.executable)
    assert "proxy" in run_one(ctx, "ports").detail
    assert asked == [sorted([*gateway, ports.policy]), sorted([*gateway, ports.policy, ports.proxy, ports.model])]


def test_without_the_metaclaw_environment_the_skill_line_is_skipped_before_anything_starts(
    ctx: check.Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(procs.ManagedProcess, "start", lambda self: pytest.fail("a process was started"))
    ctx.openclaw_problem = None
    for name in ("agent skills file-create", "agent skills drive-create"):
        result = run_one(ctx, name)
        assert result.status == "skip" and f"MetaClaw's Python {NO_METACLAW} does not exist" in result.detail
        assert "EVOLVE_IPI_METACLAW_PYTHON" in result.detail


# --- the command ------------------------------------------------------------------------------------------------------


@pytest.fixture
def command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Iterator[Path]:
    """A configuration with free ports, in an environment where nothing is installed (so nothing is started)."""
    install_world(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "check_openclaw", not_installed)
    monkeypatch.setattr(check, "check_openclaw", not_installed)
    monkeypatch.setenv("EVOLVE_IPI_METACLAW_PYTHON", str(NO_METACLAW))
    yield write_configs(tmp_path, name="check-configs")


def test_check_exits_with_its_own_status_when_a_service_program_changed(
    command: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    services = tmp_path / "services"
    shutil.copytree(state.SERVICES_DIR, services)
    with (services / "drive").open("ab") as handle:
        handle.write(b"\n")
    monkeypatch.setattr(state, "SERVICES_DIR", services)
    code = cli.main(["check", "--configs", str(command)])
    out = capsys.readouterr().out.splitlines()
    assert code == cli.EXIT_CHECK_FAILED
    failed = [line for line in out if line.startswith("FAIL")]
    assert any(line.startswith("FAIL  service programs") and "drive is not the pinned file" in line for line in failed)
    assert out[-1].startswith("checks: ") and " failed, " in out[-1] and "0 failed" not in out[-1]
    assert any(line.startswith("PASS  task data") for line in out), "the other checks still ran"


def test_check_prints_a_line_per_check_and_exits_with_zero_when_none_failed(
    command: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    thirdparty.apply_patches()
    names = ("task data", "orders", "patches", "service programs", "openclaw", "ports", "agent rl file-create")
    monkeypatch.setattr(check, "_plan", lambda: [(name, fn) for name, fn in _PLAN if name in names])
    code = cli.main(["check", "--configs", str(command)])
    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert [line.split("  ")[0] for line in out[:-1]] == ["PASS"] * 4 + ["SKIP"] * 3
    assert (
        out[0]
        == "PASS  task data"
        + " " * (check.NAME_WIDTH - len("task data") + 2)
        + "5 tasks, each with 48 adaptation items and 40 test items"
    )
    assert out[-1] == "checks: 4 passed, 0 failed, 3 skipped"


def test_check_needs_no_key_and_makes_no_paid_call(
    command: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("TINKER_API_KEY", "EVOLVER_BASE_URL", "EVOLVER_API_KEY", "EVOLVER_MODEL", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(check, "run_checks", lambda cfgs, *, report, **options: [])
    assert cli.main(["check", "--configs", str(command)]) == 0
    assert capsys.readouterr().out == "checks: 0 passed, 0 failed, 0 skipped\n"


def test_check_reports_a_configuration_mistake_before_any_check(
    command: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (command / "rl.yaml").write_text("bach_size: 4\n", encoding="utf-8")
    code = cli.main(["check", "--configs", str(command)])
    captured = capsys.readouterr()
    assert code == cli.EXIT_CONFIG and captured.out == ""
    assert "rl.yaml" in captured.err and "bach_size: unknown key" in captured.err


# --- the properties of a scripted episode: each one alone makes the check fail ---------------------------------------


def episode(kind: str, *, hit: bool, reply: str = "I did it.") -> dict:
    """The record of an episode as `run_episode` makes it, for a hit or a miss of the scripted agent."""
    record: dict = {
        "status": "ok",
        "error": None,
        "anomalies": [] if hit else [{"type": "no_calls", "detail": "behaviour, not a fault of the pipeline"}],
        "reply": reply,
        "encountered": hit,
        "session_id": "session-1",
    }
    if kind == "file":
        record["attack_success"] = hit
    else:
        calls = [
            {"program": "drive", "argv": ["read", "x"], "rc": 0},
            {"program": "drive", "argv": ["create"], "rc": 0},
        ]
        record.update(asr={"hit": hit}, access_log=calls if hit else [])
    return record


def problems(
    kind: str, record: dict, *, hit: bool, line: str = "rl", requests: list | None = None, proxy_log: str = ""
) -> list[str]:
    task = tasks.load_task("file-create" if kind == "file" else "drive-create")
    headers = [{"turn_type": "main", "session_id": "session-1"}] if requests is None else requests
    return check._problems_of(
        task, record, hit=hit, reply="I did it.", requests=headers, line=line,  # type: ignore[arg-type]
        proxy_log=proxy_log or "[main] session=session-1",
    )  # fmt: skip


@pytest.mark.parametrize("kind", ["file", "service"])
@pytest.mark.parametrize("line", ["rl", "skills"])
@pytest.mark.parametrize("hit", [True, False])
def test_a_scripted_episode_that_did_what_the_script_says_has_no_problem(kind: str, line: str, hit: bool) -> None:
    assert problems(kind, episode(kind, hit=hit), hit=hit, line=line) == []


@pytest.mark.parametrize(
    ("what", "change", "expected"),
    [
        ("the turn failed", lambda r: r.update(status="agent_error", error="exit 1"), "the turn ended agent_error: exit 1"),
        ("a pipeline anomaly", lambda r: r["anomalies"].append({"type": "timeout", "detail": "d"}), "pipeline anomalies: timeout"),
        ("the agent did not read it", lambda r: r.update(encountered=False), "encountered is False"),
        ("the reply is not the model's", lambda r: r.update(reply="something else"), "the reply is 'something else', not the model's 'I did it.'"),
    ],
)  # fmt: skip
@pytest.mark.parametrize("kind", ["file", "service"])
def test_each_property_of_a_hit_fails_the_check_when_it_alone_is_broken(
    kind: str, what: str, change: Callable[[dict], None], expected: str
) -> None:
    record = episode(kind, hit=True)
    change(record)
    assert problems(kind, record, hit=True) == [expected], what


@pytest.mark.parametrize("kind", ["file", "service"])
def test_a_hit_that_the_judge_calls_a_miss_and_a_miss_that_it_calls_a_hit_fail(kind: str) -> None:
    lost = episode(kind, hit=True)
    lost["attack_success" if kind == "file" else "asr"] = False if kind == "file" else {"hit": False}
    assert problems(kind, lost, hit=True) == ["the judge said miss"]
    found = episode(kind, hit=False)
    found["attack_success" if kind == "file" else "asr"] = True if kind == "file" else {"hit": True}
    assert problems(kind, found, hit=False) == ["the judge said hit"]


@pytest.mark.parametrize("calls", [[], [("drive", 0)], [("drive", 0), ("drive", 1)], [("mail", 0), ("drive", 0)]])
def test_the_programs_a_service_hit_calls_must_be_the_two_of_the_script_and_both_must_succeed(
    calls: list[tuple[str, int]],
) -> None:
    record = episode("service", hit=True)
    record["access_log"] = [{"program": program, "argv": [], "rc": rc} for program, rc in calls]
    assert problems("service", record, hit=True) == [f"the programs were called as {calls}"]


@pytest.mark.parametrize(
    "requests",
    [[], [{"turn_type": "side", "session_id": "session-1"}], [{"turn_type": "main", "session_id": "another"}]],
    ids=["none", "not a main turn", "another session"],
)
def test_on_the_rl_line_every_request_must_carry_the_plugins_session_headers(requests: list[dict]) -> None:
    record = episode("file", hit=True)
    assert problems("file", record, hit=True, requests=requests) == [
        "the model server did not get the session headers of the plugin"
    ]


def test_on_the_skills_line_the_proxy_must_have_seen_the_session() -> None:
    record = episode("file", hit=True)
    assert problems("file", record, hit=True, line="skills", proxy_log="[main] session=another") == [
        "the MetaClaw proxy did not see the session"
    ]


def test_the_agents_target_directory_must_be_left_empty(tmp_path: Path) -> None:
    assert check._leftovers(None) == [] and check._leftovers(tmp_path) == []
    (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
    assert check._leftovers(tmp_path) == [f"{tmp_path} was not left empty"]


# --- with OpenClaw and MetaClaw ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openclaw_ready() -> None:
    problem = openclaw_problem()
    if problem:
        pytest.skip(problem)


@pytest.fixture(scope="module")
def metaclaw_python() -> Path:
    python = RuntimePaths().metaclaw_python
    problem = metaclaw_problem(python)
    if problem:
        pytest.skip(problem)
    return python


class RealStack:
    """A configuration with free ports, a temporary home and runtime for the process, and the real python of MetaClaw
    where there is one. Nothing of a real run is touched: the test checks that its own home stays empty."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metaclaw: Path | None) -> None:
        self.tmp_path = tmp_path
        self.paths = runtime_env(monkeypatch, tmp_path, metaclaw)
        self.ports = free_ports()
        self.configs = write_configs(tmp_path, ports=self.ports, name="real-configs")
        self.cfgs = load_configs(self.configs)
        for port in (
            *range(self.ports.gateway, self.ports.gateway + 6),
            self.ports.policy,
            self.ports.proxy,
            self.ports.model,
        ):
            assert port not in {18789, 30011, 30012, 30013}

    def context(self, scratch: Path, metaclaw: Path) -> check.Context:
        scratch.mkdir()
        return check.Context(self.cfgs, metaclaw, tasks.DATA_DIR, thirdparty.THIRD_PARTY, scratch)

    def assert_nothing_left(self) -> None:
        assert (
            not (self.paths.agent_home / ".openclaw-rl").exists() and not (self.paths.agent_home / ".metaclaw").exists()
        )
        assert not self.paths.root.exists(), "the check used a runtime directory of its own, not this one"
        busy = [
            p
            for p in (
                *range(self.ports.gateway, self.ports.gateway + 6),
                self.ports.policy,
                self.ports.proxy,
                self.ports.model,
            )
            if port_in_use(p)
        ]
        assert busy == [], f"ports still in use after the check: {busy}"


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> RealStack:
    metaclaw = request.getfixturevalue("metaclaw_python") if "metaclaw" in request.keywords else None
    return RealStack(tmp_path, monkeypatch, metaclaw)


@pytest.mark.openclaw
@pytest.mark.usefixtures("openclaw_ready")
@pytest.mark.parametrize("name", ["agent rl file-create", "agent rl drive-create"])
def test_the_scripted_agent_on_the_rl_line_is_a_hit_when_it_acts_and_a_miss_when_it_does_not(
    stack: RealStack, name: str
) -> None:
    ctx = stack.context(stack.tmp_path / "scratch", NO_METACLAW)
    assert run_one(ctx, "openclaw").status == "pass" and run_one(ctx, "ports").status == "pass"
    result = run_one(ctx, name)
    assert result.status == "pass", result.detail
    assert result.detail == "OpenClaw: the request in the document was a hit when carried out, a miss when not"
    stack.assert_nothing_left()


@pytest.mark.openclaw
@pytest.mark.usefixtures("openclaw_ready")
def test_a_pipeline_that_loses_the_hit_is_a_failed_check(stack: RealStack, monkeypatch: pytest.MonkeyPatch) -> None:
    """The scripted agent creates its file next to the target instead of in it: the judge says miss where the check
    expects a hit."""
    real = check._hit_script

    def wrong_place(task: tasks.Task, item: dict, line: str) -> list[dict]:
        script = real(task, item, line)
        script[1]["tool_calls"][0]["arguments"]["command"] = f"touch ~/elsewhere-{item['tgt_name']}"
        return script

    monkeypatch.setattr(check, "_hit_script", wrong_place)
    result = run_one(stack.context(stack.tmp_path / "scratch", NO_METACLAW), "agent rl file-create")
    assert result.status == "fail" and "hit episode: the judge said miss" in result.detail
    stack.assert_nothing_left()


@pytest.mark.openclaw
@pytest.mark.usefixtures("openclaw_ready")
def test_a_text_only_agent_is_a_miss_where_the_check_expects_a_hit(
    stack: RealStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "_hit_script", lambda task, item, line: [{"content": check._HIT_REPLY}])
    result = run_one(stack.context(stack.tmp_path / "scratch", NO_METACLAW), "agent rl drive-create")
    assert result.status == "fail" and "hit episode: the judge said miss" in result.detail
    stack.assert_nothing_left()


@pytest.mark.openclaw
@pytest.mark.usefixtures("openclaw_ready")
def test_an_agent_that_leaves_a_file_in_its_target_directory_is_a_failed_check(
    stack: RealStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness removes the one file an episode is judged on; anything else the agent made stays and would stop the
    next run, so the check must not pass it."""
    real = check._hit_script

    def with_a_stray_file(task: tasks.Task, item: dict, line: str) -> list[dict]:
        script = real(task, item, line)
        command = script[1]["tool_calls"][0]["arguments"]["command"]
        script[1]["tool_calls"][0]["arguments"]["command"] = f"{command} && touch ~/.openclaw-rl/stray.txt"
        return script

    monkeypatch.setattr(check, "_hit_script", with_a_stray_file)
    result = run_one(stack.context(stack.tmp_path / "scratch", NO_METACLAW), "agent rl file-create")
    assert result.status == "fail" and "was not left empty" in result.detail
    assert "hit episode" not in result.detail, "the episodes themselves were fine"


@pytest.mark.openclaw
@pytest.mark.metaclaw
@pytest.mark.usefixtures("openclaw_ready")
@pytest.mark.parametrize("name", ["agent skills file-create", "agent skills drive-create"])
def test_the_scripted_agent_on_the_skill_line_goes_through_the_metaclaw_proxy(
    stack: RealStack, metaclaw_python: Path, name: str
) -> None:
    ctx = stack.context(stack.tmp_path / "scratch", metaclaw_python)
    assert run_one(ctx, "openclaw").status == "pass" and run_one(ctx, "ports").status == "pass"
    result = run_one(ctx, name)
    assert result.status == "pass", result.detail
    assert result.detail.startswith("OpenClaw + MetaClaw proxy: ")
    stack.assert_nothing_left()


@pytest.mark.openclaw
@pytest.mark.usefixtures("openclaw_ready")
def test_check_refuses_a_busy_port_and_starts_nothing(
    stack: RealStack, listener: int, capsys: pytest.CaptureFixture[str]
) -> None:
    held = PortsSpec(gateway=stack.ports.gateway, policy=listener, proxy=stack.ports.proxy, model=stack.ports.model)
    configs = write_configs(stack.tmp_path, ports=held, name="busy-configs")
    code = cli.main(["check", "--configs", str(configs)])
    out = capsys.readouterr().out.splitlines()
    assert code == cli.EXIT_CHECK_FAILED
    ports = next(line for line in out if line.startswith("FAIL  ports"))
    assert f"port {listener} is in use by pid {os.getpid()}" in ports
    assert all(line.startswith("SKIP") for line in out if line.startswith(("SKIP", "FAIL  agent")))
    assert sum(line.startswith("SKIP  agent") for line in out) == 4
    stack.assert_nothing_left()


@pytest.mark.openclaw
@pytest.mark.usefixtures("openclaw_ready")
def test_sigterm_stops_the_scripted_agent_and_everything_it_started(stack: RealStack, tmp_path: Path) -> None:
    """What `docker stop` does: the gateway and the scripted model are stopped, the temporary directory is removed,
    and the command says so and exits with 130."""
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    child = subprocess.Popen(
        [sys.executable, "-c", CLI_MAIN, "check", "--configs", str(stack.configs)],
        env={**os.environ, "TMPDIR": str(temp_root)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 300
        while not port_in_use(stack.ports.gateway):  # the first scripted agent has started its gateway
            assert child.poll() is None, "check ended before it started a gateway"
            assert time.monotonic() < deadline, "check did not start a gateway in time"
            time.sleep(0.3)
        child.send_signal(signal.SIGTERM)
        out, err = child.communicate(timeout=120)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    assert child.returncode == cli.EXIT_INTERRUPTED, err
    assert "evolve-ipi: interrupted by SIGTERM" in err and "Traceback" not in err
    assert "PASS  task data" in out and "agent rl file-create" not in out
    assert list(temp_root.iterdir()) == [], "the temporary home and runtime are removed"
    stack.assert_nothing_left()


@pytest.mark.openclaw
@pytest.mark.metaclaw
@pytest.mark.usefixtures("openclaw_ready")
def test_check_passes_end_to_end_and_touches_nothing_of_the_real_home(
    stack: RealStack, metaclaw_python: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    before = {path.name for path in stack.paths.agent_home.iterdir()}
    code = cli.main(["check", "--configs", str(stack.configs)])
    captured = capsys.readouterr()
    out = captured.out.splitlines()
    assert code == 0, captured.out + captured.err
    assert [line.split("  ")[1].strip() for line in out[:-1]] == CHECK_NAMES
    # outside the image the host name is not checked; without git metadata the submodule commits cannot be read
    skipped = 1 + (0 if (REPO / ".git").exists() else 1)
    assert out[-1] == f"checks: {len(CHECK_NAMES) - skipped} passed, 0 failed, {skipped} skipped"
    # Nothing new in the home but .cache, which Docker Desktop under Rosetta adds to the home of a container by itself.
    after = {path.name for path in stack.paths.agent_home.iterdir()}
    assert after - before <= {".cache"}, sorted(after - before)
    stack.assert_nothing_left()
