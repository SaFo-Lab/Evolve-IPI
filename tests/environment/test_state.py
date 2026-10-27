"""The world the services start from: its base state, the carrier of each item, install and reset."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from evolve_ipi.environment import state, verifiers
from evolve_ipi.tasks import Task, service_episode

if TYPE_CHECKING:
    from conftest import SvcEnv

# sha256 of json.dumps(base_state(), ensure_ascii=False): the whole world before any carrier is inserted.
BASE_STATE_SHA256 = "e88da641435623b7f362747732d7719d9dbe0542feb36e1a03efaf759d953909"

# sha256 of the seven state files as `reset` writes them (name, NUL, bytes, NUL, in service order), for the first
# adaptation item, the first item with extra colleagues and the first test item of each service task.
STATE_FILES_SHA256 = {
    "drive-create-adapt-000": "db40b01ae2e310ce0b8a5a6be6eb767b5bffccffeaba8cc4514b40cb28ef3175",
    "drive-create-adapt-100": "fc04bb1d817867795582011aa8d55efe1cb4dddb333ab5ccefb719ebd2a774eb",
    "drive-create-test-200": "28e3835961442d511745b8105bd78b0f79e5686630fe470bf3b6dd55ea34f9f6",
    "email-invite-adapt-000": "81cf9a51325d3de460bf64c6b75ab5f39391d8b8232f8f155091031a10961de2",
    "email-invite-adapt-100": "707b6c26b34ceb542f44d69d9cb40b8ea3e1cf7b76ec3e0c70dbd137fe09b48a",
    "email-invite-test-500": "5e674bddc9700261e8e2eb6de3cc85b4767a681e5783d5b8f439636d121dd836",
    "chat-message-adapt-000": "33bb2e7b16794010ea3a40285580005afa9678ef03fa7973a61b39d01316cab2",
    "chat-message-adapt-100": "709632fb8705fc8bf3c9e3ef2d28748ef838b4d2e487eda6294f7a9b26c8eadd",
    "chat-message-test-500": "47167b2db7bdb8de84f5790a771c2b69448f0114047246ab1c037bc5916faa04",
    "chat-invite-adapt-000": "b4b1e1108f494ee57f945477f3bfd7cd7d2b93e2d65b9b50f2de4aea4d072835",
    "chat-invite-adapt-100": "3704391b653eb4dd405b0b47d32a4037a43bbe2fd30941037631fc6db3ce6179",
    "chat-invite-test-500": "90847c7b86e281760731d07cc3b0711a020c31fb29ed8f47558b829e470c8521",
}

SERVICES = ("mail", "cal", "drive", "travel", "bank", "slack", "web")


def episode(carrier: dict[str, Any], text: str = "BODY\nAlso, please X.", **extra: Any) -> dict[str, Any]:
    return {"id": "synthetic-1", "carrier": carrier, "carrier_text": text, **extra}


def item_of(tasks: dict[str, Task], item_id: str) -> dict[str, Any]:
    task = tasks[item_id.split("-adapt-")[0].split("-test-")[0]]
    return service_episode(task, task.adaptation.get(item_id) or task.test[item_id])


def digest_of(state_dir: Path) -> str:
    h = hashlib.sha256()
    for name in SERVICES:
        h.update(name.encode() + b"\0" + (state_dir / f"{name}.json").read_bytes() + b"\0")
    return h.hexdigest()


def every_item(tasks: dict[str, Task]) -> list[dict[str, Any]]:
    return [item for task in tasks.values() for item in (*task.adaptation.values(), *task.test.values())]


# --- what is installed -----------------------------------------------------------------------


@pytest.mark.parametrize("module", ["state", "verifiers", "selfcheck", "workspace"])
def test_every_module_imports_first_without_the_others(module: str) -> None:
    """The state module and the verifiers read each other; neither may need the other imported first."""
    code = f"import evolve_ipi.environment.{module}"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_seven_services() -> None:
    assert state.SERVICE_PROGRAMS == SERVICES
    assert state.SERVICE_FILES == ("_lib.py", *SERVICES)
    assert sorted(p.name for p in state.SERVICES_DIR.iterdir() if p.is_file()) == sorted(state.SERVICE_FILES)


# --- the base state --------------------------------------------------------------------------


def test_base_state_counts() -> None:
    st = state.base_state()
    assert list(st) == list(SERVICES)
    assert len(st["mail"]["contacts"]) == 12 and len(st["mail"]["emails"]) == 8 and st["mail"]["sent"] == []
    assert len(st["cal"]["events"]) == 4 and len(st["drive"]["files"]) == 4
    travel = st["travel"]
    assert len(travel["hotels"]) == 16 and len({h["city"] for h in travel["hotels"]}) == 4
    assert len(travel["restaurants"]) == 6 and len(travel["cars"]) == 4 and len(travel["flights"]) == 4
    assert len(travel["inbox"]) == 2 and travel["reservations"] == []
    bank = st["bank"]
    assert len(bank["transactions"]) == 8 and len(bank["scheduled"]) == 2 and len(bank["docs"]) == 2
    assert bank["iban"] and bank["balance"] > 0
    slack = st["slack"]
    assert slack["me"] == "alex" and len(slack["users"]) == 12 and len(slack["channels"]) == 8
    assert all(3 <= len(ch["messages"]) <= 5 for ch in slack["channels"].values())
    assert all(slack["me"] in ch["members"] for ch in slack["channels"].values())
    assert len(slack["dms"]) == 2 and len(st["web"]["pages"]) == 3


def test_base_state_is_a_fresh_copy_and_holds_no_instructions() -> None:
    a, b = state.base_state(), state.base_state()
    a["cal"]["events"].append({"id": "c9"})
    a["slack"]["channels"]["random"]["members"].append("nobody")
    assert len(b["cal"]["events"]) == 4 and "nobody" not in b["slack"]["channels"]["random"]["members"]
    assert "Also, please" not in json.dumps(state.base_state())


def test_base_state_is_pinned() -> None:
    assert hashlib.sha256(json.dumps(state.base_state(), ensure_ascii=False).encode()).hexdigest() == BASE_STATE_SHA256


def test_the_channel_order_is_the_order_the_agent_lists_them_in() -> None:
    names = list(state.base_state()["slack"]["channels"])
    assert names == ["eng-updates", "project-atlas", "design", "ops", "random", "announcements", "social", "general"]


# --- the carrier of an item ------------------------------------------------------------------


def test_an_email_carrier_is_inserted_at_its_position_and_the_ids_renumbered() -> None:
    ep = episode(
        {
            "kind": "email",
            "sender_name": "Noah Vance",
            "sender_email": "noah.vance@brightwater.example",
            "subject": "Hi",
            "date": "2024-05-14",
            "position": 3,
        }
    )
    emails = state.initial_state(ep)["mail"]["emails"]
    assert len(emails) == 9 and [e["id"] for e in emails] == [f"e{i}" for i in range(1, 10)]
    assert emails[2]["from_name"] == "Noah Vance" and emails[2]["body"] == "BODY\nAlso, please X."
    assert list(emails[2]) == ["id", "date", "from_name", "from", "subject", "body"]
    assert state.base_state()["mail"]["emails"][2]["from_name"] != "Noah Vance"  # the base is untouched


def test_a_chat_post_is_inserted_renumbered_and_its_poster_made_a_member() -> None:
    ep = episode({"kind": "slack_post", "channel": "project-atlas", "poster": "mei", "position": 2})
    channel = state.initial_state(ep)["slack"]["channels"]["project-atlas"]
    assert [m["seq"] for m in channel["messages"]] == list(range(1, len(channel["messages"]) + 1))
    assert channel["messages"][1] == {"seq": 2, "sender": "mei", "text": "BODY\nAlso, please X."}
    assert "mei" in channel["members"] and channel["members"][-1] == "mei"  # not a member before
    already = episode({"kind": "slack_post", "channel": "eng-updates", "poster": "dana", "position": 2})
    members = state.initial_state(already)["slack"]["channels"]["eng-updates"]["members"]
    assert members == state.base_state()["slack"]["channels"]["eng-updates"]["members"]


def test_a_drive_file_is_inserted_at_its_position_and_the_other_files_stay() -> None:
    ep = episode({"kind": "drive_file", "name": "report_900.txt", "position": 2})
    files = state.initial_state(ep)["drive"]["files"]
    assert files[1] == {"name": "report_900.txt", "content": "BODY\nAlso, please X.", "shared_with": []}
    assert [f["name"] for f in files if f["name"] != "report_900.txt"] == [
        f["name"] for f in state.base_state()["drive"]["files"]
    ]


def test_a_drive_file_may_not_take_the_name_of_a_file_the_drive_holds() -> None:
    held = state.base_state()["drive"]["files"][0]["name"]
    for name in (held, held.upper()):
        with pytest.raises(ValueError, match="already holds"):
            state.initial_state(episode({"kind": "drive_file", "name": name, "position": 2}))


def test_an_unknown_carrier_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown carrier kind 'pigeon'"):
        state.initial_state(episode({"kind": "pigeon", "position": 2}))
    with pytest.raises(ValueError, match="unknown carrier kind"):
        state.initial_state(episode({"position": 2}))


def test_the_base_state_is_not_touched_by_building_a_world() -> None:
    before = json.dumps(state.base_state())
    state.initial_state(episode({"kind": "drive_file", "name": "report_900.txt", "position": 2}))
    assert json.dumps(state.base_state()) == before


# --- extra colleagues ------------------------------------------------------------------------


def chat_episode(**extra: Any) -> dict[str, Any]:
    return episode({"kind": "slack_post", "channel": "eng-updates", "poster": "dana", "position": 2}, **extra)


def test_extra_colleagues_are_appended_to_the_users_and_change_nothing_else() -> None:
    plain = state.initial_state(chat_episode())
    with_colleagues = state.initial_state(chat_episode(extra_slack_users=["abel", "adil"]))
    assert with_colleagues["slack"]["users"] == plain["slack"]["users"] + ["abel", "adil"]
    with_colleagues["slack"]["users"] = plain["slack"]["users"]
    assert with_colleagues == plain


def test_extra_colleagues_belong_to_no_channel() -> None:
    st = state.initial_state(chat_episode(extra_slack_users=["abel"]))
    assert all("abel" not in ch["members"] for ch in st["slack"]["channels"].values())


def test_an_item_with_colleagues_can_message_one_of_them(svc_env: SvcEnv, service_tasks: dict[str, Task]) -> None:
    ep = item_of(service_tasks, "chat-invite-adapt-100")
    assert ep["extra_slack_users"] == ["abel", "adrien", "afua"]
    state.reset(ep, svc_env.state)
    assert svc_env.run("slack", "dm", "adrien", "hello").returncode == 0
    state.reset(item_of(service_tasks, "chat-invite-adapt-000"), svc_env.state)
    assert svc_env.run("slack", "dm", "adrien", "hello").returncode == 1  # an item without them: no such user


@pytest.mark.parametrize("taken", ["dana", "alex"])
def test_a_colleague_the_world_already_has_is_refused(taken: str) -> None:
    with pytest.raises(ValueError, match="already exists"):
        state.initial_state(chat_episode(extra_slack_users=["abel", taken]))


@pytest.mark.parametrize("bad", [[], "abel", ["Abel"], ["abel smith"], ["abel1"], [""], [7], ["abel", "abel"]])
def test_malformed_extra_colleagues_are_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="synthetic-1"):
        state.initial_state(chat_episode(extra_slack_users=bad))


def test_the_episode_is_not_modified(service_tasks: dict[str, Task]) -> None:
    for item in every_item(service_tasks):
        before = copy.deepcopy(item)
        state.initial_state(item)
        assert item == before, item["id"]


# --- every item -----------------------------------------------------------------------------


def test_every_item_starts_from_the_base_with_its_carrier_in_place(service_tasks: dict[str, Task]) -> None:
    base = state.base_state()
    for item in every_item(service_tasks):
        st = state.initial_state(item)
        carrier, text = item["carrier"], item["carrier_text"]
        position = carrier["position"]
        if carrier["kind"] == "email":
            assert st["mail"]["emails"][position - 1]["body"] == text, item["id"]
            assert len(st["mail"]["emails"]) == len(base["mail"]["emails"]) + 1
        elif carrier["kind"] == "slack_post":
            channel = st["slack"]["channels"][carrier["channel"]]
            assert channel["messages"][position - 1]["text"] == text, item["id"]
            assert channel["messages"][position - 1]["sender"] == carrier["poster"]
        else:
            assert st["drive"]["files"][position - 1]["content"] == text, item["id"]
            assert st["drive"]["files"][position - 1]["name"] == carrier["name"]
        expected_users = base["slack"]["users"] + item.get("extra_slack_users", [])
        assert st["slack"]["users"] == expected_users, item["id"]
        for name in ("cal", "travel", "bank", "web"):
            assert st[name] == base[name], (item["id"], name)


def test_the_injected_line_is_in_the_carrier_and_nowhere_else(service_tasks: dict[str, Task]) -> None:
    for item in every_item(service_tasks):
        line = item["injection"]["line"]
        assert json.dumps(state.initial_state(item)).count(json.dumps(line)[1:-1]) == 1, item["id"]


# --- reset -----------------------------------------------------------------------------------


def test_reset_writes_the_seven_files_and_an_empty_log(tmp_path: Path, service_tasks: dict[str, Task]) -> None:
    ep = item_of(service_tasks, "drive-create-test-200")
    target = tmp_path / "state"  # does not exist yet
    state.reset(ep, target)
    assert sorted(p.name for p in target.iterdir()) == sorted([*(f"{n}.json" for n in SERVICES), "access.log"])
    assert (target / "access.log").read_text(encoding="utf-8") == ""
    assert json.loads((target / "drive.json").read_text(encoding="utf-8")) == state.initial_state(ep)["drive"]


@pytest.mark.parametrize("item_id", sorted(STATE_FILES_SHA256))
def test_the_state_files_are_pinned(tmp_path: Path, service_tasks: dict[str, Task], item_id: str) -> None:
    state.reset(item_of(service_tasks, item_id), tmp_path)
    assert digest_of(tmp_path) == STATE_FILES_SHA256[item_id]


def test_reset_undoes_what_the_agent_did_and_empties_the_log(svc_env: SvcEnv, service_tasks: dict[str, Task]) -> None:
    ep = item_of(service_tasks, "email-invite-test-500")
    state.reset(ep, svc_env.state)
    before = digest_of(svc_env.state)
    assert svc_env.run("slack", "add", "omar", "random").returncode == 0
    assert svc_env.run("slack", "dm", "priya", "a summary").returncode == 0
    (svc_env.state / "cal.json").write_text("{}", encoding="utf-8")
    assert len(state.read_access_log(svc_env.state)) == 2 and digest_of(svc_env.state) != before
    state.reset(ep, svc_env.state)
    assert digest_of(svc_env.state) == before
    assert (svc_env.state / "access.log").read_text(encoding="utf-8") == ""


def test_reset_is_loud_when_a_judge_already_fires(
    tmp_path: Path, service_tasks: dict[str, Task], monkeypatch: pytest.MonkeyPatch
) -> None:
    fires = {"hit": True, "params_exact": True, "matches": [], "new_records": 1}
    ep = item_of(service_tasks, "chat-message-test-500")
    monkeypatch.setattr(verifiers, "judge_asr", lambda ep, state_dir: fires)
    with pytest.raises(state.StateError, match="chat-message-test-500.*already satisfies a judge"):
        state.reset(ep, tmp_path)
    monkeypatch.undo()
    monkeypatch.setattr(verifiers, "judge_utility", lambda ep, state_dir: fires)
    with pytest.raises(state.StateError, match="already satisfies a judge"):
        state.reset(ep, tmp_path)


def test_reset_leaves_no_temporary_file_behind(tmp_path: Path, service_tasks: dict[str, Task]) -> None:
    state.reset(item_of(service_tasks, "chat-invite-test-500"), tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


# --- install ---------------------------------------------------------------------------------


def test_install_copies_the_programs_and_points_them_at_the_state_directory(tmp_path: Path) -> None:
    workspace, state_dir = tmp_path / "ws", tmp_path / "state"
    workspace.mkdir()
    hashes = state.install(workspace, state_dir)
    for name in state.SERVICE_FILES:
        assert (workspace / "svc" / name).read_bytes() == (state.SERVICES_DIR / name).read_bytes()
        assert hashes[name] == hashlib.sha256((state.SERVICES_DIR / name).read_bytes()).hexdigest()
    assert list(hashes) == list(state.SERVICE_FILES)
    for name in state.SERVICE_PROGRAMS:
        mode = (workspace / "svc" / name).stat().st_mode
        assert mode & stat.S_IXUSR and mode & stat.S_IXGRP and mode & stat.S_IXOTH, name
    config = workspace / "svc" / "config.json"
    assert json.loads(config.read_text(encoding="utf-8")) == {"state_dir": str(state_dir.resolve())}
    assert config.read_text(encoding="utf-8") == json.dumps({"state_dir": str(state_dir.resolve())}, indent=1) + "\n"
    assert state_dir.is_dir()
    assert sorted(p.name for p in (workspace / "svc").iterdir()) == sorted([*state.SERVICE_FILES, "config.json"])


def test_the_packaged_programs_are_the_pinned_ones() -> None:
    state.verify_programs()
    assert list(state.PROGRAM_SHA256) == list(state.SERVICE_FILES)


def test_verify_programs_names_a_changed_and_a_missing_program(tmp_path: Path) -> None:
    copy_of_programs = tmp_path / "programs"
    shutil.copytree(state.SERVICES_DIR, copy_of_programs, ignore=shutil.ignore_patterns("__pycache__"))
    state.verify_programs(copy_of_programs)
    mail = copy_of_programs / "mail"
    mail.write_bytes(mail.read_bytes() + b"# one more line\n")
    (copy_of_programs / "web").unlink()
    with pytest.raises(state.StateError) as raised:
        state.verify_programs(copy_of_programs)
    assert str(raised.value).endswith("mail is not the pinned file; web is missing")  # and nothing else is named


def test_install_refuses_programs_that_are_not_the_pinned_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    changed = tmp_path / "programs"
    shutil.copytree(state.SERVICES_DIR, changed, ignore=shutil.ignore_patterns("__pycache__"))
    (changed / "slack").write_bytes((changed / "slack").read_bytes().replace(b"slack", b"chat", 1))
    monkeypatch.setattr(state, "SERVICES_DIR", changed)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(state.StateError, match="slack is not the pinned file"):
        state.install(workspace, tmp_path / "state")
    assert not (workspace / "svc").exists() and not (tmp_path / "state").exists()  # nothing was written


def test_the_modes_install_writes(tmp_path: Path) -> None:
    """The packaged file's permission bits, plus execute for the seven programs."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    state.install(workspace, tmp_path / "state")
    for name in state.SERVICE_FILES:
        packaged = stat.S_IMODE((state.SERVICES_DIR / name).stat().st_mode)
        want = packaged | (0o111 if name in state.SERVICE_PROGRAMS else 0)
        assert state.installed_mode(name) == want, name
        assert stat.S_IMODE((workspace / "svc" / name).stat().st_mode) == want, name
    assert all(state.installed_mode(name) & 0o111 == 0o111 for name in state.SERVICE_PROGRAMS)


def test_install_is_repeatable(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    first = state.install(workspace, tmp_path / "state")
    assert state.install(workspace, tmp_path / "state") == first


def test_install_refuses_a_state_directory_inside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for inside in (workspace, workspace / "state", workspace / "svc" / "state"):
        with pytest.raises(state.StateError, match="outside the workspace"):
            state.install(workspace, inside)
    assert not (workspace / "svc").exists()  # nothing was written


def test_install_refuses_a_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(state.StateError, match="does not exist"):
        state.install(tmp_path / "missing", tmp_path / "state")


def test_install_resolves_the_paths_it_records(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    link = tmp_path / "link"
    os.symlink(tmp_path / "real-state", link)
    (tmp_path / "real-state").mkdir()
    state.install(workspace, link)
    assert json.loads((workspace / "svc" / "config.json").read_text(encoding="utf-8"))["state_dir"] == str(
        (tmp_path / "real-state").resolve()
    )


# --- the access log --------------------------------------------------------------------------


def test_a_missing_access_log_is_empty(tmp_path: Path) -> None:
    assert state.read_access_log(tmp_path) == []


def test_the_access_log_is_read_line_by_line(tmp_path: Path) -> None:
    lines = [
        {"time": "2024-05-15T10:00:00", "program": "mail", "argv": ["list"], "rc": 0},
        {"time": "2024-05-15T10:00:02", "program": "mail", "argv": ["read", "e3"], "rc": 0},
    ]
    (tmp_path / "access.log").write_text("\n".join(json.dumps(line) for line in lines) + "\n\n", encoding="utf-8")
    assert state.read_access_log(tmp_path) == lines


@pytest.mark.parametrize("line", ["5", "null", "true", '"text"', "[]", '["time", "program", "argv", "rc"]'])
def test_an_access_log_line_that_is_json_but_not_an_object_is_a_state_error(tmp_path: Path, line: str) -> None:
    (tmp_path / "access.log").write_text(line + "\n", encoding="utf-8")
    with pytest.raises(state.StateError, match="line 1 is not a JSON object"):
        state.read_access_log(tmp_path)


def test_an_unreadable_access_log_is_a_state_error(tmp_path: Path) -> None:
    (tmp_path / "access.log").write_text("garbage\n", encoding="utf-8")
    with pytest.raises(state.StateError, match="line 1 is not JSON"):
        state.read_access_log(tmp_path)
    (tmp_path / "access.log").write_text('{"program": "mail", "argv": [], "rc": 0}\n', encoding="utf-8")
    with pytest.raises(state.StateError, match="line 1 lacks fields"):
        state.read_access_log(tmp_path)
