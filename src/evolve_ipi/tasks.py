"""The five tasks: their items, the texts the agent sees, and the order adaptation presents them in.

Everything lives under ``data/``; this module defines no text of its own.

    data/tasks/<task>/task.yaml          what the task is: its kind and, for File-Create, its request,
                                         feedback and injected-line templates
    data/tasks/<task>/adaptation.jsonl   48 items the agent adapts on
    data/tasks/<task>/test.jsonl         40 items it is tested on
    data/orders/rl/<task>.json           the items of the two RL stages, in the order they are presented
    data/orders/skills/<task>.json       the 12 conversations of skill evolution, in order
    data/SHA256SUMS                      the sha256 of every other file under data/ (``verify_data``)

A File-Create item is a document: ``body`` is the text a user would want read, and the line that carries
the injected request is added by ``file_document`` for the mechanism at hand. A service item holds all it
needs to seed the services and to judge an episode; the environment reads it as it stands.

Every mistake in the data raises ``TaskDataError`` and names the file and, for the item files, the line. A file
that differs from ``SHA256SUMS`` is refused by ``verify_data``, which every command runs before it starts anything.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, get_args

import yaml

from evolve_ipi.config import TASK_NAMES, ConfigError
from evolve_ipi.paths import Kind, Line

__all__ = [
    "DATA_DIR",
    "DATA_SUMS",
    "TASKS",
    "Kind",
    "Line",
    "RLOrder",
    "Task",
    "TaskDataError",
    "file_document",
    "file_target_dir",
    "file_workspace_documents",
    "load_rl_order",
    "load_skill_order",
    "load_task",
    "request",
    "service_episode",
    "verify_data",
]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_SUMS = "SHA256SUMS"  # in DATA_DIR, in the format of sha256sum: "<sha256>  <path below data/>"
_SUMS_LINE = re.compile(r"([0-9a-f]{64})  (.+)")
TASKS: tuple[str, ...] = TASK_NAMES

_ID_INFIX = {"adaptation": "adapt", "test": "test"}
_FEEDBACK_GROUPS = ("workspace-read-calendar", "slack-summary-post")

_FILE_FIELDS = frozenset({"id", "split", "src_relpath", "tgt_name", "body"})
_SERVICE_FIELDS = frozenset(
    {
        "id",
        "group",
        "split",
        "level",
        "carrier",
        "carrier_text",
        "user_task",
        "request_test",
        "request_authorized",
        "injection",
        "judge_asr",
        "judge_utility",
        "encounter",
    }
)
_SERVICE_OPTIONAL_FIELDS = frozenset({"extra_slack_users"})
_SERVICE_OBJECT_FIELDS = ("carrier", "user_task", "injection", "judge_asr", "judge_utility", "encounter")

# task.yaml of a file task: the keys that hold strings or mappings, and the fields their templates use
_FILE_META_KEYS = frozenset({"name", "kind", "requests", "feedback", "injection_line", "target_dir"})
_REQUEST_FIELDS = frozenset({"src"})
_INJECTION_FIELDS = frozenset({"tgt", "target_dir"})


class TaskDataError(Exception):
    """A file under ``data/`` is missing, malformed or inconsistent with the others."""


@dataclass(frozen=True)
class Task:
    """One task. ``adaptation`` and ``test`` map an item's id to the item, ordered by id."""

    name: str
    kind: Kind
    meta: dict[str, Any]
    adaptation: dict[str, dict[str, Any]]
    test: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RLOrder:
    """The ids of the adaptation items in the order the two RL stages present them."""

    stage1: list[str]
    stage2: list[str]


# --- reading the files -----------------------------------------------------------------------


def _check_task_name(name: str) -> None:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; known tasks: {', '.join(TASKS)}")


def _check_line(line: str) -> None:
    if line not in get_args(Line):
        raise ValueError(f"unknown line {line!r}; expected one of {', '.join(get_args(Line))}")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise TaskDataError(f"{path}: file not found") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise TaskDataError(f"{path}: cannot read the file ({exc})") from None


def _read_json_object(path: Path, keys: frozenset[str]) -> dict[str, Any]:
    try:
        data = json.loads(_read_text(path))
    except ValueError as exc:
        raise TaskDataError(f"{path}: not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise TaskDataError(f"{path}: expected a JSON object")
    _check_keys(path, data, keys)
    return data


def _check_keys(where: Any, data: dict[str, Any], keys: frozenset[str]) -> None:
    missing, unknown = sorted(keys - data.keys()), sorted(data.keys() - keys)
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing: {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown: {', '.join(unknown)}")
        raise TaskDataError(f"{where}: expected exactly the keys {', '.join(sorted(keys))} ({'; '.join(problems)})")


def _template_fields(template: str) -> frozenset[str] | None:
    """The replacement fields of a ``str.format`` template, or None when it is not a valid template."""
    try:
        return frozenset(field for _, field, _, _ in string.Formatter().parse(template) if field is not None)
    except ValueError:
        return None


def _read_meta(directory: Path, name: str) -> dict[str, Any]:
    path = directory / "task.yaml"
    try:
        meta = yaml.safe_load(_read_text(path))
    except yaml.YAMLError as exc:
        raise TaskDataError(f"{path}: invalid YAML: {exc}") from None
    if not isinstance(meta, dict):
        raise TaskDataError(f"{path}: expected a mapping of keys to values at the top level")
    kind = meta.get("kind")
    if kind not in get_args(Kind):
        raise TaskDataError(f"{path}: kind is {kind!r}; expected one of {', '.join(get_args(Kind))}")
    if meta.get("name") != name:
        raise TaskDataError(f"{path}: name is {meta.get('name')!r} but the directory is {name!r}")
    if kind == "service":
        _check_keys(path, meta, frozenset({"name", "kind", "feedback_group"}))
        if meta["feedback_group"] not in _FEEDBACK_GROUPS:
            raise TaskDataError(
                f"{path}: feedback_group is {meta['feedback_group']!r}; expected one of {', '.join(_FEEDBACK_GROUPS)}"
            )
    else:
        _check_file_meta(path, meta)
    return meta


def _check_file_meta(path: Path, meta: dict[str, Any]) -> None:
    _check_keys(path, meta, _FILE_META_KEYS)
    for key, subkeys in (
        ("requests", frozenset({"test", "authorized"})),
        ("feedback", frozenset({"positive", "negative"})),
        ("target_dir", frozenset({"rl", "skills"})),
    ):
        if not isinstance(meta[key], dict):
            raise TaskDataError(f"{path}: {key} must be a mapping")
        _check_keys(f"{path}: {key}", meta[key], subkeys)
        _check_strings(path, key, meta[key].values())
    _check_strings(path, "injection_line", [meta["injection_line"]])
    for key, templates, fields in (
        ("requests", meta["requests"].values(), _REQUEST_FIELDS),
        ("injection_line", [meta["injection_line"]], _INJECTION_FIELDS),
    ):
        for template in templates:
            if _template_fields(template) != fields:
                wanted = ", ".join(f"{{{field}}}" for field in sorted(fields))
                raise TaskDataError(f"{path}: {key} must use exactly the fields {wanted}, not {template!r}")


def _check_strings(where: Any, key: str, values: Any) -> None:
    for value in values:
        if not isinstance(value, str) or not value:
            raise TaskDataError(f"{where}: {key} must hold non-empty strings, got {value!r}")


def _check_item(item: Any, where: str, task: str, kind: Kind, split: str, feedback_group: str | None) -> str:
    """Raise ``TaskDataError`` unless ``item`` is a well-formed item of ``split``; return its id."""
    if not isinstance(item, dict):
        raise TaskDataError(f"{where}: expected a JSON object")
    item_id = item.get("id")
    if not isinstance(item_id, str) or not re.fullmatch(rf"{re.escape(task)}-{_ID_INFIX[split]}-\d{{3}}", item_id):
        raise TaskDataError(f"{where}: id {item_id!r} is not of the form '{task}-{_ID_INFIX[split]}-NNN'")
    required = _FILE_FIELDS if kind == "file" else _SERVICE_FIELDS
    optional = frozenset() if kind == "file" else _SERVICE_OPTIONAL_FIELDS
    missing, unknown = sorted(required - item.keys()), sorted(item.keys() - required - optional)
    if missing:
        raise TaskDataError(f"{where}: item {item_id!r} lacks field(s): {', '.join(missing)}")
    if unknown:
        raise TaskDataError(f"{where}: item {item_id!r} has unknown field(s): {', '.join(unknown)}")
    if item["split"] != split:
        raise TaskDataError(f"{where}: item {item_id!r} has split {item['split']!r}, expected {split!r}")
    if kind == "file":
        _check_file_item(item, where)
    else:
        _check_service_item(item, where, feedback_group)
    return item_id


def _check_file_item(item: dict[str, Any], where: str) -> None:
    for key in ("src_relpath", "tgt_name", "body"):
        if not isinstance(item[key], str) or not item[key]:
            raise TaskDataError(f"{where}: item {item['id']!r}: {key} must be a non-empty string")
    if "/" in item["tgt_name"]:
        raise TaskDataError(f"{where}: item {item['id']!r}: tgt_name {item['tgt_name']!r} must be a file name")
    source = Path(item["src_relpath"])
    if source.is_absolute() or ".." in source.parts:
        raise TaskDataError(f"{where}: item {item['id']!r}: src_relpath must stay inside the workspace")


def _check_service_item(item: dict[str, Any], where: str, feedback_group: str | None) -> None:
    if item["group"] != feedback_group:
        raise TaskDataError(
            f"{where}: item {item['id']!r} has group {item['group']!r}, but task.yaml says "
            f"feedback_group is {feedback_group!r}"
        )
    for key in ("request_test", "request_authorized", "carrier_text"):
        if not isinstance(item[key], str) or not item[key]:
            raise TaskDataError(f"{where}: item {item['id']!r}: {key} must be a non-empty string")
    for key in _SERVICE_OBJECT_FIELDS:
        if not isinstance(item[key], dict):
            raise TaskDataError(f"{where}: item {item['id']!r}: {key} must be a JSON object")
    users = item.get("extra_slack_users")
    if "extra_slack_users" in item and (
        not isinstance(users, list) or not users or not all(isinstance(u, str) and u for u in users)
    ):
        raise TaskDataError(f"{where}: item {item['id']!r}: extra_slack_users must be a list of names")


def _read_items(
    path: Path, task: str, kind: Kind, split: str, feedback_group: str | None
) -> dict[str, dict[str, Any]]:
    lines = _read_text(path).split("\n")
    if lines[-1] == "":
        lines.pop()
    items: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(lines, start=1):
        where = f"{path}:{number}"
        if not line.strip():
            raise TaskDataError(f"{where}: blank line")
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise TaskDataError(f"{where}: not valid JSON ({exc})") from None
        item_id = _check_item(item, where, task, kind, split, feedback_group)
        if item_id in items:
            raise TaskDataError(f"{where}: id {item_id!r} appears twice")
        items[item_id] = item
    return dict(sorted(items.items()))


def load_task(name: str, data_dir: Path = DATA_DIR) -> Task:
    """Read and check ``data/tasks/<name>/``."""
    _check_task_name(name)
    directory = Path(data_dir) / "tasks" / name
    meta = _read_meta(directory, name)
    kind: Kind = meta["kind"]
    group = meta.get("feedback_group")
    return Task(
        name=name,
        kind=kind,
        meta=meta,
        adaptation=_read_items(directory / "adaptation.jsonl", name, kind, "adaptation", group),
        test=_read_items(directory / "test.jsonl", name, kind, "test", group),
    )


# --- requests and documents ------------------------------------------------------------------


def _own(task: Task, item: dict[str, Any]) -> dict[str, Any]:
    if item.get("id") not in task.adaptation and item.get("id") not in task.test:
        raise ValueError(f"item {item.get('id')!r} is not an item of {task.name}")
    return item


def _file_task(task: Task) -> None:
    if task.kind != "file":
        raise ValueError(f"{task.name} has no documents: it is a {task.kind} task")


def request(task: Task, item: dict[str, Any], authorized: bool) -> str:
    """What the user writes: the authorized request (it also asks to carry out what the content asks
    for) when ``authorized``, otherwise the test request, which only asks to read."""
    _own(task, item)
    if task.kind == "file":
        template = task.meta["requests"]["authorized" if authorized else "test"]
        return str(template.format(src=item["src_relpath"]))
    return str(item["request_authorized" if authorized else "request_test"])


def file_target_dir(task: Task, line: Line) -> str:
    """The directory the injected line of a File-Create document points at."""
    _file_task(task)
    _check_line(line)
    return str(task.meta["target_dir"][line])


def file_document(task: Task, item: dict[str, Any], line: Line) -> str:
    """The text of one File-Create document: its body, a newline and the injected line. It has no
    trailing newline."""
    _file_task(task)
    _check_line(line)
    _own(task, item)
    injected = task.meta["injection_line"].format(tgt=item["tgt_name"], target_dir=task.meta["target_dir"][line])
    return f"{item['body']}\n{injected}"


def file_workspace_documents(task: Task, line: Line) -> dict[str, str]:
    """Every document the workspace holds, adaptation items first: path inside the workspace -> text."""
    _file_task(task)
    _check_line(line)
    documents: dict[str, str] = {}
    for item in (*task.adaptation.values(), *task.test.values()):
        if item["src_relpath"] in documents:
            raise TaskDataError(f"{task.name}: the path {item['src_relpath']!r} is used by two items")
        documents[item["src_relpath"]] = file_document(task, item, line)
    return documents


def service_episode(task: Task, item: dict[str, Any]) -> dict[str, Any]:
    """The episode the environment seeds its services from and judges: the item itself."""
    if task.kind != "service":
        raise ValueError(f"{task.name} is a {task.kind} task and has no service episode")
    return _own(task, item)


# --- orders ----------------------------------------------------------------------------------


def _ids(path: Path, key: str, value: Any, task: Task) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(i, str) for i in value):
        raise TaskDataError(f"{path}: {key} must be a non-empty list of item ids")
    unknown = sorted({i for i in value if i not in task.adaptation})
    if unknown:
        raise TaskDataError(f"{path}: {key} names {', '.join(unknown)}, which is not an adaptation item of {task.name}")
    return list(value)


def load_rl_order(task: str, data_dir: Path = DATA_DIR) -> RLOrder:
    """The presentation order of the two RL stages."""
    _check_task_name(task)
    path = Path(data_dir) / "orders" / "rl" / f"{task}.json"
    data = _read_json_object(path, frozenset({"stage1", "stage2"}))
    items = load_task(task, data_dir)
    return RLOrder(
        stage1=_ids(path, "stage1", data["stage1"], items),
        stage2=_ids(path, "stage2", data["stage2"], items),
    )


def load_skill_order(task: str, data_dir: Path = DATA_DIR) -> list[str]:
    """The conversations of skill evolution, in the order they are held."""
    _check_task_name(task)
    path = Path(data_dir) / "orders" / "skills" / f"{task}.json"
    data = _read_json_object(path, frozenset({"conversations"}))
    return _ids(path, "conversations", data["conversations"], load_task(task, data_dir))


# --- the checksums -----------------------------------------------------------------------------------


def _listed_sums(root: Path) -> tuple[dict[str, str], bytes]:
    path = root / DATA_SUMS
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"cannot read {path}, which lists the sha256 of every file under {root}: {error}") from None
    listed: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _SUMS_LINE.fullmatch(line)
        if match is None:
            raise ConfigError(f"{path} line {number} is not '<sha256>  <path>': {line!r}")
        digest, name = match.groups()
        pure = PurePosixPath(name)
        if pure.is_absolute() or str(pure) != name or ".." in pure.parts or name == DATA_SUMS:
            raise ConfigError(f"{path}: line {number} names {name!r}, which is not a file below {root}")
        if name in listed:
            raise ConfigError(f"{path} lists {name} twice (line {number})")
        listed[name] = digest
    return listed, raw


def verify_data(data_dir: Path | None = None) -> str:
    """Check every file under ``data/`` against ``data/SHA256SUMS``; return the sha256 of ``SHA256SUMS`` itself.

    The texts, the items and the orders are the method, so a run reads them only as they were published. Raises
    ``ConfigError`` naming every file that changed, is missing or is not listed. ``data_dir`` defaults to ``DATA_DIR``.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    listed, raw = _listed_sums(root)
    found = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} - {DATA_SUMS}
    problems = [f"{name} is missing" for name in sorted(listed.keys() - found)]
    problems += [f"{name} is not listed" for name in sorted(found - listed.keys())]
    for name in sorted(found & listed.keys()):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != listed[name]:
            problems.append(f"{name} has changed (sha256 {digest}, {DATA_SUMS} says {listed[name]})")
    if problems:
        raise ConfigError(
            f"the data under {root} is not the data {DATA_SUMS} lists: {'; '.join(problems)}. The documents, items "
            "and orders are part of the experiment: restore them (`git checkout -- data/`)"
        )
    return hashlib.sha256(raw).hexdigest()
