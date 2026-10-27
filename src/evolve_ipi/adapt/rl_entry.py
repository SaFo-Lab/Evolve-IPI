"""The trainer process of RL adaptation: OpenClaw-RL's ``run.py``, with the model family's tool-call format.

    python -m evolve_ipi.adapt.rl_entry --method rl --model-name M ... --record-dir D [--resume-from-ckpt P]

The arguments are those of OpenClaw-RL's ``run.py``, passed on unchanged. Before ``run.py`` starts the trainer, this
process changes four things, in itself only:

1. The tool-call bridge of the model's family, in OpenClaw-RL's ``api_server``: the tool calls of the message history
   are put into the shape the chat template wants, and the calls of a generation are read with the family's parser.
   Qwen's calls are read from the whole decoded text. The harmony and Inkling markers are special tokens, which
   ``api_server`` drops when it decodes a generation, so for them the family's stop strings are added to every
   sampling request and the calls the agent is given are read from a decode that keeps the markers. The training
   tokens and log-probabilities stay the sampled ones.
2. Inkling-Small only: every rollout generation is appended to ``<record-dir>/raw_train.jsonl``, with the number of
   calls the bridge read from it, so that the run can be audited for markup that parsed to nothing.
3. The trainer's own records survive: OpenClaw-RL empties ``conversations.jsonl`` and ``prm_scores.jsonl`` at every
   update, and their bytes are copied to ``conversations_all.jsonl`` and ``prm_scores_all.jsonl`` before that.
4. With ``--resume-from-ckpt``: the resume restores the optimizer state together with the LoRA weights, and waits
   for the load, so that a load that fails stops the trainer. The installed Tinker SDK's ``load_state`` restores the
   weights only, and OpenClaw-RL never waits for its result.

The rollout, the reward model, the loss and the optimizer are OpenClaw-RL's.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from evolve_ipi.models.formats import Family, Formatter, get_formatter
from evolve_ipi.models.formats.inkling import InklingFormatter
from evolve_ipi.thirdparty import OPENCLAW_TINKER_DIR

__all__ = [
    "ENTRY_LINE",
    "RAW_DUMP_NAME",
    "RESTORED_LINE",
    "Bridge",
    "family_of",
    "install_bridge",
    "install_optimizer_resume",
    "install_raw_dump",
    "install_record_keeper",
    "main",
]

# The line this process prints once its changes are in place; the orchestrator requires it in the trainer log.
ENTRY_LINE = "[rl_entry] family={family} model={model} resume={resume}"
# The line printed after a stage-2 resume restored the stage-1 state; the checkpoint collector requires it.
RESTORED_LINE = "[rl_entry] restored LoRA weights AND optimizer state from %s (load_state_with_optimizer, awaited)"
RAW_DUMP_NAME = "raw_train.jsonl"

KEEP_SUFFIX = "_all"
APPENDERS = ("_flush_pending_record", "_append_score_record")
PURGER = "purge_record_files"

_keep_lock = threading.Lock()


def family_of(model_name: str) -> Family:
    """The tool-call family of the model OpenClaw-RL trains, read from its name as the trainer is given it."""
    name = (model_name or "").lower()
    if name.startswith("qwen/"):
        return "qwen"
    if name.startswith("openai/gpt-oss"):
        return "harmony"
    if name == "thinkingmachines/inkling-small":
        return "inkling"
    raise ValueError(
        f"no tool-call family is known for the model {model_name!r}; the trainer runs Qwen/*, openai/gpt-oss-* and "
        "thinkingmachines/Inkling-Small"
    )


# --- 1. the tool-call bridge -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bridge:
    formatter: Formatter
    stops: tuple[str, ...]
    raw_decode: bool


def install_bridge(api_server: ModuleType, formatter: Formatter) -> Bridge:
    """Install ``formatter`` at the tool-call touchpoints of OpenClaw-RL's ``api_server``.

    The history is normalized before ``api_server``'s own normalization; ``_extract_tool_calls`` reads the content
    and the calls with the formatter. A family with stop strings or with markup made of special tokens also gets a
    wrapper around ``_BaseServer._forward_to_tinker``: it adds the stops after the client's own, and replaces the
    message the agent is given by the formatter's reading of the raw tokens, decoded with the special tokens kept.
    """
    normalize = api_server._normalize_messages

    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return normalize(formatter.normalize_tool_call_arguments(messages))  # type: ignore[no-any-return]

    api_server._normalize_messages = _normalize_messages  # type: ignore[attr-defined]

    def _extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
        return formatter.clean_content(text), formatter.parse_tool_calls(text)

    api_server._extract_tool_calls = _extract_tool_calls  # type: ignore[attr-defined]

    stops = tuple(formatter.stop_sequences or ())
    raw_decode = bool(formatter.needs_raw_decode)
    if not (stops or raw_decode):
        return Bridge(formatter, stops, raw_decode)
    forward = api_server._BaseServer._forward_to_tinker

    async def _forward_to_tinker(self: Any, body: dict[str, Any]) -> dict[str, Any]:
        if stops:
            existing = body.get("stop") or []
            if isinstance(existing, str):
                existing = [existing]
            body["stop"] = list(dict.fromkeys([*existing, *stops]))
        out: dict[str, Any] = await forward(self, body)
        if raw_decode:
            raw = out.get("_raw_response_tokens")
            if raw is None or getattr(self, "_tokenizer", None) is None:
                raise RuntimeError("a rollout sampling result carries no raw tokens to decode")
            full = self._tokenizer.decode(raw, skip_special_tokens=False)
            calls, content = formatter.parse_output(full)
            message = out["choices"][0]["message"]
            message["content"] = content
            if calls:
                message["tool_calls"] = calls
                out["choices"][0]["finish_reason"] = "tool_calls"
            elif "tool_calls" in message:
                del message["tool_calls"]
        return out

    api_server._BaseServer._forward_to_tinker = _forward_to_tinker
    return Bridge(formatter, stops, raw_decode)


# --- 2. the raw dump of the rollouts (Inkling-Small) ------------------------------------------------------------


def install_raw_dump(api_server: ModuleType, path: Path) -> None:
    """Append every rollout generation to ``path``: its raw text with the special tokens kept, its length, the stop
    strings it was sampled with and the number of tool calls the bridge read from it. Nothing it returns changes."""
    inner = api_server._BaseServer._forward_to_tinker
    lock = threading.Lock()

    async def _forward_to_tinker(self: Any, body: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = await inner(self, body)
        ids = out.get("_raw_response_tokens")
        if ids is None:
            raise RuntimeError("a rollout sampling result carries no _raw_response_tokens")
        choice = out["choices"][0]
        row = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "n_tokens": len(ids),
            "stop": body.get("stop"),
            "finish_reason": choice.get("finish_reason"),
            "parsed_calls": len(choice["message"].get("tool_calls") or []),
            "raw": self._tokenizer.decode(ids, skip_special_tokens=False),
        }
        with lock:  # a short append, written as OpenClaw-RL writes its own record files
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as dump:  # noqa: ASYNC230
                dump.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out

    api_server._BaseServer._forward_to_tinker = _forward_to_tinker


# --- 3. keeping the trainer's records --------------------------------------------------------------------------


def _keep_path(path: str | Path) -> Path:
    """conversations.jsonl -> conversations_all.jsonl, in the same directory."""
    p = Path(path)
    return p.with_name(p.stem + KEEP_SUFFIX + p.suffix)


def _record_paths(server: Any) -> list[str]:
    return [p for p in (getattr(server, "_record_file", None), getattr(server, "_prm_record_file", None)) if p]


def _drain(server: Any) -> None:
    """Append what the server wrote since the last drain to the keep files.

    The server only appends to its record files, and empties them in ``purge_record_files``, which resets the
    offsets; so every byte is copied once, in the order it was written.
    """
    with _keep_lock:
        offsets = getattr(server, "_keep_offsets", None)
        if offsets is None:
            offsets = server._keep_offsets = {}
        for path in _record_paths(server):
            try:
                with open(path, "rb") as records:
                    records.seek(offsets.get(path, 0))
                    chunk = records.read()
                    offsets[path] = records.tell()
            except FileNotFoundError:  # the server has not created it yet
                continue
            if chunk:
                with open(_keep_path(path), "ab") as keep:
                    keep.write(chunk)


def _prepare_keep_files(server: Any) -> None:
    """Create the keep files; a record directory that already holds kept records belongs to another run."""
    for path in _record_paths(server):
        keep = _keep_path(path)
        if keep.exists() and keep.stat().st_size:
            raise SystemExit(
                f"rl_entry: {keep} already holds {keep.stat().st_size} bytes of an earlier run's records; "
                "every trainer start needs a record directory of its own"
            )
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep.write_bytes(b"")
    server._keep_offsets = {}
    print(
        "[rl_entry] keeping records in " + ", ".join(str(_keep_path(p)) for p in _record_paths(server)),
        file=sys.stderr,
        flush=True,
    )


def _wrap(cls: type, name: str) -> str:
    original = cls.__dict__[name]
    if name == PURGER:

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            _drain(self)  # copy what is about to be emptied
            out = original(self, *args, **kwargs)
            self._keep_offsets = {}  # both files are back to zero bytes
            return out

    else:

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            out = original(self, *args, **kwargs)
            _drain(self)
            return out

    wrapper.__name__ = name
    wrapper.__doc__ = f"{name}, then the trainer's records are copied to the keep files."
    setattr(cls, name, wrapper)
    return f"{cls.__name__}.{name}"


def install_record_keeper(api_server: ModuleType) -> list[str]:
    """Wrap every record append and the purge, on every server class of ``api_server`` that defines them.

    Raises ``SystemExit`` when no class defines one of them: the keeper would then keep nothing.
    """
    base = api_server._BaseServer
    classes = [base] + sorted(
        (c for c in vars(api_server).values() if isinstance(c, type) and issubclass(c, base) and c is not base),
        key=lambda c: c.__name__,
    )
    wrapped = []
    for cls in classes:
        for name in (*APPENDERS, PURGER):
            if name in cls.__dict__:
                wrapped.append(_wrap(cls, name))
    for name in (*APPENDERS, PURGER):
        if not any(w.endswith("." + name) for w in wrapped):
            raise SystemExit(
                f"rl_entry: no server class of OpenClaw-RL defines {name!r} any more; its record keeping has moved "
                "and the trainer's records would not be kept"
            )
    original_init = base.__init__

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _prepare_keep_files(self)

    base.__init__ = __init__
    return wrapped


# --- 4. the resume with the optimizer state ----------------------------------------------------------------------


async def _load_state_with_optimizer_async(self: Any, path: str, weights_access_token: str | None = None) -> Any:
    future = await self.load_state_with_optimizer_async(path, weights_access_token=weights_access_token)
    await future.result_async()  # a failed load raises here
    print(RESTORED_LINE % path, flush=True)
    return future


def install_optimizer_resume(training_client_cls: type | None = None) -> None:
    """Make ``TrainingClient.load_state_async``, which OpenClaw-RL's trainer resumes with, restore the optimizer
    state with the weights and wait for the load."""
    if training_client_cls is None:
        from tinker.lib.public_interfaces.training_client import TrainingClient

        training_client_cls = TrainingClient
    training_client_cls.load_state_async = _load_state_with_optimizer_async  # type: ignore[attr-defined]


# --- the process -------------------------------------------------------------------------------------------------


def _options(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evolve_ipi.adapt.rl_entry", add_help=False, allow_abbrev=False)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--resume-from-ckpt", default="")
    options, _ = parser.parse_known_args(list(argv))
    if not options.model_name:
        raise SystemExit("rl_entry: --model-name is required")
    if not options.record_dir:
        raise SystemExit("rl_entry: --record-dir is required (the trainer's records and the raw dump live there)")
    return options


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    options = _options(arguments)
    family = family_of(options.model_name)
    formatter = get_formatter(family)

    tinker_dir = Path(OPENCLAW_TINKER_DIR)
    if str(tinker_dir) not in sys.path:
        sys.path.insert(0, str(tinker_dir))
    api_server = importlib.import_module("api_server")
    if Path(api_server.__file__ or "").resolve().parent != tinker_dir.resolve():
        raise SystemExit(f"rl_entry: imported api_server from {api_server.__file__}, not from {tinker_dir}")

    bridge = install_bridge(api_server, formatter)
    if family == "inkling":
        if (
            not isinstance(bridge.formatter, InklingFormatter)
            or list(bridge.stops) != InklingFormatter().stop_sequences
            or not bridge.raw_decode
        ):
            raise SystemExit("rl_entry: the Inkling bridge did not take the Inkling formatter, stop and raw decode")
        install_raw_dump(api_server, Path(options.record_dir) / RAW_DUMP_NAME)
    if options.resume_from_ckpt:
        install_optimizer_resume()
    print(
        ENTRY_LINE.format(family=family, model=options.model_name, resume=options.resume_from_ckpt or "none"),
        flush=True,
    )
    install_record_keeper(api_server)

    run = importlib.import_module("run")
    sys.argv = [sys.argv[0], *arguments]  # run.py reads its arguments from sys.argv
    run.main()


if __name__ == "__main__":
    main()
