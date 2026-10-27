"""The model server: an OpenAI-compatible endpoint in front of a model on Tinker, with thinking on.

    python -m evolve_ipi.models.server --family qwen --base-model Qwen/Qwen3.8-27B --served-model-name policy \\
        --port 30012 --temperature 0.6 --evidence-out E.json --raw-out R.jsonl --record-dir D \\
        [--load-checkpoint tinker://.../step_0032] [--lora-rank 32] [--host 127.0.0.1]

It serves the base model or a trained LoRA checkpoint to the agent, during evaluation and behind the MetaClaw proxy.
It builds on OpenClaw-RL's ``_BaseServer``, so a prompt is tokenized exactly as in RL training (the model's own chat
template). What it adds:

- The sampling temperature is forced to ``--temperature`` and checked again on every request before it is
  forwarded; a request at any other temperature is refused and recorded in the evidence file.
- OpenClaw's ``max_completion_tokens`` and ``reasoning_effort`` are not forwarded as anything: the generation limit
  is OpenClaw-RL's ``max_tokens`` default (2048) unless a client sends ``max_tokens`` itself, and the chat template
  applies its own reasoning setting.
- Tool calls are read in the model family's format (``models.formats``) from the generation decoded with its
  special tokens kept, and the family's stop strings are added after the client's.
- Every generation is appended to the raw dump, the reasoning included, since the reply the agent gets has the
  reasoning removed.

The server prints ``EVAL PROXY READY on <host>:<port> serving base:<model>`` (or ``checkpoint:<path>``) once it
listens. Any startup failure (a missing key, a tokenizer that cannot be loaded, a checkpoint that does not load,
a port in use) ends the process with status 1 before that line.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import queue
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import HTTPException

from evolve_ipi.models.formats import Family, get_formatter
from evolve_ipi.models.launch import ready_line
from evolve_ipi.procs import PortInUse, exit_without_teardown, require_free
from evolve_ipi.thirdparty import OPENCLAW_TINKER_DIR

if not (OPENCLAW_TINKER_DIR / "api_server.py").is_file():
    raise ImportError(
        f"OpenClaw-RL is not checked out at {OPENCLAW_TINKER_DIR}; run `git submodule update --init` in the "
        "repository root"
    )
# OpenClaw-RL's Tinker port is a directory of scripts, not a package: its modules import each other by bare name.
if str(OPENCLAW_TINKER_DIR) not in sys.path:
    sys.path.insert(0, str(OPENCLAW_TINKER_DIR))

import api_server
from config import TinkerConfig

__all__ = [
    "BaseServer",
    "ModelServer",
    "ServingEvidence",
    "StartupError",
    "TemperatureMismatch",
    "TinkerConfig",
    "main",
    "ready_line",
    "wait_started",
]

logger = logging.getLogger("evolve_ipi.model_server")

BaseServer = api_server._BaseServer

EXPERIMENT = "evolve-ipi"
EVIDENCE_REQUESTS_KEPT = 3
# A request's prompt is summarized in the evidence file, not copied: these keys are left out.
_NOT_IN_EVIDENCE = ("messages", "tools", "tool_choice")
_TEMPERATURE_TOLERANCE = 1e-9
_LISTEN_TIMEOUT_S = 60.0


class StartupError(RuntimeError):
    """The server cannot start; the message says why."""


class TemperatureMismatch(RuntimeError):
    """A request was about to be sampled at a temperature other than the one this server must use."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json_atomically(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _summarize(body: dict[str, Any]) -> dict[str, Any]:
    """A forwarded request for the evidence file: its sampling parameters verbatim, its prompt as counts."""
    summary = {key: value for key, value in body.items() if key not in _NOT_IN_EVIDENCE}
    messages = body.get("messages") or []
    summary["_messages_count"] = len(messages)
    summary["_messages_chars"] = sum(len(str(message.get("content") or "")) for message in messages)
    summary["_has_tools"] = bool(body.get("tools"))
    return summary


def _same_temperature(seen: Any, expected: float) -> bool:
    if isinstance(seen, bool) or not isinstance(seen, int | float):
        return False
    return math.isfinite(seen) and abs(float(seen) - expected) <= _TEMPERATURE_TOLERANCE


class ServingEvidence:
    """The record of what the server forwarded: the evidence JSON and the raw dump.

    The evidence JSON names the server's settings, counts the forwarded requests and keeps the first three of them
    (without their prompt). It is rewritten after every forwarded request and on every refused one, which it lists
    under ``violations``. The raw dump gets one JSON line per generation: ``{at, request, messages, n_tokens,
    raw}``, ``request`` being the request's number (from 1) and ``raw`` the generation decoded with its special
    tokens kept.
    """

    def __init__(
        self,
        evidence_path: Path,
        raw_path: Path,
        *,
        family: Family,
        base_model: str,
        temperature: float,
        argv: Sequence[str],
    ) -> None:
        self.evidence_path = Path(evidence_path)
        self.raw_path = Path(raw_path)
        self.temperature = temperature
        self.forwarded = 0
        self.record: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "family": family,
            "base_model": base_model,
            "expected_temperature": temperature,
            "argv": list(argv),
            "requests": [],
            "requests_forwarded": 0,
            "raw_dump": str(self.raw_path.resolve()),
        }
        self._lock = threading.Lock()

    def open(self) -> None:
        """Start both files. A raw dump that already holds generations belongs to another run and is refused."""
        if self.raw_path.exists() and self.raw_path.stat().st_size:
            raise StartupError(
                f"{self.raw_path} already holds {self.raw_path.stat().st_size} bytes of an earlier run's "
                "generations; give this server a raw dump of its own"
            )
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        self.raw_path.write_text("", encoding="utf-8")

    def _write(self) -> None:
        self.record["requests_forwarded"] = self.forwarded
        self.record["written_at"] = _now()
        _write_json_atomically(self.evidence_path, self.record)

    def admit(self, body: dict[str, Any]) -> int:
        """Check the temperature of a request about to be forwarded; return its number, or raise and record."""
        seen = body.get("temperature")
        with self._lock:
            if not _same_temperature(seen, self.temperature):
                violation = {"seen_temperature": seen, "expected": self.temperature, "request_index": self.forwarded}
                self.record.setdefault("violations", []).append(violation)
                self._write()
                raise TemperatureMismatch(
                    f"the request would be sampled at temperature {seen!r}, but this server samples at "
                    f"{self.temperature}; refused (recorded in {self.evidence_path})"
                )
            self.forwarded += 1
            if len(self.record["requests"]) < EVIDENCE_REQUESTS_KEPT:
                self.record["requests"].append(_summarize(body))
            self._write()
            return self.forwarded

    def dump(self, request: int, body: dict[str, Any], tokens: Sequence[int], text: str) -> None:
        line = {
            "at": _now(),
            "request": request,
            "messages": len(body.get("messages") or []),
            "n_tokens": len(tokens),
            "raw": text,
        }
        with self._lock, open(self.raw_path, "a", encoding="utf-8") as dump:
            dump.write(json.dumps(line, ensure_ascii=False) + "\n")


class ModelServer(BaseServer):
    """OpenClaw-RL's server, sampling only: no records, no rewards, no training."""

    _TITLE = "evolve-ipi model server"

    def __init__(
        self,
        *,
        family: Family,
        base_model: str,
        served_model_name: str,
        temperature: float,
        evidence: ServingEvidence,
        record_dir: Path,
        host: str = "127.0.0.1",
        port: int = 0,
        sampling_client: Any = None,
    ) -> None:
        self._formatter = get_formatter(family)
        self._temperature = temperature
        self._evidence = evidence
        submission_open = threading.Event()
        submission_open.set()
        config = TinkerConfig(
            model_name=base_model,
            served_model_name=served_model_name,
            proxy_host=host,
            proxy_port=port,
            record_dir=str(record_dir),
        )
        super().__init__(config, queue.Queue(), submission_open, sampling_client)

    def _load_tokenizer(self) -> Any:
        # OpenClaw-RL logs a failed load and serves on without a tokenizer, so that every request fails with
        # HTTP 503. Here the failure ends the start.
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        except Exception as error:
            raise StartupError(
                f"cannot load the tokenizer of {self.config.model_name} from Hugging Face: "
                f"{type(error).__name__}: {error}"
            ) from error

    async def _handle_request(
        self, body: dict[str, Any], session_id: str, turn_type: str, session_done: bool
    ) -> dict[str, Any]:
        # Adapted from OpenClaw-RL openclaw-tinker/api_server.py (Apache-2.0); see THIRD_PARTY_NOTICES.md.
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        forward = {key: value for key, value in body.items() if key not in api_server._NON_STANDARD_BODY_KEYS}
        forward["stream"] = False
        forward.pop("stream_options", None)
        if "model" not in forward:
            forward["model"] = self.config.served_model_name
        forward["temperature"] = self._temperature
        forward["messages"] = self._formatter.normalize_tool_call_arguments(forward["messages"])
        family_stops = self._formatter.stop_sequences
        if family_stops:
            client_stops = forward.get("stop") or []
            if isinstance(client_stops, str):
                client_stops = [client_stops]
            # The client's stops first, then the family's, each once.
            forward["stop"] = list(dict.fromkeys([*client_stops, *family_stops]))

        output = await self._forward_to_tinker(forward)

        # OpenClaw-RL read the reply from a decode without special tokens; read it again in the family's format.
        tokens = output.get("_raw_response_tokens")
        if tokens:
            full = self._tokenizer.decode(tokens, skip_special_tokens=False)
            calls, content = self._formatter.parse_output(full)
            choice = output["choices"][0]
            choice["message"]["content"] = content
            if calls:
                choice["message"]["tool_calls"] = calls
                choice["finish_reason"] = "tool_calls"
            else:
                # The finish reason OpenClaw-RL set is left as it is.
                choice["message"].pop("tool_calls", None)
        output["session_id"] = session_id
        return {"response": output}

    async def _forward_to_tinker(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self._evidence.admit(body)
        output = await super()._forward_to_tinker(body)
        tokens = output.get("_raw_response_tokens")
        if tokens is None:
            raise RuntimeError("the sampler's reply carries no raw tokens, so the generation cannot be recorded")
        self._evidence.dump(request, body, tokens, self._tokenizer.decode(tokens, skip_special_tokens=False))
        return output


def wait_started(server: uvicorn.Server, thread: threading.Thread, timeout_s: float, what: str) -> None:
    """Wait until uvicorn listens; it cannot bind a port in use, and then its thread ends."""
    deadline = time.monotonic() + timeout_s
    while not server.started:
        if not thread.is_alive():
            raise StartupError(f"{what} did not start listening; see the error above")
        if time.monotonic() >= deadline:
            raise StartupError(f"{what} did not start listening within {timeout_s:g} s")
        time.sleep(0.05)


async def _sampling_client(base_model: str, checkpoint: str | None, lora_rank: int) -> Any:
    import tinker

    service = tinker.ServiceClient()
    if not checkpoint:
        try:
            return await service.create_sampling_client_async(base_model=base_model)
        except Exception as error:
            raise StartupError(f"cannot sample {base_model} on Tinker: {type(error).__name__}: {error}") from error
    try:
        training = await service.create_lora_training_client_async(base_model=base_model, rank=lora_rank)
        # Tinker returns a future for the load; only awaiting its result surfaces a load that failed.
        load = await training.load_state_async(checkpoint)
        await load.result_async()
    except Exception as error:
        raise StartupError(f"cannot load the checkpoint {checkpoint}: {type(error).__name__}: {error}") from error
    logger.info("loaded the LoRA weights of %s onto %s", checkpoint, base_model)
    try:
        return await training.save_weights_and_get_sampling_client_async()
    except Exception as error:
        raise StartupError(f"cannot sample the checkpoint {checkpoint}: {type(error).__name__}: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evolve_ipi.models.server",
        description="Serve a model on Tinker, base or LoRA checkpoint, as an OpenAI-compatible endpoint.",
    )
    parser.add_argument("--family", required=True, choices=("qwen", "harmony", "inkling"))
    parser.add_argument("--base-model", required=True, help="the model's Tinker (and Hugging Face) id")
    parser.add_argument("--served-model-name", required=True, help="the model name clients use (a label)")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float, help="the sampling temperature, forced")
    parser.add_argument("--evidence-out", required=True, type=Path)
    parser.add_argument("--raw-out", required=True, type=Path)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--load-checkpoint", default=None, help="a tinker:// path of LoRA weights to serve")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    return parser


async def _serve(args: argparse.Namespace, argv: Sequence[str]) -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise StartupError("TINKER_API_KEY is not set")
    if args.load_checkpoint is not None and not args.load_checkpoint.strip():
        raise StartupError("--load-checkpoint is empty; leave it out to serve the base model")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise StartupError(f"--temperature must be a non-negative number, got {args.temperature}")
    evidence = ServingEvidence(
        args.evidence_out,
        args.raw_out,
        family=args.family,
        base_model=args.base_model,
        temperature=args.temperature,
        argv=argv,
    )
    evidence.open()
    try:
        require_free([args.port])
    except PortInUse as error:
        raise StartupError(str(error)) from error
    server = ModelServer(  # loads the tokenizer
        family=args.family,
        base_model=args.base_model,
        served_model_name=args.served_model_name,
        temperature=args.temperature,
        evidence=evidence,
        record_dir=args.record_dir,
        host=args.host,
        port=args.port,
    )
    server._sampling_client = await _sampling_client(args.base_model, args.load_checkpoint, args.lora_rank)
    server.start()
    wait_started(server._server, server._thread, _LISTEN_TIMEOUT_S, f"the HTTP server on {args.host}:{args.port}")
    logger.info("%s", ready_line(args.host, args.port, args.base_model, args.load_checkpoint))
    while server._thread.is_alive():
        await asyncio.sleep(1)
    raise StartupError("the HTTP server stopped")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the server until it is stopped; return 1 when it cannot start (or stops by itself)."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(args_list)
    try:
        asyncio.run(_serve(args, args_list))
    except StartupError as error:
        print(f"model server: error: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    exit_without_teardown(main)
