"""Starting the model server as a child process of a run."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from evolve_ipi.models.formats import Family
from evolve_ipi.procs import ManagedProcess, require_free

HOST = "127.0.0.1"
READY_TIMEOUT_S = 900  # a large checkpoint takes minutes to load
PORT_TIMEOUT_S = 60


def ready_line(host: str, port: int, base_model: str, checkpoint: str | None) -> str:
    """The line the model server prints once it listens; :func:`start_model_server` waits for exactly this line."""
    serving = f"checkpoint:{checkpoint}" if checkpoint else f"base:{base_model}"
    return f"EVAL PROXY READY on {host}:{port} serving {serving} (thinking ON)"


@dataclass(frozen=True)
class ServerFiles:
    """What a model server writes into its directory."""

    log: Path  # its output
    evidence: Path  # the evidence JSON: settings, forwarded requests, refused ones
    raw: Path  # one line per generation, the reasoning included
    records: Path  # OpenClaw-RL's record directory (left empty when only sampling)


def server_files(out_dir: Path) -> ServerFiles:
    out_dir = Path(out_dir)
    return ServerFiles(
        log=out_dir / "model_server.log",
        evidence=out_dir / "evidence.json",
        raw=out_dir / "raw.jsonl",
        records=out_dir / "records",
    )


def start_model_server(
    family: Family,
    base_model: str,
    served_name: str,
    port: int,
    out_dir: Path,
    checkpoint: str | None,
    env: Mapping[str, str],
    *,
    temperature: float,
) -> ManagedProcess:
    """Start ``python -m evolve_ipi.models.server`` for one model and wait until it serves on ``port``.

    It serves ``checkpoint`` (a ``tinker://`` path of LoRA weights) or, when that is ``None``, the base model, and
    writes the files of :func:`server_files` into ``out_dir``, which must not hold a raw dump yet. ``env`` is the
    server's whole environment: it must hold ``TINKER_API_KEY``, and ``HOME`` or ``HF_HOME`` for the Hugging Face
    cache (with ``HF_TOKEN`` for a gated tokenizer). Returns once the server has printed its ready
    line and holds the port; raises ``PortInUse`` before starting anything when the port is taken, and
    ``ProcessError`` with the end of the server's log when it does not get ready (the server is stopped then).
    """
    require_free([port])
    files = server_files(out_dir)
    argv = [
        sys.executable, "-m", "evolve_ipi.models.server",
        "--family", family,
        "--base-model", base_model,
        "--served-model-name", served_name,
        "--port", str(port),
        "--temperature", str(temperature),
        "--evidence-out", str(files.evidence),
        "--raw-out", str(files.raw),
        "--record-dir", str(files.records),
        "--host", HOST,
    ]  # fmt: skip
    if checkpoint is not None:
        argv += ["--load-checkpoint", checkpoint]
    process = ManagedProcess("model server", argv, {**env, "PYTHONUNBUFFERED": "1"}, files.log)
    process.start()
    try:
        process.wait_for_line(ready_line(HOST, port, base_model, checkpoint), READY_TIMEOUT_S)
        process.wait_for_port(port, PORT_TIMEOUT_S)
    except BaseException:
        process.stop()
        raise
    return process
