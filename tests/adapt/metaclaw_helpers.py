"""Helpers of the MetaClaw tests: the MetaClaw Python where it is installed, and synthetic evolution records."""

from __future__ import annotations

import functools
import json
import os
import subprocess
import tempfile
import threading
from collections.abc import Iterable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest

from evolve_ipi.paths import RuntimePaths

ADAPT_DIR = Path(__file__).resolve().parents[2] / "src" / "evolve_ipi" / "adapt"
GUARDS = ADAPT_DIR / "metaclaw_guards.py"
PROXY = ADAPT_DIR / "metaclaw_proxy.py"
WINDOW = 300


@functools.cache
def _find_metaclaw_python() -> tuple[Path | None, str]:
    python = RuntimePaths().metaclaw_python
    if not python.exists():
        return None, f"no MetaClaw Python at {python}; set EVOLVE_IPI_METACLAW_PYTHON to run this test"
    with tempfile.TemporaryDirectory() as home:
        done = subprocess.run(
            [str(python), "-c", "import metaclaw.api_server"],
            capture_output=True,
            text=True,
            timeout=300,
            env=metaclaw_env(Path(home)),
            check=False,
        )
    if done.returncode != 0:
        return None, f"{python} cannot import metaclaw: {done.stderr.strip()[-300:]}"
    return python, ""


def metaclaw_python() -> Path:
    """The Python of the MetaClaw environment; the test is skipped where there is none."""
    python, reason = _find_metaclaw_python()
    if python is None:
        pytest.skip(reason)
    return python


def metaclaw_env(home: Path, **extra: str) -> dict[str, str]:
    """A small environment for a process of the MetaClaw environment."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "PYTHONSAFEPATH": "1"}
    env.update(extra)
    return env


# Adapted from MetaClaw metaclaw/api_server.py (MIT); see THIRD_PARTY_NOTICES.md.
def flatten(content: Any) -> str:
    """MetaClaw's own rule for the text of a message (``_flatten_message_content``), for in-process tests."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def conversation_of(messages: Iterable[Mapping[str, Any]]) -> str:
    """What MetaClaw records as a turn's conversation: every message but the system prompt."""
    return "\n".join(
        f"{m.get('role', '?')}: {flatten(m.get('content', ''))}" for m in messages if m.get("role") != "system"
    )


def prompt_of(messages: Iterable[Mapping[str, Any]]) -> str:
    """The whole model input of a turn, system prompt included: what an unpatched evolver excerpted."""
    return "\n".join(f"{m.get('role', '?')}: {flatten(m.get('content', ''))}" for m in messages)


def event(fed: int, prompts: list[str] | None = None, skills: int = 1) -> dict[str, Any]:
    """One line of an evolution history: ``fed`` turns analyzed, ``prompts`` the excerpts it rendered."""
    excerpts = [f"turn {i} excerpt" for i in range(fed)] if prompts is None else prompts
    names = [f"skill-{i}" for i in range(skills)]
    return {
        "timestamp": "2026-01-01T00:00:00",
        "num_failures_analyzed": fed,
        "num_skills_generated": skills,
        "skill_names": names,
        "skills": [{"name": n, "category": "general", "description": "d", "content": "c"} for n in names],
        "failure_prompts": excerpts,
    }


def write_jsonl(path: Path, rows: Iterable[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def turn_record(messages: list[dict[str, Any]], session: str = "s1", turn: int = 1) -> dict[str, Any]:
    """A line of MetaClaw's ``conversations.jsonl``; the guard reads only ``messages``."""
    return {"session_id": session, "turn": turn, "messages": messages, "prompt_text": prompt_of(messages)}


SYSTEM = "You are a personal assistant running inside OpenClaw. " * 20 + "Runtime: channel=webchat | capabilities=none"


def short_turn(user: str = "Summarize report.md for me.") -> list[dict[str, Any]]:
    """A first turn: the system prompt and one short user message."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def long_turn() -> list[dict[str, Any]]:
    """A turn whose conversation is longer than the excerpt window."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": "Please read the notes. " * 10}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "line of the document\n" * 30},
    ]


ANSWER = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "stub-model-2026-01-01",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
}


class OpenAIStub:
    """A local stand-in for an OpenAI-compatible endpoint: ``POST /v1/chat/completions`` answers ``status`` with
    ``reply``, and every request is kept (path, lower-case headers, JSON body)."""

    def __init__(self, status: int = 200, reply: Mapping[str, Any] = ANSWER) -> None:
        self.status, self.reply = status, reply
        self.requests: list[dict[str, Any]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                headers = {name.lower(): value for name, value in self.headers.items()}
                stub.requests.append({"path": self.path, "headers": headers, "body": json.loads(body)})
                answer = json.dumps(stub.reply).encode()
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.server.shutdown()
        self.server.server_close()
