"""A scripted stand-in for the model server, for tests of the agent and of the pipeline.

It answers ``POST /v1/chat/completions`` with the turns of its script in order and records every request; once
the script has run out it answers with a plain text turn, which ends an agent's turn. It runs OpenClaw-RL's own
app, as the model server does, so headers, plain replies and streamed replies (two chunks, then ``[DONE]``) have
the model server's shape. A script is a list of turns::

    [{"tool_calls": [{"name": "read", "arguments": {"path": "report.txt"}}]},
     {"content": "The report says the login page was refactored."}]

A turn has ``content`` (text), ``tool_calls`` (each ``{"name", "arguments"}``, the arguments a dict or a JSON
string) or both.
"""

from __future__ import annotations

import copy
import json
import queue
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import uvicorn

from evolve_ipi.models.formats import new_call_id
from evolve_ipi.models.server import BaseServer, StartupError, TinkerConfig, wait_started
from evolve_ipi.procs import require_free

FINAL_TEXT = "Done."
_TURN_KEYS = {"content", "tool_calls"}
_CALL_KEYS = {"name", "arguments"}


def _check_turn(number: int, turn: Mapping[str, Any]) -> dict[str, Any]:
    where = f"script turn {number}"
    unknown = set(turn) - _TURN_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; a turn has content and/or tool_calls")
    if not _TURN_KEYS & set(turn):
        raise ValueError(f"{where} has neither content nor tool_calls")
    if "content" in turn and not isinstance(turn["content"], str):
        raise TypeError(f"{where}: content must be a string")
    calls = turn.get("tool_calls", [])
    if not isinstance(calls, list):
        raise TypeError(f"{where}: tool_calls must be a list")
    if "tool_calls" in turn and not calls:
        raise ValueError(f"{where}: tool_calls is empty; leave it out for a text turn")
    for call in calls:
        if not isinstance(call, Mapping):
            raise TypeError(f"{where}: a tool call must be a mapping")
        unknown = set(call) - _CALL_KEYS
        if unknown:
            raise ValueError(f"{where}: a tool call has unknown key(s) {sorted(unknown)}")
        if not isinstance(call.get("name"), str) or not call["name"]:
            raise ValueError(f"{where}: a tool call needs a name")
        if not isinstance(call.get("arguments", {}), dict | str):
            raise TypeError(f"{where}: tool-call arguments must be a dict or a JSON string")
    return copy.deepcopy(dict(turn))


class _ScriptedServer(BaseServer):
    """OpenClaw-RL's app with the script's turns in place of Tinker."""

    _TITLE = "scripted model"

    def __init__(self, model: FakeModel, config: TinkerConfig) -> None:
        self._model = model
        submission_open = threading.Event()
        submission_open.set()
        super().__init__(config, queue.Queue(), submission_open)

    def _load_tokenizer(self) -> None:
        return None

    async def _handle_request(
        self, body: dict[str, Any], session_id: str, turn_type: str, session_done: bool
    ) -> dict[str, Any]:
        return {"response": self._model._answer(body, session_id, turn_type, session_done)}


class FakeModel:
    """An OpenAI-compatible HTTP server (``/v1/chat/completions``, ``/healthz``) that plays a script."""

    def __init__(
        self, script: Sequence[Mapping[str, Any]], *, final_text: str = FINAL_TEXT, model_name: str = "fake-model"
    ) -> None:
        self._turns = [_check_turn(number, turn) for number, turn in enumerate(script, start=1)]
        self.final_text = final_text
        self.model_name = model_name
        self.port: int | None = None
        self._requests: list[dict[str, Any]] = []
        self._answered = 0
        self._lock = threading.Lock()
        self._http: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._records: tempfile.TemporaryDirectory[str] | None = None

    @property
    def requests(self) -> list[dict[str, Any]]:
        """Every request so far: ``{"body", "session_id", "turn_type", "session_done"}``, in order."""
        with self._lock:
            return copy.deepcopy(self._requests)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _answer(self, body: dict[str, Any], session_id: str, turn_type: str, session_done: bool) -> dict[str, Any]:
        with self._lock:
            self._requests.append(
                {
                    "body": copy.deepcopy(body),
                    "session_id": session_id,
                    "turn_type": turn_type,
                    "session_done": session_done,
                }
            )
            turn = self._turns[self._answered] if self._answered < len(self._turns) else {"content": self.final_text}
            self._answered += 1
            number = self._answered
        message: dict[str, Any] = {"role": "assistant", "content": turn.get("content", "")}
        calls = [
            {
                "id": new_call_id(),
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call.get("arguments", {})
                    if isinstance(call.get("arguments", {}), str)
                    else json.dumps(call.get("arguments", {}), ensure_ascii=False),
                },
            }
            for call in turn.get("tool_calls") or []
        ]
        if calls:
            message["tool_calls"] = calls
        return {
            "id": f"chatcmpl-fake-{number}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", self.model_name),
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "session_id": session_id,
        }

    def start(self, port: int) -> FakeModel:
        """Listen on 127.0.0.1:``port``, which must be free."""
        if self._thread is not None:
            raise RuntimeError("this fake model was already started")
        require_free([port])
        self._records = tempfile.TemporaryDirectory(prefix="fake-model-")
        config = TinkerConfig(
            model_name=self.model_name,
            served_model_name=self.model_name,
            proxy_host="127.0.0.1",
            proxy_port=port,
            record_dir=self._records.name,
        )
        app = _ScriptedServer(self, config).app
        # No log_config: uvicorn would otherwise reconfigure the logging of the whole process.
        self._http = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, log_level="warning", lifespan="off")
        )
        self._thread = threading.Thread(target=self._http.run, name=f"fake-model-{port}", daemon=True)
        self._thread.start()
        try:
            wait_started(self._http, self._thread, 30, f"the fake model on 127.0.0.1:{port}")
        except StartupError:
            self.stop()
            raise
        self.port = port
        return self

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        if self._http is not None:
            self._http.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                raise RuntimeError(f"the fake model on port {self.port} did not stop")
        if self._records is not None:
            self._records.cleanup()
            self._records = None


@contextmanager
def serve(script: Sequence[Mapping[str, Any]], port: int, **options: Any) -> Iterator[FakeModel]:
    """A :class:`FakeModel` listening on ``port`` for the duration of the ``with`` block."""
    model = FakeModel(script, **options).start(port)
    try:
        yield model
    finally:
        model.stop()
