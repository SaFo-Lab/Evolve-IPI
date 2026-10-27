"""The scripted fake model that stands in for the model server in tests of the agent and the pipeline."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from evolve_ipi.models import fake
from evolve_ipi.models.fake import FakeModel
from evolve_ipi.procs import PortInUse

REQUEST: dict[str, Any] = {
    "model": "policy",
    "messages": [{"role": "user", "content": "Please read report.txt."}],
    "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
}


def post(model: FakeModel, body: dict[str, Any], **headers: str) -> httpx.Response:
    return httpx.post(f"{model.base_url}/chat/completions", json=body, headers=headers, timeout=30)


def test_the_fake_model_returns_the_scripted_tool_call_in_openai_shape(free_port: Callable[[], int]) -> None:
    script = [
        {"tool_calls": [{"name": "read", "arguments": {"path": "report.txt"}}]},
        {"content": "The report says the login page was refactored."},
    ]
    with fake.serve(script, free_port()) as model:
        first = post(model, REQUEST, **{"X-Session-Id": "s1", "X-Turn-Type": "main"}).json()
        second = post(model, {**REQUEST, "session_id": "s2"}).json()
        third = post(model, REQUEST).json()

    assert first["object"] == "chat.completion" and first["model"] == "policy"
    (choice,) = first["choices"]
    call = choice["message"]["tool_calls"][0]
    assert choice["message"] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call["id"], "type": "function", "function": {"name": "read", "arguments": '{"path": "report.txt"}'}}
        ],
    }
    assert call["id"].startswith("call") and len(call["id"]) == 16
    assert choice["finish_reason"] == "tool_calls"

    assert second["choices"][0]["message"] == {
        "role": "assistant",
        "content": "The report says the login page was refactored.",
    }
    assert second["choices"][0]["finish_reason"] == "stop"

    # The script ran out: a text turn ends the episode.
    assert third["choices"][0]["message"] == {"role": "assistant", "content": fake.FINAL_TEXT}
    assert third["choices"][0]["finish_reason"] == "stop"

    assert [r["body"]["messages"] for r in model.requests] == [REQUEST["messages"]] * 3
    assert [(r["session_id"], r["turn_type"], r["session_done"]) for r in model.requests] == [
        ("s1", "main", False),
        ("s2", "side", False),
        ("unknown", "side", False),
    ]


def test_arguments_given_as_a_string_are_sent_as_written(free_port: Callable[[], int]) -> None:
    script = [{"content": "Let me look.", "tool_calls": [{"name": "exec", "arguments": '{"command": "ls"}'}]}]
    with fake.serve(script, free_port(), final_text="bye") as model:
        reply = post(model, REQUEST).json()
        after = post(model, REQUEST).json()
    message = reply["choices"][0]["message"]
    assert message["content"] == "Let me look."
    assert message["tool_calls"][0]["function"] == {"name": "exec", "arguments": '{"command": "ls"}'}
    assert after["choices"][0]["message"]["content"] == "bye"


def test_a_streamed_answer_has_the_model_servers_shape(free_port: Callable[[], int]) -> None:
    script = [{"tool_calls": [{"name": "read", "arguments": {"path": "a.txt"}}]}]
    with fake.serve(script, free_port()) as model:
        response = post(model, {**REQUEST, "stream": True})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line.removeprefix("data: ") for line in response.text.split("\n\n") if line]
    assert len(events) == 3 and events[2] == "[DONE]"
    first, second = json.loads(events[0]), json.loads(events[1])
    assert first["object"] == "chat.completion.chunk"
    delta = first["choices"][0]["delta"]
    assert delta["role"] == "assistant" and delta["content"] == ""
    assert delta["tool_calls"][0]["function"] == {"name": "read", "arguments": '{"path": "a.txt"}'}
    assert second["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "tool_calls"}


def test_health_check(free_port: Callable[[], int]) -> None:
    with fake.serve([], free_port()) as model:
        assert model.alive()
        assert httpx.get(f"http://127.0.0.1:{model.port}/healthz", timeout=10).json() == {"ok": True}


def test_serve_stops_the_server_and_frees_the_port(free_port: Callable[[], int]) -> None:
    port = free_port()
    with fake.serve([], port) as model:
        pass
    assert not model.alive()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
    with fake.serve([], port):  # the port can be used again
        pass


def test_serve_refuses_a_busy_port(listener: int) -> None:
    with pytest.raises(PortInUse, match=rf"pid {os.getpid()}"), fake.serve([], listener):
        pass


@pytest.mark.parametrize(
    ("turn", "match"),
    [
        ({"text": "hi"}, "unknown key"),
        ({}, "neither content nor tool_calls"),
        ({"tool_calls": []}, "tool_calls is empty"),
        ({"tool_calls": [{"arguments": {}}]}, "name"),
        ({"tool_calls": [{"name": "read", "arguments": {}, "id": "x"}]}, "unknown key"),
    ],
)
def test_the_script_is_checked(turn: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FakeModel([turn])


@pytest.mark.parametrize(
    ("turn", "match"),
    [
        ({"content": 7}, "content must be a string"),
        ({"tool_calls": {"name": "read"}}, "tool_calls must be a list"),
        ({"tool_calls": ["read"]}, "a tool call must be a mapping"),
        ({"tool_calls": [{"name": "read", "arguments": 3}]}, "arguments must be a dict or a JSON string"),
    ],
)
def test_the_script_types_are_checked(turn: dict[str, Any], match: str) -> None:
    with pytest.raises(TypeError, match=match):
        FakeModel([turn])


def test_an_empty_text_turn_is_an_empty_reply(free_port: Callable[[], int]) -> None:
    with fake.serve([{"content": ""}], free_port()) as model:
        reply = post(model, REQUEST).json()
    assert reply["choices"][0]["message"] == {"role": "assistant", "content": ""}
    assert reply["choices"][0]["finish_reason"] == "stop"
