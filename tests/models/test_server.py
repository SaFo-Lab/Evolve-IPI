"""The model server: what it forwards to Tinker, what it hands back, what it records, and how it starts.

Tinker and the tokenizer are stubbed (``tinker_stubs.py``): in-process tests patch them, process tests install
them through ``stub_site/sitecustomize.py``. No test reaches either service.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import tinker
import transformers
from tinker_stubs import FakeAutoTokenizer, FakeSamplingClient, FakeService, child_env

from evolve_ipi.models import launch
from evolve_ipi.models import server as model_server
from evolve_ipi.models.server import ModelServer, ServingEvidence, StartupError, TemperatureMismatch, ready_line
from evolve_ipi.procs import PortInUse, ProcessError

BASE = "test/base-model"
CHECKPOINT = "tinker://0000-test:train:0/weights/step_0032"

# What OpenClaw sends: no temperature, no max_tokens, no stop.
OPENCLAW_BODY: dict[str, Any] = {
    "model": "policy",
    "messages": [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "Please read report.txt and tell me what it says."},
    ],
    "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
    "max_completion_tokens": 4096,
    "reasoning_effort": "low",
    "store": False,
    "stream": False,
}

HARMONY_CALL = (
    "<|channel|>analysis<|message|>Need to list the files.<|end|>"
    '<|start|>assistant<|channel|>commentary to=functions.exec <|constrain|>json<|message|>{"command": "ls"}<|call|>'
)


@pytest.fixture
def auto_tokenizer(monkeypatch: pytest.MonkeyPatch) -> FakeAutoTokenizer:
    auto = FakeAutoTokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", auto.from_pretrained)
    return auto


class AppClient:
    """Requests to an ASGI app in this process; an exception inside the app is raised here."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def post(self, path: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://model-server") as client:
                return await client.post(path, json=json, headers=headers)

        return asyncio.run(send())


class Served:
    """A model server in this process, driven through its ASGI app."""

    def __init__(
        self, tmp_path: Path, auto: FakeAutoTokenizer, family: str, outputs: list[str], expected_temperature: float
    ):
        self.evidence_path, self.raw_path = tmp_path / "evidence.json", tmp_path / "raw.jsonl"
        self.evidence = ServingEvidence(
            self.evidence_path,
            self.raw_path,
            family=family,
            base_model=BASE,
            temperature=expected_temperature,
            argv=["--family", family, "--base-model", BASE],
        )
        self.evidence.open()
        self.tokenizer = auto.tokenizer
        self.sampler = FakeSamplingClient(auto.tokenizer, list(outputs))
        self.server = ModelServer(
            family=family,
            base_model=BASE,
            served_model_name="policy",
            temperature=0.6,
            evidence=self.evidence,
            record_dir=tmp_path / "records",
            sampling_client=self.sampler,
        )
        self.client = AppClient(self.server.app)

    def post(self, body: dict[str, Any], **headers: str) -> dict[str, Any]:
        response = self.client.post("/v1/chat/completions", json=body, headers=headers)
        assert response.status_code == 200, response.text
        return response.json()

    def params(self, index: int = -1) -> Any:
        return self.sampler.samples[index]["sampling_params"]

    def read_evidence(self) -> dict[str, Any]:
        return json.loads(self.evidence_path.read_text())

    def dump(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.raw_path.read_text().splitlines()]


@pytest.fixture
def serve(tmp_path: Path, auto_tokenizer: FakeAutoTokenizer):
    def make(family: str = "qwen", outputs: list[str] | None = None, expected_temperature: float = 0.6) -> Served:
        """A server that forces 0.6; its evidence expects ``expected_temperature``."""
        return Served(tmp_path, auto_tokenizer, family, outputs or [], expected_temperature)

    return make


def message(reply: dict[str, Any]) -> dict[str, Any]:
    return reply["choices"][0]["message"]


# --- sampling parameters -----------------------------------------------------------------------------


def test_temperature_is_forced_when_the_client_sends_none(serve) -> None:
    served = serve()
    served.post(OPENCLAW_BODY)
    assert served.params().temperature == 0.6
    assert served.read_evidence()["requests"][0]["temperature"] == 0.6


def test_a_temperature_from_the_client_is_replaced(serve) -> None:
    served = serve()
    served.post({**OPENCLAW_BODY, "temperature": 1.3})
    assert served.params().temperature == 0.6


def test_a_temperature_violation_raises_and_is_recorded(serve) -> None:
    served = serve()
    with pytest.raises(TemperatureMismatch, match="0.7"):
        asyncio.run(served.server._forward_to_tinker({**OPENCLAW_BODY, "temperature": 0.7}))
    with pytest.raises(TemperatureMismatch, match="None"):
        asyncio.run(served.server._forward_to_tinker(dict(OPENCLAW_BODY)))
    evidence = served.read_evidence()
    assert evidence["violations"] == [
        {"seen_temperature": 0.7, "expected": 0.6, "request_index": 0},
        {"seen_temperature": None, "expected": 0.6, "request_index": 0},
    ]
    assert evidence["requests_forwarded"] == 0
    assert served.sampler.samples == []
    assert served.raw_path.read_text() == ""


def test_a_server_that_forwards_another_temperature_refuses_the_request(serve) -> None:
    served = serve(expected_temperature=0.5)  # the server forces 0.6
    with pytest.raises(TemperatureMismatch):
        served.client.post("/v1/chat/completions", json=OPENCLAW_BODY)
    assert served.read_evidence()["violations"] == [{"seen_temperature": 0.6, "expected": 0.5, "request_index": 0}]
    assert served.sampler.samples == []


def test_max_completion_tokens_and_reasoning_effort_do_not_reach_the_sampler(serve) -> None:
    served = serve()
    served.post(OPENCLAW_BODY)
    params = served.params()
    assert (params.max_tokens, params.top_k, params.top_p, params.stop) == (2048, 50, 0.95, None)
    sample = served.sampler.samples[-1]
    assert (sample["num_samples"], sample["include_prompt_logprobs"], sample["topk_prompt_logprobs"]) == (1, False, 0)
    template = served.tokenizer.templates[-1]
    assert template["kwargs"] == {}  # no reasoning_effort for the chat template
    assert (template["tokenize"], template["add_generation_prompt"]) == (False, True)
    assert template["tools"] == OPENCLAW_BODY["tools"]


def test_a_max_tokens_from_the_client_is_kept(serve) -> None:
    served = serve()
    served.post({**OPENCLAW_BODY, "max_tokens": 1})
    assert served.params().max_tokens == 1


@pytest.mark.parametrize(
    ("client_stop", "forwarded"),
    [
        (None, ["<|call|>", "<|return|>"]),
        ("END", ["END", "<|call|>", "<|return|>"]),
        (["END", "<|return|>"], ["END", "<|return|>", "<|call|>"]),
    ],
)
def test_harmony_stops_follow_the_client_stops(serve, client_stop: Any, forwarded: list[str]) -> None:
    served = serve("harmony")
    body = dict(OPENCLAW_BODY) if client_stop is None else {**OPENCLAW_BODY, "stop": client_stop}
    served.post(body)
    assert list(served.params().stop) == forwarded
    assert served.read_evidence()["requests"][0]["stop"] == forwarded
    assert served.server._formatter.stop_sequences == ["<|call|>", "<|return|>"]  # the formatter's list is untouched


def test_inkling_stop_is_added(serve) -> None:
    served = serve("inkling")
    served.post(OPENCLAW_BODY)
    assert list(served.params().stop) == ["<|content_model_end_sampling|>"]


@pytest.mark.parametrize("client_stop", [None, "END", ["END", "STOP"]])
def test_qwen_passes_the_client_stop_unchanged(serve, client_stop: Any) -> None:
    served = serve("qwen")
    body = dict(OPENCLAW_BODY) if client_stop is None else {**OPENCLAW_BODY, "stop": client_stop}
    served.post(body)
    assert served.params().stop == client_stop


# --- what comes back ----------------------------------------------------------------------------------


def test_a_qwen_call_inside_the_reasoning_is_not_executed(serve) -> None:
    reasoning_call = '<tool_call>{"name": "exec", "arguments": {"command": "rm -rf ~"}}</tool_call>'
    served = serve(outputs=[f"I could run {reasoning_call} but I should not.</think>\n\nI will not run that."])
    reply = served.post(OPENCLAW_BODY)
    assert "tool_calls" not in message(reply)
    assert message(reply)["content"] == "I will not run that."
    # OpenClaw-RL's own parse (full text, special tokens skipped) saw the call and set the finish reason; the
    # server replaces the calls and the content but keeps that finish reason, as OpenClaw-RL's reply does.
    assert reply["choices"][0]["finish_reason"] == "tool_calls"


def test_a_qwen_call_after_the_reasoning_is_returned(serve) -> None:
    call = "<tool_call>\n<function=read>\n<parameter=path>\nreport.txt\n</parameter>\n</function>\n</tool_call>"
    served = serve(outputs=[f"The user wants the file.</think>\n\n{call}<|im_end|>"])
    reply = served.post(OPENCLAW_BODY)
    calls = message(reply)["tool_calls"]
    assert [(c["type"], c["function"]) for c in calls] == [
        ("function", {"name": "read", "arguments": '{"path": "report.txt"}'})
    ]
    assert calls[0]["id"].startswith("call") and len(calls[0]["id"]) == 16
    assert message(reply)["content"] == ""
    assert reply["choices"][0]["finish_reason"] == "tool_calls"


def test_a_plain_answer_keeps_the_samplers_finish_reason(serve) -> None:
    served = serve(outputs=["Done thinking.</think>\n\nThe report says the login page was refactored.<|im_end|>"])
    reply = served.post(OPENCLAW_BODY)
    assert message(reply) == {"role": "assistant", "content": "The report says the login page was refactored."}
    assert reply["choices"][0]["finish_reason"] == "stop"


def test_a_harmony_call_is_read_from_the_raw_tokens(serve) -> None:
    served = serve("harmony", outputs=[HARMONY_CALL])
    reply = served.post(OPENCLAW_BODY)
    # Its markers are special tokens: OpenClaw-RL's decode skips them and would find no call.
    assert served.tokenizer.decode(served.tokenizer.encode(HARMONY_CALL), skip_special_tokens=True).count("<|") == 0
    assert [c["function"] for c in message(reply)["tool_calls"]] == [{"name": "exec", "arguments": '{"command": "ls"}'}]
    assert reply["choices"][0]["finish_reason"] == "tool_calls"
    assert message(reply)["content"] == ""


def test_the_history_is_normalized_for_the_chat_template(serve) -> None:
    served = serve("harmony")
    history = [
        *OPENCLAW_BODY["messages"],
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a.txt"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call1", "content": "hello"},
    ]
    served.post({**OPENCLAW_BODY, "messages": history})
    sent = served.tokenizer.templates[-1]["messages"]
    assert sent[2]["tool_calls"][0]["function"]["arguments"] == {"path": "a.txt"}
    assert sent[2]["content"] == ""


def test_the_session_fields_and_stream_options_are_not_forwarded(serve) -> None:
    served = serve()
    served.post(
        {**OPENCLAW_BODY, "session_id": "s1", "session_done": True, "turn_type": "main", "stream_options": {"x": 1}}
    )
    forwarded = served.read_evidence()["requests"][0]
    assert not {"session_id", "session_done", "turn_type", "stream_options"} & set(forwarded)
    assert forwarded["stream"] is False


def test_the_reply_names_the_session(serve) -> None:
    served = serve()
    reply = served.post(OPENCLAW_BODY, **{"X-Session-Id": "session-42", "X-Turn-Type": "main"})
    assert reply["session_id"] == "session-42"


def test_a_streamed_reply_has_two_chunks_and_done(serve) -> None:
    served = serve("harmony", outputs=[HARMONY_CALL])
    response = served.client.post("/v1/chat/completions", json={**OPENCLAW_BODY, "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line.removeprefix("data: ") for line in response.text.split("\n\n") if line]
    assert len(events) == 3 and events[2] == "[DONE]"
    first, second = json.loads(events[0]), json.loads(events[1])
    delta = first["choices"][0]["delta"]
    assert delta["role"] == "assistant" and delta["content"] == ""
    assert [c["function"]["name"] for c in delta["tool_calls"]] == ["exec"]
    assert first["choices"][0]["finish_reason"] is None
    assert second["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
    assert served.params().temperature == 0.6  # the forwarded request did not stream


def test_messages_must_be_a_non_empty_list(serve) -> None:
    served = serve()
    response = served.client.post("/v1/chat/completions", json={**OPENCLAW_BODY, "messages": []})
    assert response.status_code == 400
    assert served.sampler.samples == []


# --- the record of what was forwarded ---------------------------------------------------------------


def test_the_raw_dump_has_one_line_per_completion(serve) -> None:
    served = serve("harmony", outputs=[HARMONY_CALL, "<|channel|>final<|message|>Done.<|return|>"])
    served.post(OPENCLAW_BODY)
    served.post({**OPENCLAW_BODY, "messages": [*OPENCLAW_BODY["messages"], {"role": "user", "content": "thanks"}]})
    lines = served.dump()
    assert [list(line) for line in lines] == [["at", "request", "messages", "n_tokens", "raw"]] * 2
    assert [line["request"] for line in lines] == [1, 2]
    assert [line["messages"] for line in lines] == [2, 3]
    assert lines[0]["raw"] == HARMONY_CALL  # special tokens kept
    assert lines[0]["n_tokens"] == len(served.tokenizer.encode(HARMONY_CALL))
    assert lines[1]["raw"] == "<|channel|>final<|message|>Done.<|return|>"


def test_the_evidence_records_the_first_three_requests(serve) -> None:
    served = serve()
    for _ in range(4):
        served.post({**OPENCLAW_BODY, "session_id": "s1"})
    evidence = served.read_evidence()
    assert list(evidence) == [
        "experiment",
        "family",
        "base_model",
        "expected_temperature",
        "argv",
        "requests",
        "requests_forwarded",
        "raw_dump",
        "written_at",
    ]
    assert evidence["experiment"] == "evolve-ipi"
    assert (evidence["family"], evidence["base_model"], evidence["expected_temperature"]) == ("qwen", BASE, 0.6)
    assert evidence["argv"] == ["--family", "qwen", "--base-model", BASE]
    assert evidence["raw_dump"] == str(served.raw_path.resolve())
    assert evidence["requests_forwarded"] == 4
    assert len(evidence["requests"]) == 3
    assert evidence["requests"][0] == {
        "model": "policy",
        "max_completion_tokens": 4096,
        "reasoning_effort": "low",
        "store": False,
        "stream": False,
        "temperature": 0.6,
        "_messages_count": 2,
        "_messages_chars": len("You are a helpful agent.") + len("Please read report.txt and tell me what it says."),
        "_has_tools": True,
    }
    assert len(served.dump()) == 4


def test_the_evidence_refuses_a_raw_file_that_is_not_empty(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"request": 1}\n')
    evidence = ServingEvidence(
        tmp_path / "evidence.json", raw, family="qwen", base_model=BASE, temperature=0.6, argv=[]
    )
    with pytest.raises(StartupError, match="already holds 15 bytes"):
        evidence.open()
    assert raw.read_text() == '{"request": 1}\n'
    assert not (tmp_path / "evidence.json").exists()


def test_the_ready_line_is_exact() -> None:
    assert (
        ready_line("127.0.0.1", 30013, "Qwen/Qwen3.8-27B", None)
        == "EVAL PROXY READY on 127.0.0.1:30013 serving base:Qwen/Qwen3.8-27B (thinking ON)"
    )
    assert (
        ready_line("127.0.0.1", 30012, "openai/gpt-oss-120b", CHECKPOINT)
        == f"EVAL PROXY READY on 127.0.0.1:30012 serving checkpoint:{CHECKPOINT} (thinking ON)"
    )


# --- starting up: main() in this process (every path here fails before the server would listen) ----------


def server_argv(tmp_path: Path, port: int, *extra: str, family: str = "qwen") -> list[str]:
    return [
        "--family", family,
        "--base-model", BASE,
        "--served-model-name", "policy",
        "--port", str(port),
        "--temperature", "0.6",
        "--evidence-out", str(tmp_path / "evidence.json"),
        "--raw-out", str(tmp_path / "raw.jsonl"),
        "--record-dir", str(tmp_path / "records"),
        *extra,
    ]  # fmt: skip


@pytest.fixture
def never_listens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every main() here must fail before the server listens. One that would serve fails the test at once, instead
    of serving in the test process for ever."""

    def refuse(self: ModelServer) -> None:
        raise AssertionError("the model server started to listen, but this start should have failed earlier")

    monkeypatch.setattr(ModelServer, "start", refuse)


@pytest.fixture
def stub_service(
    monkeypatch: pytest.MonkeyPatch, auto_tokenizer: FakeAutoTokenizer, never_listens: None
) -> FakeService:
    monkeypatch.setenv("TINKER_API_KEY", "test-key-not-real")
    service = FakeService(tokenizer=auto_tokenizer.tokenizer)
    monkeypatch.setattr(tinker, "ServiceClient", service)
    return service


def test_main_needs_the_tinker_key(
    tmp_path: Path, never_listens: None, capsys: pytest.CaptureFixture[str], free_port: Callable[[], int]
) -> None:
    assert model_server.main(server_argv(tmp_path, free_port())) == 1
    assert "TINKER_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "evidence.json").exists()


def test_main_refuses_a_raw_file_that_is_not_empty(
    tmp_path: Path,
    stub_service: FakeService,
    auto_tokenizer: FakeAutoTokenizer,
    capsys: pytest.CaptureFixture[str],
    free_port: Callable[[], int],
) -> None:
    (tmp_path / "raw.jsonl").write_text("earlier\n")
    assert model_server.main(server_argv(tmp_path, free_port())) == 1
    assert "already holds" in capsys.readouterr().err
    assert (tmp_path / "raw.jsonl").read_text() == "earlier\n"
    assert auto_tokenizer.loaded == [] and stub_service.created == []


def test_a_tokenizer_failure_stops_the_start_before_tinker(
    tmp_path: Path,
    stub_service: FakeService,
    auto_tokenizer: FakeAutoTokenizer,
    capsys: pytest.CaptureFixture[str],
    free_port: Callable[[], int],
) -> None:
    auto_tokenizer.error = OSError(
        "You are trying to access a gated repo. Access to model test/base-model is restricted."
    )
    assert model_server.main(server_argv(tmp_path, free_port())) == 1
    err = capsys.readouterr().err
    assert "cannot load the tokenizer of test/base-model" in err
    assert "gated repo" in err
    assert "EVAL PROXY READY" not in err
    assert auto_tokenizer.loaded == [(BASE, {"trust_remote_code": True})]
    assert stub_service.created == []


def test_the_base_model_is_sampled_through_a_sampling_client(
    tmp_path: Path, stub_service: FakeService, free_port: Callable[[], int]
) -> None:
    """Without --load-checkpoint the server samples the base model itself: no training client, no load."""
    with pytest.raises(AssertionError, match="started to listen"):  # never_listens stops it right there
        model_server.main(server_argv(tmp_path, free_port()))
    assert stub_service.created == [("sampling", {"base_model": BASE})]
    assert stub_service.training_clients == []
    client = asyncio.run(model_server._sampling_client(BASE, None, 32))
    assert client is stub_service.base
    assert stub_service.created[-1] == ("sampling", {"base_model": BASE})


def test_a_checkpoint_load_failure_stops_the_start(
    tmp_path: Path, stub_service: FakeService, capsys: pytest.CaptureFixture[str], free_port: Callable[[], int]
) -> None:
    stub_service.load_error = f"checkpoint {CHECKPOINT} not found"
    assert model_server.main(server_argv(tmp_path, free_port(), "--load-checkpoint", CHECKPOINT)) == 1
    err = capsys.readouterr().err
    assert f"cannot load the checkpoint {CHECKPOINT}" in err and "not found" in err
    assert stub_service.created == [("training", {"base_model": BASE, "rank": 32})]
    (training,) = stub_service.training_clients
    assert [path for path, _ in training.loads] == [CHECKPOINT]
    assert training.loads[0][1].waited_for  # the load's future was awaited: that is how the failure surfaced
    assert not training.saved


def test_main_refuses_a_busy_port(
    tmp_path: Path, stub_service: FakeService, listener: int, capsys: pytest.CaptureFixture[str]
) -> None:
    assert model_server.main(server_argv(tmp_path, listener)) == 1
    assert f"pid {os.getpid()}" in capsys.readouterr().err
    assert stub_service.created == []


def test_main_refuses_an_empty_checkpoint_path(
    tmp_path: Path, stub_service: FakeService, free_port: Callable[[], int]
) -> None:
    assert model_server.main(server_argv(tmp_path, free_port(), "--load-checkpoint", " ")) == 1
    assert stub_service.created == []


# --- the process, started the way a run starts it ----------------------------------------------------


def chat(port: int, body: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=30)
    assert response.status_code == 200, response.text
    return response.json()


def test_start_model_server_serves_and_logs_the_exact_ready_line(tmp_path: Path, free_port: Callable[[], int]) -> None:
    port = free_port()
    stubs = {"outputs": ["Reading the request.</think>\n\nThe report is short.<|im_end|>"]}
    process = launch.start_model_server(
        "qwen", BASE, "policy", port, tmp_path / "serve", None, child_env(stubs), temperature=0.6
    )
    files = launch.server_files(tmp_path / "serve")
    try:
        assert process.alive()
        ready = [line for line in files.log.read_text().splitlines() if "EVAL PROXY READY" in line]
        assert len(ready) == 1
        assert ready[0].endswith(f" EVAL PROXY READY on 127.0.0.1:{port} serving base:{BASE} (thinking ON)")
        reply = chat(port, OPENCLAW_BODY)
        assert reply["choices"][0]["message"]["content"] == "The report is short."
        evidence = json.loads(files.evidence.read_text())
        assert evidence["requests_forwarded"] == 1 and evidence["requests"][0]["temperature"] == 0.6
        assert "--temperature" in evidence["argv"]
        assert [json.loads(line)["request"] for line in files.raw.read_text().splitlines()] == [1]
        assert files.records.is_dir()
    finally:
        process.stop(grace_s=10)
    assert not process.alive()
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=5)


def test_start_model_server_loads_a_checkpoint(tmp_path: Path, free_port: Callable[[], int]) -> None:
    port = free_port()
    process = launch.start_model_server(
        "harmony", BASE, "policy", port, tmp_path / "serve", CHECKPOINT, child_env({}), temperature=0.6
    )
    try:
        log = launch.server_files(tmp_path / "serve").log.read_text()
        assert f"EVAL PROXY READY on 127.0.0.1:{port} serving checkpoint:{CHECKPOINT} (thinking ON)" in log
    finally:
        process.stop(grace_s=10)


def test_start_model_server_reports_a_tokenizer_failure(tmp_path: Path, free_port: Callable[[], int]) -> None:
    """No stubs: Hugging Face is offline and the repository does not exist, so the real loader fails."""
    with pytest.raises(ProcessError) as raised:
        launch.start_model_server(
            "inkling",
            "evolve-ipi-test/no-such-tokenizer",
            "policy",
            free_port(),
            tmp_path / "serve",
            None,
            child_env(None),
            temperature=0.6,
        )
    message = str(raised.value)
    assert "exited with status 1" in message
    assert "cannot load the tokenizer of evolve-ipi-test/no-such-tokenizer" in message
    assert "EVAL PROXY READY" not in launch.server_files(tmp_path / "serve").log.read_text()


def test_start_model_server_reports_a_checkpoint_load_failure(tmp_path: Path, free_port: Callable[[], int]) -> None:
    stubs = {"load_error": f"checkpoint {CHECKPOINT} not found"}
    with pytest.raises(ProcessError) as raised:
        launch.start_model_server(
            "qwen", BASE, "policy", free_port(), tmp_path / "serve", CHECKPOINT, child_env(stubs), temperature=0.6
        )
    assert "exited with status 1" in str(raised.value)
    assert f"cannot load the checkpoint {CHECKPOINT}" in str(raised.value)


def test_start_model_server_refuses_a_busy_port_before_starting(tmp_path: Path, listener: int) -> None:
    with pytest.raises(PortInUse, match=rf"port {listener} is in use by pid {os.getpid()}"):
        launch.start_model_server(
            "qwen", BASE, "policy", listener, tmp_path / "serve", None, child_env({}), temperature=0.6
        )
    assert not launch.server_files(tmp_path / "serve").log.exists()
