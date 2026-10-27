"""The trainer entry: the family's tool-call bridge, the Inkling raw dump, the record keeper and the resume with the
optimizer state, installed into stand-ins for OpenClaw-RL's modules and Tinker's training client."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from evolve_ipi.adapt import rl_entry
from evolve_ipi.config import load_configs
from evolve_ipi.models.formats import get_formatter

# What the trainer entry imports by name from OpenClaw-RL's directory; the model-server tests leave them loaded.
FRAMEWORK_MODULES = ("api_server", "run", "config", "data_formatter", "trainer", "rollout", "scorers")

# A stand-in for OpenClaw-RL's api_server: the touchpoints the entry patches, behaving as the real ones do. Its
# _forward_to_tinker decodes the sampled tokens without the special tokens and reads them with the module's
# _extract_tool_calls, looked up when it runs, as the real one does.
STUB_API_SERVER = """
import copy
import os

SPECIAL = ("<|channel|>", "<|message|>", "<|call|>", "<|return|>", "<|end|>", "<|start|>", "<|constrain|>",
           "<|message_model|>", "<|content_text|>", "<|content_thinking|>", "<|end_message|>",
           "<|content_invoke_tool_json|>", "<|content_model_end_sampling|>", "<|im_end|>")
FIRST_SPECIAL = 0x110000


class StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for number, marker in enumerate(SPECIAL):
                if text.startswith(marker, i):
                    ids.append(FIRST_SPECIAL + number)
                    i += len(marker)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for token in ids:
            if token >= FIRST_SPECIAL:
                if not skip_special_tokens:
                    out.append(SPECIAL[token - FIRST_SPECIAL])
            else:
                out.append(chr(token))
        return "".join(out)


def _normalize_messages(messages):
    return [dict(m, normalized_by_framework=True) for m in messages]


def _extract_tool_calls(text):
    return "framework parser", []


class _BaseServer:
    def __init__(self, record_dir, generation="", tokenizer=None):
        os.makedirs(record_dir, exist_ok=True)
        self._record_file = os.path.join(record_dir, "conversations.jsonl")
        self._prm_record_file = os.path.join(record_dir, "prm_scores.jsonl")
        open(self._record_file, "w").close()
        open(self._prm_record_file, "w").close()
        self._tokenizer = tokenizer if tokenizer is not None else StubTokenizer()
        self.generation = generation
        self.bodies = []

    async def _forward_to_tinker(self, body):
        self.bodies.append(copy.deepcopy(body))
        tokens = self._tokenizer.encode(self.generation)
        content, calls = _extract_tool_calls(self._tokenizer.decode(tokens, skip_special_tokens=True))
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        return {
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
            "_raw_prompt_ids": [1, 2, 3],
            "_raw_response_tokens": tokens,
            "_raw_response_logprobs": [-0.5] * len(tokens),
        }

    def _flush_pending_record(self, session_id, next_state):
        with open(self._record_file, "a", encoding="utf-8") as f:
            f.write("base %s %s\\n" % (session_id, next_state))

    def _append_score_record(self, record):
        with open(self._prm_record_file, "a", encoding="utf-8") as f:
            f.write("score %s\\n" % record)

    def purge_record_files(self):
        for path in (self._record_file, self._prm_record_file):
            open(path, "w").close()


class OpenClawRLServer(_BaseServer):
    def _flush_pending_record(self, session_id, next_state):
        with open(self._record_file, "a", encoding="utf-8") as f:
            f.write("rl %s %s\\n" % (session_id, next_state))
"""

# A stand-in for OpenClaw-RL's run.py: main() keeps the arguments it saw and serves one rollout through the
# patched api_server, as the trainer would.
STUB_RUN = """
import asyncio
import sys

import api_server

seen = {}


def main():
    seen["argv"] = list(sys.argv[1:])
    record_dir = sys.argv[sys.argv.index("--record-dir") + 1]
    server = api_server.OpenClawRLServer(record_dir, generation=seen.get("generation", "Hello."))
    seen["out"] = asyncio.run(server._forward_to_tinker({"messages": [{"role": "user", "content": "hi"}]}))
    seen["bodies"] = server.bodies
"""

QWEN_CALL = '<tool_call>\n{"name": "exec", "arguments": {"command": "ls"}}\n</tool_call>'
HARMONY_CALL = (
    "<|channel|>analysis<|message|>The user wants the file.<|end|><|start|>assistant<|channel|>commentary "
    'to=functions.read <|constrain|>json<|message|>{"path": "notes.txt"}<|call|>'
)
INKLING_CALL = (
    "<|content_thinking|>Read it first.<|end_message|><|message_model|>read<|content_invoke_tool_json|>"
    '{"name": "read", "args": {"path": "notes.txt"}}<|end_message|><|content_model_end_sampling|>'
)


@pytest.fixture
def api(tmp_path: Path) -> types.ModuleType:
    """A fresh stand-in api_server module, loaded from a file of its own (never registered in sys.modules)."""
    path = tmp_path / "stub_api_server.py"
    path.write_text(STUB_API_SERVER, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("stub_api_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def without_ids(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for call in calls:
        assert call["id"].startswith("call") and len(call["id"]) == 16
        out.append({key: value for key, value in call.items() if key != "id"})
    return out


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Each framework module this test loads is dropped afterwards, and the ones loaded before come back."""
    for name in FRAMEWORK_MODULES:
        monkeypatch.setitem(sys.modules, name, None)  # remembers what was there, or that nothing was
        del sys.modules[name]
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["rl_entry"])
    return monkeypatch


@pytest.fixture
def stub_tree(tmp_path: Path, isolated: pytest.MonkeyPatch) -> Path:
    tree = tmp_path / "openclaw-tinker"
    tree.mkdir()
    (tree / "api_server.py").write_text(STUB_API_SERVER, encoding="utf-8")
    (tree / "run.py").write_text(STUB_RUN, encoding="utf-8")
    isolated.setattr(rl_entry, "OPENCLAW_TINKER_DIR", tree)
    return tree


@pytest.fixture
def resume_target(monkeypatch: pytest.MonkeyPatch) -> type:
    """Tinker's real TrainingClient, whose resume method main() replaces; the original comes back afterwards."""
    from tinker.lib.public_interfaces.training_client import TrainingClient

    monkeypatch.setattr(TrainingClient, "load_state_async", TrainingClient.load_state_async)
    return TrainingClient


def flags(model: str, record_dir: Path, *extra: str) -> list[str]:
    return [
        "--method", "rl", "--model-name", model, "--batch-size", "4", "--prm-m", "3", "--lora-rank", "32",
        "--learning-rate", "1e-4", "--max-steps", "100", "--save-interval", "32", "--proxy-host", "127.0.0.1",
        "--proxy-port", "30012", "--served-model-name", "policy", "--record-dir", str(record_dir), *extra,
    ]  # fmt: skip


# --- the family -----------------------------------------------------------------------------------------------


def test_the_family_of_every_shipped_rl_model_is_the_one_models_yaml_names(configs_dir: Path) -> None:
    models = load_configs(configs_dir).models
    for spec in models.values():
        assert rl_entry.family_of(spec.tinker_id) == spec.family


@pytest.mark.parametrize("name", ["meta-llama/Llama-3.1-8B", "", "thinkingmachines/Inkling", "gpt-oss-120b"])
def test_a_model_without_a_family_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="no tool-call family"):
        rl_entry.family_of(name)


def test_main_refuses_a_model_without_a_family_before_importing_anything(stub_tree: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no tool-call family"):
        rl_entry.main(flags("meta-llama/Llama-3.1-8B", tmp_path / "records"))
    assert "api_server" not in sys.modules and "run" not in sys.modules


@pytest.mark.parametrize("missing", ["--model-name", "--record-dir"])
def test_main_needs_the_model_and_the_record_dir(stub_tree: Path, tmp_path: Path, missing: str) -> None:
    argv = flags("Qwen/Qwen3.8-27B", tmp_path / "records")
    index = argv.index(missing)
    del argv[index : index + 2]
    with pytest.raises(SystemExit, match=missing):
        rl_entry.main(argv)


# --- the Qwen bridge -------------------------------------------------------------------------------------------


def test_qwen_reads_a_call_written_inside_the_reasoning(api: types.ModuleType) -> None:
    rl_entry.install_bridge(api, get_formatter("qwen"))
    text = f"I will list the files.\n{QWEN_CALL}\nand then answer.</think>\n\nHere is what I found."
    content, calls = api._extract_tool_calls(text)
    assert content == "Here is what I found."
    assert without_ids(calls) == [
        {"type": "function", "function": {"name": "exec", "arguments": json.dumps({"command": "ls"})}}
    ]


def test_qwen_history_is_normalized_before_the_framework_normalizes_it(api: types.ModuleType) -> None:
    rl_entry.install_bridge(api, get_formatter("qwen"))
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "exec", "arguments": '{"command": "ls"}'}}]},
    ]  # fmt: skip
    out = api._normalize_messages(history)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert all(message["normalized_by_framework"] for message in out)
    assert history[1]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'  # the caller's copy is untouched


def test_qwen_sampling_is_left_as_the_framework_does_it(api: types.ModuleType) -> None:
    original = api._BaseServer._forward_to_tinker
    bridge = rl_entry.install_bridge(api, get_formatter("qwen"))
    assert api._BaseServer._forward_to_tinker is original
    assert bridge.stops == () and bridge.raw_decode is False


def test_qwen_rollout_gives_the_agent_the_call_read_from_the_full_text(api: types.ModuleType, tmp_path: Path) -> None:
    rl_entry.install_bridge(api, get_formatter("qwen"))
    server = api.OpenClawRLServer(str(tmp_path), generation=f"thinking {QWEN_CALL}</think>\n\nDone.<|im_end|>")
    out = asyncio.run(server._forward_to_tinker({"messages": []}))
    assert server.bodies == [{"messages": []}]  # no stop added
    message = out["choices"][0]["message"]
    assert message["content"] == "Done."
    assert [call["function"]["name"] for call in message["tool_calls"]] == ["exec"]


# --- the harmony and Inkling bridges ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("client_stop", "expected"),
    [
        (None, ["<|call|>", "<|return|>"]),
        ("END", ["END", "<|call|>", "<|return|>"]),
        (["<|return|>", "END"], ["<|return|>", "END", "<|call|>"]),
    ],
)
def test_harmony_adds_its_stops_after_the_clients(
    api: types.ModuleType, tmp_path: Path, client_stop: Any, expected: list[str]
) -> None:
    rl_entry.install_bridge(api, get_formatter("harmony"))
    server = api.OpenClawRLServer(str(tmp_path), generation="<|channel|>final<|message|>Hi.<|return|>")
    body: dict[str, Any] = {"messages": []}
    if client_stop is not None:
        body["stop"] = client_stop
    asyncio.run(server._forward_to_tinker(body))
    assert server.bodies[0]["stop"] == expected


def test_harmony_reads_the_call_from_the_raw_tokens(api: types.ModuleType, tmp_path: Path) -> None:
    bridge = rl_entry.install_bridge(api, get_formatter("harmony"))
    assert bridge.stops == ("<|call|>", "<|return|>") and bridge.raw_decode is True
    server = api.OpenClawRLServer(str(tmp_path), generation=HARMONY_CALL)
    # without the special tokens nothing can be read: what the framework alone would hand the agent
    plain = api._extract_tool_calls(server._tokenizer.decode(server._tokenizer.encode(HARMONY_CALL), True))
    assert plain == ("", [])
    out = asyncio.run(server._forward_to_tinker({"messages": []}))
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    assert without_ids(choice["message"]["tool_calls"]) == [
        {"type": "function", "function": {"name": "read", "arguments": '{"path": "notes.txt"}'}}
    ]
    assert out["_raw_response_tokens"] == server._tokenizer.encode(HARMONY_CALL)  # the training tokens stay as sampled


def test_harmony_without_a_call_keeps_the_finish_reason(api: types.ModuleType, tmp_path: Path) -> None:
    rl_entry.install_bridge(api, get_formatter("harmony"))
    server = api.OpenClawRLServer(str(tmp_path), generation="<|channel|>final<|message|>Nothing to do.<|return|>")
    out = asyncio.run(server._forward_to_tinker({"messages": []}))
    choice = out["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Nothing to do."}
    assert choice["finish_reason"] == "stop"


class PlainTextCalls:
    """A format whose calls show in the text without the special tokens, and not in the raw text."""

    stop_sequences = None
    needs_raw_decode = True

    def normalize_tool_call_arguments(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        return [{"id": "call000000000000", "type": "function", "function": {"name": "x", "arguments": "{}"}}]

    def clean_content(self, text: str) -> str:
        return text

    def parse_output(self, full: str) -> tuple[list[dict[str, Any]], str]:
        return [], "raw reading"


def test_calls_the_raw_text_does_not_hold_are_dropped(api: types.ModuleType, tmp_path: Path) -> None:
    rl_entry.install_bridge(api, PlainTextCalls())
    server = api.OpenClawRLServer(str(tmp_path), generation="anything")
    out = asyncio.run(server._forward_to_tinker({"messages": []}))
    assert server.bodies[0] == {"messages": []}  # no stop strings: none to add
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "raw reading"}
    assert out["choices"][0]["finish_reason"] == "tool_calls"  # the framework's, as the bridge leaves it


def test_inkling_dumps_every_rollout_generation(api: types.ModuleType, tmp_path: Path) -> None:
    bridge = rl_entry.install_bridge(api, get_formatter("inkling"))
    dump = tmp_path / "records" / rl_entry.RAW_DUMP_NAME
    rl_entry.install_raw_dump(api, dump)
    server = api.OpenClawRLServer(str(tmp_path / "records"), generation=INKLING_CALL)
    out = asyncio.run(server._forward_to_tinker({"messages": [], "stop": ["END"]}))
    assert bridge.stops == ("<|content_model_end_sampling|>",)
    assert [call["function"]["name"] for call in out["choices"][0]["message"]["tool_calls"]] == ["read"]
    rows = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {"at", "n_tokens", "stop", "finish_reason", "parsed_calls", "raw"}
    assert row["raw"] == INKLING_CALL
    assert row["n_tokens"] == len(out["_raw_response_tokens"])
    assert row["stop"] == ["END", "<|content_model_end_sampling|>"]
    assert row["finish_reason"] == "tool_calls" and row["parsed_calls"] == 1


# --- the record keeper ------------------------------------------------------------------------------------------


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_the_keeper_copies_the_records_before_every_purge(
    api: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wrapped = rl_entry.install_record_keeper(api)
    assert wrapped == [
        "_BaseServer._flush_pending_record",
        "_BaseServer._append_score_record",
        "_BaseServer.purge_record_files",
        "OpenClawRLServer._flush_pending_record",
    ]
    server = api.OpenClawRLServer(str(tmp_path))
    assert "[rl_entry] keeping records in" in capsys.readouterr().err
    conversations, scores = tmp_path / "conversations.jsonl", tmp_path / "prm_scores.jsonl"
    kept, kept_scores = tmp_path / "conversations_all.jsonl", tmp_path / "prm_scores_all.jsonl"
    assert read(kept) == "" and read(kept_scores) == ""
    server._flush_pending_record("s1", 1)
    server._append_score_record("a")
    server.purge_record_files()
    assert read(conversations) == "" and read(scores) == ""
    server.purge_record_files()  # nothing new: nothing copied twice
    server._flush_pending_record("s1", 2)
    server._flush_pending_record("s2", 3)
    server._append_score_record("b")
    server.purge_record_files()
    server._flush_pending_record("s3", 4)
    assert read(kept) == "rl s1 1\nrl s1 2\nrl s2 3\nrl s3 4\n"
    assert read(kept_scores) == "score a\nscore b\n"
    assert read(conversations) == "rl s3 4\n"


def test_the_keeper_copies_a_write_whose_own_copy_a_purge_overtook(api: types.ModuleType, tmp_path: Path) -> None:
    # The server appends from its request thread and the trainer purges from its own: a purge can come between an
    # append's write and the copy that follows it. What was written must be copied before the files are emptied.
    rl_entry.install_record_keeper(api)
    server = api.OpenClawRLServer(str(tmp_path))
    server._flush_pending_record("s1", 1)
    with open(tmp_path / "conversations.jsonl", "a", encoding="utf-8") as records:
        records.write("rl s2 2\n")  # written, not yet copied
    with open(tmp_path / "prm_scores.jsonl", "a", encoding="utf-8") as scores:
        scores.write("score b\n")
    server.purge_record_files()
    server._flush_pending_record("s3", 3)
    assert read(tmp_path / "conversations_all.jsonl") == "rl s1 1\nrl s2 2\nrl s3 3\n"
    assert read(tmp_path / "prm_scores_all.jsonl") == "score b\n"


def test_the_keeper_refuses_a_record_dir_with_kept_records(api: types.ModuleType, tmp_path: Path) -> None:
    rl_entry.install_record_keeper(api)
    (tmp_path / "conversations_all.jsonl").write_text("an earlier run\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="earlier run"):
        api.OpenClawRLServer(str(tmp_path))


def test_the_keeper_fails_loud_when_the_framework_moved_its_record_keeping(api: types.ModuleType) -> None:
    del api._BaseServer._append_score_record
    with pytest.raises(SystemExit, match="_append_score_record"):
        rl_entry.install_record_keeper(api)


# --- the resume with the optimizer state ------------------------------------------------------------------------


class StubFuture:
    def __init__(self, error: BaseException | None) -> None:
        self.error = error
        self.was_awaited = False

    async def result_async(self) -> str:
        self.was_awaited = True
        if self.error is not None:
            raise self.error
        return "loaded"


class StubTrainingClient:
    error: BaseException | None = None

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.futures: list[StubFuture] = []

    async def load_state_async(self, path: str) -> StubFuture:  # what the SDK's weights-only resume would do
        raise AssertionError("the weights-only load must not be used")

    async def load_state_with_optimizer_async(self, path: str, weights_access_token: str | None = None) -> StubFuture:
        self.calls.append((path, {"weights_access_token": weights_access_token}))
        future = StubFuture(self.error)
        self.futures.append(future)
        return future


PATH = "tinker://run-1:train:0/weights/step_0032"


def test_the_resume_restores_the_optimizer_state_waits_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(StubTrainingClient, "load_state_async", StubTrainingClient.load_state_async)
    rl_entry.install_optimizer_resume(StubTrainingClient)
    client = StubTrainingClient()
    result = asyncio.run(client.load_state_async(PATH))
    assert client.calls == [(PATH, {"weights_access_token": None})]
    assert client.futures[0].was_awaited and result is client.futures[0]
    assert capsys.readouterr().out == (
        f"[rl_entry] restored LoRA weights AND optimizer state from {PATH} (load_state_with_optimizer, awaited)\n"
    )


def test_a_failed_resume_raises_and_says_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(StubTrainingClient, "load_state_async", StubTrainingClient.load_state_async)
    monkeypatch.setattr(StubTrainingClient, "error", RuntimeError("checkpoint not found"))
    rl_entry.install_optimizer_resume(StubTrainingClient)
    client = StubTrainingClient()
    with pytest.raises(RuntimeError, match="checkpoint not found"):
        asyncio.run(client.load_state_async(PATH))
    assert client.futures[0].was_awaited
    assert capsys.readouterr().out == ""


def test_the_restored_line_is_the_one_the_checkpoint_collector_reads() -> None:
    from evolve_ipi.models.checkpoints import parse_trainer_log

    log = parse_trainer_log(rl_entry.RESTORED_LINE % PATH + "\n")
    assert [path for _, path in log.restored_from] == [PATH]


# --- the whole entry ------------------------------------------------------------------------------------------


def test_main_installs_the_bridge_and_the_keeper_then_runs_run_py(
    stub_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], resume_target: type
) -> None:
    resume_before = resume_target.load_state_async
    records = tmp_path / "records"
    argv = flags("Qwen/Qwen3.8-27B", records)
    rl_entry.main(argv)
    run = sys.modules["run"]
    api = sys.modules["api_server"]
    assert Path(api.__file__).parent == stub_tree
    assert run.seen["argv"] == argv  # run.py parses exactly the arguments the entry was given
    assert run.seen["bodies"] == [{"messages": [{"role": "user", "content": "hi"}]}]
    out, err = capsys.readouterr()
    assert out.splitlines().count("[rl_entry] family=qwen model=Qwen/Qwen3.8-27B resume=none") == 1
    assert f"[rl_entry] keeping records in {records / 'conversations_all.jsonl'}" in err
    assert (records / "conversations_all.jsonl").exists()
    assert not (records / rl_entry.RAW_DUMP_NAME).exists()
    assert resume_target.load_state_async is resume_before  # no resume: Tinker's client is left alone


def test_main_for_inkling_dumps_the_rollouts(
    stub_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = tmp_path / "records"
    sys.path.insert(0, str(stub_tree))
    import run as stub_run  # the same module main() imports

    stub_run.seen["generation"] = INKLING_CALL
    rl_entry.main(flags("thinkingmachines/Inkling-Small", records))
    out = stub_run.seen["out"]
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert stub_run.seen["bodies"][0]["stop"] == ["<|content_model_end_sampling|>"]
    rows = [json.loads(line) for line in (records / rl_entry.RAW_DUMP_NAME).read_text(encoding="utf-8").splitlines()]
    assert [row["parsed_calls"] for row in rows] == [1]
    assert "[rl_entry] family=inkling model=thinkingmachines/Inkling-Small resume=none" in capsys.readouterr().out


def test_main_with_a_resume_installs_the_optimizer_resume(
    stub_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], resume_target: type
) -> None:
    rl_entry.main(flags("openai/gpt-oss-120b", tmp_path / "records", "--resume-from-ckpt", PATH))
    assert resume_target.load_state_async is rl_entry._load_state_with_optimizer_async
    assert f"[rl_entry] family=harmony model=openai/gpt-oss-120b resume={PATH}" in capsys.readouterr().out
    assert sys.modules["run"].seen["bodies"][0]["stop"] == ["<|call|>", "<|return|>"]


def test_main_refuses_an_api_server_from_another_directory(stub_tree: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "api_server.py").write_text(STUB_API_SERVER, encoding="utf-8")
    sys.path.insert(0, str(elsewhere))
    import api_server  # noqa: F401  (loaded from the wrong place before main() runs)

    with pytest.raises(SystemExit, match="not from"):
        rl_entry.main(flags("Qwen/Qwen3.8-27B", tmp_path / "records"))
