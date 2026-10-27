"""Tool calls of the gpt-oss models, which speak OpenAI's harmony format.

A reply is a sequence of channel messages:

    <|channel|>analysis<|message|>reasoning<|end|>
    <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}<|call|>
    <|start|>assistant<|channel|>final<|message|>answer<|return|>

A call is marked by the ``<|call|>`` terminator and a ``to=functions.NAME`` recipient, not by its channel. gpt-oss
should put calls on ``commentary`` but often writes them on ``analysis``, and dropping those would turn a missed call
into an empty reply that reads like a refusal. So a call on any channel is read. Reasoning ends in ``<|end|>`` and
can never match. The visible answer is the ``final`` channel only.
"""

from __future__ import annotations

import re
from typing import Any

from evolve_ipi.models.formats.qwen import new_call_id
from evolve_ipi.models.formats.qwen import normalize_tool_call_arguments as _dict_arguments

_TOOL_CALL = re.compile(
    r"<\|channel\|>\w+\b[^<]*?to=functions\.([A-Za-z0-9_.\-]+)"
    r".*?<\|message\|>(.*?)<\|call\|>",
    re.DOTALL,
)
_FINAL = re.compile(
    r"<\|channel\|>final\b[^<]*?<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|\Z)",
    re.DOTALL,
)


class HarmonyFormatter:
    # <|channel|>, <|message|>, <|call|> and <|constrain|> are special tokens: a decode that skips them leaves
    # "commentary to=functions.exec json{...}", which nothing here can match.
    needs_raw_decode = True

    def __init__(self) -> None:
        # A call turn ends at <|call|> and an answer turn at <|return|>. Unlike Qwen's <|im_end|>, <|call|> is not
        # among the tokenizer's end-of-turn tokens, so without these stops the model runs on past a call into an
        # invented final message that would pollute the history. The sampler returns the stop text with the output,
        # so the <|call|> and <|return|> anchors in the patterns still match.
        self.stop_sequences: list[str] | None = ["<|call|>", "<|return|>"]

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        calls = []
        for match in _TOOL_CALL.finditer(text):
            name = match.group(1).strip()
            arguments = match.group(2).strip()
            # The arguments stay a string, as in an OpenAI tool call. A payload that is not JSON is passed on as
            # written: a malformed call must stay visible, and dropping it would read as a refusal.
            calls.append({"id": new_call_id(), "type": "function", "function": {"name": name, "arguments": arguments}})
        return calls

    def clean_content(self, text: str) -> str:
        """The first ``final`` message. A turn that only calls a tool has none, and then the content is empty."""
        match = _FINAL.search(text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Qwen's fix for the arguments, and ``content: None`` becomes ``""``.

        The chat template evaluates ``"<|channel|>analysis<|message|>" in message.content`` and raises on None,
        which is what OpenClaw sends for an assistant turn that only calls a tool. The arguments must be a dict,
        or the template encodes the string a second time and the model reads it as text.
        """
        out = _dict_arguments(messages)
        for message in out:
            if isinstance(message, dict) and "content" in message and message["content"] is None:
                message["content"] = ""
        return out

    def parse_output(self, full: str) -> tuple[list[dict[str, Any]], str]:
        return self.parse_tool_calls(full), self.clean_content(full)
