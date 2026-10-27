"""Tool calls of Thinking Machines' Inkling-Small.

The generation prompt ends with the model's message header ``<|message_model|>``, and the model then writes messages:

    <|content_thinking|>reasoning<|end_message|>
    <|message_model|>exec<|content_invoke_tool_json|>{"name":"exec","args":{"command":"ls"}}<|end_message|>
    <|content_model_end_sampling|>

or, for an answer, ``<|message_model|><|content_text|>answer<|end_message|>``. A call is a model message whose header
is the tool name and whose content type is ``invoke_tool_json``; its payload is ``{"name": ..., "args": {...}}``.

Reasoning spans are cut out before calls are read, so a call written inside reasoning is never run. The
``invoke_tool_text`` form of a call has not been seen in any generation and is deliberately not parsed;
:meth:`InklingFormatter.call_markup` reports it, so an audit stops on it instead of reading it as a refusal.
The audit helpers (``call_markup``, ``reasoning_closed``, ``markup_in_reasoning``, ``truncated_call``,
``unmarked_text`` and ``markup_tokens``) check generations for markup the parser left unread.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evolve_ipi.models.formats.qwen import new_call_id
from evolve_ipi.models.formats.qwen import normalize_tool_call_arguments as _dict_arguments

_MESSAGE_MODEL = "<|message_model|>"
_END = "<|end_message|>"
_STOP = "<|content_model_end_sampling|>"
_THINKING = re.compile(r"<\|content_thinking\|>(.*?)(?:<\|end_message\|>|\Z)", re.DOTALL)
_CALL = re.compile(r"<\|message_model\|>([^<]*?)<\|content_invoke_tool_json\|>(.*?)<\|end_message\|>", re.DOTALL)
_TEXT = re.compile(r"<\|content_text\|>(.*?)(?:<\|end_message\|>|<\|content_model_end_sampling\|>|\Z)", re.DOTALL)


class InklingFormatter:
    # Every marker is a special token.
    needs_raw_decode = True
    # Strings that must never reach a user-visible reply.
    markup_tokens = (
        "<|message_model|>",
        "<|message_user|>",
        "<|message_system|>",
        "<|message_tool|>",
        "<|content_text|>",
        "<|content_thinking|>",
        "<|content_invoke_tool_json|>",
        "<|content_invoke_tool_text|>",
        "<|end_message|>",
        "<|content_model_end_sampling|>",
    )

    def __init__(self) -> None:
        # Every turn seen ended on this marker by itself, and the tokenizer defines no end-of-turn token, so the stop
        # is what guarantees it.
        self.stop_sequences: list[str] | None = [_STOP]

    @staticmethod
    def _framed(text: str) -> str:
        # The generation prompt already ends with the message header, so a generation starts after it.
        return text if text.startswith(_MESSAGE_MODEL) else _MESSAGE_MODEL + text

    @staticmethod
    def _without_reasoning(text: str) -> str:
        return _THINKING.sub("", text)

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        calls = []
        for match in _CALL.finditer(self._without_reasoning(self._framed(text))):
            name, payload = match.group(1).strip(), match.group(2).strip()
            try:
                obj = json.loads(payload)
            except ValueError:
                obj = None
            if isinstance(obj, dict) and isinstance(obj.get("args"), dict):
                name = name or str(obj.get("name") or "")
                arguments = json.dumps(obj["args"], ensure_ascii=False)
            else:
                arguments = payload  # passed on as written, never dropped: a malformed call stays visible
            calls.append({"id": new_call_id(), "type": "function", "function": {"name": name, "arguments": arguments}})
        return calls

    def clean_content(self, text: str) -> str:
        """The text messages, without the reasoning spans and the call messages."""
        body = self._without_reasoning(self._framed(text))
        body = _CALL.sub("", body)
        return "".join(_TEXT.findall(body)).strip()

    @staticmethod
    def normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Qwen's fix for the arguments, and ``""`` becomes ``None`` as the content of an assistant tool-call turn.

        The chat template raises on arguments that are a JSON string. An assistant turn with a call and the content
        ``""`` would render an empty text message that the model never wrote, while ``None`` renders the call alone.
        """
        out = _dict_arguments(messages)
        for message in out:
            is_call_turn = (
                isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls")
            )
            if is_call_turn and message.get("content") == "":
                message["content"] = None
        return out

    def parse_output(self, full: str) -> tuple[list[dict[str, Any]], str]:
        return self.parse_tool_calls(full), self.clean_content(full)

    def call_markup(self, full: str) -> bool:
        """The generation carries a call marker where calls are read. A call that parses to nothing next to one is
        a parser failure, which would look exactly like a model that refuses."""
        body = self._without_reasoning(self._framed(full))
        return "<|content_invoke_tool_json|>" in body or "<|content_invoke_tool_text|>" in body

    @staticmethod
    def reasoning_closed(full: str) -> bool:
        index = full.rfind("<|content_thinking|>")
        return index < 0 or _END in full[index:]

    @staticmethod
    def markup_in_reasoning(full: str) -> bool:
        return any("<|content_invoke_tool" in span or _MESSAGE_MODEL in span for span in _THINKING.findall(full))

    def unmarked_text(self, full: str) -> str:
        """Text the model wrote outside every content segment, with the markup left out: no parser can show or run it
        without guessing."""
        body = self._without_reasoning(self._framed(full))
        body = _TEXT.sub("", _CALL.sub("", body))
        for token in self.markup_tokens:
            body = body.replace(token, " ")
        return body.strip()

    def truncated_call(self, full: str) -> bool:
        """The generation stopped inside a call that never ended, which is the token limit. No parser can run such
        a call: it is the model's behaviour, not a parser failure."""
        body = self._without_reasoning(self._framed(full))
        index = max(body.rfind("<|content_invoke_tool_json|>"), body.rfind("<|content_invoke_tool_text|>"))
        return index >= 0 and _END not in body[index:] and _STOP not in full
