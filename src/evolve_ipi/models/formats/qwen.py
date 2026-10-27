"""Tool calls of the Qwen3 models.

A call is written between ``<tool_call>`` and ``</tool_call>``. Qwen3.8 puts XML inside,
``<function=NAME><parameter=P>VALUE</parameter></function>``; the earlier Qwen3 models put a JSON object,
``{"name": ..., "arguments": {...}}``. Both are read.

The chat template of Qwen3.8 starts the reply inside the reasoning (it prefills ``<think>``), so a ``</think>`` ends
the reasoning. A generation without one has no closed reasoning, and its whole text is read as the answer.

This module also holds the two helpers the other families build on: the id of a call and the fix for tool calls
in the message history.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from typing import Any

# Adapted from OpenClaw-RL openclaw-tinker/api_server.py (Apache-2.0); see THIRD_PARTY_NOTICES.md.
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL)


def new_call_id() -> str:
    """The id of a tool call. Every family uses this scheme."""
    return "call" + uuid.uuid4().hex[:12]


def normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of ``messages`` in which every tool call carries its arguments as a dict.

    OpenAI-style messages hold them as a JSON string, but the chat templates iterate over them
    (``arguments | items``), which fails on a string. Empty or invalid JSON becomes ``{}``.
    """
    out = copy.deepcopy(messages)
    for message in out:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function", call) if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments) if arguments.strip() else {}
                except Exception:  # not valid JSON
                    function["arguments"] = {}
    return out


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """The complete ``<tool_call>`` blocks of ``text`` as OpenAI-style tool calls, in order.

    A block that is neither the JSON nor the XML form, or that names no function, is skipped. A block without its
    closing tag is not read.
    """
    calls = []
    for block in _TOOL_CALL.findall(text):
        block = block.strip()
        name, args = None, "{}"
        try:  # the JSON form
            payload = json.loads(block)
            name = payload.get("name") or payload.get("function", {}).get("name")
            raw_args = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
            args = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
        except Exception:  # not a JSON object, so try the XML form
            function = _FUNCTION.search(block)
            if not function:
                continue
            name = function.group(1).strip()
            parameters = {p.strip(): v.strip() for p, v in _PARAMETER.findall(function.group(2))}
            args = json.dumps(parameters, ensure_ascii=False)
        if not name:
            continue
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append({"id": new_call_id(), "type": "function", "function": {"name": str(name), "arguments": args}})
    return calls


def clean_content(text: str) -> str:
    """What the user sees of a generation: the text after the reasoning, without tool-call blocks or ``<|im_end|>``."""
    body = text.split("</think>")[-1] if "</think>" in text else text
    body = _TOOL_CALL.sub("", body)
    return body.replace("<|im_end|>", "").strip()


class QwenFormatter:
    # The model ends a turn with <|im_end|>, its end-of-turn token, which already follows any call.
    stop_sequences: list[str] | None = None
    # <tool_call> and <think> are ordinary text, so they survive a decode that skips special tokens.
    needs_raw_decode = False

    normalize_tool_call_arguments = staticmethod(normalize_tool_call_arguments)
    parse_tool_calls = staticmethod(parse_tool_calls)
    clean_content = staticmethod(clean_content)

    def parse_output(self, full: str) -> tuple[list[dict[str, Any]], str]:
        """The model server's rule: calls come from the text after the last ``</think>`` (the whole text when
        there is none), so a call written inside closed reasoning is never run. The content is cleaned from the full
        text. RL training instead reads calls from the full text with :func:`parse_tool_calls`."""
        visible = full.split("</think>")[-1] if "</think>" in full else full
        return parse_tool_calls(visible), clean_content(full)
