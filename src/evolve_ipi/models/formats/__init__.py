"""The tool-call format of each model family.

A model writes its tool calls in its own markup, and its chat template wants the calls of the history in its own
shape. A ``Formatter`` holds both for one family. The model server parses every completion with ``parse_output``;
RL training reads calls with ``parse_tool_calls`` and ``clean_content``. ``get_formatter`` picks the formatter of a
family as ``models.yaml`` names it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

from evolve_ipi.models.formats.harmony import HarmonyFormatter
from evolve_ipi.models.formats.inkling import InklingFormatter
from evolve_ipi.models.formats.qwen import QwenFormatter, new_call_id

__all__ = ["Family", "Formatter", "get_formatter", "new_call_id"]

Family = Literal["qwen", "harmony", "inkling"]


@runtime_checkable
class Formatter(Protocol):
    """What the model server and the trainer need from a family's tool-call format.

    ``stop_sequences`` are added to the stop strings of every request (treat them as read-only). ``needs_raw_decode``
    says the family's markup is made of special tokens, so a generation must be decoded with them kept before it is
    parsed. A call is an OpenAI-style dict: ``{"id", "type": "function", "function": {"name", "arguments"}}``.

    ``parse_tool_calls`` and ``clean_content`` read a text as it stands. ``parse_output`` applies the model server's
    rule to one whole generation and returns ``(tool_calls, content)``; for Qwen it reads calls only from after the
    reasoning, which is the one place it differs from the two.
    """

    stop_sequences: list[str] | None
    needs_raw_decode: bool

    def normalize_tool_call_arguments(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]]: ...

    def clean_content(self, text: str) -> str: ...

    def parse_output(self, full: str) -> tuple[list[dict[str, Any]], str]: ...


_FORMATTERS: dict[str, Callable[[], Formatter]] = {
    "qwen": QwenFormatter,
    "harmony": HarmonyFormatter,
    "inkling": InklingFormatter,
}


def get_formatter(family: Family) -> Formatter:
    """The formatter of a model family. Anything but the family names raises ``ValueError``."""
    factory = _FORMATTERS.get(family)
    if factory is None:
        raise ValueError(
            f"no tool-call format for the model family {family!r}; the families are {', '.join(_FORMATTERS)}"
        )
    return factory()
