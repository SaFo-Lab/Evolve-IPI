"""Stand-ins for Tinker and the Hugging Face tokenizer, so that no test reaches either.

In-process tests install them with ``monkeypatch``. A test that runs a model-server or checkpoint process puts
``stub_site/`` and this directory on the child's ``PYTHONPATH`` and the stub settings, as JSON, in
``EVOLVE_IPI_TEST_STUBS``; ``stub_site/sitecustomize.py`` then calls :func:`install` before the child runs anything.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STUB_SITE = Path(__file__).resolve().parent / "stub_site"
STUBS_ENV = "EVOLVE_IPI_TEST_STUBS"

# Markup that real tokenizers hold as single special tokens: a decode that skips special tokens drops them.
SPECIAL_TOKENS = (
    "<|im_end|>",
    "<|start|>",
    "<|end|>",
    "<|channel|>",
    "<|message|>",
    "<|call|>",
    "<|return|>",
    "<|constrain|>",
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|end_message|>",
    "<|content_invoke_tool_json|>",
    "<|content_model_end_sampling|>",
)
_FIRST_SPECIAL = 0x110000  # above every Unicode code point


class FakeTokenizer:
    """One token per character; each special token is one id above the Unicode range."""

    def __init__(self) -> None:
        self.templates: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: Any = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        self.templates.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        return "".join(f"<{m.get('role')}>{m.get('content')}" for m in messages) + "<assistant>"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            for number, special in enumerate(SPECIAL_TOKENS):
                if text.startswith(special, i):
                    ids.append(_FIRST_SPECIAL + number)
                    i += len(special)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        out = []
        for token in ids:
            if token >= _FIRST_SPECIAL:
                if not skip_special_tokens:
                    out.append(SPECIAL_TOKENS[token - _FIRST_SPECIAL])
            else:
                out.append(chr(token))
        return "".join(out)


class FakeAutoTokenizer:
    """``AutoTokenizer.from_pretrained``: a :class:`FakeTokenizer`, or the configured error."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.loaded: list[tuple[str, dict[str, Any]]] = []
        self.tokenizer = FakeTokenizer()

    def from_pretrained(self, name: str, **kwargs: Any) -> FakeTokenizer:
        self.loaded.append((name, kwargs))
        if self.error is not None:
            raise self.error
        return self.tokenizer


# --- Tinker ------------------------------------------------------------------------------------------


class FakeFuture:
    """An ``APIFuture``: ``result()`` and ``result_async()`` return the value or raise the error."""

    def __init__(self, value: Any = None, error: BaseException | None = None) -> None:
        self.value, self.error = value, error
        self.waited_for = False

    def result(self) -> Any:
        self.waited_for = True
        if self.error is not None:
            raise self.error
        return self.value

    async def result_async(self) -> Any:
        return self.result()


@dataclass
class FakeSequence:
    tokens: list[int]
    logprobs: list[float] | None
    stop_reason: str | None


@dataclass
class FakeSampleResponse:
    sequences: list[FakeSequence]


@dataclass
class FakeSamplingClient:
    """Answers ``sample_async`` with the next scripted text and ``compute_logprobs`` with fixed values."""

    tokenizer: FakeTokenizer
    outputs: list[str] = field(default_factory=list)
    logprobs: list[float | None] = field(default_factory=list)
    stop_reason: str = "stop"
    samples: list[dict[str, Any]] = field(default_factory=list)

    async def sample_async(
        self,
        prompt: Any,
        num_samples: int,
        sampling_params: Any,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> FakeSampleResponse:
        self.samples.append(
            {
                "prompt_tokens": prompt.to_ints(),
                "num_samples": num_samples,
                "sampling_params": sampling_params,
                "include_prompt_logprobs": include_prompt_logprobs,
                "topk_prompt_logprobs": topk_prompt_logprobs,
            }
        )
        text = self.outputs.pop(0) if self.outputs else "Hello."
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return FakeSampleResponse([FakeSequence(tokens, [-0.25] * len(tokens), self.stop_reason)])

    def compute_logprobs(self, model_input: Any) -> FakeFuture:
        self.samples.append({"logprobs_of": model_input.to_ints()})
        return FakeFuture(list(self.logprobs))


@dataclass
class FakeCheckpoint:
    tinker_path: str
    checkpoint_type: str = "training"
    size_bytes: int | None = 2_900_000_000


@dataclass
class FakeCheckpointList:
    checkpoints: list[FakeCheckpoint]


class FakeRestClient:
    def __init__(self, listings: list[list[FakeCheckpoint]], unloadable: set[str] | None = None) -> None:
        self.listings = listings
        self.unloadable = unloadable or set()
        self.listed: list[str] = []
        self.checked: list[str] = []

    def list_checkpoints(self, training_run_id: str) -> FakeFuture:
        self.listed.append(training_run_id)
        listing = self.listings.pop(0) if len(self.listings) > 1 else self.listings[0]
        return FakeFuture(FakeCheckpointList(listing))

    def get_weights_info_by_tinker_path(self, path: str) -> FakeFuture:
        self.checked.append(path)
        if path in self.unloadable:
            return FakeFuture(error=RuntimeError(f"not found: {path}"))
        return FakeFuture({"base_model": "stub"})


class FakeTrainingClient:
    def __init__(self, service: FakeService) -> None:
        self.service = service
        self.loads: list[tuple[str, FakeFuture]] = []
        self.saved = False

    def _load(self, path: str) -> FakeFuture:
        error = RuntimeError(self.service.load_error) if self.service.load_error else None
        future = FakeFuture(error=error)
        self.loads.append((path, future))
        return future

    def load_state(self, path: str) -> FakeFuture:
        return self._load(path)

    async def load_state_async(self, path: str) -> FakeFuture:
        return self._load(path)

    def save_weights_and_get_sampling_client(self) -> FakeSamplingClient:
        self.saved = True
        return self.service.adapted

    async def save_weights_and_get_sampling_client_async(self) -> FakeSamplingClient:
        return self.save_weights_and_get_sampling_client()


class FakeService:
    """``tinker.ServiceClient``: one shared instance, so a test can look at what the code under test did.

    ``base`` serves the base model and ``adapted`` a loaded checkpoint; both sample ``outputs`` in turn.
    ``logprobs`` and ``shift`` set the probe log-probabilities of the base model and of the checkpoint.
    """

    def __init__(
        self,
        tokenizer: FakeTokenizer | None = None,
        outputs: list[str] | None = None,
        load_error: str | None = None,
        logprobs: list[float | None] | None = None,
        shift: float = 0.3,
        rest: FakeRestClient | None = None,
    ) -> None:
        self.tokenizer = tokenizer or FakeTokenizer()
        self.load_error = load_error
        self.rest = rest
        self.created: list[tuple[str, dict[str, Any]]] = []
        self.training_clients: list[FakeTrainingClient] = []
        outputs = list(outputs or [])
        base_logprobs = list(logprobs if logprobs is not None else [None, -1.0, -2.0, -0.5])
        adapted_logprobs = [None if value is None else value - shift for value in base_logprobs]
        self.base = FakeSamplingClient(self.tokenizer, outputs, base_logprobs)
        self.adapted = FakeSamplingClient(self.tokenizer, outputs, adapted_logprobs)

    def __call__(self) -> FakeService:  # stands in for the class: tinker.ServiceClient()
        return self

    def create_sampling_client(self, base_model: str | None = None, **kwargs: Any) -> FakeSamplingClient:
        self.created.append(("sampling", {"base_model": base_model, **kwargs}))
        return self.base

    async def create_sampling_client_async(self, base_model: str | None = None, **kwargs: Any) -> FakeSamplingClient:
        return self.create_sampling_client(base_model=base_model, **kwargs)

    def create_lora_training_client(self, base_model: str, rank: int = 32, **kwargs: Any) -> FakeTrainingClient:
        self.created.append(("training", {"base_model": base_model, "rank": rank, **kwargs}))
        client = FakeTrainingClient(self)
        self.training_clients.append(client)
        return client

    async def create_lora_training_client_async(
        self, base_model: str, rank: int = 32, **kwargs: Any
    ) -> FakeTrainingClient:
        return self.create_lora_training_client(base_model=base_model, rank=rank, **kwargs)

    def create_rest_client(self) -> FakeRestClient:
        if self.rest is None:
            raise AssertionError("this test set up no REST client")
        return self.rest


# --- installing them in a child process -----------------------------------------------------------------


def child_env(stubs: Mapping[str, Any] | None, **extra: str) -> dict[str, str]:
    """The environment of a model-server or checkpoint process under test.

    Hugging Face is offline and Tinker's address is a closed local port, so even a stub that failed to install
    could not reach either service.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/"),
        "TINKER_API_KEY": "test-key-not-real",
        "TINKER_BASE_URL": "http://127.0.0.1:9",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        **extra,
    }
    if stubs is not None:
        env["PYTHONPATH"] = os.pathsep.join([str(STUB_SITE), str(Path(__file__).resolve().parent)])
        env[STUBS_ENV] = json.dumps(stubs)
    return env


def install(settings: Mapping[str, Any]) -> None:
    """Replace ``tinker.ServiceClient`` and ``AutoTokenizer.from_pretrained`` in this process."""
    import tinker
    import transformers

    tokenizer_error = settings.get("tokenizer_error")
    auto = FakeAutoTokenizer(OSError(tokenizer_error) if tokenizer_error else None)
    transformers.AutoTokenizer.from_pretrained = auto.from_pretrained
    tinker.ServiceClient = FakeService(
        tokenizer=auto.tokenizer,
        outputs=list(settings.get("outputs", [])),
        load_error=settings.get("load_error"),
        logprobs=settings.get("logprobs"),
        shift=float(settings.get("shift", 0.3)),
    )
    print(f"[test stubs] Tinker and the tokenizer are stubbed: {sorted(settings)}", flush=True)
