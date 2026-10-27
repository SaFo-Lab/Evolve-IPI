"""Run the MetaClaw proxy script as far as it goes before it serves, and write what it built as JSON.

    <MetaClaw python> metaclaw_dump.py <out.json> <metaclaw_proxy.py>

The script runs as ``__main__`` with the environment it is given. Its worker's ``resume_submission`` (the call right
before serving) is replaced: it writes the config, the skill manager, the skill evolver and the worker to ``out.json``
and stops the script. Runs in the MetaClaw environment: standard library and ``metaclaw`` only.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import queue
import runpy
import sys
import threading
from typing import Any

import metaclaw.rollout
from metaclaw.config import MetaClawConfig


class _Built(Exception):
    pass


def describe(obj: Any, depth: int = 0) -> Any:
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if depth > 4:
        return {"type": type(obj).__name__}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(key): describe(value, depth + 1) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [describe(value, depth + 1) for value in obj]
    if isinstance(obj, queue.Queue):
        return {"type": "Queue", "maxsize": obj.maxsize}
    if isinstance(obj, threading.Event):
        return {"type": "Event", "set": obj.is_set()}
    if isinstance(obj, itertools.count):
        return repr(obj)
    module = type(obj).__module__
    if module.startswith("openai"):
        return {"type": type(obj).__name__, "base_url": str(obj.base_url), "api_key": obj.api_key}
    if module.startswith("metaclaw"):
        fields = {name: describe(value, depth + 1) for name, value in sorted(vars(obj).items()) if name != "app"}
        return {"type": type(obj).__name__, **fields}
    return {"type": type(obj).__name__}


def main() -> None:
    out, script = sys.argv[1], sys.argv[2]

    def capture(worker: Any) -> None:
        server = worker._server
        built = {
            "config": describe(worker.config),
            "config_defaults": dataclasses.asdict(MetaClawConfig()),
            "skill_manager": describe(server.skill_manager),
            "skill_evolver": describe(server.skill_evolver),
            "worker": describe(worker),
            "same_objects": {
                "server_config": server.config is worker.config,
                "evolver_history_is_config_history": (
                    server.skill_evolver is None
                    or server.skill_evolver.history_path == worker.config.skill_evolution_history_path
                ),
            },
            "tokenizer_is_none": server._tokenizer is None,
        }
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(built, handle, indent=1, sort_keys=True)
        raise _Built

    metaclaw.rollout.AsyncRolloutWorker.resume_submission = capture  # type: ignore[method-assign]
    sys.argv = [script]
    try:
        runpy.run_path(script, run_name="__main__")
    except _Built:
        return
    raise SystemExit("the proxy script returned without building its worker")


if __name__ == "__main__":
    main()
