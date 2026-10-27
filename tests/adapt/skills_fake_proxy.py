"""A stand-in for ``metaclaw_proxy.py`` in the tests: the same ready lines, a tiny HTTP server, and an "evolver" that
appends one history line per closed session, after a short delay.

``start_proxy`` gives the proxy an environment of its own, so the test passes the stand-in's behaviour in
``$HOME/fake_proxy.json``: ``exit_before_ready`` (a status), ``hang_before_ready``, ``no_listen``, ``evolve_delay_s``.
The environment the stand-in got is written to ``$RECORD_DIR/env.json``, every request to ``requests.jsonl``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ENV = dict(os.environ)
RECORD_DIR = Path(ENV["RECORD_DIR"])
SKILLS_DIR = Path(ENV["SKILLS_DIR"])
_settings = Path(ENV["HOME"]) / "fake_proxy.json"
BEHAVIOUR: dict[str, Any] = json.loads(_settings.read_text(encoding="utf-8")) if _settings.is_file() else {}
_lock = threading.Lock()


def evolve(session_id: str) -> None:
    with _lock:
        skill = SKILLS_DIR / f"skill-{session_id}"
        skill.mkdir(parents=True, exist_ok=True)
        (skill / "SKILL.md").write_text(f"# {session_id}\n", encoding="utf-8")
        line = {"num_failures_analyzed": 2, "failure_prompts": ["a", "b"], "num_skills_generated": 1}
        with open(SKILLS_DIR / "evolution_history.jsonl", "a", encoding="utf-8") as history:
            history.write(json.dumps(line) + "\n")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        seen = {"method": "POST", "path": self.path, "headers": dict(self.headers.items()), "body": body.decode()}
        with _lock, open(RECORD_DIR / "requests.jsonl", "a", encoding="utf-8") as log:
            log.write(json.dumps(seen) + "\n")
        if self.headers.get("X-Session-Done") == "true":
            delay = float(BEHAVIOUR.get("evolve_delay_s", 0.05))
            threading.Timer(delay, evolve, args=(self.headers.get("X-Session-Id", ""),)).start()
        reply = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"00:00:00 | INFO | fake | {format % args}", flush=True)


def main() -> int:
    (RECORD_DIR / "env.json").write_text(json.dumps(ENV, indent=1, sort_keys=True), encoding="utf-8")
    (RECORD_DIR / "pid").write_text(str(os.getpid()), encoding="utf-8")
    print("fake proxy starting", flush=True)
    if "exit_before_ready" in BEHAVIOUR:
        return int(BEHAVIOUR["exit_before_ready"])
    print(
        f"METACLAW patched /fake skill_evolver={ENV['EXPECT_SHA256_SKILL_EVOLVER']} "
        f"api_server={ENV['EXPECT_SHA256_API_SERVER']}",
        flush=True,
    )
    if BEHAVIOUR.get("hang_before_ready"):
        time.sleep(3600)
    port = int(ENV["PROXY_PORT"])
    server = None if BEHAVIOUR.get("no_listen") else ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"00:00:00 | INFO | fake | skills_only proxy READY on 127.0.0.1:{port} -> {ENV['LLM_BASE']}", flush=True)
    if server is None:
        time.sleep(3600)
        return 0
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
