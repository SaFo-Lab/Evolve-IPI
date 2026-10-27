"""A stand-in for the RL trainer process in the tests of ``evolve_ipi.adapt.rl``.

    python fake_trainer.py --control CONTROL.json <the trainer's flags>

It logs what OpenClaw-RL's trainer logs, holds the policy port as the trainer does, and is told by a test's scripted
episodes how many samples each episode submitted:

    POST /fake/episode {"samples": n, "scores": [s, ...]}

logs one "[Server] submitted" line per sample and, as in the trainer, one update for every batch the samples complete,
then answers {"submitted": ..., "updates": ...}. The control file says what each start does (``behaviors``, one entry
per start, in order) and where the events go: every start, update, SIGTERM and exit is appended to ``events`` as a
JSON line. The process exits when its parent is gone, and after ten minutes in any case.

A start's behavior can hold:
    omit_entry_line, omit_restored   leave out the trainer entry's line or the restored-state line
    fatal_after: n                   log a traceback once n samples were submitted
    exit_after: n                    exit (status 3) while the n-th sample is being submitted
    exit_after_episode: n            exit (status 3) right after answering the n-th episode
    exit_at_start: status            exit before getting ready
    never_ready                      never log "collecting batch"
    step_delay_s, delay_from_update  wait this long before each update from that update on (default: every update);
                                     the updates are then logged in order by a worker thread, after the answer
    raw_rows, raw_text               rows (and verbatim text) written to the record dir's raw_train.jsonl at start
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

LIFETIME_S = 600
GREEN, RESET = "\033[32m", "\033[0m"

_print_lock = threading.Lock()


def say(line: str) -> None:
    with _print_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def stamp(logger: str) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S") + f" INFO {logger}: "


class Events:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def add(self, **event: Any) -> None:
        event.setdefault("t", time.time())
        event.setdefault("pid", os.getpid())
        with self.lock, open(self.path, "a", encoding="utf-8") as out:
            out.write(json.dumps(event) + "\n")
            out.flush()

    def starts(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if '"event": "start"' in line)


class Trainer:
    def __init__(self, args: argparse.Namespace, behavior: dict[str, Any], events: Events) -> None:
        self.args = args
        self.behavior = behavior
        self.events = events
        self.lock = threading.Lock()
        self.submitted = 0
        self.drained = 0
        self.update = 0
        self.episodes = 0
        self.pending: list[float] = []
        self.batches: queue.Queue[list[float]] = queue.Queue()
        self.delay = float(behavior.get("step_delay_s", 0))
        self.delay_from = int(behavior.get("delay_from_update", 1))
        if self.delay:
            threading.Thread(target=self.train_in_order, daemon=True).start()

    def train_in_order(self) -> None:
        """The delayed updates, one after the other, as the trainer's loop takes them."""
        while True:
            batch = self.batches.get()
            if self.update + 1 >= self.delay_from:
                time.sleep(self.delay)
            with self.lock:
                self.log_update(batch)

    def log_update(self, scores: list[float]) -> None:
        self.update += 1
        k, args = self.update, self.args
        for text in (
            f"[Trainer] step {k}: forward_backward ({args.batch_size} datums) ...",
            f"[Trainer] step {k}: forward_backward done — metrics={{}}",
            f"[Trainer] step {k}: optim_step ...",
            f"[Trainer] step {k}: optim_step done — metrics={{}}",
            f"[Trainer] step {k}: pausing inference for weight swap ...",
            f"[Trainer] step {k}: inference resumed with updated weights",
        ):
            say(stamp("trainer") + text)
        if k % args.save_interval == 0 or k == args.max_steps:
            say(stamp("trainer") + "[Trainer] checkpoint saved: ")
        mean = sum(scores) / len(scores)
        success = sum(1 for s in scores if s > 0) / len(scores)
        say(
            stamp("trainer")
            + f"{GREEN}[Trainer] step {k} done | batch={len(scores)} mean_reward={mean:.3f} success={success:.2f}{RESET}"
        )
        self.events.add(event="update", update=k)
        say(
            stamp("trainer")
            + f"[Trainer] step {k + 1}/{args.max_steps} - collecting batch (size={args.batch_size}) ..."
        )

    def submit(self, scores: list[float], session: str) -> dict[str, int]:
        with self.lock:
            for score in scores:
                say(
                    stamp("api_server")
                    + f"[Server] submitted session={session} idx={self.submitted} score={score:.1f} exclude=False"
                )
                self.submitted += 1
                self.pending.append(score)
                if self.behavior.get("fatal_after") == self.submitted:
                    say("Traceback (most recent call last):")
                    say('  File "trainer.py", line 1, in <module>')
                    say("RuntimeError: the trainer broke")
                if self.behavior.get("exit_after") == self.submitted:
                    self.events.add(event="exit", reason="exit_after")
                    os._exit(3)
            batches = []
            while len(self.pending) >= self.args.batch_size:
                batches.append(self.pending[: self.args.batch_size])
                del self.pending[: self.args.batch_size]
            self.episodes += 1
            for batch in batches:
                if self.delay:
                    self.batches.put(batch)
                else:
                    self.log_update(batch)
            return {"submitted": self.submitted, "updates": self.update, "episode": self.episodes}


def handler_for(trainer: Trainer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            say(f'INFO:     127.0.0.1 - "{self.command} {self.path} HTTP/1.1" {args[1] if len(args) > 1 else ""}')

        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            samples = int(body.get("samples", 0))
            scores = [float(s) for s in body.get("scores") or [1.0] * samples]
            answer = trainer.submit(scores[:samples], body.get("session") or str(uuid.uuid4()))
            data = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            if trainer.behavior.get("exit_after_episode") == answer["episode"]:
                trainer.events.add(event="exit", reason="exit_after_episode")
                threading.Timer(0.05, os._exit, args=(3,)).start()

    return Handler


def watch_parent(parent: int) -> None:
    started = time.monotonic()
    while True:
        if os.getppid() != parent or time.monotonic() - started > LIFETIME_S:
            os._exit(1)
        time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--prm-m", required=True, type=int)
    parser.add_argument("--lora-rank", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--save-interval", required=True, type=int)
    parser.add_argument("--proxy-host", required=True)
    parser.add_argument("--proxy-port", required=True, type=int)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--resume-from-ckpt", default="")
    args = parser.parse_args()
    threading.Thread(target=watch_parent, args=(os.getppid(),), daemon=True).start()

    control = json.loads(args.control.read_text(encoding="utf-8"))
    events = Events(Path(control["events"]))
    index = events.starts()
    behaviors = control.get("behaviors", [])
    behavior = behaviors[index] if index < len(behaviors) else {}
    events.add(event="start", index=index, argv=sys.argv[sys.argv.index("--method") :], env=dict(os.environ))

    def on_sigterm(signum: int, frame: Any) -> None:
        events.add(event="sigterm")
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_sigterm)
    if "exit_at_start" in behavior:
        events.add(event="exit", reason="exit_at_start")
        sys.exit(int(behavior["exit_at_start"]))

    from evolve_ipi.adapt.rl_entry import family_of

    resume = args.resume_from_ckpt
    if not behavior.get("omit_entry_line"):
        say(f"[rl_entry] family={family_of(args.model_name)} model={args.model_name} resume={resume or 'none'}")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    with open(args.record_dir / "raw_train.jsonl", "a", encoding="utf-8") as dump:
        dump.writelines(json.dumps(row) + "\n" for row in behavior.get("raw_rows", []))
        dump.write(behavior.get("raw_text", ""))
    say(stamp("trainer") + "[Trainer] wandb init skipped")
    say(stamp("trainer") + f"[Trainer] loading tokenizer for {args.model_name} ...")
    say(stamp("trainer") + "[Trainer] connecting to Tinker ...")
    client = f"{uuid.uuid4()}:train:0"
    say(stamp("tinker.lib.public_interfaces.service_client") + f"TrainingClient initialized for model {client}")
    if resume:
        say(stamp("trainer") + f"[Trainer] resuming from checkpoint: {resume}")
        if not behavior.get("omit_restored"):
            say(
                f"[rl_entry] restored LoRA weights AND optimizer state from {resume} (load_state_with_optimizer, awaited)"
            )
    say(stamp("trainer") + "[Trainer] initial sampling client ready")
    say(f"[rl_entry] keeping records in {args.record_dir / 'conversations_all.jsonl'}")
    if behavior.get("never_ready"):
        time.sleep(LIFETIME_S)
    trainer = Trainer(args, behavior, events)
    server = ThreadingHTTPServer((args.proxy_host, args.proxy_port), handler_for(trainer))
    say(stamp("trainer") + f"[Trainer] proxy starting at {args.proxy_host}:{args.proxy_port} (method={args.method})")
    say(stamp("trainer") + f"[Trainer] step 1/{args.max_steps} - collecting batch (size={args.batch_size}) ...")
    server.serve_forever(poll_interval=0.05)


if __name__ == "__main__":
    main()
