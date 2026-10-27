"""A stand-in for ``python -m evolve_ipi.models.checkpoints`` in the tests of ``evolve_ipi.adapt.rl``.

    python rl_fake_checkpoints.py --control CONTROL.json collect LOG --stage N --base-model M --out J [--resume-from P]
    python rl_fake_checkpoints.py --control CONTROL.json fingerprint PATH --base-model M --out J --min-diff D

``collect`` runs the real collector on the trainer log; only Tinker's answer is made up: the listing holds the one
checkpoint the log's training client saved, and it loads. ``fingerprint`` writes a passing record, or a failing one
for the calls the control file's ``fail_fingerprint`` counts (0 is the first fingerprint of the test). Every call is
appended to the control file's ``events``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from evolve_ipi.models.checkpoints import CheckpointError, ListedCheckpoint, collect_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("log", type=Path)
    collect.add_argument("--stage", required=True, type=int)
    collect.add_argument("--base-model", required=True)
    collect.add_argument("--out", required=True, type=Path)
    collect.add_argument("--resume-from", default=None)
    fingerprint = commands.add_parser("fingerprint")
    fingerprint.add_argument("path")
    fingerprint.add_argument("--base-model", required=True)
    fingerprint.add_argument("--out", required=True, type=Path)
    fingerprint.add_argument("--min-diff", required=True, type=float)
    args = parser.parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))
    events = Path(control["events"])

    def note(**event: object) -> None:
        with open(events, "a", encoding="utf-8") as out:
            out.write(json.dumps({"t": time.time(), "pid": os.getpid(), **event}) + "\n")

    if args.command == "collect":
        note(event="collect", stage=args.stage, log=str(args.log), resume_from=args.resume_from, env=dict(os.environ))
        try:
            record = collect_stage(
                args.log,
                args.stage,
                args.base_model,
                resume_from=args.resume_from,
                list_fn=lambda client: [ListedCheckpoint(f"tinker://{client}/weights/step_0032", 2_900_000_000)],
                verify_fn=lambda path: None,
            )
        except CheckpointError as error:
            print(f"checkpoints: error: {error}", file=sys.stderr)
            return 1
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return 0

    done = sum(1 for line in events.read_text(encoding="utf-8").splitlines() if '"event": "fingerprint"' in line)
    note(event="fingerprint", path=args.path, index=done)
    passing = done not in control.get("fail_fingerprint", [])
    diff = 0.3 if passing else 0.001
    record = {
        "path": args.path,
        "base_model": args.base_model,
        "probe_tokens": 46,
        "mean_abs_logprob_diff_vs_base": diff,
        "min_diff": args.min_diff,
        "non_base": diff > args.min_diff,
        "seconds": 0.1,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if not passing:
        print(f"checkpoints: error: {args.path} is indistinguishable from the base model", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
