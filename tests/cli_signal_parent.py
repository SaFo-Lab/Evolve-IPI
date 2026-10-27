"""A command line that is stopped by a signal, for the signal tests: `adapt rl` with its startup and its adaptation
replaced by a stand-in that starts one child process and waits.

    python cli_signal_parent.py WORKDIR polite|stubborn [GRACE_S]

A signal named in ``IGNORE_SIGNAL`` is ignored before the command starts, as `nohup` ignores SIGHUP.

It prints ``ready`` once the child runs and has written its pid to ``WORKDIR/child.pid``. A ``stubborn`` child ignores
SIGTERM, so stopping it takes the whole grace period and ends with SIGKILL, as with a trainer that does not stop.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from evolve_ipi import cli, runner
from evolve_ipi.procs import ManagedProcess

CHILD = """
import os, signal, sys, time
if sys.argv[2] == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path = __import__("pathlib").Path
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(300)
"""


def main() -> int:
    if os.environ.get("IGNORE_SIGNAL"):
        signal.signal(getattr(signal, os.environ["IGNORE_SIGNAL"]), signal.SIG_IGN)
    work, mode = Path(sys.argv[1]), sys.argv[2]
    grace = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    pid_file = work / "child.pid"

    def adapt(start: object, combo: object, *, progress: runner.Progress) -> None:
        progress.begin("qwen3.8-27b / file-create / rl", work)
        child = ManagedProcess(
            "the child",
            [sys.executable, "-c", CHILD, str(pid_file), mode],
            {"PATH": os.environ["PATH"]},
            work / "child.log",
        ).start()
        try:
            while not pid_file.exists():
                time.sleep(0.05)
            print("ready", flush=True)
            time.sleep(300)
        finally:
            child.stop(grace_s=grace)

    runner.startup = lambda *args, **options: None  # type: ignore[assignment]
    runner.adapt = adapt  # type: ignore[assignment]
    return cli.main(["adapt", "rl", "--model", "qwen3.8-27b", "--task", "file-create"])


if __name__ == "__main__":
    sys.exit(main())
