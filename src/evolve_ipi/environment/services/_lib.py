"""Shared helpers for the seven service programs.

Standard library only; syntax must stay Python 3.9 compatible because the agent's
exec tool inherits the gateway daemon's PATH, which may resolve python3 to the
macOS system interpreter (3.9.6).

State lives OUTSIDE the workspace, one JSON per service, in the directory named
by svc/config.json (written by services_core.install). Every invocation of every
program - including --help and failures - appends one line to access.log there;
that log is the only evidence of "the agent read the carrier" and of the
program error rate, so it must never be skipped.
"""
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY = "2024-05-15"
ME_NAME = "Alex Morgan"
ME_EMAIL = "alex.morgan@northwind.example"
ME_SLACK = "alex"


class ServiceError(Exception):
    """A semantic error (unknown id, bad date, insufficient funds): exit code 1."""


def state_dir():
    cfg = os.path.join(HERE, "config.json")
    try:
        with open(cfg, encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        raise ServiceError("svc/config.json is missing; run setup_workspace.py first")
    sd = d.get("state_dir")
    if not sd or not os.path.isdir(sd):
        raise ServiceError("state dir %r (from svc/config.json) does not exist" % sd)
    return sd


def load(name):
    path = os.path.join(state_dir(), name + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ServiceError("state file %s is missing; the environment was not reset" % path)


def save(name, obj):
    path = os.path.join(state_dir(), name + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def log_access(program, argv, rc):
    line = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "program": program,
            "argv": list(argv), "rc": rc}
    with open(os.path.join(state_dir(), "access.log"), "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def check_date(s):
    try:
        datetime.date.fromisoformat(s)
    except (TypeError, ValueError):
        raise ServiceError("dates must be written as YYYY-MM-DD, got %r" % (s,))
    return s


def check_time(s):
    try:
        datetime.datetime.strptime(s, "%H:%M")
    except (TypeError, ValueError):
        raise ServiceError("times must be written as HH:MM, got %r" % (s,))
    return s


def check_amount(s):
    try:
        v = float(str(s).replace(",", ""))
    except ValueError:
        raise ServiceError("amount must be a number, got %r" % (s,))
    if v <= 0:
        raise ServiceError("amount must be positive, got %r" % (s,))
    return round(v, 2)


def find(items, key, value, what):
    """Case-insensitive lookup of items[*][key] == value; ServiceError if absent."""
    want = str(value).strip().casefold()
    for it in items:
        if str(it.get(key, "")).strip().casefold() == want:
            return it
    raise ServiceError("no %s %r" % (what, value))


def next_id(items, prefix):
    n = 0
    for it in items:
        v = it.get("id", "")
        if isinstance(v, str) and v.startswith(prefix) and v[len(prefix):].isdigit():
            n = max(n, int(v[len(prefix):]))
    return "%s%d" % (prefix, n + 1)


def run(program, main):
    """Entry wrapper: main(argv) -> exit code. Logs every invocation, then exits."""
    argv = sys.argv[1:]
    rc = 0
    try:
        rc = main(argv) or 0
    except SystemExit as e:                 # argparse: --help exits 0, usage errors exit 2
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except ServiceError as e:
        print("error: %s" % e, file=sys.stderr)
        rc = 1
    finally:
        try:
            log_access(program, argv, rc)
        except Exception as e:              # a call that cannot be logged is itself a failure
            print("error: cannot write access.log: %s" % e, file=sys.stderr)
            rc = rc or 1
    sys.exit(rc)
