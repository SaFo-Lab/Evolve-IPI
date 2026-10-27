"""Hygiene of the public tree: nothing private, no secret, nothing large, and a README that says what the code does.

Every file git would put in the repository (tracked, or untracked and not ignored) is scanned, except the contents
of the two framework submodules. A failure prints one line per finding: ``path:line: rule: matched text``. A secret
is printed only by its first characters.

The rules are boundary regexes. A name must not be glued to a letter or a digit, so that a longer word, a hash or an id
that merely contains a codename is not flagged; an underscore or a hyphen does not glue, so that a file or an identifier
named after a codename is caught. ``test_rules_*`` pins what each rule catches and what it lets through.

The names that the rules look for are stored rot13-encoded and decoded when the module loads: the scan covers this file
as well, and a file in a public repository that spelled out the private names would publish them.
"""

from __future__ import annotations

import codecs
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from hygiene_rules import KEY_RULES, decode, make_rule

PUB = Path(__file__).resolve().parents[1]
SUBMODULES = ("third_party/OpenClaw-RL", "third_party/MetaClaw")
MAX_BYTES = 5 * 1024 * 1024
README = PUB / "README.md"


def _name(encoded: str) -> str:
    return codecs.decode(encoded, "rot13")


def _lit(encoded: str) -> str:
    """The regex for a rot13-encoded name, matched literally."""
    return re.escape(_name(encoded))


_W = r"(?<![A-Za-z0-9_])"  # not preceded by a word character
_E = r"(?![A-Za-z0-9_])"  # not followed by one
_WA = r"(?<![A-Za-z0-9])"  # not preceded by a letter or a digit: an underscore joins a codename to its file's name
_EA = r"(?![A-Za-z0-9])"

# The names, for the rules and for the samples that the rules are tested on.
HOME_PATH = _name("/ubzr/kyvh316")
MAC_PATH = _name("/Hfref/kvalnajnat")
MAC_PATH_2 = _name("/Hfref/kvnbtratyvh")
LOGIN = _name("kyvh316")
AZURE_HOST = _name("bcranv.nmher.pbz")
AUTHOR = _name("Punbjrv")  # an author's first name: allowed only as part of "<AUTHOR> Xiao" or "Xiao, <AUTHOR>"
SERVICES_NAME = _name("freivprf_pber")
SETUP_NAME = _name("frghc_jbexfcnpr")
BENCHMARK_NAME = _name("NtragQbwb")


# Names and places that identify a person or a machine.
PRIVATE_RULES = (
    make_rule("private-path", "|".join(_lit(e) for e in ("/ubzr/kyvh316", "/Hfref/kvalnajnat", "/Hfref/kvnbtratyvh"))),
    make_rule("private-name", _lit("kyvh316")),
    make_rule("private-host", _lit("bcranv.nmher.pbz")),
)
# A machine is named after a co-author. In any case and any context the name is private; only the author's name itself
# is not: the capitalised first name, directly before or after the surname ("Chaowei Xiao", "Xiao, Chaowei").
_AUTHOR_NAME = re.compile(_lit("punbjrv"), re.IGNORECASE)
_SURNAME_AFTER = re.compile(r" Xiao(?![A-Za-z])")  # not "Xiaogeng"
_SURNAME_BEFORE = re.compile(r"(?<![A-Za-z])Xiao, $")

# Internal codenames, as identifiers, file names, labels or comments. An underscore joins a codename to another word
# ("<name>_driver.py", "run_<name>.sh"); a letter or a digit does ("Inkling", "te1x").
CODENAME_RULES = (
    make_rule("codename", _WA + _lit("j64") + _EA),
    make_rule("codename", _lit("jrvtug-64")),
    make_rule("codename", _WA + _lit("znez") + _EA),
    make_rule("codename", _WA + _lit("anez") + _EA),
    make_rule("codename", _WA + _lit("bnez") + _EA),
    make_rule("codename", _WA + _lit("vax") + _EA),
    make_rule("codename", _WA + _lit("gr") + _EA),
    make_rule("codename", _WA + _lit("genva") + "[-_]" + _lit("riny") + _EA),
    make_rule("codename", _lit("fip_qevire")),
    make_rule("codename", "(?:" + _W + _lit("frrq1") + r"[/\\]|[/\\]" + _lit("frrq1") + _E + ")"),
    make_rule("codename", r"\b(?:" + _lit("Yvar N") + "|" + _lit("Yvar O") + ")" + _E, flags=0),
)
# Two experiment names and a third that is also a mail id: the same letters and digits inside a hash, a uuid, an id
# like "call" + 12 hex digits, or a longer word are not codenames.
EXPERIMENT_RULE = make_rule("codename", _WA + "[eE][17]" + _EA)

# Names that the byte-identical service programs keep from the code they were taken from.
SERVICE_NAME_RULES = (
    make_rule("internal-name", _W + re.escape(SERVICES_NAME) + _E, flags=0),
    make_rule("internal-name", _W + re.escape(SETUP_NAME) + _E, flags=0),
    make_rule("internal-name", _W + re.escape(BENCHMARK_NAME) + _E),
)
SERVICES = "src/evolve_ipi/environment/services"
# (file, string) pairs. The programs are pinned byte for byte by a hash, so they keep these strings; nothing else may.
# test_the_allowlist_is_exactly_what_the_programs_contain fails when an entry no longer matches.
ALLOWED: tuple[tuple[str, str], ...] = (
    (f"{SERVICES}/_lib.py", SERVICES_NAME),
    (f"{SERVICES}/travel", SERVICES_NAME),
    (f"{SERVICES}/_lib.py", SETUP_NAME),
    (f"{SERVICES}/web", BENCHMARK_NAME),
)

# A long run of base64 is a key too (KEY_RULES, in hygiene_rules, holds the patterns of the services we use). Pure hex
# is a hash, and base64 that lacks a lower-case letter, an upper-case letter or a digit is not random bytes: the file
# paths of uv.lock and a Dockerfile directive are the long runs that this lets through.
BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{40,}")

# The content-addressed URLs of uv.lock hold two hex path segments, which can read like an experiment name.
_PACKAGE_URL = re.compile(r"https?://files\.pythonhosted\.org/packages/[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{60}/")
# The e-mail ids of the e-mail task's data are JSON strings of a letter and digits.
_MAIL_ID = re.compile(r'"[eE][0-9]+"')


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.text}"


def _shown(text: str, secret: bool) -> str:
    text = text.strip()
    if secret:
        return f"{text[:6]}... ({len(text)} characters)"
    return text if len(text) <= 80 else text[:77] + "..."


def _is_random_base64(run: str) -> bool:
    return (
        not re.fullmatch(r"[0-9a-fA-F]+", run)
        and re.search("[a-z]", run) is not None
        and re.search("[A-Z]", run) is not None
        and re.search("[0-9]", run) is not None
    )


def _author_findings(path: str, number: int, line: str) -> list[Finding]:
    found = []
    for match in _AUTHOR_NAME.finditer(line):
        listed = _SURNAME_AFTER.match(line, match.end()) or _SURNAME_BEFORE.search(line, 0, match.start())
        if match.group() != AUTHOR or not listed:
            found.append(Finding(path, number, "private-name", match.group()))
    return found


def scan_line(
    path: str,
    number: int,
    line: str,
    allowed: set[tuple[str, str]] | None = None,
    *,
    private_only: bool = False,
) -> list[Finding]:
    """The findings of one line of ``path``; the (file, string) pairs of ``ALLOWED`` that matched go to ``allowed``.

    ``private_only`` is for what is not text (a binary file read as latin-1 is full of short accidental matches): only
    the names, paths and hosts that identify a person or a machine are looked for."""
    found: list[Finding] = []
    for rule in PRIVATE_RULES if private_only else (*PRIVATE_RULES, *CODENAME_RULES):
        for match in rule.pattern.finditer(line):
            found.append(Finding(path, number, rule.name, _shown(match.group(), rule.secret)))
    found += _author_findings(path, number, line)
    if private_only:
        return found
    experiment_line = _PACKAGE_URL.sub("", line)
    if path.startswith("data/"):
        experiment_line = _MAIL_ID.sub("", experiment_line)
    for match in EXPERIMENT_RULE.pattern.finditer(experiment_line):
        found.append(Finding(path, number, EXPERIMENT_RULE.name, match.group()))
    for rule in SERVICE_NAME_RULES:
        for match in rule.pattern.finditer(line):
            if (path, match.group()) in ALLOWED:
                if allowed is not None:
                    allowed.add((path, match.group()))
            else:
                found.append(Finding(path, number, rule.name, match.group()))
    for rule in KEY_RULES:
        for match in rule.pattern.finditer(line):
            found.append(Finding(path, number, rule.name, _shown(match.group(), True)))
    for match in BASE64_RUN.finditer(line):
        if _is_random_base64(match.group()):
            found.append(Finding(path, number, "base64-run", _shown(match.group(), True)))
    return found


def scan_path(path: str) -> list[Finding]:
    """The names a file path holds: a file named after a codename is as private as a line that says it."""
    return scan_line(path, 0, path)


# --- the tree ---------------------------------------------------------------------------------------------------------


def tree_files(root: Path = PUB) -> list[str]:
    """Every file git would put in the repository, relative to ``root``; none inside a framework submodule."""
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("not a git checkout: there is no list of the files the repository holds")
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    paths = sorted({item.decode("utf-8") for item in listing.split(b"\0") if item})
    return [
        path
        for path in paths
        if not any(path == sub or path.startswith(sub + "/") for sub in SUBMODULES) and (root / path).is_file()
    ]


def scan_bytes(path: str, data: bytes, allowed: set[tuple[str, str]] | None = None) -> list[Finding]:
    text, is_text = decode(data)
    found: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        found += scan_line(path, number, line, allowed, private_only=not is_text)
    return found


def scan_tree(root: Path = PUB) -> tuple[list[Finding], set[tuple[str, str]]]:
    findings: list[Finding] = []
    allowed: set[tuple[str, str]] = set()
    for path in tree_files(root):
        findings += scan_path(path)
        findings += scan_bytes(path, (root / path).read_bytes(), allowed)
    return findings, allowed


@pytest.fixture(scope="module")
def scanned() -> tuple[list[Finding], set[tuple[str, str]]]:
    return scan_tree()


def _report(findings: list[Finding]) -> str:
    return "\n" + "\n".join(str(finding) for finding in findings)


def test_the_scan_sees_the_repository() -> None:
    """An empty file list would make every scan below pass."""
    files = tree_files()
    assert len(files) > 100, f"only {len(files)} files listed"
    for expected in (
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "tests/test_hygiene.py",
        "src/evolve_ipi/cli.py",
        "data/tasks/file-create/test.jsonl",
    ):
        assert expected in files
    assert not [path for path in files if path.startswith(SUBMODULES)], "a submodule's files are scanned"


def test_nothing_private_and_no_secret_in_the_tree(scanned: tuple[list[Finding], set[tuple[str, str]]]) -> None:
    findings, _ = scanned
    assert not findings, _report(findings)


def test_the_allowlist_is_exactly_what_the_programs_contain(
    scanned: tuple[list[Finding], set[tuple[str, str]]],
) -> None:
    _, allowed = scanned
    assert allowed == set(ALLOWED), f"allowed but not found: {sorted(set(ALLOWED) - allowed)}; found: {sorted(allowed)}"


def test_no_file_is_larger_than_5_mb() -> None:
    sizes = {path: (PUB / path).stat().st_size for path in tree_files()}
    large = [f"{path}:0: size: {size} bytes" for path, size in sizes.items() if size > MAX_BYTES]
    assert not large, "\n" + "\n".join(large)


# Files whose bytes a hash or a test pins: git must not rewrite their line endings on any platform (.gitattributes).
PINNED = (
    "src/evolve_ipi/environment/services/",
    "src/evolve_ipi/environment/workspace_files/",
    "data/",
    "third_party/patches/",
    "docker/entrypoint.sh",
)


def test_git_never_rewrites_the_line_endings_of_pinned_files() -> None:
    files = [path for path in tree_files() if path.startswith(PINNED)]
    assert all(any(path.startswith(prefix) for path in files) for prefix in PINNED), "a pinned path is gone"
    answer = subprocess.run(
        ["git", "check-attr", "-z", "--stdin", "text"],
        cwd=PUB,
        input="\0".join(files).encode("utf-8"),
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    fields = answer.split("\0")[:-1]
    state = {fields[i]: fields[i + 2] for i in range(0, len(fields), 3)}
    bad = [f"{path}:0: line-endings: text is {state[path]!r}, not 'unset'" for path in files if state[path] != "unset"]
    assert not bad, "\n" + "\n".join(bad)


def test_env_example_holds_no_value() -> None:
    values = []
    for number, line in enumerate((PUB / ".env.example").read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() and not line.startswith("#"):
            name, _, value = line.partition("=")
            if value.strip():
                values.append(f".env.example:{number}: value: {name} has a value")
    assert not values, "\n" + "\n".join(values)


def test_env_example_says_not_to_quote_values() -> None:
    """``docker run --env-file`` keeps the quotes as part of the value, so a quoted key would be sent with them."""
    comments = " ".join(
        line for line in (PUB / ".env.example").read_text(encoding="utf-8").splitlines() if line.startswith("#")
    )
    assert "without quotes" in comments and "--env-file" in comments


def test_no_env_file_is_in_the_tree() -> None:
    assert [path for path in tree_files() if Path(path).name.startswith(".env") and path != ".env.example"] == []


# --- the README -------------------------------------------------------------------------------------------------------


def headings(text: str) -> Iterator[tuple[int, str]]:
    """The headings outside code fences, as ``(line number, text)``: ``#`` headings and underlined ones."""
    fenced = False
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        if line.startswith("```"):
            fenced = not fenced
        elif fenced:
            continue
        elif re.match(r"#{1,6}\s", line):
            yield number, line
        elif re.fullmatch(r"(=+|-{2,})\s*", line) and number > 1:
            above = lines[number - 2]
            if above.strip() and not re.match(r"\s*([|>#*+-]|\d+[.)]\s)", above):
                yield number - 1, above


# What a README that tells how the runs differ from the paper's, or what was changed in a framework, would say.
HEADING_TRIPWIRE = re.compile(r"differ|deviat|modif|patch|pars(?:er|ing)", re.IGNORECASE)
PROSE_TRIPWIRE = re.compile(
    r"\bpatch\w*|\bdiffers? from the paper|\bdifferences from\b|\bwe modified\b|\btool[- ]calls?\s+pars\w*",
    re.IGNORECASE,
)


def readme_tripwires(text: str) -> list[str]:
    bad = [f"README.md:{number}: heading: {line}" for number, line in headings(text) if HEADING_TRIPWIRE.search(line)]
    for number, line in enumerate(text.splitlines(), start=1):
        bad += [f"README.md:{number}: prose: {match.group()}" for match in PROSE_TRIPWIRE.finditer(line)]
    return bad


def test_readme_has_no_section_about_differences_patches_or_parsing() -> None:
    text = README.read_text(encoding="utf-8")
    bad = [f"README.md:{number}: heading: {line}" for number, line in headings(text) if HEADING_TRIPWIRE.search(line)]
    assert not bad, "\n" + "\n".join(bad)


def test_readme_does_not_describe_patches_or_parsing_rules() -> None:
    bad = [line for line in readme_tripwires(README.read_text(encoding="utf-8")) if ": prose: " in line]
    assert not bad, "\n" + "\n".join(bad)


@pytest.mark.parametrize(
    "text",
    [
        "## Differences from the paper\n",
        "## How this differs\n",
        "## Deviations\n",
        "## Modifications to the frameworks\n",
        "### Tool-call parsing\n",
        "## Parser rules\n",
        "## Patches\n",
        "Parser notes\n------\n",
        "Differences\n===========\n",
        "Framework patches\n---\n",
        "Our runs differ from the paper in two ways.\n",
        "Here the setup differs from the paper's.\n",
        "We modified OpenClaw-RL.\n",
        "The tool calls parsing is described below.\n",
        "Tool-call parsing follows a rule.\n",
        "A patch to the framework.\n",
    ],
)
def test_the_readme_tripwires_catch_what_they_are_for(text: str) -> None:
    assert readme_tripwires(text), text


def test_the_readme_tripwires_let_the_readme_and_ordinary_text_through() -> None:
    assert readme_tripwires(README.read_text(encoding="utf-8")) == []
    table_and_rule = "## Usage\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n- item\n---\n\nThe images differ in size.\n"
    assert readme_tripwires(table_and_rule) == []


def test_readme_names_no_local_path() -> None:
    bad = [
        f"README.md:{number}: local-path: {match.group()}"
        for number, line in enumerate(README.read_text(encoding="utf-8").splitlines(), start=1)
        for match in re.finditer(r"/Users/|/home/", line)
    ]
    assert not bad, "\n" + "\n".join(bad)


def test_readme_has_no_unfinished_marker() -> None:
    """The arXiv link is the one placeholder (the id in the BibTeX entry); a to-do note is not."""
    bad = [
        f"README.md:{number}: unfinished: {match.group()}"
        for number, line in enumerate(README.read_text(encoding="utf-8").splitlines(), start=1)
        for match in re.finditer(r"\bTODO\b|\bTBD\b|\bFIXME\b", line)
    ]
    assert not bad, "\n" + "\n".join(bad)


# --- the rules themselves ---------------------------------------------------------------------------------------------

_KEY_BODY = "abcdefghijklmnopqrstuvwxyz"
HOST = AUTHOR.lower()
_SAMPLES_CAUGHT = [
    (f"path = '{HOME_PATH}/runs'", "private-path"),
    (f"cd {MAC_PATH}/Documents", "private-path"),
    (f"cd {MAC_PATH_2}/x", "private-path"),
    (f"user {LOGIN} logged in", "private-name"),
    (f"https://x.{AZURE_HOST}/openai", "private-host"),
    (f"host={HOST}", "private-name"),
    (f"Host {HOST}", "private-name"),
    (f"ssh {HOST}", "private-name"),
    (f"ssh -p 22 {HOST}", "private-name"),
    (f"scp a {HOST}.cs.example.edu:/x", "private-name"),
    (f"root@{HOST}", "private-name"),
    (f"/data/{HOST}/runs", "private-name"),
    (f"{HOST}-gpu", "private-name"),
    (f"the machine {HOST} is busy", "private-name"),
    (HOST, "private-name"),
    (HOST.upper(), "private-name"),
    (f"thanks to {AUTHOR}.", "private-name"),
    (f"{AUTHOR} says", "private-name"),
    (f"Xiao {AUTHOR}", "private-name"),
    (f"{AUTHOR} Xiaogeng", "private-name"),  # the surname must be a whole word
    (f"ZXiao, {AUTHOR}", "private-name"),
    (AUTHOR[:-3] + AUTHOR[-3:].upper(), "private-name"),
    (f"{_name('j64')} arm", "codename"),
    (f"the {_name('jrvtug-64')} run", "codename"),
    (f"a {_name('znez')} run", "codename"),
    (f"{_name('znez')}_notes.txt", "codename"),
    (f"run_{_name('anez')}.sh", "codename"),
    (f"{_name('j64')}_train.py", "codename"),
    (f"{_name('r1')}_driver.py", "codename"),
    (f"run_{_name('r1')}.sh", "codename"),
    (f"{_name('R7')}_arm", "codename"),
    (f"{_name('vax')}_run = 1", "codename"),
    (f"{_name('gr')}_driver", "codename"),
    (f"{_name('VAX')}_TEXT = 'x'", "codename"),
    (_name("anez"), "codename"),
    (f"--arm {_name('bnez')}", "codename"),
    (f"label: {_name('vax')}", "codename"),
    (f"dir/{_name('gr')}/file", "codename"),
    (f"the {_name('genva')}-{_name('riny')} layer", "codename"),
    (f"{_name('genva')}_{_name('riny')}.py", "codename"),
    (f"import {_name('fip_qevire')}", "codename"),
    (f"runs/{_name('frrq1')}/x", "codename"),
    (f"see {_name('frrq1')}/records", "codename"),
    (f"{_name('Yvar N')} of the paper", "codename"),
    (_name("Yvar O"), "codename"),
    (_name("R1"), "codename"),
    (f"the {_name('r1')} seeds", "codename"),
    (f"{_name('R7')} arm", "codename"),
    (f"{_name('r1')}-seeds/stock", "codename"),
    (f"import {SERVICES_NAME}", "internal-name"),
    (f"python {SETUP_NAME}.py", "internal-name"),
    (f"mirrors {BENCHMARK_NAME}'s suite", "internal-name"),
    ("key " + "s" + "k-" + _KEY_BODY, "api-key"),
    ("key " + "tm" + "l-" + _KEY_BODY, "api-key"),
    ("token h" + "f_" + "a1B2c3D4e5F6" * 3, "api-key"),
    ("AKI" + "AABCDEFGHIJKLMNOP", "api-key"),
    ("Zm9vYmFy" * 7, "base64-run"),
]
_SAMPLES_PASSED = [
    "Xinyan Wang, Xiaogeng Liu, Jun Shao, Chaowei Xiao",
    "author = {Wang, Xinyan and Liu, Xiaogeng and Shao, Jun and Xiao, Chaowei},",
    "Chaowei Xiao",
    "Chaowei Xiao, Jun Shao and Chaowei Xiao.",
    "Corresponding authors: Xinyan Wang and Chaowei Xiao.",
    "inkling-small and Inkling-Small, INKLING_TEXT, thinkingmachines/Inkling-Small",
    "tests, tell, terminal, the, text, step, te1, tex_x, disarm, alarm, marmalade, narms",
    _name("1r1 naq gr1k naq pnyy_r1nop naq 0kR7 naq \\kr7 naq #R1R1R1"),
    "sha256 " + "e1b2" + "c3d4" * 14 + " in a hash",
    "id 1b9d6bcd-" + "e1a2-4b2d-9b5d-ab8dfbbd4bed",
    "https://files.pythonhosted.org/packages/18/" + _name("r1") + "/" + "9" * 60 + "/x-1.0.tar.gz",
    "the file seed10/x and seed1_notes",
    "Line Breaks, line a, Lineage",
    "The disk-usage risk-based task-force html-escape-xxxxxxxxxxxxxxxxxxxxxxxx",
    "sk-short",
    "a sha256: " + "ab12" * 16,
    "check=skip=FromPlatformFlagConstDisallowed",
    f"{SERVICES_NAME}_x and my_{SETUP_NAME}",
]


@pytest.mark.parametrize("line, rule", _SAMPLES_CAUGHT)
def test_rules_catch_what_they_are_for(line: str, rule: str) -> None:
    assert rule in {finding.rule for finding in scan_line("some/file.txt", 1, line)}


@pytest.mark.parametrize("line", _SAMPLES_PASSED)
def test_rules_let_ordinary_text_through(line: str) -> None:
    assert scan_line("some/file.txt", 1, line) == []


def test_the_mail_ids_of_the_email_data_are_allowed_in_data_only() -> None:
    line = '{"argv_prefix": ["read", "' + _name("r7") + '"]}'
    assert scan_line("data/tasks/email-invite/test.jsonl", 1, line) == []
    assert [f.rule for f in scan_line("src/evolve_ipi/x.py", 1, line)] == ["codename"]


def test_the_allowlist_is_by_file_and_string() -> None:
    suite = f"present only to mirror {BENCHMARK_NAME}'s slack suite"
    assert scan_line(f"{SERVICES}/web", 1, suite) == []
    assert [f.rule for f in scan_line(f"{SERVICES}/mail", 1, suite)] == ["internal-name"]
    assert [f.rule for f in scan_line(f"{SERVICES}/web", 1, f"import {SERVICES_NAME}")] == ["internal-name"]
    assert [f.rule for f in scan_line("README.md", 1, f"run {SETUP_NAME}.py")] == ["internal-name"]


def test_a_secret_is_not_printed_whole() -> None:
    (finding,) = scan_line("x", 1, "key " + "s" + "k-" + _KEY_BODY)
    assert _KEY_BODY[3:] not in str(finding)
    assert str(finding).startswith("x:1: api-key: sk-abc...")


def test_a_path_is_scanned_like_a_line() -> None:
    assert [f.rule for f in scan_path(f"runs/{_name('znez')}/notes.txt")] == ["codename"]
    assert scan_path(f"{SERVICES}/travel") == []


def test_a_file_that_is_not_utf8_is_still_scanned_for_private_names() -> None:
    image = b"\x89PNG\r\n\x1a\n\x00\xff" + f"Author: {HOME_PATH}".encode() + b"\x00\x81" + f"\nowner {LOGIN}".encode()
    assert {f.rule for f in scan_bytes("a.png", image)} == {"private-path", "private-name"}
    assert [f.rule for f in scan_bytes("a.png", b"\x00\xfe" + HOST.encode() + b"\xff")] == ["private-name"]


def test_a_utf16_file_with_a_byte_order_mark_is_scanned() -> None:
    for bom, encoding in ((codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be")):
        data = bom + f"line one\n{MAC_PATH}/x\n".encode(encoding)
        assert [(f.line, f.rule) for f in scan_bytes("a.txt", data)] == [(2, "private-path")]
    assert decode(codecs.BOM_UTF16_LE + "ok".encode("utf-16-le")) == ("ok", True)


def test_binary_noise_does_not_trip_the_codename_and_key_rules() -> None:
    noise = bytes(range(256)) * 40 + b" " + _name("gr").encode() + b" " + _name("r1").encode() + b" sk-" + b"a1B2" * 12
    assert decode(noise)[1] is False
    assert scan_bytes("a.png", noise) == []


# The three places adapted from a framework's source: the file, the comment it carries, and what the notice names.
ADAPTED_FROM_OPENCLAW_RL = "OpenClaw-RL openclaw-tinker/api_server.py (Apache-2.0)"
ADAPTED_FROM_METACLAW = "MetaClaw metaclaw/api_server.py (MIT)"
ADAPTED = (
    ("src/evolve_ipi/models/server.py", ADAPTED_FROM_OPENCLAW_RL, "openclaw-tinker/api_server.py", "_handle_request"),
    (
        "src/evolve_ipi/models/formats/qwen.py",
        ADAPTED_FROM_OPENCLAW_RL,
        "openclaw-tinker/api_server.py",
        "_extract_tool_calls",
    ),
    ("tests/adapt/metaclaw_helpers.py", ADAPTED_FROM_METACLAW, "metaclaw/api_server.py", "_flatten_message_content"),
)
NOTICES = PUB / "THIRD_PARTY_NOTICES.md"


@pytest.mark.parametrize("path, origin, upstream, function", ADAPTED)
def test_an_adapted_spot_says_so_and_the_notice_names_it(path: str, origin: str, upstream: str, function: str) -> None:
    assert f"# Adapted from {origin}; see THIRD_PARTY_NOTICES.md." in (PUB / path).read_text(encoding="utf-8")
    notice = NOTICES.read_text(encoding="utf-8")
    path_in, upstream_in = f"`{path}`" in notice, upstream in notice
    assert path_in and upstream_in, f"the notice must name `{path}` and `{upstream}`"
    assert function in notice
    framework = "OpenClaw-RL" if "OpenClaw-RL" in origin else "MetaClaw"
    source = PUB / "third_party" / framework / upstream
    if source.is_file():  # the submodule is checked out: the function the notice names is really there
        assert function in source.read_text(encoding="utf-8")


def test_the_notice_holds_both_license_texts_of_the_frameworks() -> None:
    notice = NOTICES.read_text(encoding="utf-8")
    for framework in ("OpenClaw-RL", "MetaClaw"):
        license_file = PUB / "third_party" / framework / "LICENSE"
        if license_file.is_file():
            assert license_file.read_text(encoding="utf-8").strip() in notice, framework
    assert "Copyright 2026 Gen-Verse" in notice
    assert "Copyright (c) 2026 Peng Xia" in notice


def test_every_adapted_from_comment_is_in_the_list_above() -> None:
    found = sorted(
        path
        for path in tree_files()
        if path.endswith(".py")
        and path != "tests/test_hygiene.py"
        and re.search(r"^\s*# Adapted from ", (PUB / path).read_text(encoding="utf-8"), re.MULTILINE)
    )
    assert found == sorted(path for path, *_ in ADAPTED), found
