"""The Docker image: what a build is sent, what the Dockerfile pins, and what the image holds.

The tests without the ``docker`` marker read the Dockerfile, the entrypoint and ``.dockerignore`` and need no Docker.
The tests marked ``docker`` build the image for linux/amd64 and run it. A plain ``pytest`` leaves them out; run with
``-m docker`` to run them. They are skipped when ``docker info`` fails.
On an arm64 machine the first build runs under emulation and takes up to half an hour. No test passes a real ``.env``
or key to a container: the ``.env`` files that appear here are fakes with a canary in them.

Set ``EVOLVE_IPI_TEST_IMAGE`` to test an image that is already built instead of building ``evolve-ipi``.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from hygiene_rules import KEY_RULES, decode  # the one list of key patterns; hygiene scans the sources with it

from evolve_ipi.agent.openclaw import OPENCLAW_VERSION
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.thirdparty import PATCHES, THIRD_PARTY

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "Dockerfile"
DOCKERIGNORE = REPO / ".dockerignore"
ENTRYPOINT = REPO / "docker" / "entrypoint.sh"
REQUIREMENTS = REPO / "docker" / "requirements-metaclaw.txt"

IMAGE = os.environ.get("EVOLVE_IPI_TEST_IMAGE") or "evolve-ipi"
HOSTNAME = "evolve-ipi"
NODE_VERSION = "v22.23.2"
BUILD_TIMEOUT_S = 30 * 60
RUN_TIMEOUT_S = 20 * 60

# The names under which a key would be set: none may appear in the Dockerfile or in the image.
SECRET_NAMES = re.compile(
    r"\b(TINKER_API_KEY|HF_TOKEN|EVOLVER_[A-Z_]+|OPENAI_[A-Z_]+|AZURE_[A-Z_]+"
    r"|[A-Z_]*API_KEY|[A-Z_]*TOKEN|[A-Z_]*SECRET[A-Z_]*|[A-Z_]*PASSWORD)\b"
)


def key_values(lines: Sequence[str] | Iterator[str]) -> dict[str, str]:
    """``name=value`` lines as a dict."""
    pairs: dict[str, str] = {}
    for line in lines:
        name, _, value = line.partition("=")
        pairs[name] = value
    return pairs


# --- the Dockerfile ----------------------------------------------------------------------------------------


def instructions(text: str) -> list[tuple[str, str]]:
    """The ``(keyword, arguments)`` of each instruction: continuation lines joined, comments dropped."""
    found: list[tuple[str, str]] = []
    pending = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") or (not line and not pending):
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        whole, pending = (pending + line).strip(), ""
        keyword, _, arguments = whole.partition(" ")
        found.append((keyword.upper(), arguments.strip()))
    return found


@pytest.fixture(scope="module")
def dockerfile() -> list[tuple[str, str]]:
    return instructions(DOCKERFILE.read_text(encoding="utf-8"))


def arguments_of(parsed: Sequence[tuple[str, str]], keyword: str) -> list[str]:
    return [arguments for word, arguments in parsed if word == keyword]


def final_environment(parsed: Sequence[tuple[str, str]]) -> dict[str, str]:
    """The ``ENV`` of the last stage, which is the image."""
    last_from = max(index for index, (word, _) in enumerate(parsed) if word == "FROM")
    values: dict[str, str] = {}
    for word, arguments in parsed[last_from:]:
        if word == "ENV":
            values.update(key_values(shlex.split(arguments)))
    return values


def pinned_python(parsed: Sequence[tuple[str, str]]) -> str:
    versions = {version for _, arguments in parsed for version in re.findall(r"\b3\.12\.\d+\b", arguments)}
    assert len(versions) == 1, f"the Dockerfile must pin one Python 3.12 patch version, it names {sorted(versions)}"
    return versions.pop()


def pinned_uv(parsed: Sequence[tuple[str, str]]) -> str:
    versions = {
        match.group(1)
        for arguments in arguments_of(parsed, "FROM")
        if (match := re.search(r"astral-sh/uv:(\d+\.\d+\.\d+)@sha256:", arguments))
    }
    assert len(versions) == 1, f"the Dockerfile must pin one uv version, it names {sorted(versions)}"
    return versions.pop()


def test_every_image_is_for_linux_amd64_and_pinned_by_digest(dockerfile: list[tuple[str, str]]) -> None:
    stages: set[str] = set()
    for arguments in arguments_of(dockerfile, "FROM"):
        tokens = shlex.split(arguments)
        if tokens[0].startswith("--platform="):
            assert tokens[0] == "--platform=linux/amd64", arguments
            assert re.fullmatch(r"[\w./-]+:[\w.-]+@sha256:[0-9a-f]{64}", tokens[1]), f"{tokens[1]}: no tag and digest"
            assert not tokens[1].split("@")[0].endswith(":latest")
        else:  # a stage built earlier in the file takes the platform of its base
            assert tokens[0] in stages, f"{arguments}: an image from a registry must say --platform=linux/amd64"
        if "AS" in tokens:
            stages.add(tokens[-1])


def test_node_openclaw_and_uv_are_pinned_to_the_versions_of_the_experiments(dockerfile: list[tuple[str, str]]) -> None:
    froms = arguments_of(dockerfile, "FROM")
    assert f"node:{NODE_VERSION.removeprefix('v')}-bookworm-slim@sha256:" in froms[0]
    runs = "\n".join(arguments_of(dockerfile, "RUN"))
    assert f"npm install -g openclaw@{OPENCLAW_VERSION} " in runs
    assert f'test "$(node --version)" = "{NODE_VERSION}"' in runs
    assert f'test "$(openclaw --version)" = "{OPENCLAW_VERSION}"' in runs
    assert re.fullmatch(r"\d+\.\d+\.\d+", pinned_uv(dockerfile))
    assert re.fullmatch(r"3\.12\.\d+", pinned_python(dockerfile))


def test_python_packages_come_from_the_lock_and_the_pinned_requirements(dockerfile: list[tuple[str, str]]) -> None:
    runs = arguments_of(dockerfile, "RUN")
    syncs = [run for run in runs if run.startswith("uv sync")]
    assert len(syncs) == 2  # the dependencies first, then the project
    for sync in syncs:
        assert "--locked" in sync and "--no-dev" in sync, sync
    assert "--no-install-project" in syncs[0] and "--no-install-project" not in syncs[1]
    everything = "\n".join(runs)
    assert "COPY docker/requirements-metaclaw.txt " in "\n".join(f"COPY {a}" for a in arguments_of(dockerfile, "COPY"))
    assert "--no-deps --index-url https://download.pytorch.org/whl/cpu" in everything
    assert 'uv pip install --python "$python" --no-deps -r ' in everything
    assert "-e /work/third_party/MetaClaw" in everything and "uv pip check" in everything
    assert "-m evolve_ipi.thirdparty apply" in everything and "-m evolve_ipi.thirdparty verify" in everything


def test_the_metaclaw_requirements_pin_the_torch_of_the_package() -> None:
    requirements = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    (torch,) = [line for line in requirements if line.startswith("torch==")]
    assert f'"{torch}"' in (REPO / "pyproject.toml").read_text(encoding="utf-8"), "both environments run one torch"
    options = [line for line in requirements if line.startswith(("-", "git+")) or "://" in line]
    assert options == [], f"the Dockerfile installs these with --no-deps; no option or URL belongs in them: {options}"


def test_the_image_fixes_what_the_agent_sees_of_the_machine(dockerfile: list[tuple[str, str]]) -> None:
    environment = final_environment(dockerfile)
    assert environment["TZ"] == "America/New_York"
    assert environment["SHELL"] == "/bin/bash"
    assert environment["LANG"] == "C.UTF-8"
    assert arguments_of(dockerfile, "USER") == [], "the agent runs as root, as in the experiments"
    assert environment["UV_PYTHON_INSTALL_DIR"].startswith("/opt/"), "no interpreter may land in the agent's home"
    assert not SECRET_NAMES.search("\n".join(f"{name}=" for name in environment)), sorted(environment)
    assert arguments_of(dockerfile, "ARG") == []


def test_the_image_says_that_it_is_the_image(dockerfile: list[tuple[str, str]]) -> None:
    # The CLI checks what the image cannot fix itself (the host name, an init process, a mounted outputs directory) when
    # this is set. Unlike /.dockerenv it is there under Podman and Kubernetes too.
    assert final_environment(dockerfile)["EVOLVE_IPI_IMAGE"] == "1"


def test_the_image_keeps_the_path_of_its_base_image(dockerfile: list[tuple[str, str]]) -> None:
    # An environment on PATH would change the python3 the agent's tools run, and what `check` runs the services with.
    names = {name for word, arguments in dockerfile if word == "ENV" for name in key_values(shlex.split(arguments))}
    assert "PATH" not in names


@pytest.mark.parametrize(
    "name",
    ["TINKER_API_KEY", "OPENAI_API_KEY", "EVOLVER_MODEL", "HF_TOKEN", "GITHUB_TOKEN", "ACCESS_TOKEN", "DB_PASSWORD"],
)
def test_the_secret_name_pattern_knows_the_usual_names(name: str) -> None:
    assert SECRET_NAMES.search(f"{name}=")


@pytest.mark.parametrize(
    "name",
    [
        "PATH",
        "TZ",
        "HF_HOME",
        "PYTHONUNBUFFERED",
        "UV_PYTHON_INSTALL_DIR",
        "EVOLVE_IPI_IMAGE",
        "TOKENIZERS_PARALLELISM",
    ],
)
def test_the_secret_name_pattern_leaves_the_image_environment_alone(name: str) -> None:
    assert not SECRET_NAMES.search(f"{name}=")


def test_the_environments_are_where_the_package_looks_for_them(
    dockerfile: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("EVOLVE_IPI_RUNTIME", "EVOLVE_IPI_METACLAW_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    paths = RuntimePaths()
    environment = final_environment(dockerfile)
    runs = "\n".join(arguments_of(dockerfile, "RUN"))
    assert f"uv venv --python {pinned_python(dockerfile)} {paths.metaclaw_python.parent.parent};" in runs
    assert f"python={paths.metaclaw_python};" in runs
    assert Path(environment["HF_HOME"]) == paths.hf_home
    assert f"mkdir -p /root/.openclaw-rl /root/.metaclaw {paths.root} " in runs
    assert environment["UV_PROJECT_ENVIRONMENT"] + "/bin/evolve-ipi" in ENTRYPOINT.read_text(encoding="utf-8")


def test_the_dockerfile_checks_the_submodules_before_anything_is_installed(dockerfile: list[tuple[str, str]]) -> None:
    froms = [index for index, (word, _) in enumerate(dockerfile) if word == "FROM"]
    (source,) = [index for index in froms if dockerfile[index][1].endswith(" AS source")]
    stage = dockerfile[source : froms[froms.index(source) + 1]]
    assert source < froms[-1], "stages build in file order with the classic builder: the check must come first"
    (check,) = [arguments for word, arguments in stage if word == "RUN"]
    assert "OpenClaw-RL" in check and "MetaClaw" in check and "ls -A" in check and "exit 1" in check
    assert "git submodule update --init --recursive" in check
    assert "--from=source /work /work" in "\n".join(arguments_of(dockerfile[froms[-1] :], "COPY"))


def test_the_image_starts_in_work_and_hands_every_argument_to_the_entrypoint(dockerfile: list[tuple[str, str]]) -> None:
    assert arguments_of(dockerfile, "WORKDIR")[-1] == "/work"
    assert arguments_of(dockerfile, "ENTRYPOINT") == ['["/work/docker/entrypoint.sh"]']
    assert arguments_of(dockerfile, "CMD") == ['["--help"]']


def test_the_entrypoint_passes_every_argument_on_unchanged(tmp_path: Path) -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert os.access(ENTRYPOINT, os.X_OK), "docker/entrypoint.sh must be executable"
    assert re.search(r'^exec /opt/evolve-ipi/venv/bin/evolve-ipi "\$@"$', text, re.MULTILINE)
    # Run it with a stub in place of evolve-ipi: the arguments arrive one by one, whatever they hold.
    stub = tmp_path / "evolve-ipi"
    stub.write_text('#!/bin/sh\nfor argument in "$@"; do printf "[%s]\\n" "$argument"; done\n', encoding="utf-8")
    stub.chmod(0o755)
    script = tmp_path / "entrypoint.sh"
    script.write_text(text.replace("/opt/evolve-ipi/venv/bin/evolve-ipi", str(stub)), encoding="utf-8")
    script.chmod(0o755)
    arguments = ["check", "two words", "--model=qwen3.8-27b", "", "$HOME", "it's", "*"]
    done = subprocess.run([str(script), *arguments], capture_output=True, text=True, check=True)
    assert done.stdout.splitlines() == [f"[{argument}]" for argument in arguments]


def collected(*options: str) -> list[str]:
    """The test ids that ``pytest`` collects from this file with ``options``, in a process of its own."""
    argv = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *options, __file__]
    done = subprocess.run(argv, cwd=REPO, check=False, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return [line for line in done.stdout.splitlines() if "::" in line]


def test_a_plain_pytest_run_builds_no_image_and_dash_m_docker_still_selects_the_docker_tests() -> None:
    everything = collected("-m", "")  # an empty expression selects everything, whatever the default says
    by_default, docker = collected(), collected("-m", "docker")
    assert docker, "-m docker must select the docker tests"
    assert any(name.endswith("::test_check_passes_in_the_image") for name in docker)
    assert not set(by_default) & set(docker), "a plain run must leave the docker tests out"
    assert sorted(by_default + docker) == sorted(everything)


def test_the_docker_marker_says_what_it_does(pytestconfig: pytest.Config) -> None:
    markers = dict(entry.partition(":")[::2] for entry in pytestconfig.getini("markers"))
    assert markers["docker"].strip() == "builds and runs the evolve-ipi image (minutes)"


def test_pytest_turns_warnings_into_errors() -> None:
    settings = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    assert settings["filterwarnings"] == ["error"]


# --- .dockerignore --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    pattern: str
    regex: re.Pattern[str]
    exception: bool  # a line that starts with "!"


def glob_regex(pattern: str) -> re.Pattern[str]:
    """A ``.dockerignore`` pattern as BuildKit reads it: ``*`` is no ``/``, ``**/`` is any number of directories."""
    unsupported = re.sub(r"[A-Za-z0-9_./*-]", "", pattern)
    if unsupported:
        raise ValueError(f"the pattern {pattern!r} uses {unsupported!r}, which this test does not model")
    out, i = "^", 0
    while i < len(pattern):
        char = pattern[i]
        if char == "*" and pattern[i + 1 : i + 2] == "*":
            i += 1
            if pattern[i + 1 : i + 2] == "/":
                i += 1
            out += ".*" if i + 1 == len(pattern) else "(.*/)?"
        elif char == "*":
            out += "[^/]*"
        else:
            out += re.escape(char)
        i += 1
    return re.compile(out + "$")


def parse_dockerignore(text: str) -> list[Rule]:
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        exception = line.startswith("!")
        pattern = posixpath.normpath(line.removeprefix("!").strip()).lstrip("/")
        if exception and "/" in pattern:
            raise ValueError(f"{line!r} puts a file back below a directory; this test does not model that")
        rules.append(Rule(pattern, glob_regex(pattern), exception))
    return rules


def excluded(path: str, rules: Sequence[Rule]) -> bool:
    """Whether the rules, read from the first line to the last, leave exactly ``path`` out."""
    left_out = False
    for rule in rules:
        if rule.exception == left_out and rule.regex.match(path):
            left_out = not rule.exception
    return left_out


def left_out(path: str, rules: Sequence[Rule]) -> bool:
    """Whether ``path`` is not part of the build context: it, or a directory above it, is excluded."""
    parts = path.split("/")
    return any(excluded("/".join(parts[:count]), rules) for count in range(1, len(parts) + 1))


def context_files(root: Path, rules: Sequence[Rule]) -> list[str]:
    """The files below ``root`` that a build is sent, as ``/``-separated relative paths."""
    sent: list[str] = []
    for directory, directories, files in os.walk(root):
        relative = Path(directory).relative_to(root).as_posix()
        prefix = "" if relative == "." else relative + "/"
        directories[:] = sorted(name for name in directories if not excluded(prefix + name, rules))
        sent += [prefix + name for name in sorted(files) if not excluded(prefix + name, rules)]
    return sorted(sent)


@pytest.fixture(scope="module")
def rules() -> list[Rule]:
    return parse_dockerignore(DOCKERIGNORE.read_text(encoding="utf-8"))


# Files a developer's clone can hold. The first group must stay out of the image, the second must go in.
SECRETS = [
    ".env",
    ".env.local",
    ".env.production",
    "nested/.env",
    "nested/deeper/.env.staging",
    "third_party/MetaClaw/.env",
    "docker/.env",
    "tinker.env",
    "nested/secrets/tinker.env",
]
LOCAL_STATE = [
    ".git/HEAD",
    "nested/.git/config",
    "third_party/MetaClaw/.git",
    "third_party/OpenClaw-RL/.git",
    ".github/workflows/ci.yml",
    ".venv/lib/site.py",
    ".venv-metaclaw/bin/python",
    "nested/.venv/x.py",
    "outputs/run/summary.json",
    "scratch/notes.txt",
    "tmp/t.txt",
    "src/pkg/__pycache__/m.cpython-312.pyc",
    "a/b/c/__pycache__/deep.cpython-312.pyc",
    "nested/__pycache__/note.txt",
    "__pycache__/top.cpython-312.pyc",
    "stray.pyc",
    "nested/stray.pyc",
    "third_party/MetaClaw/aiming_metaclaw.egg-info/PKG-INFO",
    ".pytest_cache/CACHEDIR.TAG",
    "nested/.mypy_cache/x",
    ".ruff_cache/x",
    ".claude/settings.json",
    ".idea/workspace.xml",
    ".vscode/settings.json",
    ".DS_Store",
    "nested/.DS_Store",
]
KEPT = [
    ".env.example",
    ".dockerignore",
    "Dockerfile",
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "docker/entrypoint.sh",
    "docker/requirements-metaclaw.txt",
    "src/evolve_ipi/__init__.py",
    "data/tasks/file-create/task.yaml",
    "third_party/MetaClaw/metaclaw/api_server.py",
    "third_party/OpenClaw-RL/openclaw-tinker/rollout.py",
    "third_party/patches/SHA256SUMS",
    "nested/outputs/keep.txt",
    "nested/tmp/keep.txt",
    "nested/scratch/keep.txt",
    "outputs.txt",
    ".envrc",
    "env.txt",
]
# Neither asserted nor forbidden; the comparison with Docker covers them.
UNDECIDED = ["nested/.env.example", "nested/.env.d/inside.txt"]


def fake_clone(root: Path, canary: str) -> None:
    """A tree with the files above; the files that hold a secret hold the canary."""
    for name in [*SECRETS, *LOCAL_STATE, *KEPT, *UNDECIDED]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"TINKER_API_KEY={canary}\n" if name in SECRETS else f"{name}\n", encoding="utf-8")
    (root / ".dockerignore").write_text(DOCKERIGNORE.read_text(encoding="utf-8"), encoding="utf-8")


def test_dockerignore_leaves_out_every_env_file_but_the_example(tmp_path: Path, rules: list[Rule]) -> None:
    canary = f"canary-{uuid.uuid4().hex}"
    fake_clone(tmp_path, canary)
    sent = context_files(tmp_path, rules)
    assert [name for name in SECRETS if name in sent] == []
    assert ".env.example" in sent
    # Whatever the names, no file that is sent holds the canary.
    assert [name for name in sent if canary in (tmp_path / name).read_text(encoding="utf-8")] == []


def test_dockerignore_leaves_out_version_control_environments_results_and_caches(
    tmp_path: Path, rules: list[Rule]
) -> None:
    fake_clone(tmp_path, "canary")
    sent = context_files(tmp_path, rules)
    assert [name for name in LOCAL_STATE if name in sent] == []
    assert [name for name in KEPT if name not in sent] == []


def test_dockerignore_leaves_out_a_directory_as_a_whole(rules: list[Rule]) -> None:
    assert left_out(".git/objects/ab/cdef", rules)
    assert left_out(".venv-metaclaw/lib/python3.12/site-packages/torch/__init__.py", rules)
    assert left_out("outputs/qwen3.8-27b/file-create/rl/adapt/stage1/episodes.jsonl", rules)
    assert not left_out("src/evolve_ipi/adapt/outputs.py", rules)
    assert not left_out("third_party/OpenClaw-RL/slime/outputs/keep.txt", rules)


def test_the_context_of_this_repository_holds_no_secret_and_no_local_state(rules: list[Rule]) -> None:
    sent = context_files(REPO, rules)
    for path in sent:
        parts = path.split("/")
        name = parts[-1]
        assert name != ".env" and not (name.startswith(".env.") and path != ".env.example"), path
        assert ".git" not in parts and not any(part.startswith(".venv") for part in parts), path
        assert parts[0] != "outputs" and "__pycache__" not in parts and not path.endswith(".pyc"), path
    size = sum((REPO / path).stat().st_size for path in sent)
    assert size < 200 * 1024 * 1024, f"the build context is {size / 2**20:.0f} MB: something large is not excluded"


def lines_with_a_key(files: Iterator[tuple[str, bytes]]) -> list[str]:
    """``path:line`` of every line of a text file that a key pattern of the hygiene test matches; never the key."""
    found = []
    for path, data in files:
        text, is_text = decode(data)
        if not is_text:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if any(rule.pattern.search(line) for rule in KEY_RULES):
                found.append(f"{path}:{number}")
    return found


def made_up_key() -> str:
    """Something the key patterns match, built here so that this file holds no such string."""
    return "s" + "k-" + "x" * 24


def test_the_key_scan_finds_a_key_and_names_only_the_line() -> None:
    files = iter(
        [
            ("a.py", f'TOKEN = "{made_up_key()}"\n'.encode()),
            ("b.py", b"nothing\n"),
            ("c.bin", b"\xff\xfe" + made_up_key().encode()),
        ]
    )
    assert lines_with_a_key(files) == ["a.py:1"]


def test_the_build_context_holds_no_api_key(rules: list[Rule]) -> None:
    sent = context_files(REPO, rules)
    found = lines_with_a_key((path, (REPO / path).read_bytes()) for path in sent)
    assert found == [], "a file that is sent to the build holds a key (the key is not shown)"


def test_the_context_holds_everything_the_dockerfile_uses(rules: list[Rule], dockerfile: list[tuple[str, str]]) -> None:
    sent = set(context_files(REPO, rules))
    copied = []
    for arguments in arguments_of(dockerfile, "COPY"):
        tokens = shlex.split(arguments)
        if not any(token.startswith("--from=") for token in tokens):
            copied += [name for name in tokens[:-1] if name != "."]
    assert copied, "the Dockerfile copies the dependency files by name"
    assert [name for name in copied if name not in sent] == []
    needed = [
        "docker/entrypoint.sh",
        "README.md",  # the project's readme, read when the package is installed
        "LICENSE",
        "src/evolve_ipi/__init__.py",
        "third_party/patches/SHA256SUMS",
        "third_party/OpenClaw-RL/extensions",
        "third_party/MetaClaw/pyproject.toml",
        *(f"third_party/{patch.path}" for patch in PATCHES),
        *(f"third_party/patches/{patch.file}" for patch in PATCHES),
    ]
    assert [name for name in needed if not any(path == name or path.startswith(name + "/") for path in sent)] == []


# --- Docker -------------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def daemon() -> None:
    """Skips the test unless a Docker daemon answers. Only a run with ``-m docker`` gets here."""
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed; install it, then run with `-m docker`")
    try:
        done = subprocess.run(["docker", "info"], check=False, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.skip("`docker info` did not answer within 60 s; start Docker and run with `-m docker` again")
    if done.returncode != 0:
        lines = (done.stderr or done.stdout).strip().splitlines()
        pytest.skip(
            f"no Docker daemon: `docker info` failed ({lines[-1] if lines else done.returncode}); "
            "start Docker and run with `-m docker` again"
        )


def tail(text: str, lines: int = 60) -> str:
    return "\n".join(text.splitlines()[-lines:])


def docker_build(context: Path, tag: str, *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    """``docker build`` for linux/amd64 with plain progress; the output holds stdout and stderr."""
    argv = ["docker", "build", "--platform", "linux/amd64", "--progress=plain", "-t", tag, str(context)]
    done = subprocess.run(argv, check=False, capture_output=True, text=True, errors="replace", timeout=BUILD_TIMEOUT_S)
    if expect_success and done.returncode != 0:
        pytest.fail(f"docker build of {tag} failed with status {done.returncode}:\n{tail(done.stderr + done.stdout)}")
    return done


def run_in(
    image: str, *command: str, entrypoint: str | None = None, timeout: float = RUN_TIMEOUT_S
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` in a throwaway container, started the way the README starts it (without an ``.env``).

    ``--init`` and ``--hostname`` are always given. The CLI also gets an outputs directory on a mount, which is what the
    README gives it; a run with an ``entrypoint`` of its own is a probe of the image and has none.
    """
    argv = ["docker", "run", "--rm", "--init", "--platform", "linux/amd64", "--hostname", HOSTNAME]
    with tempfile.TemporaryDirectory(prefix="evolve-ipi-outputs-", ignore_cleanup_errors=True) as outputs:
        if entrypoint is None:
            argv += ["-v", f"{os.path.realpath(outputs)}:/work/outputs"]
        else:
            argv += ["--entrypoint", entrypoint]
        return subprocess.run(
            [*argv, image, *command], capture_output=True, text=True, errors="replace", timeout=timeout, check=False
        )


def sh(image: str, script: str) -> str:
    """The output of a shell script in the image; the script must succeed."""
    done = run_in(image, "-c", script, entrypoint="sh")
    assert done.returncode == 0, f"{script!r} exited with {done.returncode}:\n{done.stdout}\n{done.stderr}"
    return done.stdout


@pytest.fixture(scope="session")
def image(daemon: None) -> str:
    """The image under test: ``EVOLVE_IPI_TEST_IMAGE`` as it is, or ``evolve-ipi`` built from this checkout."""
    if not os.environ.get("EVOLVE_IPI_TEST_IMAGE"):
        docker_build(REPO, IMAGE)
    return IMAGE


def copy_repository(destination: Path) -> None:
    """A copy of this checkout to build from: without version control, environments and results, and without any
    ``.env`` file of the developer's."""

    def skip(directory: str, names: list[str]) -> set[str]:
        at_root = Path(directory) == REPO
        return {
            name
            for name in names
            if name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", ".env"}
            or name.startswith(".venv")
            or (at_root and name == "outputs")
            or (name.startswith(".env.") and name != ".env.example")
        }

    shutil.copytree(REPO, destination, ignore=skip, symlinks=True)


def docker_history(tag: str) -> subprocess.CompletedProcess[str]:
    argv = ["docker", "history", "--no-trunc", tag]
    return subprocess.run(argv, check=False, capture_output=True, text=True, timeout=120)


def remove_image(tag: str) -> None:
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, text=True, timeout=300, check=False)


@pytest.mark.docker
def test_docker_sends_the_files_that_the_dockerignore_rules_say(
    daemon: None, tmp_path: Path, rules: list[Rule]
) -> None:
    """The rules above are this file's model of BuildKit; here Docker itself lists what a build is sent."""
    context, exported = tmp_path / "context", tmp_path / "exported"
    canary = f"canary-{uuid.uuid4().hex}"
    fake_clone(context, canary)
    recipe = tmp_path / "Dockerfile.list"  # outside the context, so that the context's own .dockerignore applies
    recipe.write_text("FROM scratch\nCOPY . /context\n", encoding="utf-8")
    argv = ["docker", "build", "-f", str(recipe), "--output", f"type=local,dest={exported}", str(context)]
    done = subprocess.run(argv, check=False, capture_output=True, text=True, errors="replace", timeout=600)
    assert done.returncode == 0, tail(done.stderr)
    root = exported / "context"
    sent = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() or path.is_symlink())
    assert sent == context_files(context, rules)
    assert [name for name in SECRETS if name in sent] == []
    assert [name for name in sent if canary in (root / name).read_text(encoding="utf-8")] == []


@pytest.mark.docker
@pytest.mark.parametrize("framework", ["OpenClaw-RL", "MetaClaw"])
def test_the_build_stops_with_a_clear_message_when_a_framework_checkout_is_empty(
    daemon: None, tmp_path: Path, framework: str
) -> None:
    context = tmp_path / "context"
    copy_repository(context)
    shutil.rmtree(context / "third_party" / framework)
    (context / "third_party" / framework).mkdir()  # what `git clone` without --recursive leaves
    tag = f"evolve-ipi-test-empty-{framework.lower()}"
    try:
        done = docker_build(context, tag, expect_success=False)
        output = done.stderr + done.stdout
        assert done.returncode != 0, "the build must fail"
        assert f"third_party/{framework} is empty" in output, tail(output)
        assert "git submodule update --init --recursive" in output
        inspected = subprocess.run(["docker", "image", "inspect", tag], check=False, capture_output=True)
        assert inspected.returncode != 0, "no image may be left behind"
    finally:
        remove_image(tag)


@pytest.fixture
def fresh_clone_image(daemon: None, tmp_path: Path) -> Iterator[tuple[str, str]]:
    """An image built from a copy of the checkout as a fresh clone has it: the frameworks as upstream ships them, with
    the patches not applied yet, and with ``.env`` files, a results directory and an environment lying around.
    Yields the image and the canary that every one of those files holds."""
    context = tmp_path / "context"
    copy_repository(context)
    sums = (THIRD_PARTY / "patches/SHA256SUMS").read_text(encoding="utf-8").splitlines()
    recorded = {name: digest for digest, _, name in (line.partition("  ") for line in sums)}
    for patch in PATCHES:
        upstream = subprocess.run(
            ["git", "-C", str(THIRD_PARTY / patch.submodule), "cat-file", "blob", f"HEAD:{patch.target}"],
            capture_output=True,
            check=True,
        ).stdout
        (context / "third_party" / patch.path).write_bytes(upstream)
        patched = hashlib.sha256(upstream).hexdigest() == recorded[patch.path]
        assert not patched, f"{patch.path}: upstream is already the patched file"
    canary = f"canary-{uuid.uuid4().hex}"
    for name in [
        ".env",
        ".env.local",
        "tinker.env",
        "third_party/MetaClaw/.env",
        ".venv-canary/lib/x.py",
        "outputs/canary.json",
    ]:
        (context / name).parent.mkdir(parents=True, exist_ok=True)
        (context / name).write_text(f"TINKER_API_KEY={canary}\n", encoding="utf-8")
    tag = "evolve-ipi-test-fresh-clone"
    try:
        docker_build(context, tag)
        yield tag, canary
    finally:
        remove_image(tag)


@pytest.mark.docker
def test_a_build_from_a_fresh_clone_patches_the_frameworks_and_takes_no_secret(
    fresh_clone_image: tuple[str, str],
) -> None:
    tag, canary = fresh_clone_image
    verified = run_in(tag, "-m", "evolve_ipi.thirdparty", "verify", entrypoint="/opt/evolve-ipi/venv/bin/python")
    assert verified.returncode == 0, verified.stderr
    for line in (THIRD_PARTY / "patches/SHA256SUMS").read_text(encoding="utf-8").splitlines():
        assert line in verified.stdout
    found = sh(
        tag,
        f"find /work \\( -name '.env*' -o -name '*.env' \\); "
        f"grep -rIl {canary} /work /root /etc /var/lib/evolve-ipi || true",
    )
    assert found.split() == ["/work/.env.example"], found
    history = docker_history(tag)
    assert canary not in history.stdout


@pytest.mark.docker
def test_the_machine_is_what_the_experiments_pin(image: str, dockerfile: list[tuple[str, str]]) -> None:
    probes = {
        "openclaw": "openclaw --version",
        "node": "node --version",
        "zone": "date +%Z",
        "system_python": "/usr/bin/python3 --version",
        "hostname": "hostname",
        "machine": "uname -m",
        "system": "uname -s",
        "shell": 'echo "$SHELL"',
        "tz": 'echo "$TZ"',
        "user": "id -u",
        "home": 'echo "$HOME"',
        "uv": "uv --version",
        "package_python": "/opt/evolve-ipi/venv/bin/python --version",
        "metaclaw_python": "/opt/evolve-ipi/venv-metaclaw/bin/python --version",
    }
    # What each command prints on stdout, as the package reads it (`installed_version` ignores stderr).
    script = "; ".join(f'echo "{name}=$({command} 2>/dev/null)"' for name, command in probes.items())
    facts = key_values(sh(image, script).splitlines())
    assert facts["openclaw"] == OPENCLAW_VERSION
    assert facts["node"] == NODE_VERSION
    assert facts["tz"] == "America/New_York" and facts["zone"] in {"EDT", "EST"}
    assert re.fullmatch(r"Python 3\.11\.\d+", facts["system_python"]), facts["system_python"]
    assert facts["hostname"] == HOSTNAME
    assert (facts["system"], facts["machine"]) == ("Linux", "x86_64")
    assert facts["shell"] == "/bin/bash"
    assert (facts["user"], facts["home"]) == ("0", "/root")
    assert facts["uv"].split()[:2] == ["uv", pinned_uv(dockerfile)], facts["uv"]
    assert facts["package_python"] == facts["metaclaw_python"] == f"Python {pinned_python(dockerfile)}"


@pytest.mark.docker
def test_the_agents_tools_run_the_python_of_debian_and_no_environment_is_on_the_path(image: str) -> None:
    # The PATH the gateway gives the agent: the directory of node and openclaw, then these two.
    agent_path = "/usr/local/bin:/usr/bin:/bin"
    facts = key_values(
        sh(
            image,
            f"""
            echo "python3=$(command -v python3)"
            echo "node=$(command -v node)"
            echo "openclaw=$(command -v openclaw)"
            echo "agent_python3=$(env -i PATH={agent_path} sh -c 'command -v python3')"
            echo "agent_bash=$(env -i PATH={agent_path} sh -c 'command -v bash')"
            echo "path=$PATH"
            """,
        ).splitlines()
    )
    assert facts["python3"] == facts["agent_python3"] == "/usr/bin/python3"
    assert (facts["node"], facts["openclaw"]) == ("/usr/local/bin/node", "/usr/local/bin/openclaw")
    assert facts["agent_bash"].endswith("/bash")
    assert "/opt/evolve-ipi" not in facts["path"]


@pytest.mark.docker
def test_the_agents_home_holds_nothing_but_the_two_empty_target_directories(image: str) -> None:
    listing = (
        "ls -A /root; echo ---; find /root/.openclaw-rl /root/.metaclaw -mindepth 1; "
        "echo ---; find /root/.cache -mindepth 1 2>/dev/null; true"
    )
    home, targets, cache = sh(image, listing).split("---")
    entries = set(home.split()) - {".bashrc", ".profile"}
    # Docker Desktop on Apple silicon runs x86 programs with Rosetta, which adds its translation cache to the home of
    # every container; a native x86 host does not. It is the emulator's file, not the image's.
    if set(cache.split()) <= {"/root/.cache/rosetta"}:
        entries -= {".cache"}
    assert entries == {".openclaw-rl", ".metaclaw"}, home
    assert targets.split() == [], "the target directories must be empty"


@pytest.mark.docker
def test_the_runtime_lies_outside_any_git_checkout_and_no_results_are_baked_in(image: str) -> None:
    output = sh(
        image,
        "test -d /var/lib/evolve-ipi && echo runtime; "
        "for d in /var/lib/evolve-ipi /var/lib /var /; do test -e $d/.git && echo git:$d; done; "
        "test -e /work/.git && echo git:/work; "
        "test -e /work/third_party/MetaClaw/.git && echo git:MetaClaw; "
        "test -e /work/outputs && echo outputs; "
        "true",
    )
    assert output.split() == ["runtime"]


@pytest.mark.docker
def test_both_environments_hold_the_package_and_the_frameworks_with_the_patches(image: str) -> None:
    torch = "torch " + REQUIREMENTS.read_text(encoding="utf-8").split("torch==")[1].split()[0] + "+cpu"
    package = (
        "import sys, importlib; import evolve_ipi, torch; "
        "from evolve_ipi.paths import RuntimePaths; "
        "from evolve_ipi.thirdparty import OPENCLAW_TINKER_DIR, verify_patches; "
        "verify_patches(); "
        "sys.path.insert(0, str(OPENCLAW_TINKER_DIR)); "
        "[importlib.import_module(name) for name in ('api_server', 'run')]; "
        "print('package', evolve_ipi.__file__); print('torch', torch.__version__); "
        "print('metaclaw_python_exists', RuntimePaths().metaclaw_python.is_file())"
    )
    metaclaw = (
        "import torch, metaclaw, metaclaw.api_server, metaclaw.skill_evolver; "
        "print('metaclaw', metaclaw.__file__); print('torch', torch.__version__)"
    )
    first = run_in(image, "-c", package, entrypoint="/opt/evolve-ipi/venv/bin/python")
    assert first.returncode == 0, first.stderr
    assert "package /work/src/evolve_ipi/__init__.py" in first.stdout, "evolve_ipi must be installed editable"
    assert torch in first.stdout and "metaclaw_python_exists True" in first.stdout
    second = run_in(image, "-c", metaclaw, entrypoint="/opt/evolve-ipi/venv-metaclaw/bin/python")
    assert second.returncode == 0, second.stderr
    assert "metaclaw /work/third_party/MetaClaw/metaclaw/__init__.py" in second.stdout
    assert torch in second.stdout


@pytest.mark.docker
def test_no_secret_and_no_env_file_is_in_the_image(image: str) -> None:
    configured = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .Config.Env}} {{json .Config.Cmd}}", image],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    assert not SECRET_NAMES.search(configured), configured
    history = docker_history(image)
    assert not SECRET_NAMES.search(history.stdout), "the name of a secret is in the image history"
    where = "/work /root /home /etc /var/lib/evolve-ipi"
    assert sh(image, f"find {where} \\( -name '.env*' -o -name '*.env' \\)").split() == ["/work/.env.example"]


@pytest.mark.docker
def test_no_api_key_is_in_the_files_of_the_image(image: str) -> None:
    # grep -P takes one pattern: the patterns of the hygiene test as one alternation
    patterns = shlex.quote("|".join(f"(?:{rule.pattern.pattern})" for rule in KEY_RULES))
    # grep -l names the files and never shows a line; status 0 and 1 are "found" and "not found", 2 is an error. The
    # control file proves that the patterns match something in this grep.
    script = (
        f"printf '%s\\n' {shlex.quote(made_up_key())} > /tmp/control.txt; "
        "export LC_ALL=C; "
        f"echo control:; grep -rIlP {patterns} /tmp; test $? -le 1; "
        f"echo work:; grep -rIlP {patterns} /work; test $? -le 1"
    )
    output = sh(image, script)
    control, _, work = output.partition("work:")
    assert control.split() == ["control:", "/tmp/control.txt"], "the key patterns do not match in the image's grep"
    assert work.split() == [], f"files of the image hold a key (the key is not shown): {work.split()}"


@pytest.mark.docker
def test_the_default_command_prints_the_help(image: str) -> None:
    done = run_in(image)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage: evolve-ipi")


@pytest.mark.docker
def test_check_passes_in_the_image(image: str) -> None:
    done = run_in(image, "check")
    output = tail(done.stdout + done.stderr, 80)
    assert done.returncode == 0, f"evolve-ipi check exited with {done.returncode}:\n{output}"
