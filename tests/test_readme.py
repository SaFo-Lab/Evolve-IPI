"""The README tells the truth about the command line, the configs, the data and the outputs.

Every ``evolve-ipi`` command the README shows (in a code block, written out or through its alias) must be accepted by
the real argument parser and name a model and a task that exist; every flag it mentions must be in the help of the
command it belongs to; ``--help`` of the program and of every command it names must run. The numbers it states are
compared with the shipped configs and data. The one test that runs the README's ``docker run`` line carries the
``docker`` marker, so a plain ``pytest`` leaves it out; run with ``-m docker`` to run it. It skips when Docker or the
image ``evolve-ipi`` is not there, and it never builds the image.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from evolve_ipi import cli, evaluate, records, runner
from evolve_ipi.adapt import rl as adapt_rl
from evolve_ipi.adapt import skills as adapt_skills
from evolve_ipi.config import TASK_NAMES
from evolve_ipi.records import RunDir
from evolve_ipi.runner import IMAGE_HOSTNAME

REPO = Path(__file__).resolve().parents[1]
README_TEXT = (REPO / "README.md").read_text(encoding="utf-8")
DOCKER_PREFIX = (
    'docker run --rm --init --hostname evolve-ipi --env-file .env -v "$PWD/outputs:/work/outputs" evolve-ipi'
)
SUBCOMMANDS: tuple[tuple[str, ...], ...] = (("check",), ("eval",), ("adapt", "rl"), ("adapt", "skills"), ("run",))
FLAG = re.compile(r"(?<![\w-])--[a-z][a-z-]*")
# Flags in the prose that are not the program's: git's, in the clone command, and docker's, those of the README's own
# `docker run` line.
EXTERNAL_FLAGS = {"--recursive"} | set(FLAG.findall(DOCKER_PREFIX))


# --- reading the README ----------------------------------------------------------------------------------------------


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    return re.findall(r"^```(\w*)\n(.*?)^```", text, flags=re.DOTALL | re.MULTILINE)


def prose(text: str) -> str:
    return re.sub(r"^```\w*\n.*?^```\n", "", text, flags=re.DOTALL | re.MULTILINE)


def shell_commands(text: str) -> list[str]:
    """The commands of the bash blocks, a continued line joined to its first line; comments and blanks dropped."""
    commands: list[str] = []
    for language, body in fenced_blocks(text):
        if language != "bash":
            continue
        pending = ""
        for raw in body.splitlines():
            line = raw.strip()
            if line.endswith("\\"):
                pending += line[:-1].strip() + " "
                continue
            line = (pending + line).strip()
            pending = ""
            if line and not line.startswith("#"):
                commands.append(re.sub(r"\s+#\s.*$", "", line))
    return commands


@dataclass(frozen=True)
class Invocation:
    command: str  # the line as the README writes it
    args: tuple[str, ...]  # what follows ``evolve-ipi``

    @property
    def subcommand(self) -> tuple[str, ...]:
        return subcommand_of(self.args)

    @property
    def flags(self) -> set[str]:
        return {token.split("=")[0] for token in self.args if token.startswith("--")}


def subcommand_of(tokens: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    for known in SUBCOMMANDS:
        if tuple(tokens[: len(known)]) == known:
            return known
    raise AssertionError(f"{' '.join(tokens)!r} does not start with a command of evolve-ipi")


_DOCKER_OPTIONS_WITH_VALUE = {"--hostname", "--env-file", "-v", "--volume", "-e", "--env", "--name", "-w", "--workdir"}


def invocations(text: str | None = None) -> list[Invocation]:
    found: list[Invocation] = []
    for command in shell_commands(README_TEXT if text is None else text):
        tokens = shlex.split(command)
        if tokens[:2] == ["docker", "run"]:
            rest, skip = tokens[2:], False
            for index, token in enumerate(rest):
                if skip:
                    skip = False
                elif token in _DOCKER_OPTIONS_WITH_VALUE:
                    skip = True
                elif not token.startswith("-"):
                    assert token == "evolve-ipi", f"{command!r} runs the image {token!r}, not evolve-ipi"
                    found.append(Invocation(command, tuple(rest[index + 1 :])))
                    break
        elif tokens[0] == "evolve-ipi":
            found.append(Invocation(command, tuple(tokens[1:])))
    return found


# --- running the program ---------------------------------------------------------------------------------------------


def _program() -> list[str]:
    script = Path(sys.executable).parent / "evolve-ipi"
    if script.exists():
        return [str(script)]
    return [sys.executable, "-c", "import sys; from evolve_ipi.cli import main; sys.exit(main(sys.argv[1:]))"]


def run_help(*words: str) -> str:
    result = subprocess.run(
        [*_program(), *words, "--help"], cwd=REPO, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, f"evolve-ipi {' '.join(words)} --help exited {result.returncode}:\n{result.stderr}"
    assert result.stdout.startswith("usage: evolve-ipi"), result.stdout[:200]
    return result.stdout


@pytest.fixture(scope="module")
def helps() -> dict[tuple[str, ...], str]:
    return {words: run_help(*words) for words in ((), *SUBCOMMANDS)}


def flags_of(help_text: str) -> set[str]:
    return set(FLAG.findall(help_text))


# --- the commands ----------------------------------------------------------------------------------------------------


def test_the_readme_shows_every_command() -> None:
    shown = {invocation.subcommand for invocation in invocations()}
    assert shown == set(SUBCOMMANDS), f"the README shows {sorted(shown)}"


CONFIGS_MOUNT = '-v "$PWD/configs:/work/configs"'
DOCKER_SUBSET = DOCKER_PREFIX.removesuffix(" evolve-ipi") + f" {CONFIGS_MOUNT} evolve-ipi"


def test_the_docker_commands_have_one_form() -> None:
    """One form everywhere; the one exception is the `run` of a subset, which also mounts the edited configs."""
    lines = [command for command in shell_commands(README_TEXT) if command.startswith("docker run")]
    assert len(lines) >= 3
    plain = [line for line in lines if CONFIGS_MOUNT not in line]
    assert all(line.startswith(DOCKER_PREFIX + " ") for line in plain), plain
    assert len(plain) == len(lines) - 1
    (alias,) = [command for command in shell_commands(README_TEXT) if command.startswith("alias ")]
    assert alias == f"alias evolve-ipi='{DOCKER_PREFIX}'"


def test_the_subset_command_mounts_the_configs_the_image_would_otherwise_hide() -> None:
    """The image holds its own copy of configs/, so editing configs/main.yaml on the host changes nothing without it."""
    (subset,) = [command for command in shell_commands(README_TEXT) if CONFIGS_MOUNT in command]
    assert subset.startswith(DOCKER_SUBSET + " run "), subset
    paragraph = README_TEXT[: README_TEXT.index(subset)].split("\n\n")[-2]  # the text above the code block
    assert "subset" in paragraph and "configs/main.yaml" in paragraph
    (invocation,) = [i for i in invocations() if i.command == subset]
    assert invocation.args == ("run", "configs/main.yaml")
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    baked = re.search(r"^COPY\s+(--from=\w+\s+)?(/work|\.)\s+/work\b", dockerfile, re.MULTILINE)
    assert baked, "the image no longer copies the repository, so the mount of configs/ may no longer be needed"


def test_help_of_the_program_and_of_every_command_runs(helps: dict[tuple[str, ...], str]) -> None:
    assert set(helps) == {(), *SUBCOMMANDS}
    for sub in ("check", "eval", "adapt", "run"):
        assert sub in helps[()]


@pytest.mark.parametrize("invocation", invocations(), ids=lambda invocation: " ".join(invocation.args[:3]))
def test_every_command_is_accepted_by_the_parser(invocation: Invocation) -> None:
    parsed = cli.build_parser().parse_args(list(invocation.args))
    models = yaml.safe_load((REPO / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]
    if hasattr(parsed, "model"):
        assert parsed.model in models, f"{invocation.command!r}: no model {parsed.model!r} in models.yaml"
        assert parsed.task in TASK_NAMES
        mechanism = invocation.subcommand[1] if invocation.subcommand[0] == "adapt" else None
        if mechanism is not None:
            assert mechanism in models[parsed.model]["mechanisms"], f"{parsed.model} does not take {mechanism}"
        if getattr(parsed, "adapted", None) is not None:
            model, task, line = Path(parsed.adapted).parts[-3:]
            assert (model, task) == (parsed.model, parsed.task) and line in models[model]["mechanisms"]
            assert Path(parsed.adapted).parts[-4] == "outputs"
    if invocation.subcommand == ("run",):
        assert (REPO / parsed.config).is_file()


def test_every_flag_of_a_command_is_in_its_help(helps: dict[tuple[str, ...], str]) -> None:
    missing = [
        f"{invocation.command}: {flag} is not in `evolve-ipi {' '.join(invocation.subcommand)} --help`"
        for invocation in invocations()
        for flag in sorted(invocation.flags - flags_of(helps[invocation.subcommand]))
    ]
    assert not missing, "\n" + "\n".join(missing)


def test_every_flag_in_the_prose_is_a_flag_of_the_program(helps: dict[tuple[str, ...], str]) -> None:
    known = set().union(*(flags_of(text) for text in helps.values()))
    missing = []
    for line in prose(README_TEXT).splitlines():
        spans = re.findall(r"`([^`]+)`", line)
        flags = {flag for span in spans for flag in FLAG.findall(span)}
        row = re.match(r"\|\s*`([^`]+)`\s*\|", line)
        if row and row.group(1).split()[0] in {s[0] for s in SUBCOMMANDS}:
            allowed = flags_of(helps[subcommand_of(row.group(1).split())])  # a table row: the flags of its command
        else:
            allowed = known | EXTERNAL_FLAGS
        missing += [f"{line.strip()[:70]}: {flag}" for flag in sorted(flags - allowed)]
    assert not missing, "\n" + "\n".join(missing)


def test_the_environment_variables_the_readme_names_exist() -> None:
    example = {
        line.partition("=")[0]
        for line in (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    program = run_help()
    named = set(re.findall(r"\b(?:TINKER_API_KEY|HF_TOKEN|EVOLVER_[A-Z_]+|EVOLVE_IPI_[A-Z_]+)\b", README_TEXT))
    assert {"TINKER_API_KEY", "HF_TOKEN"} <= named
    unknown = [name for name in sorted(named) if name not in example and name not in program]
    assert not unknown, f"named in the README but in neither .env.example nor the program's help: {unknown}"


# --- what the README says about the experiment ------------------------------------------------------------------


def _yaml(name: str) -> dict[str, Any]:
    return yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))


def test_the_tasks_and_models_are_the_shipped_ones() -> None:
    section = README_TEXT.split("## Tasks", 1)[1].split("## Requirements", 1)[0]
    in_table = re.findall(r"^\| `([a-z-]+)` \|", section, flags=re.MULTILINE)
    assert in_table == list(TASK_NAMES)
    models_paragraph = re.search(r"^Models: .*?\n\n", README_TEXT, flags=re.DOTALL | re.MULTILINE)
    assert models_paragraph is not None
    assert re.findall(r"`([a-z0-9.-]+)`", models_paragraph.group()) == list(_yaml("models.yaml")["models"])


def test_the_numbers_of_the_readme_match_the_configs_and_the_data() -> None:
    combos = _yaml("main.yaml")["combos"]
    assert len(combos) == 35 and "35 model" in README_TEXT
    rl_models = sorted({c["model"] for c in combos if c["mechanism"] == "rl"})
    skills_models = sorted({c["model"] for c in combos if c["mechanism"] == "skills"})
    assert (rl_models, skills_models) == (
        ["gpt-oss-120b", "inkling-small", "qwen3.8-27b"],
        ["gpt-oss-120b", "gpt-oss-20b", "inkling-small", "qwen3.8-27b"],
    )
    assert all(sum(1 for c in combos if c["model"] == m and c["mechanism"] == "rl") == 5 for m in rl_models)
    assert all(sum(1 for c in combos if c["model"] == m and c["mechanism"] == "skills") == 5 for m in skills_models)
    items = re.search(r"(\d+) adaptation items and (\d+) test documents", README_TEXT)
    assert items is not None
    for task in TASK_NAMES:
        counts = tuple(
            len((REPO / "data" / "tasks" / task / f"{part}.jsonl").read_text(encoding="utf-8").splitlines())
            for part in ("adaptation", "test")
        )
        assert counts == (int(items.group(1)), int(items.group(2))), task
    updates = re.search(r"(\d+) RL updates in two stages of (\d+)", README_TEXT)
    assert updates is not None and _yaml("rl.yaml")["updates_per_stage"] == int(updates.group(2))
    assert int(updates.group(1)) == 2 * int(updates.group(2))
    conversations = re.search(r"\| `adapt skills` \| (\d+) conversations", README_TEXT)
    assert conversations is not None and _yaml("skills.yaml")["conversations"] == int(conversations.group(1))
    assert _yaml("eval.yaml")["setting3_base_passes"] >= 1 and "40 new test documents" in README_TEXT


def test_the_output_layout_is_the_one_the_code_writes() -> None:
    tree = next(body for language, body in fenced_blocks(README_TEXT) if body.startswith("outputs/<model>"))
    run = RunDir(Path("run"))
    expected = [
        run.config_path.relative_to(run.root).as_posix(),
        run.adapt_dir.relative_to(run.root).as_posix() + "/",
        run.summary_path.relative_to(run.root).as_posix(),
        run.void_dir.relative_to(run.root).as_posix() + "/",
        adapt_rl.CHECKPOINTS_FILE,
        adapt_skills.MARKER,
    ]
    for name in expected:
        assert name in tree, f"{name} is not in the README's output layout"
    pass_dir = run.pass_dir("base", 3, 1).relative_to(run.root).as_posix()
    assert pass_dir == "eval/base/setting3/pass1"
    assert "eval/<base|adapted>/setting<s>/pass<k>/" in tree and "<item>.json" in tree
    for name in (records.COMPLETE, records.LOCK, run.topup_path.relative_to(run.root).as_posix()):
        assert name in tree, f"{name} is not in the README's output layout"
    assert "outputs/<model>/<task>/<mechanism>/" in tree and "limit<N>/" in README_TEXT
    written = Path(evaluate.__file__).read_text(encoding="utf-8")
    for name in ("verdict.json", "server", "unused_passes"):  # as evaluate.py names what a pass and the summary hold
        assert name in tree, f"{name} is not in the README's output layout"
        assert f'"{name}"' in written, f"evaluate.py no longer writes {name}"
    assert "data/SHA256SUMS" in tree and '"data_sha256sums"' in Path(runner.__file__).read_text(encoding="utf-8")


def test_the_files_and_directories_the_readme_names_exist() -> None:
    layout = next(body for language, body in fenced_blocks(README_TEXT) if "src/evolve_ipi/" in body)
    listed = {line.split()[0].rstrip("/") for line in layout.splitlines()}
    for name in sorted(listed):
        assert (REPO / name).exists(), name
    if shutil.which("git") and (REPO / ".git").exists():  # everything at the top of the tree is in the layout
        files = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=REPO, capture_output=True, check=True,
        ).stdout.decode("utf-8").split("\0")  # fmt: skip
        top = {path.split("/")[0] for path in files if "/" in path and not path.startswith(".")} | {"Dockerfile"}
        shown = {name.split("/")[0] for name in listed}
        assert top <= shown, f"the layout does not list {sorted(top - shown)}"
    for target in re.findall(r"\]\(([^)#][^)]*)\)", README_TEXT) + re.findall(r'src="([^"]+)"', README_TEXT):
        if not target.startswith("http"):
            assert (REPO / target).is_file(), target
    for name in re.findall(r"`((?:configs/)?[a-z]+\.yaml)`", README_TEXT):
        assert (REPO / "configs" / Path(name).name).is_file(), name
    for name in ("LICENSE", "LICENSE-DATA", "THIRD_PARTY_NOTICES.md", "Dockerfile", ".env.example", "CITATION.bib"):
        assert (REPO / name).is_file(), name
    assert all(f"`{name}`" in README_TEXT for name in ("LICENSE", "LICENSE-DATA", "THIRD_PARTY_NOTICES.md"))


def test_the_readme_does_not_make_gpt_5_2_a_requirement() -> None:
    """The evolver is any OpenAI-compatible endpoint; GPT-5.2 is what the paper used, and its name has no default."""
    sentences = re.findall(r"[^.]*GPT-5\.2[^.]*\.", prose(README_TEXT).replace("\n", " "))
    assert sentences and all("paper used" in sentence for sentence in sentences), sentences
    assert "`EVOLVER_MODEL` has no default" in prose(README_TEXT).replace("\n  ", " ")


def test_the_readme_says_what_the_drive_judge_checks() -> None:
    """The judge needs the injected name and the injected content (tests/environment/test_verifiers.py)."""
    (row,) = [line for line in README_TEXT.splitlines() if line.startswith("| `drive-create` |")]
    assert row.rstrip().endswith("| a new drive file has the injected name and content |"), row


def test_the_readme_marks_the_two_equal_contributors() -> None:
    """The author line stars the first two authors, as the paper's title page does, and a note says what the star means.

    The stars are escaped: two bare ``*`` in one paragraph pair up as emphasis and italicise the text between them.
    """
    lines = README_TEXT.splitlines()
    assert "Xinyan Wang<sup>\\*</sup>, Xiaogeng Liu<sup>\\*</sup>, Jun Shao, Chaowei Xiao" in lines
    assert "<sup>\\*</sup>Equal contribution" in lines
    author_lines = lines[lines.index("Xinyan Wang<sup>\\*</sup>, Xiaogeng Liu<sup>\\*</sup>, Jun Shao, Chaowei Xiao") :][:3]
    assert "*" not in "\n".join(author_lines).replace("\\*", "")


def test_the_docker_notes_of_the_readme_match_the_image() -> None:
    """The host name the README demands is the one the code demands, and its cache volume is where the image keeps the
    Hugging Face cache."""
    text = prose(README_TEXT)
    assert f"--hostname {IMAGE_HOSTNAME}" in DOCKER_PREFIX and f"`--hostname {IMAGE_HOSTNAME}`" in text
    cache = re.search(r"`-v evolve-ipi-hf:(\S+)`", text)
    assert cache is not None, "the README no longer names a volume for the tokenizer cache"
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(rf"\bHF_HOME={re.escape(cache.group(1))}(?:\s|\\|$)", dockerfile), cache.group(1)
    assert "`docker stop -t 120 <container>`" in text


def test_the_citation_in_the_readme_is_the_citation_file() -> None:
    bibtex = next(body for language, body in fenced_blocks(README_TEXT) if language == "bibtex")
    assert bibtex.strip() == (REPO / "CITATION.bib").read_text(encoding="utf-8").strip()
    assert "Xiao, Chaowei" in bibtex and "arXiv:XXXX.XXXXX" in bibtex


# --- the one test that needs Docker -----------------------------------------------------------------------------------


@pytest.mark.docker
def test_the_readme_check_command_runs_in_the_image(tmp_path: Path) -> None:
    """The ``docker run ... check`` line of the README, as written, in a directory with a ``.env`` and ``outputs/``.

    Run with ``-m docker``; a plain ``pytest`` leaves it out."""
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed; install it, then run with `-m docker`")
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=60, check=False).returncode != 0:
            pytest.skip("the docker daemon is not running; start it, then run with `-m docker` again")
        image = subprocess.run(
            ["docker", "image", "inspect", "evolve-ipi"], capture_output=True, timeout=60, check=False
        )
    except subprocess.TimeoutExpired:
        pytest.skip("docker does not answer; run with `-m docker` again when it does")
    if image.returncode != 0:
        pytest.skip("the image evolve-ipi is not built; run `docker build -t evolve-ipi .`, then run with `-m docker`")
    (command,) = [i.command for i in invocations() if i.args == ("check",)]
    (tmp_path / ".env").write_text("TINKER_API_KEY=\n", encoding="utf-8")  # `check` needs no key
    (tmp_path / "outputs").mkdir()
    result = subprocess.run(
        ["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True, timeout=30 * 60, check=False
    )
    tails = f"--- stdout\n{result.stdout[-3000:]}\n--- stderr\n{result.stderr[-3000:]}"
    assert result.returncode == 0, f"{command}\n{tails}"
    assert "FAIL" not in result.stdout
