# check=skip=FromPlatformFlagConstDisallowed

# The Evolve-IPI image: the environment the experiments run in.
#
#   docker build -t evolve-ipi .
#   docker run --rm --init --hostname evolve-ipi --env-file .env -v "$PWD/outputs:/work/outputs" evolve-ipi <command>
#
# OpenClaw writes the machine it runs on into the agent's system prompt: operating system, Node version, shell,
# host name and time zone. The image fixes them, so that every machine shows the agent the same facts; the host
# name is the one thing a container only takes from `docker run --hostname`. Every image and tool below is pinned to
# a version, and the base images to a digest. The image is for linux/amd64 and runs under emulation on other CPUs.
#
# Two Python environments, because the two frameworks pin different versions of tinker:
#   /opt/evolve-ipi/venv            this package, OpenClaw-RL's trainer and the model server
#   /opt/evolve-ipi/venv-metaclaw   MetaClaw's proxy and guards
# Neither is on PATH. The agent's own tools run the python3 of Debian.

FROM --platform=linux/amd64 node:22.23.2-bookworm-slim@sha256:48e4b67d85f87bd551df43704e24d252f56cc5f8e9718841aace50f19948f0f9 AS base

FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:0.10.11@sha256:3472e43b4e738cf911c99d41bb34331280efad54c73b1def654a6227bb59b2b4 AS uv


# The repository, checked for its submodules first: a clone made without them stops the build here, within
# seconds, and not after the installs below.
FROM base AS source
WORKDIR /work
COPY . /work
RUN for framework in OpenClaw-RL MetaClaw; do \
        if [ -z "$(ls -A "third_party/$framework" 2>/dev/null)" ]; then \
            echo "error: third_party/$framework is empty. It is a git submodule: run 'git submodule update --init --recursive' in the repository root and build again. (A zip archive of the repository has no submodules; use 'git clone --recursive'.)" >&2; \
            exit 1; \
        fi; \
    done


FROM base

# What the agent's system prompt and tools see of the machine.
ENV TZ=America/New_York \
    SHELL=/bin/bash \
    LANG=C.UTF-8

# Tells the CLI that it runs in this image, so that it can check what the image cannot fix itself: the host name
# (docker run --hostname), an init process (--init) and an outputs directory on a mount (-v).
ENV EVOLVE_IPI_IMAGE=1

# The Hugging Face cache lies in the runtime directory, and output appears as it is written. uv keeps its interpreters
# and the package's environment under /opt, and leaves no cache: nothing it installs may land in the agent's home.
ENV HF_HOME=/var/lib/evolve-ipi/hf \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/evolve-ipi/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/evolve-ipi/venv \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 git ca-certificates tzdata \
 && ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime \
 && echo "$TZ" > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

# node-llama-cpp, a dependency of OpenClaw, loads or compiles a llama.cpp binary when it is installed. The runs use no
# local model, and a CPU that the emulator does not fully offer would stop the build there, so that step is skipped.
# The versions are probed with a scratch home, so that nothing OpenClaw writes lands in the agent's.
RUN NODE_LLAMA_CPP_SKIP_DOWNLOAD=true npm install -g openclaw@2026.3.7 \
 && npm cache clean --force \
 && rm -rf /root/.npm \
 && mkdir -p /tmp/probe-home \
 && export HOME=/tmp/probe-home \
 && test "$(node --version)" = "v22.23.2" \
 && test "$(openclaw --version)" = "2026.3.7" \
 && rm -rf /tmp/probe-home

COPY --from=uv /uv /uvx /usr/local/bin/
RUN uv python install --no-bin 3.12.13

# The dependencies come first, so that a change in the sources does not install them again.
WORKDIR /work
COPY pyproject.toml uv.lock /work/
RUN uv sync --python 3.12.13 --locked --no-dev --no-install-project

# MetaClaw's environment: the pinned packages, with torch alone from the CPU index. Nothing is resolved here;
# `uv pip check` stops the build if the pins miss a package that another one needs.
COPY docker/requirements-metaclaw.txt /tmp/requirements-metaclaw.txt
RUN set -eu; \
    python=/opt/evolve-ipi/venv-metaclaw/bin/python; \
    uv venv --python 3.12.13 /opt/evolve-ipi/venv-metaclaw; \
    torch="$(grep -E '^torch==' /tmp/requirements-metaclaw.txt)"; \
    test -n "$torch"; \
    grep -v -E '^torch==' /tmp/requirements-metaclaw.txt > /tmp/requirements-rest.txt; \
    uv pip install --python "$python" --no-deps --index-url https://download.pytorch.org/whl/cpu "$torch"; \
    uv pip install --python "$python" --no-deps -r /tmp/requirements-rest.txt; \
    rm /tmp/requirements-metaclaw.txt /tmp/requirements-rest.txt

# The repository, with the frameworks under third_party/. The patches are applied to the checkouts and the
# patched files are compared with their recorded hashes; the build stops if one differs.
COPY --from=source /work /work
RUN uv sync --python 3.12.13 --locked --no-dev
RUN /opt/evolve-ipi/venv/bin/python -m evolve_ipi.thirdparty apply \
 && /opt/evolve-ipi/venv/bin/python -m evolve_ipi.thirdparty verify
RUN uv pip install --python /opt/evolve-ipi/venv-metaclaw/bin/python --no-deps -e /work/third_party/MetaClaw \
 && uv pip check --python /opt/evolve-ipi/venv-metaclaw/bin/python

# Both environments import what the runs import, and both have the CPU build of torch. The home is a scratch
# directory, so that nothing an import writes lands in the agent's.
RUN set -eu; \
    mkdir -p /tmp/smoke-home; \
    export HOME=/tmp/smoke-home; \
    /opt/evolve-ipi/venv/bin/python -c "import evolve_ipi, torch, tinker, transformers, fastapi, uvicorn; assert torch.__version__.endswith('+cpu'), torch.__version__; print('venv: torch', torch.__version__)"; \
    /opt/evolve-ipi/venv-metaclaw/bin/python -c "import torch, metaclaw.api_server, metaclaw.skill_evolver; assert torch.__version__.endswith('+cpu'), torch.__version__; print('venv-metaclaw: torch', torch.__version__)"; \
    rm -rf /tmp/smoke-home

# The agent runs as root, with the home of root. The injected requests of the File-Create task point into the two
# directories below, which are empty; the runtime directory holds what a run writes besides its results. Nothing else
# may be in the home, and the build stops if something is. (Rosetta, the x86 emulator of Docker Desktop on Apple
# silicon, adds .cache/rosetta to the home of whatever runs on it; that is not part of the image.)
RUN mkdir -p /root/.openclaw-rl /root/.metaclaw /var/lib/evolve-ipi \
 && chmod 755 /work/docker/entrypoint.sh \
 && test -z "$(find /root/.openclaw-rl /root/.metaclaw -mindepth 1)" \
 && test "$(ls -A /root | grep -v '^\.cache$' | sort | tr '\n' ' ')" = ".bashrc .metaclaw .openclaw-rl .profile " \
 && rm -rf /tmp/* /root/.cache

WORKDIR /work
ENTRYPOINT ["/work/docker/entrypoint.sh"]
CMD ["--help"]
