#!/bin/sh
# The entrypoint of the image: every argument goes to the evolve-ipi command.
# The environment is reached by its absolute path. It is not on PATH, because the agent's own tools run the
# python3 of Debian, as they did in the experiments.
set -eu
exec /opt/evolve-ipi/venv/bin/evolve-ipi "$@"
