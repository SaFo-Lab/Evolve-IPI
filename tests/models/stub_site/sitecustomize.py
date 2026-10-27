"""Run at the start of a child process under test: stub out Tinker and the tokenizer (see ``tinker_stubs.py``)."""

import json
import os

if os.environ.get("EVOLVE_IPI_TEST_STUBS"):
    import tinker_stubs

    tinker_stubs.install(json.loads(os.environ["EVOLVE_IPI_TEST_STUBS"]))
