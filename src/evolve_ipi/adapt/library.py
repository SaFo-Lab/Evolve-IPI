"""The hash a skill library is held to once its evolution has finished."""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["lib_sha"]


def lib_sha(skills_dir: Path) -> str:
    """sha256 over the library's ``SKILL.md`` files: each one's path below ``skills_dir``, a NUL byte and its bytes.

    The files are taken in ``Path`` order, which compares the path components (``a/`` before ``a-b/``, unlike the
    plain strings). The evolution history is not part of it. A library that does not exist has no hash.
    """
    root = Path(skills_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"there is no skill library at {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("SKILL.md")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()
