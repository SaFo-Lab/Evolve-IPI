"""What the hygiene test and the Docker test share: the patterns of a key, and how a file's bytes are read as text.

``test_hygiene`` scans the sources with ``KEY_RULES``; ``test_docker`` scans the files of the image with the same
list, so that there is one definition of what a key looks like.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    secret: bool = False  # the finding shows only the first characters of the match


def make_rule(name: str, pattern: str, flags: int = re.IGNORECASE, secret: bool = False) -> Rule:
    return Rule(name, re.compile(pattern, flags), secret)


# A key of the services we use. Pure hex is a hash and is not matched.
KEY_RULES = (
    make_rule("api-key", r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}", flags=0, secret=True),
    make_rule("api-key", r"(?<![A-Za-z0-9])tml-[A-Za-z0-9_-]{20,}", flags=0, secret=True),
    make_rule("api-key", r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}", flags=0, secret=True),
    make_rule("api-key", r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])", flags=0, secret=True),
)


def decode(data: bytes) -> tuple[str, bool]:
    """The text of a file, and whether it is text. UTF-16 when the file starts with a byte order mark, else UTF-8;
    whatever neither reads (an image) is read as latin-1, which never fails, and is not text."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return data.decode("utf-16"), True
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8"), True
    except UnicodeDecodeError:
        return data.decode("latin-1"), False
