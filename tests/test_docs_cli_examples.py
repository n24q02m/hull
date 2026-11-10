"""Docs contract: no example passes a token to ``hull token hash`` on argv.

The CLI refuses a token on the command line (it would land in shell history),
so a quick start that taught ``hull token hash mysecret`` would document a
command that cannot run. Every fenced shell block in the shipped docs must use
the piped form.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "docs/HANDOVER.md", "packages/core-py/README.md")
SHELL_LANGS = {"", "bash", "sh", "shell", "console"}
# A word (not a shell operator) after "token hash" is an argv token.
_TOKEN_ARGUMENT = re.compile(r"token\s+hash\s+([^\s&|;)]+)")


def _shell_lines(text: str) -> list[str]:
    """Code lines from fenced shell blocks, inline comments removed."""
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            # Opening fence carries the language; a closing fence is bare.
            fence = stripped[3:].strip() if fence is None else None
            continue
        if fence in SHELL_LANGS and "token hash" in line:
            lines.append(line.split("#", 1)[0].strip())
    return lines


@pytest.mark.parametrize("doc", DOCS)
def test_token_hash_example_is_the_piped_form(doc: str) -> None:
    text = (ROOT / doc).read_text(encoding="utf-8")
    commands = _shell_lines(text)
    assert commands, f"{doc} documents no `token hash` command"
    for command in commands:
        match = _TOKEN_ARGUMENT.search(command)
        assert match is None, f"{doc} passes the token on argv ({match.group(1)}): {command}"
