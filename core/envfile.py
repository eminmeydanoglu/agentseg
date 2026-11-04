"""Minimal reader for the repo's gitignored `.env` file (KEY=VALUE lines).

Used for secrets such as META_API_KEY so `agentseg up` and the agent launcher
can pick them up without the key ever being written into tracked files.
"""
from __future__ import annotations

from pathlib import Path


def read_env_file(path: Path | str) -> dict[str, str]:
    """Parse `KEY=VALUE` lines; ignores blanks, comments and `export ` prefixes."""
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            out[key] = value
    return out
