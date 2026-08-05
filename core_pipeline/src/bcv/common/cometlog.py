"""Comet logging helper — one CometLogger per stage, key loaded from a gitignored .env.

Returns ``None`` (graceful no-op) when there's no ``COMET_API_KEY``, so training still
runs offline. A separate Comet *project* per stage keeps runs from mixing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_env(path: str | Path = ".env") -> None:
    """Load ``KEY=VALUE`` lines from a .env file into ``os.environ`` (without overriding)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def make_logger(project: str, *, name: str | None = None) -> Any | None:
    """Build a CometLogger for ``project`` (workspace from env), or None if no API key."""
    load_env()
    if not os.environ.get("COMET_API_KEY"):
        print("[comet] no COMET_API_KEY found (.env) — logging disabled")
        return None
    try:
        from lightning.pytorch.loggers import CometLogger

        logger = CometLogger(
            api_key=os.environ["COMET_API_KEY"],
            workspace=os.environ.get("COMET_WORKSPACE"),
            project=project,
        )
        if name:
            logger.experiment.set_name(name)
        print(f"[comet] logging to {logger.experiment.url}")
        return logger
    except Exception as e:
        print(f"[comet] disabled ({type(e).__name__}: {e})")
        return None
