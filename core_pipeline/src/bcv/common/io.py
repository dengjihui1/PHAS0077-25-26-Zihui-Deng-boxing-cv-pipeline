"""Atomic, crash-safe artifact IO — the explicit fix for the old append-mode race.

The legacy pipeline truncated a JSONL file on init and appended one line per frame
(``input_analyser.py`` lines 126/224); two concurrent runs corrupted each other.
Here every artifact is written *whole* to a temp file, fsync'd, ``os.replace``'d into
place, and the containing directory is fsync'd — so a reader never sees a partial file
and there is no append/truncate to race on.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import ArtifactMeta

META_NAME = "meta.json"


def atomic_write(path: str | Path, writer_fn: Callable[[Path], None]) -> Path:
    """Write ``path`` atomically: ``writer_fn`` fills a temp file, then we fsync+replace.

    ``writer_fn`` receives the temp path and must write the full file content to it.
    Guarantees: never appends, never truncates-in-place, and the rename is durable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        writer_fn(tmp)
        # fsync the file contents before the rename.
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        # fsync the directory so the rename itself survives a crash.
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def write_json(path: str | Path, obj: Any) -> Path:
    def _w(tmp: Path) -> None:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=False))

    return atomic_write(path, _w)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_parquet(path: str | Path, df: pd.DataFrame) -> Path:
    return atomic_write(path, lambda tmp: df.to_parquet(tmp, index=False))


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_meta(artifact_dir: str | Path, meta: ArtifactMeta) -> Path:
    """Write ``meta.json`` (last, atomically) — its presence marks the artifact complete."""
    return write_json(Path(artifact_dir) / META_NAME, meta.model_dump())


def read_meta(artifact_dir: str | Path) -> ArtifactMeta:
    return ArtifactMeta.model_validate(read_json(Path(artifact_dir) / META_NAME))


class MetaMismatchError(RuntimeError):
    """Raised when a consumed artifact's meta does not match the expected source."""


def validate_meta(
    meta: ArtifactMeta,
    *,
    source_video: str | None = None,
    num_frames: int | None = None,
) -> None:
    """Hard-fail (not silent-skip) if a downstream stage is handed a mismatched artifact."""
    if source_video is not None and meta.source_video != str(source_video):
        raise MetaMismatchError(
            f"source_video mismatch: artifact={meta.source_video!r} expected={source_video!r}"
        )
    if num_frames is not None and meta.num_frames != int(num_frames):
        raise MetaMismatchError(
            f"num_frames mismatch: artifact={meta.num_frames} expected={num_frames}"
        )
