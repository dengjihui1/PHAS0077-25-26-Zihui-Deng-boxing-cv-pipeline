from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bcv.common import io
from bcv.common.contracts import DETECTION_SCHEMA, ArtifactMeta, missing_columns


def _meta() -> ArtifactMeta:
    return ArtifactMeta(
        stage="stage1_detect",
        source_video="/data/Bout 122_Split 1-4/split_0.mp4",
        fps=29.97,
        width=1920,
        height=1080,
        num_frames=100,
        producer={"backend": "chain", "min_cls_conf": 0.5},
    )


def test_meta_write_read_roundtrip(tmp_path: Path):
    d = tmp_path / "stage1_detect"
    io.write_meta(d, _meta())
    back = io.read_meta(d)
    assert back.stage == "stage1_detect"
    assert back.num_frames == 100
    assert back.producer["backend"] == "chain"
    assert back.schema_version == "1.0"


def test_validate_meta_mismatch_raises(tmp_path: Path):
    d = tmp_path / "stage1_detect"
    io.write_meta(d, _meta())
    m = io.read_meta(d)
    io.validate_meta(m, source_video="/data/Bout 122_Split 1-4/split_0.mp4", num_frames=100)
    with pytest.raises(io.MetaMismatchError):
        io.validate_meta(m, num_frames=99)
    with pytest.raises(io.MetaMismatchError):
        io.validate_meta(m, source_video="/data/other.mp4")


def test_detection_parquet_roundtrip_schema(tmp_path: Path):
    n = 5
    df = pd.DataFrame({col: [0] * n for col in DETECTION_SCHEMA})
    df["frame"] = range(n)
    df = df.astype(DETECTION_SCHEMA)
    p = tmp_path / "detections.parquet"
    io.write_parquet(p, df)
    back = io.read_parquet(p)
    assert list(back.columns) == list(DETECTION_SCHEMA)
    assert missing_columns(list(back.columns), DETECTION_SCHEMA) == []
    assert back["frame"].tolist() == list(range(n))


def test_atomic_write_leaves_no_tmp_and_overwrites(tmp_path: Path):
    p = tmp_path / "a.json"
    io.write_json(p, {"v": 1})
    io.write_json(p, {"v": 2})
    assert io.read_json(p)["v"] == 2
    # no stray temp files left behind
    assert not list(tmp_path.glob(".a.json.tmp*"))


def test_atomic_write_failure_cleans_tmp(tmp_path: Path):
    p = tmp_path / "boom.bin"

    def bad_writer(_tmp: Path) -> None:
        raise RuntimeError("writer blew up")

    with pytest.raises(RuntimeError):
        io.atomic_write(p, bad_writer)
    assert not p.exists()
    assert not list(tmp_path.glob(".boom.bin.tmp*"))
