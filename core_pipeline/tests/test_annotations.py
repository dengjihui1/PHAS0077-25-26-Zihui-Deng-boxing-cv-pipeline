from __future__ import annotations

from pathlib import Path

from bcv.common import annotations as A


def test_discovery_both_filename_patterns(bout_dir_legacy: Path, bout_dir_split: Path):
    f1 = A.discover_annotation_file(bout_dir_legacy)
    f2 = A.discover_annotation_file(bout_dir_split)
    assert f1 is not None and f1.name == "annotations.json"
    assert f2 is not None and f2.name == "Bout 122_Split 1-4.json"


def test_discovery_ignores_bbox_sidecar(bout_dir_legacy: Path):
    # the legacy dir also contains split_0_fighter_bboxes.json which must not be picked
    assert A.discover_annotation_file(bout_dir_legacy).name == "annotations.json"


def test_load_lowercases_labels(bout_dir_split: Path):
    doc = A.load_annotations(A.discover_annotation_file(bout_dir_split))
    labels = {a.label for a in doc.annotations}
    assert labels == {"blue_strike_missed", "red_head_landed"}
    assert doc.num_frames == 50


def test_group_into_runs_per_label_no_cross_merge(bout_dir_split: Path):
    doc = A.load_annotations(A.discover_annotation_file(bout_dir_split))
    runs = A.group_into_runs(doc.annotations)
    assert len(runs) == 2
    blue = next(r for r in runs if r.label == "blue_strike_missed")
    red = next(r for r in runs if r.label == "red_head_landed")
    assert (blue.start_frame, blue.end_frame, blue.length) == (10, 13, 4)
    assert (red.start_frame, red.end_frame, red.length) == (20, 21, 2)


def test_binary_target_span_membership_and_dilation(bout_dir_split: Path):
    doc = A.load_annotations(A.discover_annotation_file(bout_dir_split))
    runs = A.group_into_runs(doc.annotations)
    t = A.frame_binary_target(runs, 50, dilate=0)
    assert t.sum() == 4 + 2
    assert t[10] == 1 and t[13] == 1 and t[14] == 0 and t[9] == 0
    td = A.frame_binary_target(runs, 50, dilate=1)
    assert td[9] == 1 and td[14] == 1  # dilation widens each run by 1 on both sides


def test_group_validate_rejects_unknown_label(bout_dir_split: Path):
    import pytest

    from bcv.common.contracts import Annotation

    bad = [Annotation(frame=1, label="purple_uppercut")]
    with pytest.raises(ValueError):
        A.group_into_runs(bad, validate=True)
