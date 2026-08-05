"""Keyframe box model for the bbox placer: sparse red/blue keyframes -> dense per-frame GT.

The labeller stores SPARSE keyframes per fighter (the frames the human actually touched);
in-between frames are linearly interpolated (CVAT-style), so a fight is labelled in minutes.
``box_at`` resolves any frame; ``to_fighter_bboxes`` expands the whole track to the canonical
``split_S_fighter_bboxes.json`` JSONL the pipeline already ingests via ``import_bboxes``.

A keyframe value is either ``[x1,y1,x2,y2]`` (present) or ``None`` (an explicit ABSENT
keyframe that ends interpolation — the fighter left frame). Between two present keyframes we
interpolate; a present keyframe with no later keyframe is held forward; before the first
keyframe the fighter is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

Box = list[int]  # [x1, y1, x2, y2]
FIGHTERS = ("red", "blue")


class BBoxProject(BaseModel):
    """Editable keyframe state for one split's red+blue fighter boxes."""

    model_config = ConfigDict(extra="forbid")

    bout: int
    split: int
    num_frames: int
    # fighter -> {frame_index(str): box | None}.  str keys so it round-trips through JSON.
    keyframes: dict[str, dict[str, Box | None]] = {"red": {}, "blue": {}}

    def set_key(self, fighter: str, frame: int, box: Box | None) -> None:
        self.keyframes.setdefault(fighter, {})[str(frame)] = box

    def del_key(self, fighter: str, frame: int) -> None:
        self.keyframes.get(fighter, {}).pop(str(frame), None)

    def _keys(self, fighter: str) -> list[tuple[int, Box | None]]:
        kf = self.keyframes.get(fighter, {})
        return sorted(((int(f), b) for f, b in kf.items()), key=lambda x: x[0])

    def box_at(self, fighter: str, frame: int) -> Box | None:
        """Resolve the fighter's box at ``frame`` (interpolate / hold / absent)."""
        keys = self._keys(fighter)
        if not keys:
            return None
        prev = next_ = None
        for f, b in keys:
            if f <= frame:
                prev = (f, b)
            elif next_ is None:
                next_ = (f, b)
                break
        if prev is None:
            return None  # before the first keyframe
        pf, pb = prev
        if pb is None:
            return None  # explicit absent keyframe in force
        if pf == frame or next_ is None:
            return list(pb)  # exact hit or held forward (no later keyframe)
        nf, nb = next_
        if nb is None:
            return list(pb)  # held up to an explicit absent keyframe
        t = (frame - pf) / (nf - pf)
        return [round(pb[i] + t * (nb[i] - pb[i])) for i in range(4)]

    def frame_state(self, frame: int) -> dict:
        """Boxes + whether each is an exact keyframe at ``frame`` (for the UI)."""
        out: dict = {}
        for c in FIGHTERS:
            out[c] = {
                "box": self.box_at(c, frame),
                "is_key": str(frame) in self.keyframes.get(c, {}),
            }
        return out

    def to_fighter_bboxes(self) -> list[list[dict]]:
        """Expand to the canonical JSONL: line i = frame i, list of red/blue candidates."""
        rows: list[list[dict]] = []
        for f in range(self.num_frames):
            frame_cands: list[dict] = []
            for c in FIGHTERS:
                b = self.box_at(c, f)
                if b is not None:
                    frame_cands.append({
                        "bbox": [int(v) for v in b],
                        "det_conf": 1.0,
                        "cls_confs": {"red": float(c == "red"), "blue": float(c == "blue"),
                                      "unlabeled": 0.0},
                    })
            rows.append(frame_cands)
        return rows


def project_path(out_root: Path, bout: int, split: int) -> Path:
    return out_root / f"bbox_keyframes_bout{bout}_split{split}.json"


def load_project(path: Path, *, bout: int, split: int, num_frames: int) -> BBoxProject:
    if path.exists():
        proj = BBoxProject.model_validate_json(path.read_text())
        proj.num_frames = num_frames  # keep in sync with the actual video
        return proj
    return BBoxProject(bout=bout, split=split, num_frames=num_frames)


def save_project(path: Path, proj: BBoxProject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(proj.model_dump_json(indent=1))


def export_fighter_bboxes(path: Path, proj: BBoxProject) -> None:
    """Write the canonical split_S_fighter_bboxes.json (JSONL, one frame per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for frame_cands in proj.to_fighter_bboxes():
            fh.write(json.dumps(frame_cands) + "\n")
