"""Track-class memory — hold a person's red/blue label across short classifier dips.

Per-frame crop classification is jittery: during clinches, motion blur, or profile
views a fighter often scores ``unlabeled``, which would drop them from that frame. But
the detector tracks people across frames with stable ``track_id``s, so once a track is
*confidently* classified red/blue we remember it and re-inject that label for up to
``hold_frames`` subsequent frames where the live confidence falls below the gate. A high
gate sets the label (precision); the hold maintains continuity (recall).
"""

from __future__ import annotations


class TrackClassMemory:
    def __init__(self, *, min_cls_conf: float = 0.5, hold_frames: int = 15) -> None:
        self.min_cls_conf = min_cls_conf
        self.hold_frames = hold_frames
        # track_id -> (color, conf, last_confident_frame_idx)
        self._mem: dict[int, tuple[str, float, int]] = {}

    def reset(self) -> None:
        self._mem.clear()

    def resolve(
        self, track_id: int | None, cls_conf: dict[str, float], frame_idx: int
    ) -> dict[str, float]:
        """Return ``cls_conf`` augmented with a held red/blue label when applicable."""
        red, blue = cls_conf.get("red", 0.0), cls_conf.get("blue", 0.0)
        best_color = "red" if red >= blue else "blue"
        best_conf = max(red, blue)

        if best_conf >= self.min_cls_conf:
            if track_id is not None:
                self._mem[track_id] = (best_color, best_conf, frame_idx)
            return cls_conf

        # Below gate: re-inject a recent confident label for this track, if any.
        if track_id is not None and track_id in self._mem:
            color, conf, last = self._mem[track_id]
            if frame_idx - last <= self.hold_frames:
                out = dict(cls_conf)
                out[color] = max(out.get(color, 0.0), conf)
                return out
        return cls_conf
