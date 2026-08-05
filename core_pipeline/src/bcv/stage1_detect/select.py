"""Red/blue SLOT SELECTION — collapse candidate boxes into at most one red + one blue.

This is the net-new piece the old repo never had in one place (its logic was scattered
across ``labels.py``). Both backends feed their candidates through here, so fighter
slotting is never assumed free — a 2-class detector can still emit multiple reds/blues
or miss one.

Policy (greedy, highest-confidence-first, distinct candidates):
- a color slot is filled by the highest ``cls_conf`` candidate for that color that
  clears ``min_cls_conf`` and is not already used by the other slot;
- if the two strongest candidates both classify as the same color, the higher-conf one
  takes that color and the other falls back to its next-best color (if it clears the gate);
- ties broken by detector confidence, then box area.
"""

from __future__ import annotations

from ..common.geometry import area
from .interface import Box, Candidate

FIGHTER_COLORS = ("red", "blue")


def select_fighters(
    candidates: list[Candidate], *, min_cls_conf: float = 0.5
) -> tuple[Box | None, Box | None]:
    """Return ``(red_box, blue_box)`` resolved from ``candidates`` (either may be None)."""
    # Every (candidate, color) pairing that clears the gate becomes an option.
    options: list[tuple[float, float, float, str, Candidate]] = []
    for c in candidates:
        for color in FIGHTER_COLORS:
            conf = float(c.cls_conf.get(color, 0.0))
            if conf >= min_cls_conf:
                options.append((conf, float(c.det_conf), area(c.bbox), color, c))

    # Highest cls_conf first; ties → det_conf, then area.
    options.sort(key=lambda o: (o[0], o[1], o[2]), reverse=True)

    slots: dict[str, Box | None] = {"red": None, "blue": None}
    used: set[int] = set()
    for conf, _det, _ar, color, c in options:
        if slots[color] is None and id(c) not in used:
            x1, y1, x2, y2 = c.bbox
            slots[color] = Box(x1, y1, x2, y2, det_conf=c.det_conf, cls_conf=conf)
            used.add(id(c))
            if slots["red"] is not None and slots["blue"] is not None:
                break
    return slots["red"], slots["blue"]
