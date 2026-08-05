from __future__ import annotations

from bcv.stage2_crop.cropper import Cropper, Stage2Config

W, H = 960, 540


def _cfg(**kw):
    # single_fighter_scale=1.0 here so the existing tight-crop tests are unaffected; the
    # union cropper in run_stage2 uses the configured (>1) scale, per-fighter croppers use 1.0.
    base = dict(ema_alpha=1.0, pad_frac=0.0, max_staleness=30, single_fighter_scale=1.0)
    base.update(kw)
    return Stage2Config(**base)


def test_both_present_square_centered_on_union():
    c = Cropper(_cfg())
    r = c.step((100, 100, 200, 300), (300, 100, 400, 300), W, H)
    assert r.crop_valid and r.staleness == 0
    assert r.crop_box == (100, 50, 400, 350)  # center (250,200), half 150
    x1, y1, x2, y2 = r.crop_box
    assert (x2 - x1) == (y2 - y1)  # square


def test_single_fighter_uses_its_box():
    c = Cropper(_cfg())  # scale 1.0
    r = c.step((100, 100, 200, 300), None, W, H)
    assert r.crop_valid
    assert r.crop_box == (50, 100, 250, 300)  # center (150,200), half 100


def test_single_fighter_scale_expands_to_cover_opponent():
    # only red present -> the union crop is enlarged so a missing opponent is still covered
    c = Cropper(_cfg(single_fighter_scale=2.0))
    r = c.step((100, 100, 200, 300), None, W, H)
    # center (150,200), half 0.5*max(100,200)=100 -> *2.0 = 200 -> box (-50..350) clamped
    assert (r.crop_box[2] - r.crop_box[0]) == 400  # 2x the scale-1.0 crop (200)
    tight = Cropper(_cfg(single_fighter_scale=1.0)).step((100, 100, 200, 300), None, W, H)
    assert (r.crop_box[2] - r.crop_box[0]) > (tight.crop_box[2] - tight.crop_box[0])


def test_carry_forward_when_neither_present():
    c = Cropper(_cfg())
    first = c.step((100, 100, 200, 300), (300, 100, 400, 300), W, H)
    second = c.step(None, None, W, H)
    assert not second.crop_valid and second.staleness == 1
    assert second.crop_box == first.crop_box  # frozen on last good window


def test_fallback_to_full_frame_after_max_staleness():
    c = Cropper(_cfg(max_staleness=2))
    c.step((100, 100, 200, 300), (300, 100, 400, 300), W, H)
    for _ in range(2):
        c.step(None, None, W, H)
    r = c.step(None, None, W, H)  # staleness now 3 > 2
    assert not r.crop_valid
    assert r.crop_box == (210, 0, 750, 540)  # full-frame center square (half = min(W,H)/2)


def test_no_detection_at_start_snaps_to_full_frame():
    c = Cropper(_cfg())
    r = c.step(None, None, W, H)
    assert not r.crop_valid and r.staleness == 1
    assert r.crop_box == (210, 0, 750, 540)


def test_ema_smooths_center():
    c = Cropper(_cfg(ema_alpha=0.5))
    c.step((100, 100, 200, 300), (300, 100, 400, 300), W, H)  # target cx 250
    r = c.step((200, 100, 300, 300), (400, 100, 500, 300), W, H)  # target cx 350
    assert r.ema_cx == 300.0  # 0.5*350 + 0.5*250


def test_per_fighter_crops_isolate_each_fighter():
    # red-only and blue-only croppers (as run_stage2 uses them) crop to their own fighter
    red_c, blue_c = Cropper(_cfg()), Cropper(_cfg())
    rr = red_c.step((100, 100, 200, 300), None, W, H)
    br = blue_c.step(None, (300, 100, 400, 300), W, H)
    assert rr.crop_valid and rr.crop_box == (50, 100, 250, 300)   # centered on red
    assert br.crop_valid and br.crop_box == (250, 100, 450, 300)  # centered on blue
    # a fighter absent -> carry forward, marked invalid
    assert not red_c.step(None, None, W, H).crop_valid
