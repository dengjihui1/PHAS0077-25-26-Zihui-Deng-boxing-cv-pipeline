"""Profile the Stage-1 CHAIN detector to locate its speed bottleneck.

Stage 1's ChainDetector (src/bcv/stage1_detect/backend_chain.py) runs, per frame:
  - ONE   self.det.track(frame, ...)      # YOLO detect + botsort tracking
  - up to max_det  self.cls(crop, ...)     # one classifier call PER person crop

This script does NOT modify backend_chain.py. It monkeypatches the *instances*
``detector.det.track`` and ``detector.cls`` with thin timing wrappers (perf_counter
accumulators + call counts), and separately times frame decode (next(reader)) and the
remaining per-frame Python/numpy work ("other"). It samples nvidia-smi GPU utilisation
on a background thread during the run.

Run with the project venv:
  /home/ubuntu/boxing-cv-pipeline/.venv/bin/python scripts/profile_stage1.py
Optional: --frames N  --bout B  --split S
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections.abc import Callable

from bcv.common.config import load_config, load_pipeline_config
from bcv.common.video import VideoReader
from bcv.stage1_detect.run import Stage1Config, build_detector


class Accum:
    """Wall-clock accumulator + call counter for a monkeypatched callable."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0

    def wrap(self, fn: Callable):
        def inner(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.seconds += time.perf_counter() - t0
                self.calls += 1

        return inner


class GpuSampler(threading.Thread):
    """Poll nvidia-smi GPU + memory utilisation on a background thread."""

    def __init__(self, gpu_index: int = 0, period_s: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.period_s = period_s
        self._stop_evt = threading.Event()
        self.util: list[float] = []
        self.mem_used: list[float] = []

    def run(self) -> None:
        query = (
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            f"--id={self.gpu_index}",
        )
        while not self._stop_evt.is_set():
            try:
                out = subprocess.check_output(
                    ("nvidia-smi", *query), text=True, timeout=5
                ).strip()
                u, m = (x.strip() for x in out.splitlines()[0].split(","))
                self.util.append(float(u))
                self.mem_used.append(float(m))
            except Exception:
                pass
            self._stop_evt.wait(self.period_s)

    def stop(self) -> dict[str, float]:
        self._stop_evt.set()
        self.join(timeout=2)

        def stats(xs: list[float]) -> dict[str, float]:
            if not xs:
                return {"mean": 0.0, "max": 0.0, "n": 0}
            return {"mean": sum(xs) / len(xs), "max": max(xs), "n": len(xs)}

        return {"util_pct": stats(self.util), "mem_used_mib": stats(self.mem_used)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--bout", type=int, default=120)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10, help="frames excluded from timing")
    args = ap.parse_args()

    stage_cfg = load_config("configs/stage1_detect.yaml", Stage1Config)
    pipe = load_pipeline_config("configs/pipeline.yaml")
    video_path = pipe.split_video(args.bout, args.split)
    print(f"video: {video_path}")
    print(
        f"cfg: det_imgsz={stage_cfg.det_imgsz} cls_imgsz={stage_cfg.cls_imgsz} "
        f"max_det={stage_cfg.max_det} tracker={stage_cfg.tracker} half={stage_cfg.half}"
    )

    detector = build_detector(stage_cfg)
    detector.reset()

    # Monkeypatch the two GPU calls on the *instances* (not backend_chain.py).
    det_acc, cls_acc = Accum(), Accum()
    detector.det.track = det_acc.wrap(detector.det.track)  # type: ignore[attr-defined]
    detector.cls = cls_acc.wrap(detector.cls)  # type: ignore[attr-defined]

    reader = VideoReader(video_path)
    it = iter(reader)

    decode_s = 0.0
    detect_s = 0.0  # wall time of detector.detect() (includes det_track + cls + other)
    frames_done = 0

    # ---- warmup (cudnn autotune, tracker init, first-call JIT) ----
    for _ in range(args.warmup):
        frame = next(it)
        detector.detect(frame, frames_done)
        frames_done += 1
    det_acc.seconds = det_acc.calls = 0
    cls_acc.seconds = cls_acc.calls = 0

    sampler = GpuSampler(gpu_index=0)
    sampler.start()

    t_wall0 = time.perf_counter()
    timed_frames = 0
    try:
        while timed_frames < args.frames:
            t0 = time.perf_counter()
            try:
                frame = next(it)
            except StopIteration:
                break
            decode_s += time.perf_counter() - t0

            t1 = time.perf_counter()
            detector.detect(frame, frames_done)
            detect_s += time.perf_counter() - t1

            frames_done += 1
            timed_frames += 1
    finally:
        wall_s = time.perf_counter() - t_wall0
        gpu = sampler.stop()
        reader.release()

    # detector.detect wall time decomposes into det_track + classifier + "other"
    # (crop_person numpy slicing, Boxes iteration, .tolist() host copies, select, mem).
    other_in_detect = detect_s - det_acc.seconds - cls_acc.seconds
    total_loop = decode_s + detect_s  # ~= wall_s; loop overhead is negligible

    def pct(x: float) -> float:
        return 100.0 * x / total_loop if total_loop else 0.0

    fps = timed_frames / wall_s if wall_s else 0.0
    cls_per_frame = cls_acc.calls / timed_frames if timed_frames else 0.0

    det_pct = pct(det_acc.seconds)
    cls_pct = pct(cls_acc.seconds)
    dec_pct = pct(decode_s)
    oth_pct = pct(other_in_detect)

    components = {
        "det_track": det_pct,
        "classifier": cls_pct,
        "decode": dec_pct,
        "other": oth_pct,
    }
    dominant = max(components, key=components.get)

    report = {
        "frames_profiled": timed_frames,
        "wall_s": round(wall_s, 3),
        "fps": round(fps, 2),
        "det_track_pct": round(det_pct, 1),
        "classifier_pct": round(cls_pct, 1),
        "decode_pct": round(dec_pct, 1),
        "other_pct": round(oth_pct, 1),
        "dominant_cost": dominant,
        "classifier_calls_per_frame": round(cls_per_frame, 2),
        "ms_per_frame": {
            "det_track": round(1000 * det_acc.seconds / timed_frames, 2),
            "classifier": round(1000 * cls_acc.seconds / timed_frames, 2),
            "decode": round(1000 * decode_s / timed_frames, 2),
            "other": round(1000 * other_in_detect / timed_frames, 2),
        },
        "det_track_calls": det_acc.calls,
        "classifier_calls": cls_acc.calls,
        "gpu": gpu,
    }
    print("\n===== STAGE-1 CHAIN PROFILE =====")
    print(json.dumps(report, indent=2))
    print("=================================")


if __name__ == "__main__":
    main()
