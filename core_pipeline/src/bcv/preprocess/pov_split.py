"""Split a 2x2 quad-view boxing video into 4 per-POV videos (the ``new_splits`` layout).

The raw recordings tile 4 camera angles as a 2x2 grid (e.g. 1920x1080 -> 4x 960x540):

    split_0 = top-left      split_1 = top-right
    split_2 = bottom-left   split_3 = bottom-right

split_2 (bottom-left) is **horizontally flipped** — that camera is mirrored, and the
fighter boxes were generated against the flipped frame, so we must reproduce the flip.

ffmpeg is supplied by the ``imageio-ffmpeg`` dependency, so no system ffmpeg is required.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Even half-dims keep H.264 happy: half_w = 2*trunc(iw/4), half_h = 2*trunc(ih/4)
_HALF_W, _HALF_H = "2*trunc(iw/4)", "2*trunc(ih/4)"


def ffmpeg_exe() -> str:
    """Locate an ffmpeg binary: prefer the bundled imageio-ffmpeg, fall back to PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        raise RuntimeError(
            "ffmpeg not found — install the 'imageio-ffmpeg' dependency or a system ffmpeg"
        ) from None


def crop_filter(split_index: int) -> str:
    """ffmpeg ``-vf`` string cropping the given quadrant (split_2 also h-flips)."""
    if split_index not in (0, 1, 2, 3):
        raise ValueError("split_index must be 0..3")
    x = _HALF_W if split_index % 2 == 1 else "0"
    y = _HALF_H if split_index >= 2 else "0"
    vf = f"crop={_HALF_W}:{_HALF_H}:{x}:{y}"
    if split_index == 2:
        vf += ",hflip"
    return vf


def _encode_split(
    ffmpeg: str, input_path: Path, out_path: Path, split_index: int,
    *, crf: int, preset: str, overwrite: bool,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", str(input_path), "-map", "0:v:0", "-an",
        "-vf", crop_filter(split_index),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-y" if overwrite else "-n", str(out_path),
    ]
    subprocess.run(cmd, check=True)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no output for split_{split_index}: {out_path}")
    return out_path


def split_quad_video(
    input_path: str | Path, out_dir: str | Path,
    *, splits: tuple[int, ...] = (0, 1, 2, 3), crf: int = 17,
    preset: str = "fast", overwrite: bool = True, max_workers: int = 4,
) -> dict[int, Path]:
    """Split ``input_path`` into ``out_dir/<stem>/split_N.mp4`` for each requested split."""
    ffmpeg = ffmpeg_exe()
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    bout_dir = Path(out_dir) / input_path.stem
    targets = {s: bout_dir / f"split_{s}.mp4" for s in splits}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as pool:
        futures = {
            pool.submit(
                _encode_split, ffmpeg, input_path, out, s,
                crf=crf, preset=preset, overwrite=overwrite,
            ): s
            for s, out in targets.items()
        }
        for fut in futures:
            fut.result()  # propagate the first error
    return targets


def main() -> None:
    p = argparse.ArgumentParser(description="Split a 2x2 quad POV video into 4 POV videos")
    p.add_argument("--input", "-i", required=True, help="Path to the 2x2 quad-view .mp4")
    p.add_argument("--out-dir", "-o", required=True, help="Output root (writes <stem>/split_N.mp4)")
    p.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--crf", type=int, default=17)
    p.add_argument("--preset", default="fast")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    outs = split_quad_video(
        args.input, args.out_dir, splits=tuple(args.splits),
        crf=args.crf, preset=args.preset, overwrite=args.overwrite,
    )
    for s, o in sorted(outs.items()):
        print(f"[OK] split_{s} -> {o}")


if __name__ == "__main__":
    main()
