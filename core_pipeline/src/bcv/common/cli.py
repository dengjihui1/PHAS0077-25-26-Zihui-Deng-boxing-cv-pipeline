"""Shared argparse scaffold so every stage CLI exposes the same core flags."""

from __future__ import annotations

import argparse


def base_parser(description: str) -> argparse.ArgumentParser:
    """Build the common argument parser used by every ``bcv-*`` stage entrypoint."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="Path to this stage's YAML config")
    p.add_argument(
        "--pipeline-config", default="configs/pipeline.yaml", help="Shared pipeline config"
    )
    p.add_argument("--bout", type=int, required=True, help="Bout number")
    p.add_argument("--split", type=int, default=0, help="Camera split index (0-3)")
    p.add_argument("--output-root", default=None, help="Override pipeline output_root")
    p.add_argument(
        "--debug-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit the stage's decorated debug.mp4",
    )
    return p
