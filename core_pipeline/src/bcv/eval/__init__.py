"""Evaluation: frame ROC/PR, window event-detection, and Stage-1 detection metrics + plots."""

from .detection import detection_metrics, plot_detection_eval
from .frame import frame_metrics, plot_frame_eval, pr_curve, roc_curve
from .window import match_events, plot_window_eval, window_metrics

__all__ = [
    "detection_metrics",
    "frame_metrics",
    "match_events",
    "plot_detection_eval",
    "plot_frame_eval",
    "plot_window_eval",
    "pr_curve",
    "roc_curve",
    "window_metrics",
]
