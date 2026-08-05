"""Shared building blocks for all five stages (contracts, IO, geometry, annotations, …)."""

# NB: no ``from __future__ import annotations`` here — it would bind a package
# attribute ``annotations`` (the __future__ feature) that shadows the submodule.
from . import annotations, config, contracts, geometry, io, splits
from .contracts import (
    STRIKE_LABELS,
    Annotation,
    AnnotationDoc,
    ArtifactMeta,
    StrikeRun,
)
from .splits import BoutSplits, LeakageError, assert_no_leakage

__all__ = [
    "STRIKE_LABELS",
    "Annotation",
    "AnnotationDoc",
    "ArtifactMeta",
    "BoutSplits",
    "LeakageError",
    "StrikeRun",
    "annotations",
    "assert_no_leakage",
    "config",
    "contracts",
    "geometry",
    "io",
    "splits",
]
