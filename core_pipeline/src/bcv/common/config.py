"""One typed config layer for all five stages — no Hydra, no LightningCLI sprawl.

``pipeline.yaml`` is the single source of truth for data roots and bout splits; each
stage config ``extends`` it. Stage configs are plain Pydantic models loaded by
``load_config`` and instantiate their (Lightning or plain) components from typed
objects — there is no second config system and no global/env state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

from .splits import BoutSplits


class PipelineConfig(BaseModel):
    """Shared roots + splits referenced by every stage (``configs/pipeline.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    data_root: Path
    output_root: Path
    bouts: dict[int, str]  # bout number -> directory name under data_root
    splits: BoutSplits
    num_views: int = 4

    def bout_dir(self, bout: int) -> Path:
        return self.data_root / self.bouts[bout]

    def split_video(self, bout: int, split: int) -> Path:
        return self.bout_dir(bout) / f"split_{split}.mp4"

    def artifact_dir(self, bout: int, split: int, stage: str) -> Path:
        # Grouped by STAGE first so all of a stage's outputs live together:
        # output/<stage>/<fight>/split_<N>/
        return self.output_root / stage / self.bouts[bout] / f"split_{split}"

    def stage_summary_dir(self, stage: str) -> Path:
        """Per-stage summary dir: output/<stage>/summary/ (scorecards, montages)."""
        return self.output_root / stage / "summary"


T = TypeVar("T", bound=BaseModel)


_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../boxing-cv-pipeline


def _set_path_defaults() -> None:
    """Portable data/model roots: default beside the repo, overridable via env (or .env).

    Lets ``configs/*.yaml`` reference ``${BCV_DATA_ROOT}`` / ``${BCV_MODELS_ROOT}`` so a
    co-worker who unzips the data bundle just lays ``data/`` and ``moughton/models/`` next
    to the repo (or sets the two env vars) — no editing of YAMLs or code.
    """
    os.environ.setdefault("BCV_REPO_ROOT", str(_REPO_ROOT))
    os.environ.setdefault("BCV_DATA_ROOT", str(_REPO_ROOT.parent / "data"))
    os.environ.setdefault("BCV_MODELS_ROOT", str(_REPO_ROOT.parent / "moughton" / "models"))


def load_yaml(path: str | Path) -> dict:
    _set_path_defaults()
    # Expand ${BCV_DATA_ROOT} etc. so configs are portable across machines.
    data = yaml.safe_load(os.path.expandvars(Path(path).read_text()))
    return data or {}


def load_config(path: str | Path, model: type[T]) -> T:
    """Load a YAML file and validate it into ``model``."""
    return model.model_validate(load_yaml(path))


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    return load_config(path, PipelineConfig)
