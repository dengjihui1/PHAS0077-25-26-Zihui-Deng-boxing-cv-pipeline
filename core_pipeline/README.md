# boxing-cv-pipeline

> **Historical snapshot.** This README and the package state reflect the inherited
> project at handoff. The final submitted work (multi-view Stage 4 consensus and the
> Stage 5 fighter-query classifier) lives in `../stage4_multiview_consensus_final/` and
> `../stage5_fighter_query_final/`; see the repository root README for the final results.
> References below to "Stage 5 NOT built" and AUROC 0.887 are the pre-improvement
> baseline, not the final numbers.

A 5-stage computer-vision pipeline that detects **punches** in boxing footage and (eventually)
classifies strike type + which fighter. Package `bcv`, uv-managed, Python ≥3.11.

```
Stage 1 detect fighters → Stage 2 crop to fighters → Stage 3 per-frame P(punch)
   → Stage 4 windowing (events) → Stage 5 strike-type + fighter   [Stage 5 NOT built]
```

**Headline:** the Stage-3 frame classifier learns a real, cross-fight punch signal —
held-out **AUROC 0.887 / AP 0.628** on bout 115 (trained on 116+117).

> ### 👉 Your first job: **improve the punch finder (Stage 3).** See [§9](#9-your-first-task-improve-the-punch-finder).
> AUROC generalizes well, but **AP and downstream window-F1 (0.36) are ceiling'd** by an
> over-confident probability curve and weak crops. The highest-ratio fixes are cheap retrains.

Deeper context: **`HANDOFF.md`** (current state + prioritized roadmap), **`CLAUDE.md`** (the
authoritative data inventory).

---

## 1. Status at a glance
All three gates are green (run them yourself to trust the tree):
```bash
uv run pytest -q        # 80 passed (~5s)
uv run ruff check .     # All checks passed!
uv run mypy src         # Success: no issues found in 56 source files
```
> The working tree may have uncommitted edits — check `git status`.

## 2. Quickstart (fresh clone + data bundle)
1. **Lay out the data + models** (the defaults expect them *beside* the repo):
   ```
   <parent>/
   ├── boxing-cv-pipeline/        # this repo
   ├── data/                      # = the unzipped bundle's data/  (videos + labels)
   └── moughton/models/           # = the bundle's models/  (yolo26x.pt, boxer classifier, ...)
   ```
   …or point anywhere via env (see [§5](#5-configuration--paths)):
   `export BCV_DATA_ROOT=/path/to/data BCV_MODELS_ROOT=/path/to/models`.
2. **Install deps** ([uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync --extra detect --extra train --extra label
   ```
   Extras: `detect` (ultralytics), `train` (torch cu124 + lightning + comet), `label` (FastAPI GUI).
3. **(optional) Comet logging:** put `COMET_API_KEY=…` + `COMET_WORKSPACE=…` in a gitignored `.env`.
   Without it, training runs but **isn't logged** (no crash).
4. **Smoke test** — see a number:
   ```bash
   uv run bcv-eval --pipeline-config configs/pipeline.yaml summary   # per-stage scorecard
   ```

## 3. The 5 stages + CLI
Every stage is a `src/bcv/stageN_*/` package, runnable standalone, emitting a typed artifact
(+ a debug video). Entrypoints (see `pyproject.toml [project.scripts]`):

| Command | Stage | Purpose | Example |
|---|---|---|---|
| `bcv-split` | pre | split a 2×2 quad POV mp4 → 4 per-camera videos | `uv run bcv-split -i orig.mp4 -o data/preprocessed/new_splits/` |
| `bcv-detect` | 1 | chain detector (YOLO track + red/blue classifier) → `detections.parquet` | `uv run bcv-detect --bout 117 --split 1` |
| `bcv-crop` | 2 | union-crop the fighters → fixed-square `crop.mp4` | `uv run bcv-crop --bout 117 --split 1 --no-debug-video` |
| `bcv-frame-clf` | 3 | per-frame punch classifier (`fit`/`predict`) → `frame_probs.parquet` | `uv run bcv-frame-clf predict --bout 115 --split 1 --ckpt <ckpt>` |
| `bcv-window` | 4 | hysteresis on P(punch) → strike `windows.json` | `uv run bcv-window --bout 115 --split 1` |
| `bcv-eval` | — | `detection` / `frame` / `window` / `summary` metrics + plots | `uv run bcv-eval --bout 115 --split 1 frame` |
| `bcv-label` | — | **browser bbox-labelling GUI** (SSH tunnel) | see [§6](#6-labelling-gui-bcv-label) |
| `bcv-label-prefill` | — | precompute initial boxes for the GUI | `uv run bcv-label-prefill --bout 120` |

> ⚠️ **`bcv-strike-clf` (Stage 5) is not implemented** — the package is empty (ImportError on run).
> ⚠️ **`bcv-detector-finetune` (Backend B) is a stub** — raises `SystemExit`; blocked on Melik's
> `train15` detector dataset/weights, which are **not on this machine**.

## 4. Data layout
The code expects `data_root` (`${BCV_DATA_ROOT}/preprocessed/new_splits/`) to contain one
directory per fight named `Bout N_Split 1-4/`. **Three distinct label types — don't conflate**
(full detail in `CLAUDE.md`):

| Fight | Videos | Punch event labels | Rounds | Fighter-box GT |
|---|:---:|:---:|:---:|:---:|
| **115, 116, 117** | split_0..3.mp4 | `annotations.json` | `rounds.json` (inferred) | **`split_S_fighter_bboxes.json` ✅** |
| **120, 121, 122** | split_0..3.mp4 | `Bout N_Split 1-4.json` *(diff. filename!)* | `rounds.json` (manual) | ❌ (run the detector) |

- **Only 115/116/117 have hand-labelled fighter boxes** → the only fights usable for Stage-1/2
  *detection* evaluation. 120/121/122's boxes come from the chain detector.
- **`rounds.json`** (fight-level — the 4 splits are frame-synced) gives `[start,end]` frame spans;
  training/eval use **only in-round frames** (drops ~16–19% of rest/walk-in/post — `bcv/common/rounds.py`).
- If the bundle isn't unpacked to the expected path, `build_dataset` returns **None silently** — check paths first.

## 5. Configuration & paths
All YAML flows through one choke point (`src/bcv/common/config.py: load_yaml`), which expands env
vars and sets repo-relative defaults:

| Var | Default | Used by |
|---|---|---|
| `BCV_DATA_ROOT` | `<repo>/../data` | `configs/pipeline.yaml: data_root` |
| `BCV_MODELS_ROOT` | `<repo>/../moughton/models` | `configs/stage1_detect.yaml` weights |
| `BCV_REPO_ROOT` | the repo dir | `output_root` |

To repoint after unzipping: lay `data/` + `moughton/models/` beside the repo, or set the vars (shell or `.env`).

## 6. Labelling GUI (`bcv-label`)
Browser tool to grow labelled data (served over an SSH tunnel):
```bash
uv run bcv-label-prefill --bout 120            # precompute initial boxes (instant from detections)
uv run bcv-label --bout 120 --split 0 --port 8000
# locally:  ssh -L 8000:localhost:8000 <host>  →  http://localhost:8000
```
Keys: **R/B** pick fighter · **drag** place box · press **`y`** then **click a grey box** to assign a
raw-YOLO detection · **X** absent · **`[`/`]`** jump keyframes · **`g`/`h`** mark round start/end ·
**Space** play · **S** save (exports `split_S_fighter_bboxes.json`). The green **CROP** box previews
what Stage 2 will crop (enlarges when only one fighter is found).
> ⚠️ Box edits are in-memory until you press **Save** (rounds auto-save). Save often.

## 7. Current results (with honest caveats)
- **Stage 3 (punch):** held-out **115 → AUROC 0.887 / AP 0.628**. Pooled on chain-cropped
  **120/121/122 → ~0.76 / ~0.21** (AUROC generalizes; AP collapses ~3× because the chain crops are
  weak — it misses the RED fighter ~40%).
- **Stage 1 (chain detector) on 117 vs GT:** presence recall **0.72**, precision **0.94**, mean IoU
  **0.96** — localization is great; the gap is recall, almost entirely the **RED fighter (0.60 vs
  blue 0.88)**, caused by the red/blue classifier+gate (not the detector).
- **Caveats:** Stage-4 thresholds + temperature were fit *in-sample*; 116/117 overlays read
  optimistically (the model trained on them); only **115** (and box-less 120/121/122) are genuinely held out.

## 8. Known issues & gotchas
- **Three near-duplicate Stage-3 training entrypoints** (`scripts/cross_bout_experiment.py`,
  `scripts/temporal_experiment.py`, `src/bcv/stage3_frame_classifier/run.py`) diverge on
  backbone/calibration/val. **The 0.887 came from `cross_bout_experiment.py`** (`channel_stack_2d`
  @112px, `pos_weight`, `temperature=1.0`), **not** `bcv-frame-clf fit` defaults (`small3d`).
- **Eval leakage in `run.py: run_predict`** — fits temperature on the *same* split it scores, and
  aliases `val=train`. Don't trust `run.py` metrics until reconciled ([§9](#9-your-first-task-improve-the-punch-finder) task 4).
- **`min_cls_conf` was just lowered 0.5→0.4** but existing crops + the 0.887 model were built at
  0.5 — re-run detection/crop to propagate it.
- The 0.887 checkpoint is `.cometml-runs/boxing-stage3-frame/0f0def58.../checkpoints/epoch=7-step=11136.ckpt`.
- `HANDOFF.md` uses `[[wiki-links]]` to an external notes vault — ignore the brackets.

## 9. YOUR FIRST TASK: improve the punch finder
Reproduce the baseline **before changing anything** (the canonical run, not the CLI):
```bash
uv run python scripts/cross_bout_experiment.py --train-bouts 116 117 --eval-bout 115
uv run python scripts/make_eval_probs.py --eval-bout 115 --eval-split 1   # predict-only, no OOM
```
Ranked levers (detail/rationale in `HANDOFF.md`):

1. **[LOW effort, HIGH impact — do first] Remove the double positive-weighting.** The balanced
   sampler **and** `pos_weight≈neg/pos` both up-weight positives → an over-confident curve that
   ceilings Stage-4 (window-F1 0.36) and wrecks calibration. Set `pos_weight≈1` when the sampler is
   on (`run.py: _auto_pos_weight` + the `PunchDataModule` sampler). One cheap ~8-epoch retrain.
2. **[LOW to start, HIGH+PROVEN] Add 120/121/122 to the training mix.** They have punch labels +
   chain crops. More data is the most-proven lever (8× data moved AP 0.32→0.63). Pairs with task 3.
3. **[MEDIUM, HIGH — dominant AP lever] Fix detector→crop→punch.** Propagate `min_cls_conf=0.4`,
   improve RED recall (or build Backend B), re-crop → better crops → better AP (it collapsed ~3× on weak crops).
4. **[MEDIUM, prerequisite for trust] Reconcile the 3 training entrypoints into the CLI and remove
   the `run.py` eval leakage** so `bcv-frame-clf` reproduces 0.887.
5. **[MEDIUM, uncertain — last] Stronger/temporal backbone + higher resolution** (needs the
   `crop_windows.parquet` refactor to avoid OOM).

## 10. The data bundle
Build a zippable bundle with `scripts/make_data_bundle.sh` (see `--help`):
- **Recommended (~13.5 GB):** the 6 fights' videos + all labels + model weights + the 0.887 checkpoint
  + configs + tiny regenerable `*.parquet`/`meta.json` (skips ~16 GB of regenerable debug mp4s).
- **Full:** also includes `crop.mp4` so Stage 3 runs without re-cropping.

The bundle ships a `MANIFEST.txt` (sizes + sha256) and its own unpack README. Lay its `data/` and
`models/` beside the repo (or set the env vars in [§5](#5-configuration--paths)).

## 11. Repo map
```
src/bcv/   common/ (config, io, video, geometry, rounds, annotations, viz)
           stage1_detect/  stage2_crop/  stage3_frame_classifier/  stage4_windowing/
           eval/ (detection, frame, window, summary) · label/ (the GUI) · preprocess/
scripts/   cross_bout_experiment.py (the real generalization test), make_eval_probs.py,
           sweep_stage4.py, seed_rounds.py, import_gt.py, detect_parallel.sh, … (self-documented)
configs/   pipeline.yaml + 5 stage configs        tests/   80 tests (pytest)
```
