#!/usr/bin/env bash
# Assemble a zippable data bundle to hand the project to a co-worker.
#
#   bash scripts/make_data_bundle.sh                 # recommended tier, stage only
#   TIER=full bash scripts/make_data_bundle.sh       # also include crop.mp4 (skip re-cropping)
#   ARCHIVE=1 bash scripts/make_data_bundle.sh       # also produce the .tar.gz
#   OUT=/path/to/stage bash scripts/make_data_bundle.sh
#
# Layout produced (lay `data/` + `models/` beside the repo on the co-worker's box, or point
# BCV_DATA_ROOT / BCV_MODELS_ROOT at them — see README §5):
#   <OUT>/data/preprocessed/new_splits/Bout N_Split 1-4/...   (videos + labels + bbox GT + rounds)
#   <OUT>/models/...                                          (yolo26x.pt, boxer classifier)
#   <OUT>/checkpoints/stage3_0.887.ckpt                       (the held-out 0.887 model)
#   <OUT>/configs/                                            (the 5 stage YAMLs)
#   <OUT>/output/...                                          (regenerable parquets [+crop.mp4 if full])
#   <OUT>/MANIFEST.txt, <OUT>/README_BUNDLE.md
#
# Big videos are SYMLINKED into the stage tree (no disk doubling); archive with `tar -czhf`
# (the -h follows symlinks into real contents).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
TIER="${TIER:-recommended}"        # recommended | full
OUT="${OUT:-$REPO/../bcv-data-bundle}"
DATA_ROOT="${BCV_DATA_ROOT:-$REPO/../data}"
MODELS_ROOT="${BCV_MODELS_ROOT:-$REPO/../moughton/models}"
CKPT="$REPO/.cometml-runs/boxing-stage3-frame/0f0def58ba61412ab2e4937c3b1ef645/checkpoints/epoch=7-step=11136.ckpt"
BOUTS=(115 116 117 120 121 122)

echo "[bundle] tier=$TIER  out=$OUT  data=$DATA_ROOT  models=$MODELS_ROOT"
[ -d "$DATA_ROOT/preprocessed/new_splits" ] || { echo "no new_splits under $DATA_ROOT"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"/{data/preprocessed/new_splits,models/boxer_yolo26_classifier/weights,checkpoints,configs}

# --- fights: symlink videos (big), copy labels/rounds/bbox GT (small) ---
for b in "${BOUTS[@]}"; do
  src="$DATA_ROOT/preprocessed/new_splits/Bout ${b}_Split 1-4"
  dst="$OUT/data/preprocessed/new_splits/Bout ${b}_Split 1-4"
  [ -d "$src" ] || { echo "  WARN: missing $src"; continue; }
  mkdir -p "$dst"
  for f in "$src"/*; do
    case "$f" in
      *.mp4) ln -sf "$f" "$dst/$(basename "$f")" ;;   # symlink source videos
      *) cp -f "$f" "$dst/" ;;                          # copy labels / rounds / bbox GT
    esac
  done
  echo "  fight $b staged"
done

# --- weights + checkpoint + configs ---
cp -f "$MODELS_ROOT/yolo26x.pt" "$OUT/models/" 2>/dev/null || echo "  WARN: yolo26x.pt missing"
cp -f "$MODELS_ROOT/boxer_yolo26_classifier/weights/best.pt" "$OUT/models/boxer_yolo26_classifier/weights/" 2>/dev/null || echo "  WARN: classifier best.pt missing"
[ -f "$CKPT" ] && cp -f "$CKPT" "$OUT/checkpoints/stage3_0.887.ckpt" || echo "  WARN: 0.887 ckpt missing"
cp -f "$REPO"/configs/*.yaml "$OUT/configs/"

# --- regenerable outputs: tiny parquets+meta always; crop.mp4 only on full ---
copy_rel() { mkdir -p "$OUT/$(dirname "$1")"; cp -f "$REPO/$1" "$OUT/$1"; }
while IFS= read -r f; do copy_rel "${f#"$REPO"/}"; done < <(find "$REPO/output/stage1_detect" "$REPO/output/stage2_crop" \
  \( -name '*.parquet' -o -name 'meta.json' \) 2>/dev/null)
if [ "$TIER" = "full" ]; then
  while IFS= read -r f; do d="$OUT/${f#"$REPO"/}"; mkdir -p "$(dirname "$d")"; ln -sf "$f" "$d"; done \
    < <(find "$REPO/output/stage2_crop" -name 'crop.mp4' 2>/dev/null)
fi

# --- manifest + bundle README ---
{ echo "# boxing-cv data bundle — tier=$TIER"; echo "# path  size  sha256"
  cd "$OUT"; find . \( -type f -o -type l \) | sort | while IFS= read -r f; do
    case "$f" in
      *.mp4) sha="(video; size-only)" ;;                       # skip hashing the big videos
      *) sha="$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)" ;;
    esac
    printf '%s\t%s\t%s\n' "$f" "$(du -L -h "$f" | cut -f1)" "$sha"
  done; } > "$OUT/MANIFEST.txt"

cat > "$OUT/README_BUNDLE.md" <<'EOF'
# boxing-cv-pipeline — data bundle

Unpack so `data/` and `models/` sit **beside** the cloned repo:
```
<parent>/
├── boxing-cv-pipeline/   # git clone
├── data/                 # this bundle's data/
└── moughton/models/      # this bundle's models/   (rename `models/` -> `moughton/models/`)
```
…or point env vars at wherever you put them:
`export BCV_DATA_ROOT=$PWD/data BCV_MODELS_ROOT=$PWD/models`.

Then: `uv sync --extra detect --extra train --extra label` and `uv run bcv-eval summary`.
The 0.887 Stage-3 checkpoint is `checkpoints/stage3_0.887.ckpt`.
Fights 115/116/117 have hand-labelled fighter boxes; 120/121/122 do not. See the repo README §4.
Verify integrity against `MANIFEST.txt` (sha256). Regenerable artifacts (detections/crops) are
included to save compute; everything else is the source videos + labels + model weights.
EOF

echo "[bundle] staged -> $OUT"
du -sLh "$OUT" 2>/dev/null | cut -f1 | sed 's/^/[bundle] total (resolving symlinks): /'
if [ "${ARCHIVE:-0}" = "1" ]; then
  ARC="$OUT.tar.gz"; echo "[bundle] archiving (follows symlinks) -> $ARC"
  tar -czhf "$ARC" -C "$(dirname "$OUT")" "$(basename "$OUT")"
  echo "[bundle] done: $ARC ($(du -h "$ARC" | cut -f1))"
else
  echo "[bundle] to archive: tar -czhf $OUT.tar.gz -C $(dirname "$OUT") $(basename "$OUT")"
fi
