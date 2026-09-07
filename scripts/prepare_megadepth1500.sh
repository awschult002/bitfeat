#!/usr/bin/env bash
# Prepare a disk-friendly MegaDepth subset for BitFeat training/eval.
#
# Disk constraint: full MegaDepth is ~200GB; typical boxes have ~80GB free.
# We use ETH megadepth1500 images + (optional) depth h5 + remapped scene_info.
#
# Download URLs:
#   Images:  https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip
#   Prep:    Parskatt prep_scene_info.tar (pairs / poses metadata)
#   LoFTR:   megadepth_test_1500_scene_info (official LoFTR assets)
#   Depths:  subset of MegaDepth v1 dense depths for scenes 0015 / 0022
#            (place under megadepth1500/depths/{0015,0022}/*.h5)
#
# Target layout (BITFEAT_DATA default /workspace/datasets/megadepth):
#
#   $ROOT/
#     megadepth1500/images/{0015,0022}/*.jpg
#     megadepth1500/depths/{0015,0022}/*.h5
#     megadepth1500/{pairs.txt,pairs_calibrated.txt,views.txt}
#     Undistorted_SfM/{0015,0022}/images -> ../../megadepth1500/images/{scene}
#     Undistorted_SfM/{0015,0022}/depths -> ../../megadepth1500/depths/{scene}
#     index/train_local/*.npz   # PREFERRED train index (remapped local paths)
#     index/megadepth_test_1500_scene_info/*.npz
#     prep_scene_info/*.npy
#
# train_local npz paths are relative to $ROOT, e.g.
#   megadepth1500/images/0015/foo.jpg
#   megadepth1500/depths/0015/foo.h5
# Keys: image_paths, depth_paths, intrinsics, poses, pair_infos
#   pair_infos[k] = (array([i, j]), overlap)
#
set -euo pipefail

ROOT="${BITFEAT_DATA:-/workspace/datasets/megadepth}"
DL="${BITFEAT_DOWNLOADS:-/workspace/datasets/downloads}"
mkdir -p "$ROOT" "$DL"

echo "==> Root: $ROOT"

if [[ -f "$DL/megadepth1500.zip" ]]; then
  echo "==> Unpacking megadepth1500.zip"
  unzip -n "$DL/megadepth1500.zip" -d "$ROOT"
  # Zip contains megadepth1500/{images,pairs*.txt,views.txt}
else
  echo "Missing $DL/megadepth1500.zip — download from:"
  echo "  https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip"
fi

if [[ -f "$DL/prep_scene_info.tar" ]]; then
  echo "==> Extracting prep_scene_info.tar"
  mkdir -p "$ROOT/prep_scene_info"
  tar -xf "$DL/prep_scene_info.tar" -C "$ROOT"
fi

# Symlink Undistorted_SfM layout expected by classic LoFTR paths
for scene in 0015 0022; do
  mkdir -p "$ROOT/Undistorted_SfM/$scene"
  if [[ -d "$ROOT/megadepth1500/images/$scene" ]]; then
    ln -sfn "$ROOT/megadepth1500/images/$scene" "$ROOT/Undistorted_SfM/$scene/images"
  fi
  if [[ -d "$ROOT/megadepth1500/depths/$scene" ]]; then
    ln -sfn "$ROOT/megadepth1500/depths/$scene" "$ROOT/Undistorted_SfM/$scene/depths"
  fi
done

# Expect train_local index already built (remapped paths). Document only:
#   $ROOT/index/train_local/*.npz
if [[ -d "$ROOT/index/train_local" ]]; then
  echo "==> Found index/train_local ($(ls "$ROOT/index/train_local"/*.npz 2>/dev/null | wc -l) npz)"
else
  echo "NOTE: place remapped train npz under $ROOT/index/train_local/"
  echo "      (paths like megadepth1500/images/0015/foo.jpg relative to ROOT)"
fi

echo "==> Done. Train with:"
echo "  python -m bitfeat.train --config configs/default.yaml \\"
echo "    --data-root $ROOT --index-dir index/train_local --device cuda"
