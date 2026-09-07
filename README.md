# BitFeat

**BitFeat** is a standalone **1.58-bit local-feature** project: BitNet b1.58 ternary weights `{-1,0,+1}` for a **new deeper/wider** keypoint + descriptor architecture — **not** a quantized ALIKE clone.

Literature suggests ~2× hidden width (and/or more depth) can recover FP-quality at small scales ([BitNet b1.58 Reloaded](https://arxiv.org/abs/2412.11845), *When are 1.58 bits enough?*). BitFeat follows that recipe: FP32 stem + BitConv residual stages at ~2× a tiny baseline width, with **FP32** score and descriptor heads.

This repo is **separate from FastGS** and other projects.

## Layout

```
bitfeat/
  layers/          # BitLinear, BitConv2d (absmean / absmedian + STE)
  models/          # Backbone, heads, BitFeatNet
  losses/          # Depth-warp + dual-softmax / circle matching losses
  data/            # MegaDepth ImagePairDataset + synthetic smoke generator
  eval/            # HPatches MMA CLI
  train.py         # python -m bitfeat.train
  extract.py       # python -m bitfeat.extract
configs/default.yaml
scripts/prepare_megadepth1500.sh
tests/
DESIGN.md
```

## Install

```bash
git clone https://github.com/awschult002/bitfeat.git
cd bitfeat
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# depths need: pip install h5py
```

## Smoke train (no MegaDepth / GPU required)

```bash
python -m bitfeat.train --smoke
```

## Disk constraint + data downloads

Full MegaDepth is ~**200GB**; typical boxes have ~**80GB** free. BitFeat trains on a **megadepth1500** subset:

| Asset | URL / source | Destination |
|-------|----------------|-------------|
| Images | https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip | `$ROOT/megadepth1500/` |
| Depths | MegaDepth v1 dense h5 for scenes `0015`, `0022` | `$ROOT/megadepth1500/depths/{0015,0022}/` |
| Train index | remapped LoFTR-style npz | `$ROOT/index/train_local/*.npz` |
| LoFTR scene_info | LoFTR assets `megadepth_test_1500_scene_info` | `$ROOT/index/megadepth_test_1500_scene_info/` |
| Prep metadata | Parskatt `prep_scene_info.tar` | `$ROOT/prep_scene_info/` |
| HPatches | [hpatches-sequences-release](https://github.com/hpatches/hpatches-dataset) | `/workspace/datasets/hpatches/hpatches-sequences-release/` |

Unpack / symlink helper:

```bash
bash scripts/prepare_megadepth1500.sh
# BITFEAT_DATA=/workspace/datasets/megadepth BITFEAT_DOWNLOADS=/workspace/datasets/downloads
```

**Expected train layout** (`$ROOT=/workspace/datasets/megadepth`):

```
$ROOT/megadepth1500/images/{0015,0022}/*.jpg
$ROOT/megadepth1500/depths/{0015,0022}/*.h5
$ROOT/Undistorted_SfM/{0015,0022}/{images,depths}   # symlinks
$ROOT/index/train_local/*.npz                  # ~1500 pairs; paths relative to $ROOT
```

`train_local` npz keys: `image_paths`, `depth_paths`, `intrinsics`, `poses`, `pair_infos`
where each pair is `(array([i, j]), overlap)`.

## Train (real data)

```bash
python -m bitfeat.train --config configs/default.yaml \
  --data-root /workspace/datasets/megadepth \
  --index-dir index/train_local \
  --device cuda
```

`--smoke` still works without any dataset. Losses warp with depth+pose+K; if a batch has no valid depth/correspondences the geometric terms are skipped gracefully.

## HPatches MMA

```bash
python -m bitfeat.eval.hpatches \
  --hpatches-root /workspace/datasets/hpatches/hpatches-sequences-release \
  --checkpoint checkpoints/bitfeat.pt
```

## Tests (CPU)

```bash
pytest -q
```

## Extract keypoints + descriptors

```bash
python -m bitfeat.extract --image path/to/image.jpg --out feats.npz
```

## Design notes

See [DESIGN.md](DESIGN.md).

## License

MIT — see [LICENSE](LICENSE).
