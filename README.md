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

### Checkpoints, resume, and mixed precision

Every `train.checkpoint_every` steps (default 1000) the trainer writes
`<output_dir>/ckpt_stepXXXXXXXX.pt` (model, optimizer, GradScaler, step, config,
RNG states), keeps the newest `train.keep_last` (default 3), and writes
`<output_dir>/final.pt` at `max_steps`. Ctrl-C / SIGTERM finishes the current
step and checkpoints before exiting. Resume exactly (same data order and loss
sampling) with `--resume auto` (newest checkpoint in `--output-dir`) or
`--resume path/to/ckpt.pt`. Metrics go to `<output_dir>/metrics.jsonl`.

`--amp` enables `torch.autocast` + `GradScaler` (CUDA; `--amp-dtype float16`
by default, `bfloat16` on Ampere+). Ternarization (gamma, divide, round/STE)
runs in FP32 on the FP32 shadow weights and all losses are computed in FP32;
non-finite losses skip the update and abort after `max_nonfinite_steps` in a
row. `--grad-checkpoint` recomputes the full-resolution stem + first stage in
backward (BatchNorm running stats in those layers are updated twice per step).

On a 4 GB GPU (Quadro T2000) the default batch 2 @ 480x480 fits only with
`--amp --grad-checkpoint` (~3.0 GB peak). Note that the T2000 (TU117) has no
fast FP16 tensor-core path: cuDNN's FP16 convs run ~4x slower than FP32 there,
so AMP buys memory, not speed. On that card FP32 `--batch-size 1 --image-size
384 384` (~1.8 it/s) or FP32 `--batch-size 1 --grad-checkpoint` at 480 are
the faster choices; on Ampere+ GPUs prefer `--amp --amp-dtype bfloat16`.

```bash
python -m bitfeat.train --config configs/default.yaml \
  --data-root ~/data/megadepth --device cuda --amp --grad-checkpoint \
  --output-dir runs/bitfeat_main --resume auto
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
