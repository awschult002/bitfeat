# BitFeat

**BitFeat** is a standalone **1.58-bit local-feature** project: BitNet b1.58 ternary weights `{-1,0,+1}` for a **new deeper/wider** keypoint + descriptor architecture — **not** a quantized ALIKE clone.

Literature suggests ~2× hidden width (and/or more depth) can recover FP-quality at small scales ([BitNet b1.58 Reloaded](https://arxiv.org/abs/2412.11845), *When are 1.58 bits enough?*). BitFeat follows that recipe: FP32 stem + BitConv residual stages at ~2× a tiny baseline width, with **FP32** score and descriptor heads.

This repo is **separate from FastGS** and other projects.

## Layout

```
bitfeat/
  layers/          # BitLinear, BitConv2d (absmean / absmedian + STE)
  models/          # Backbone, heads, BitFeatNet
  losses/          # Matching / reprojection stubs (MegaDepth-ready TODOs)
  data/            # Pair dataset stub + synthetic smoke generator
  train.py         # python -m bitfeat.train
  extract.py       # python -m bitfeat.extract
configs/default.yaml
tests/
DESIGN.md
```

## Install

```bash
git clone https://github.com/awschult002/bitfeat.git
cd bitfeat
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
```

## Smoke train (no MegaDepth / GPU required)

```bash
python -m bitfeat.train --smoke
# or after install:
bitfeat-train --smoke
```

## Tests (CPU)

```bash
pytest -q
```

## Extract keypoints + descriptors

```bash
python -m bitfeat.extract --image path/to/image.jpg --out feats.npz
# bitfeat-extract --image path/to/image.jpg
```

Outputs an `.npz` with `keypoints`, `scores`, `descriptors`.

## Train (real data — next steps)

1. Download / prepare **MegaDepth** (or DISK-style) image pairs with depth + poses.
2. Implement loading in `bitfeat/data/pairs.py` (`ImagePairDataset`).
3. Replace loss stubs in `bitfeat/losses/matching.py` with warp + dual-softmax / circle loss.
4. Point the config at your data root:

```bash
# configs/default.yaml → data.root: /path/to/megadepth
python -m bitfeat.train --config configs/default.yaml --device cuda
```

## Design notes

See [DESIGN.md](DESIGN.md) for architecture rationale, separation from FastGS, and the HPatches MMA + matching-recall eval plan.

## License

MIT — see [LICENSE](LICENSE).
