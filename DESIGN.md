# BitFeat design notes

## Why a separate repo

BitFeat is a **standalone** 1.58-bit local-feature research codebase. It is **not** part of FastGS, vid2scene, or any Gaussian / novel-view stack. Keeping it separate avoids coupling training data, licenses, and release cadence to those projects.

## Ternary weights (BitNet b1.58)

Shadow (trainable) weights stay in FP32. On the forward pass we ternarize:

```
gamma = mean(|W|)  or  median(|W|)
W_q   = round(clip(W / (gamma + eps), -1, 1))
```

Effective weight is `gamma * W_q`. Gradients use a straight-through estimator (STE) through the quantizer (`bitfeat/layers/bitlinear.py`, `bitconv.py`).

## Scale width / depth (not "quantize ALIKE")

Quantizing an existing tiny detector usually underperforms. Following *BitNet b1.58 Reloaded* / *When are 1.58 bits enough?*, BitFeat uses a **new** CNN:

| Piece | Precision | Notes |
|-------|-----------|--------|
| Stem | FP32 Conv | Stable early filters |
| Stages | BitConv residuals | Channels e.g. `64→128→256→256` (~2× tiny) |
| Depth | 2 blocks / stage (configurable) | Extra capacity vs. shallow ternaries |
| Score + descriptor heads | FP32 | Detection / desc quality sensitive |

Default is deeper+wider than ALIKE-tiny-class backbones; hyperparameters live in `configs/default.yaml`.

## Heads

- **Score map**: dense sigmoid confidence (DKD-style soft-NMS top-k stub for extraction).
- **Descriptors**: dense L2-normalized vectors (default dim 128), bilinearly upsampled to input resolution for losses / export.

## Data + disk constraint

Full MegaDepth (~200GB) does **not** fit typical boxes (~80GB free). Training uses:

- ETH **megadepth1500** images for scenes `0015` / `0022`
- Matching dense **depth h5** under `megadepth1500/depths/`
- Remapped LoFTR-style index at `index/train_local/*.npz` (paths relative to the MegaDepth root)

See `scripts/prepare_megadepth1500.sh` and README download table.

`ImagePairDataset` (`bitfeat/data/pairs.py`) prefers `index/train_local`, then classic LoFTR scene_info, prep_scene_info npy, and finally `pairs_calibrated.txt`. It does **not** invent homography/synthetic poses when only RGB exists — depth h5 or npz poses are required for geometric supervision.

## Training plan

1. **Smoke**: synthetic pairs (`SyntheticPairDataset`) via `--smoke`.
2. **Real**: MegaDepth subset with depth + relative pose (`T_0to1`).
3. Losses in `losses/matching.py`:
   - Warp sampled pixels with depth + `T_0to1` + `K`.
   - Score heatmap BCE at projected correspondences.
   - Dual-softmax (default) or circle loss on gathered descriptors.
   - Graceful zero / skip when a batch has no valid depth/correspondences.

Train command:

```bash
python -m bitfeat.train --config configs/default.yaml \
  --data-root /workspace/datasets/megadepth \
  --index-dir index/train_local --device cuda
```

## Evaluation plan

1. **HPatches** — MMA at 1/3/5 px (`python -m bitfeat.eval.hpatches`).
2. **Matching recall** on MegaDepth / IMC-style splits when available.
3. Optional: IMC Phototourism / Aachen pose AUC once descriptors are competitive.

Compare against FP32 baselines of similar width and a width-matched ternary ablation.

## Non-goals (v0)

- Shipping pretrained MegaDepth weights in-repo.
- CUDA ternary kernels / fused BitGEMM (PyTorch STE path is enough to start).
- Coupling to FastGS or any other monorepo package.
