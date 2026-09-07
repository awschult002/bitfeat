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

## Training plan

1. **Smoke**: synthetic pairs (`SyntheticPairDataset`) — already wired via `--smoke`.
2. **Real**: MegaDepth (or similar) pairs with depth + relative pose.
3. Losses to flesh out in `losses/matching.py`:
   - Reprojection / heatmap supervision from warped depth.
   - Dual-softmax or circle loss on corresponded descriptors.
   - Optional peakiness / reliability terms.

## Evaluation plan

Primary outdoor / viewpoint benchmarks:

1. **HPatches** — Mean Matching Accuracy (MMA) at multiple pixel thresholds; also repeatability / matching score where useful.
2. **Matching recall** — fraction of GT correspondences recovered after mutual nearest-neighbor matching (MegaDepth / IMC-style splits as available).
3. Optional: IMC Phototourism / Aachen for pose AUC once descriptors are competitive.

Compare against FP32 baselines of similar width and against a width-matched ternary ablation (same architecture, FP vs BitConv) to isolate the 1.58-bit cost.

## Non-goals (v0 scaffold)

- Shipping pretrained MegaDepth weights in this commit.
- CUDA ternary kernels / fused BitGEMM (PyTorch STE path is enough to start).
- Coupling to FastGS or any other monorepo package.
