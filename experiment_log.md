# Experiment Log — xAI Domain Gap Score

---

## Step A — Code Review
**Date:** 2026-05-02

### Pipeline confirmed
- Backbone: ResNet18 (pretrained ImageNet)
- Preprocessing at inference: center crop to 224×224 → ToTensor → Normalize
- Normalization: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- Labels: 0 = real (Cityscapes), 1 = synthetic (GTA5)
- Split: 800 train / 100 val / 100 test per dataset, random shuffle (no city-based split)

### Confirmed Grad-CAM target layer
`model.backbone.layer4[-1]` — last BasicBlock of ResNet18 layer4.
Accessed via `get_target_layers()` in model.py.

### Test set performance
- Accuracy: 1.0000 (200/200)
- Real accuracy: 1.0000 (100/100)
- Synthetic accuracy: 1.0000 (100/100)
- F1: 1.0000
- Loss: 0.5199

### Issues flagged
1. **Splits use HPC paths** — splits.json paths point to `/cluster/...` and cannot be used directly for local inference. Local data paths must be remapped:
   - Cityscapes: `leftImg8bit_trainvaltest/leftImg8bit/{train,val,test}/{city}/`
   - GTA5: `images/` (flat, 2500 files)
2. **Cityscapes split is not city-based** — random shuffle across all cities. Documented design decision; to note in methodology.
3. **RandomErasing applied before Normalize in training** — non-standard order but consistent (model saw this during training). Inference has no erasing, so no impact on xAI.

### Files confirmed
- `best_model.pt` — confirmed present
- `config.json` — confirmed, matches the above
- `utils/preprocessing.py` — created this step with confirmed constants

---

## Step B — Grad-CAM Smoke Test
**Date:** 2026-05-02

### Implementation
- File: `xai/gradcam.py`
- Target layer: `model.backbone.layer4[-1]` (last BasicBlock of ResNet18 layer4)
- Backpropagates through predicted-class score (not always synthetic logit) to avoid ReLU collapse on real images

### Bug found and fixed
Initial version backpropped through raw logit (synthetic direction) for both images. For the real image (logit << 0), real-contributing features have negative gradients; after ReLU the CAM was near-zero. Fixed by using `-logit` for real predictions, `logit` for synthetic predictions.

### Results

| Image | Logit | Predicted | Grad mean | CAM max (post-ReLU) |
|---|---|---|---|---|
| Real (aachen_000000) | -0.5053 | real | 1.84e-4 | 0.0270 |
| Synthetic (00001.png) | +0.2972 | synthetic | 1.80e-4 | 0.0278 |

Gradients confirmed flowing. Feature activations healthy (mean ~0.9–1.0, max ~16–19).

### Observations
- Both heatmaps are spatially concentrated and different from each other — correct behaviour
- Synthetic: strong activation upper-left (sky/lighting texture typical of GTA5 rendering)
- Real: two hot spots at lower-left and lower-right (road surface and building texture)
- Both logits are weak in magnitude, suggesting these specific crop patches are near the decision boundary — expected for single-image smoke test; full extraction averages over hundreds of images

### Status
PASS — heatmaps reasonable. Ready for Step C (Integrated Gradients).
