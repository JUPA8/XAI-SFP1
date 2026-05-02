# xAI Domain Gap Scoring

A binary image classifier (ResNet18) is trained to distinguish Cityscapes (real) from GTA5 (synthetic) driving images. The xAI part extracts attribution maps from that classifier and uses them to compute a Domain Gap Score — a measure of where and how much the two domains differ according to the model's decisions.

**Research question:** Can explainable AI be used to compute a semantically meaningful domain gap score between real and synthetic image datasets?

---

## Datasets

**Cityscapes** (real, label 0): `leftImg8bit_trainvaltest/leftImg8bit/{train,val,test}/{city}/`  
**GTA5 / Playing for Data** (synthetic, label 1): `images/` — flat directory, ~2500 files

Splits: 800 train / 100 val / 100 test per dataset, random shuffle. The model was trained on an HPC cluster. `best_model.pt` is in the project root. Test accuracy: 100% (200/200).

---

## xAI methods tested

Three attribution methods were implemented and evaluated with the Adebayo et al. (2018) sanity check. The test progressively randomizes model layers from output to input and measures whether the attribution map changes. A method passes if the Spearman ρ drops below 0.3 after full randomization.

| Method | REAL ρ | SYNTH ρ | Result |
| --- | --- | --- | --- |
| Grad-CAM layer3 | −0.43 | −0.11 | **PASS — selected** |
| Integrated Gradients + SmoothGrad | 0.19 | 0.47 | fail |
| ScoreCAM top-128 channels | 0.11 | 0.45 | fail |

IG and ScoreCAM fail on synthetic images. GTA5's regular computer-rendered textures create spatially consistent feature maps even after full model randomization, so the attribution maps reflect input structure rather than model decisions. Grad-CAM avoids this because it uses gradient flow rather than activation patterns directly.

**Selected method: Grad-CAM at `model.backbone.layer3[-1]` (14×14 resolution, 256 channels).**

---

## Environment

```bash
conda activate ki
```

All scripts must be run from the project root. On macOS, PyTorch and the system OpenMP library conflict, so every command needs the prefix:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/ki/bin/python <script>
```

---

## Key scripts

**Grad-CAM heatmaps:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/ki/bin/python xai/gradcam.py
```
Saves `outputs/figures/gradcam_real_layer3.png` and `gradcam_synthetic_layer3.png`.

**Sanity checks:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/ki/bin/python xai/sanity_checks.py
```
Edit the `METHODS` list at the top of the file to choose what to test (`"gradcam"`, `"ig"`, `"scorecam"`, or any combination).

**Insertion / deletion curves:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/ki/bin/python xai/insertion_deletion.py
```
Saves `outputs/figures/insertion_deletion.png`. Grad-CAM insertion AUC = 0.629 vs random 0.562.

**Qualitative grid + mean attribution maps:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/ki/bin/python xai/gradcam_qualitative.py
```
Saves a 4×5 grid (10 real, 10 synthetic) and mean attribution maps over 20 images each.

---

## File structure

```
step1/
  best_model.pt               trained checkpoint
  config.json                 training config from HPC run
  experiment_log.md           full research log — steps A through E

  xai/
    gradcam.py                Grad-CAM implementation and smoke test
    integrated_gradients.py   IG with trapezoidal integration and completeness check
    scorecam.py               ScoreCAM — gradient-free, 512 forward passes per image
    sanity_checks.py          Adebayo et al. cascading randomization test
    insertion_deletion.py     Petsiuk et al. insertion/deletion curves
    gradcam_qualitative.py    qualitative grid and mean attribution maps

  utils/
    preprocessing.py          load_image(), normalization constants — shared across all xAI scripts

  outputs/
    figures/                  saved heatmaps and plots

  scoring/
    (Step F onward — DGS computation, not yet started)
```

---

## Status

Steps A–E complete. Grad-CAM layer3 is confirmed as the attribution method for DGS.  
Next: Step F — extract Grad-CAM maps over the full test set and compute the DGS.
