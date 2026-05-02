"""
Sanity Checks — Step E.
Adebayo et al. (2018): https://arxiv.org/abs/1810.03292

Cascading model randomization test: progressively randomize layers from
output to input. Attribution maps that stay correlated with the original
after full randomization are not truly model-dependent (FAIL — excluded
from DGS computation).

Tests: Grad-CAM, Integrated Gradients, ScoreCAM.
"""

import sys, io
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from model import create_model
from utils.preprocessing import load_image
from xai.gradcam import gradcam as compute_gradcam, resolve_layer
from xai.scorecam import scorecam as compute_scorecam
from xai.integrated_gradients import integrated_gradients, smoothgrad_ig

# ---- config ----
CHECKPOINT     = "best_model.pt"
PASS_THRESHOLD = 0.3    # fully-randomized Spearman ρ must drop below this (Adebayo et al. 2018)
IG_STEPS       = 50     # reduced steps for IG here — precision not needed, direction is
IG_SMOOTHGRAD  = True   # SmoothGrad: blurred baseline failed (ρ↑); averaging noisy IG maps reduces input-structure bias
SCORECAM_K     = 64     # top-k channels by activation magnitude; dropped from 128 (ρ=0.449 still FAIL)
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
GRADCAM_LAYER  = "layer3"   # layer being tested: "layer3" or "layer4"
METHODS        = ["gradcam"]     # confirmed final method; add "ig", "scorecam" to reproduce comparison

REAL_IMG  = "leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
SYNTH_IMG = "images/00001.png"

# Cascading order: output layer first, then work backward into the backbone
LAYER_GROUPS = [
    ("classifier",       lambda m: m.classifier),
    ("+ layer4",         lambda m: m.backbone.layer4),
    ("+ layer3",         lambda m: m.backbone.layer3),
    ("+ layer2",         lambda m: m.backbone.layer2),
    ("+ layer1  (full)", lambda m: m.backbone.layer1),
]
# ----------------


def load_model():
    model = create_model(backbone="resnet18", hidden_dim=512, dropout_rate=0.3)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(DEVICE)


def randomize(layer):
    for m in layer.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def get_map(method, model, img):
    # internal prints from gradcam/scorecam suppressed so sanity output stays readable
    with redirect_stdout(io.StringIO()):
        if method == "gradcam":
            layer = resolve_layer(model, GRADCAM_LAYER)
            cam, _ = compute_gradcam(model, img, layer=layer)   # (H, W) numpy
            mag = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze().numpy()

        elif method == "ig":
            # black baseline (baseline=None default); SmoothGrad averages over noisy copies
            if IG_SMOOTHGRAD:
                attrs, _ = smoothgrad_ig(model, img, n_steps=IG_STEPS)
            else:
                attrs, _ = integrated_gradients(model, img, n_steps=IG_STEPS)
            mag = attrs.abs().sum(0).cpu().numpy()      # already (224, 224)

        elif method == "scorecam":
            cam, _ = compute_scorecam(model, img, k=SCORECAM_K)
            mag = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze().numpy()

    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


def spearman(a, b):
    # returns 0 for constant inputs (e.g. all-zero map after ReLU wipes everything)
    a, b = a.flatten(), b.flatten()
    if a.std() < 1e-10 or b.std() < 1e-10:
        return 0.0
    return float(spearmanr(a, b).statistic)


def check_method(method, img_paths, img_labels):
    print(f"\n{'='*56}")
    print(f"  {method.upper()}")
    print(f"{'='*56}")

    method_pass = True

    for img_path, img_label in zip(img_paths, img_labels):
        print(f"\n  [{img_label}]")

        model = load_model()     # fresh model for each run
        img   = load_image(img_path)

        print("  computing original attribution...", end=" ", flush=True)
        orig = get_map(method, model, img)
        print("done.")

        print(f"\n  {'Layer group':<30}  {'ρ':>6}")
        print(f"  {'-'*40}")

        last_rho = None
        for label, get_layer in LAYER_GROUPS:
            randomize(get_layer(model))
            rnd  = get_map(method, model, img)
            rho  = spearman(orig, rnd)
            last_rho = rho
            flag = "  ← still high" if rho >= PASS_THRESHOLD else ""
            print(f"  {label:<30}  {rho:>6.4f}{flag}")

        ok = last_rho < PASS_THRESHOLD
        if not ok:
            method_pass = False
        verdict = "PASS" if ok else "FAIL"
        print(f"\n  Fully-randomized ρ = {last_rho:.4f}  →  {verdict}")
        if not ok:
            print(f"  [WARNING] {method.upper()} attributions remain correlated after full "
                  f"randomization. This method should be excluded from DGS computation — "
                  f"its maps may reflect input structure rather than model decisions.")

    return method_pass


if __name__ == "__main__":
    img_paths  = [REAL_IMG, SYNTH_IMG]
    img_labels = ["REAL", "SYNTHETIC"]

    results = {}
    for method in METHODS:
        results[method] = check_method(method, img_paths, img_labels)

    print(f"\n{'='*56}")
    print("  SUMMARY")
    print(f"{'='*56}")
    print(f"  {'Method':<12}  {'Result':<6}  Note")
    print(f"  {'-'*50}")
    for method, passed in results.items():
        note = "eligible for DGS" if passed else "EXCLUDED — not model-dependent"
        print(f"  {method.upper():<12}  {'PASS' if passed else 'FAIL':<6}  {note}")
    print()
