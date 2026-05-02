"""
Insertion / Deletion curves — Step E.3.
Petsiuk et al. (2018): https://arxiv.org/abs/1806.07421

Deletion:  progressively replace most-important pixels with blurred baseline.
           Confidence should drop fast → lower AUC = better.
Insertion: progressively reveal most-important pixels from blurred baseline.
           Confidence should rise fast → higher AUC = better.
"""

import sys, io
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from model import create_model
from utils.preprocessing import load_image
from xai.gradcam import gradcam as compute_gradcam, resolve_layer
from xai.scorecam import scorecam as compute_scorecam

# ---- config ----
CHECKPOINT  = "best_model.pt"
OUT_DIR     = Path("outputs/figures")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS     = 100    # checkpoints per curve
N_IMGS      = 5      # images per class
BLUR_SIGMA  = 10     # blurred baseline (insertion start / deletion end)
SCORECAM_K  = 128    # must match sanity-check config

# add "ig" to include SmoothGrad — slow (~1000 forward passes per image)
METHODS = ["gradcam", "scorecam", "random"]

REAL_IMGS  = sorted(Path("leftImg8bit_trainvaltest/leftImg8bit/train/aachen").glob("*.png"))[:N_IMGS]
SYNTH_IMGS = sorted(Path("images").glob("*.png"))[:N_IMGS]
# ----------------


def load_model():
    model = create_model(backbone="resnet18", hidden_dim=512, dropout_rate=0.3)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(DEVICE)


def make_blur(img_tensor):
    arr = img_tensor.numpy()
    return torch.tensor(gaussian_filter(arr, sigma=[0, BLUR_SIGMA, BLUR_SIGMA]),
                        dtype=img_tensor.dtype).to(DEVICE)


def class_score(model, x, is_synthetic):
    """Sigmoid probability for the predicted class, held fixed at original prediction."""
    with torch.no_grad():
        logit = model(x.unsqueeze(0))[0, 0]
    return torch.sigmoid(logit if is_synthetic else -logit).item()


def get_saliency(method, model, img):
    with redirect_stdout(io.StringIO()):
        if method == "gradcam":
            layer = resolve_layer(model, "layer3")
            cam, _ = compute_gradcam(model, img, layer=layer)
            sal = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze().numpy()
        elif method == "ig":
            from xai.integrated_gradients import smoothgrad_ig
            attrs, _ = smoothgrad_ig(model, img, n_steps=50)
            sal = attrs.abs().sum(0).cpu().numpy()
        elif method == "scorecam":
            cam, _ = compute_scorecam(model, img, k=SCORECAM_K)
            sal = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze().numpy()
        elif method == "random":
            sal = np.random.rand(224, 224)
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal


def run_curves(model, img_tensor, saliency, is_synthetic):
    # deletion and insertion share the same pixel ordering — run both in one loop
    x    = img_tensor.to(DEVICE)
    base = make_blur(img_tensor)
    H, W = x.shape[1], x.shape[2]
    order = np.argsort(saliency.flatten())[::-1].copy()   # most important first
    step  = max(1, H * W // N_STEPS)

    x_del = x.clone()     # deletion: full image, progressively masked
    x_ins = base.clone()  # insertion: blurred,  progressively revealed

    fracs, del_s, ins_s = [], [], []
    for i in range(N_STEPS + 1):
        fracs.append(i / N_STEPS)
        del_s.append(class_score(model, x_del, is_synthetic))
        ins_s.append(class_score(model, x_ins, is_synthetic))
        if i < N_STEPS:
            idx = order[i * step : (i + 1) * step]
            rows, cols = idx // W, idx % W
            x_del[:, rows, cols] = base[:, rows, cols]
            x_ins[:, rows, cols] = x[:, rows, cols]

    return np.array(fracs), np.array(del_s), np.array(ins_s)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)   # reproducible random baseline

    print("Loading model...")
    model = load_model()
    print(f"  device: {DEVICE}  |  {N_IMGS} images/class  |  methods: {METHODS}")

    if not REAL_IMGS or not SYNTH_IMGS:
        print("[ERROR] No images found — check REAL_IMGS / SYNTH_IMGS paths.")
        raise SystemExit(1)

    img_list = [(str(p), False) for p in REAL_IMGS] + \
               [(str(p), True)  for p in SYNTH_IMGS]

    acc = {m: {"del": [], "ins": [], "fracs": None} for m in METHODS}

    for img_path, is_synth in img_list:
        label = "SYNTH" if is_synth else "REAL "
        print(f"\n  [{label}] {Path(img_path).name}")
        img = load_image(img_path)

        for method in METHODS:
            print(f"    {method:<10}", end=" ", flush=True)
            sal = get_saliency(method, model, img)
            fracs, del_s, ins_s = run_curves(model, img, sal, is_synth)
            acc[method]["del"].append(del_s)
            acc[method]["ins"].append(ins_s)
            acc[method]["fracs"] = fracs
            del_auc = np.trapezoid(del_s, fracs)
            ins_auc = np.trapezoid(ins_s, fracs)
            print(f"del={del_auc:.3f}  ins={ins_auc:.3f}")

    # Summary
    print(f"\n{'='*52}")
    print(f"  SUMMARY (mean over {len(img_list)} images)")
    print(f"{'='*52}")
    print(f"  {'Method':<12}  {'Del AUC':>8}  {'Ins AUC':>8}  Note")
    print(f"  {'-'*46}")
    for method in METHODS:
        fracs    = acc[method]["fracs"]
        del_mean = np.mean(acc[method]["del"], axis=0)
        ins_mean = np.mean(acc[method]["ins"], axis=0)
        del_auc  = np.trapezoid(del_mean, fracs)
        ins_auc  = np.trapezoid(ins_mean, fracs)
        note = "PASS (sanity)" if method == "gradcam" else "EXCLUDED (sanity FAIL)" if method in ("ig", "scorecam") else "random baseline"
        print(f"  {method.upper():<12}  {del_auc:>8.4f}  {ins_auc:>8.4f}  {note}")
        acc[method]["del_mean"] = del_mean
        acc[method]["ins_mean"] = ins_mean

    # Plot
    colors = {"gradcam": "steelblue", "ig": "forestgreen",
              "scorecam": "darkorange", "random": "gray"}
    lstyle = {"gradcam": "-", "ig": "--", "scorecam": "-.", "random": ":"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for method in METHODS:
        fracs = acc[method]["fracs"]
        c, ls = colors.get(method, "black"), lstyle.get(method, "-")
        ax1.plot(fracs, acc[method]["ins_mean"], color=c, ls=ls, lw=2, label=method.upper())
        ax2.plot(fracs, acc[method]["del_mean"], color=c, ls=ls, lw=2, label=method.upper())

    for ax, title, xlabel in [
        (ax1, "Insertion  (higher AUC = better)", "Fraction of pixels revealed"),
        (ax2, "Deletion   (lower  AUC = better)", "Fraction of pixels removed"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Predicted-class probability")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    plt.suptitle("Insertion / Deletion — Grad-CAM layer3 vs baselines", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "insertion_deletion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  saved → {out}")
    print("\nDone.")
