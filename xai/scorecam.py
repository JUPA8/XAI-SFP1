"""
ScoreCAM — Step D.
Wang et al. (2020): https://arxiv.org/abs/1910.01279

Gradient-free: uses model confidence on masked inputs as channel weights
instead of backpropagated gradients. Good cross-check for Grad-CAM.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F

from model import create_model
from utils.preprocessing import load_image, denormalize, MEAN, STD
from xai.gradcam import gradcam as compute_gradcam

# ---- config ----
CHECKPOINT = "best_model.pt"
REAL_IMG   = "leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
SYNTH_IMG  = "images/00001.png"
OUT_DIR    = Path("outputs/figures")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------


def load_model():
    model = create_model(backbone="resnet18", hidden_dim=512, dropout_rate=0.3)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(DEVICE)


def make_baseline(img_tensor):
    """Black image baseline in normalized space — same reference point as IG."""
    black = torch.tensor([(0.0 - m) / s for m, s in zip(MEAN, STD)]).view(3, 1, 1)
    return black.expand_as(img_tensor).clone()


def scorecam(model, img_tensor, k=512):
    """
    ScoreCAM: for each channel in the target layer, upsample and normalize its
    activation map, mask the input with it, run a forward pass, and use the
    resulting class score as the channel weight. (Wang et al. 2020, eq. 6-7)

    Gradient-free — avoids gradient saturation issues.
    """
    activations = {}
    target = model.backbone.layer4[-1]  # same layer as Grad-CAM

    h = target.register_forward_hook(
        lambda m, inp, out: activations.update({"feat": out.detach()})
    )

    x    = img_tensor.to(DEVICE)
    base = make_baseline(img_tensor).to(DEVICE)

    with torch.no_grad():
        logit = model(x.unsqueeze(0))
    h.remove()

    feat = activations["feat"].squeeze(0)   # (512, 7, 7)
    C    = feat.shape[0]
    H, W = x.shape[1], x.shape[2]

    # Use predicted-class score direction — consistent with Grad-CAM convention
    score_sign = 1.0 if logit.item() > 0 else -1.0

    # Top-k channel selection: rank by mean activation magnitude.
    # Low-activation channels have near-random spatial structure that inflates
    # input-structure correlation — skipping them reduces sanity-check bias.
    channel_mag = feat.abs().mean(dim=(1, 2))          # (C,)
    k_actual    = min(k, C)
    top_ch      = torch.argsort(channel_mag, descending=True)[:k_actual]

    print(f"  logit: {logit.item():.4f}  →  predicted: {'synthetic' if logit.item() > 0 else 'real'}")
    print(f"  scoring {k_actual}/{C} channels (top-{k_actual} by activation)...", end="", flush=True)

    weights = torch.zeros(C)

    with torch.no_grad():
        for c in top_ch:
            # Upsample channel c to input size
            mask = F.interpolate(
                feat[c].unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False
            ).squeeze()   # (H, W)

            # Normalize channel map to [0, 1] so it acts as a soft mask
            lo, hi = mask.min(), mask.max()
            mask = (mask - lo) / (hi - lo) if hi > lo else torch.zeros_like(mask)

            # Masked input: baseline + mask * (input - baseline) (Wang et al. 2020 eq. 6)
            x_masked = base + mask.unsqueeze(0) * (x - base)

            weights[c] = score_sign * model(x_masked.unsqueeze(0))[0, 0]

    print(" done.")

    # Weighted sum of feature maps + ReLU (Wang et al. 2020 eq. 7)
    cam = (weights.to(DEVICE)[:, None, None] * feat).sum(dim=0)
    cam = F.relu(cam).cpu().numpy()
    return cam, logit.item()


def make_overlay(img_tensor, cam):
    rgb = denormalize(img_tensor).permute(1, 2, 0).numpy()
    H, W = rgb.shape[:2]

    cam_up = F.interpolate(
        torch.tensor(cam).unsqueeze(0).unsqueeze(0),
        size=(H, W), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    if cam_up.max() > 0:
        cam_up = cam_up / cam_up.max()

    heatmap = cm.jet(cam_up)[..., :3]
    blended = np.clip(0.5 * rgb + 0.5 * heatmap, 0, 1)
    return blended, cam_up


def cosine_sim(a, b):
    a, b = a.flatten().astype(np.float32), b.flatten().astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def run(label, img_path, model, out_path, k=512):
    print(f"\n[{label}] {img_path}")
    img = load_image(img_path)

    # ScoreCAM
    scam, _   = scorecam(model, img, k=k)
    print(f"  ScoreCAM — min: {scam.min():.4f}  max: {scam.max():.4f}  mean: {scam.mean():.4f}")

    # Grad-CAM on same image for cosine similarity comparison
    gcam, _ = compute_gradcam(model, img)

    # Both maps are (7, 7) before upsampling — compare at that resolution
    sim = cosine_sim(scam, gcam)
    print(f"  cosine similarity (ScoreCAM vs Grad-CAM, 7x7): {sim:.4f}")

    blended, _ = make_overlay(img, scam)
    plt.figure(figsize=(5, 5))
    plt.imshow(blended)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    TOP_K = 128   # top-k channels by activation magnitude; change to 64 if sanity check fails

    print("Loading model...")
    model = load_model()
    print(f"  device: {DEVICE}  |  top-k channels: {TOP_K}")

    run("REAL",      REAL_IMG,  model, OUT_DIR / "scorecam_topk_real.png",      k=TOP_K)
    run("SYNTHETIC", SYNTH_IMG, model, OUT_DIR / "scorecam_topk_synthetic.png", k=TOP_K)

    print("\nDone.")
