"""
Grad-CAM smoke test — one real image vs one synthetic image.
Selvaraju et al. (2017): https://arxiv.org/abs/1610.02391
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from PIL import Image

from model import create_model
from utils.preprocessing import load_image, denormalize

# ---- paths and settings ----
CHECKPOINT  = "best_model.pt"
REAL_IMG    = "leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
SYNTH_IMG   = "images/00001.png"
OUT_DIR     = Path("outputs/figures")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------


def load_model():
    model = create_model(backbone="resnet18", hidden_dim=512, dropout_rate=0.3)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(DEVICE)


def gradcam(model, img_tensor):
    """
    Grad-CAM: global-average-pooled gradients weight the feature maps
    at the target layer. (Selvaraju et al. 2017, eq. 1-3)

    Backprops through the predicted class score (not always the synthetic
    logit) so that real-image CAMs are not killed by the ReLU.

    Returns (cam, logit_val) where cam is a 2D numpy array (pre-resize).
    """
    activations = {}
    gradients   = {}

    target = model.backbone.layer4[-1]  # last BasicBlock of ResNet18 layer4

    h_fwd = target.register_forward_hook(
        lambda m, inp, out: activations.update({"feat": out.clone()})
    )
    h_bwd = target.register_full_backward_hook(
        lambda m, gi, go: gradients.update({"grad": go[0].clone()})
    )

    x = img_tensor.unsqueeze(0).to(DEVICE)
    logit = model(x)  # (1, 1) — positive = synthetic, negative = real

    # Use predicted-class score so gradients are positive for the winning class.
    # If logit > 0 → synthetic: backprop through logit.
    # If logit < 0 → real:      backprop through -logit (real-class confidence).
    score = logit if logit.item() > 0 else -logit

    model.zero_grad()
    score.backward()

    h_fwd.remove()
    h_bwd.remove()

    feat = activations["feat"].squeeze(0)   # (512, 7, 7)
    grad = gradients["grad"].squeeze(0)     # (512, 7, 7)

    # Diagnostics — confirm gradients are non-zero
    print(f"  logit: {logit.item():.4f}  →  predicted: {'synthetic' if logit.item() > 0 else 'real'}")
    print(f"  grad  |mean|: {grad.abs().mean():.6f}   max: {grad.abs().max():.6f}")
    print(f"  feat  |mean|: {feat.abs().mean():.4f}    max: {feat.abs().max():.4f}")

    # alpha_k = mean gradient per channel (Selvaraju et al. 2017 eq. 1)
    weights = grad.mean(dim=(1, 2))                        # (512,)
    cam_raw = (weights[:, None, None] * feat).sum(dim=0)  # (7, 7)
    print(f"  CAM pre-ReLU — min: {cam_raw.min():.4f}  max: {cam_raw.max():.4f}")

    cam = F.relu(cam_raw).detach().cpu().numpy()
    return cam, logit.item()


def overlay(img_tensor, cam):
    """Resize CAM to image size and blend with original image for display."""
    rgb = denormalize(img_tensor).permute(1, 2, 0).numpy()   # (H, W, 3)
    h, w = rgb.shape[:2]

    cam_up = np.array(
        Image.fromarray(cam).resize((w, h), Image.BILINEAR)
    )
    if cam_up.max() > 0:
        cam_up = cam_up / cam_up.max()

    heatmap = cm.jet(cam_up)[..., :3]              # (H, W, 3)
    blended = np.clip(0.5 * rgb + 0.5 * heatmap, 0, 1)
    return blended, cam_up


def run(label, img_path, model, out_path):
    print(f"\n[{label}] {img_path}")
    img = load_image(img_path)
    cam, _ = gradcam(model, img)

    print(f"  raw CAM (post-ReLU) — min: {cam.min():.6f}  max: {cam.max():.6f}  mean: {cam.mean():.6f}")

    blended, _ = overlay(img, cam)

    plt.figure(figsize=(5, 5))
    plt.imshow(blended)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model()
    print(f"  device: {DEVICE}")

    run("REAL",      REAL_IMG,  model, OUT_DIR / "gradcam_real.png")
    run("SYNTHETIC", SYNTH_IMG, model, OUT_DIR / "gradcam_synthetic.png")

    print("\nDone.")
