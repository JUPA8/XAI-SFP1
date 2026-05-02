"""
Grad-CAM layer3 qualitative validation.

1. 4×5 grid: 10 real (rows 0-1) + 10 synthetic (rows 2-3)
   → outputs/figures/gradcam_layer3_grid.png

2. Mean Grad-CAM maps over 20 real and 20 synthetic images
   → outputs/figures/mean_gradcam_real.png
   → outputs/figures/mean_gradcam_synthetic.png
"""

import sys, io
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import torch
import torch.nn.functional as F

from model import create_model
from utils.preprocessing import load_image, denormalize
from xai.gradcam import gradcam as compute_gradcam, resolve_layer

# ---- config ----
CHECKPOINT = "best_model.pt"
OUT_DIR    = Path("outputs/figures")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_GRID     = 10    # images per class for grid
N_MEAN     = 20    # images per class for mean map

REAL_DIR   = Path("leftImg8bit_trainvaltest/leftImg8bit/train/aachen")
SYNTH_DIR  = Path("images")
# ----------------


def load_model():
    model = create_model(backbone="resnet18", hidden_dim=512, dropout_rate=0.3)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(DEVICE)


def get_cam(model, img):
    with redirect_stdout(io.StringIO()):
        layer = resolve_layer(model, "layer3")
        raw, logit = compute_gradcam(model, img, layer=layer)
    cam = F.interpolate(
        torch.tensor(raw).unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, logit


def blend(img_tensor, cam):
    rgb = denormalize(img_tensor).permute(1, 2, 0).numpy()
    return np.clip(0.5 * rgb + 0.5 * mpl_cm.jet(cam)[..., :3], 0, 1)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model()
    print(f"  device: {DEVICE}")

    n_load     = max(N_GRID, N_MEAN)
    real_paths = sorted(REAL_DIR.glob("*.png"))[:n_load]
    synth_paths = sorted(SYNTH_DIR.glob("*.png"))[:n_load]

    if len(real_paths) < N_MEAN or len(synth_paths) < N_MEAN:
        print(f"[WARN] Found {len(real_paths)} real, {len(synth_paths)} synthetic — need {N_MEAN} each.")

    # ---- 1. Qualitative grid ----
    print(f"\nBuilding grid ({N_GRID} real + {N_GRID} synthetic)...")
    paths  = list(real_paths[:N_GRID])  + list(synth_paths[:N_GRID])
    labels = ["real"]  * N_GRID + ["synth"] * N_GRID
    n_cols = 5
    n_rows = (2 * N_GRID) // n_cols   # 4 rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    for idx, (path, lbl) in enumerate(zip(paths, labels)):
        row, col = idx // n_cols, idx % n_cols
        img = load_image(str(path))
        cam, logit = get_cam(model, img)
        ax = axes[row][col]
        ax.imshow(blend(img, cam))
        ax.set_title(f"{lbl}  {logit:+.2f}", fontsize=7)
        ax.axis("off")
        # blue border = real, red border = synthetic
        color = "royalblue" if lbl == "real" else "tomato"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

    plt.suptitle("Grad-CAM layer3   (blue = real, red = synthetic)", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "gradcam_layer3_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")

    # ---- 2. Mean Grad-CAM maps ----
    for class_paths, class_label, out_name in [
        (real_paths,  "real",      "mean_gradcam_real"),
        (synth_paths, "synthetic", "mean_gradcam_synthetic"),
    ]:
        print(f"\nComputing mean Grad-CAM over {N_MEAN} {class_label} images...")
        cams     = []
        imgs_rgb = []
        for i, path in enumerate(class_paths[:N_MEAN]):
            img = load_image(str(path))
            cam, _ = get_cam(model, img)
            cams.append(cam)
            imgs_rgb.append(denormalize(img).permute(1, 2, 0).numpy())
            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{N_MEAN}...")

        mean_cam = np.mean(cams, axis=0)
        mean_img = np.mean(imgs_rgb, axis=0)

        mean_cam_norm = mean_cam / mean_cam.max() if mean_cam.max() > 0 else mean_cam
        blended = np.clip(0.5 * mean_img + 0.5 * mpl_cm.jet(mean_cam_norm)[..., :3], 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(blended)
        axes[0].set_title(f"Mean Grad-CAM overlay\n({N_MEAN} {class_label} images)", fontsize=10)
        axes[0].axis("off")

        im = axes[1].imshow(mean_cam_norm, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Mean attention (jet, normalized)", fontsize=10)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        out = OUT_DIR / f"{out_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved → {out}")

    print("\nDone.")
