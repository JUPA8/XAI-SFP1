# Step 1 — Train a Domain Discriminator

Trains a binary image classifier (ResNet18/ConvNeXt/etc.) to discriminate between two image datasets (e.g. real vs. synthetic). Uses ImageNet-pretrained backbones, weighted random sampling, cosine LR scheduling, and early stopping.

## Requirements

Install dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision tqdm numpy pillow
```

## Data layout

Each dataset must be a flat directory of images (`.jpg`, `.jpeg`, `.png`, `.bmp`). Subdirectories are not traversed. Example:

```
data/
  dataset_a/   ← label 0 ("real")
    img001.jpg
    img002.jpg
    ...
  dataset_b/   ← label 1 ("synthetic")
    img001.jpg
    ...
```

## Running training

```bash
python train.py \
  --cityscapes-path /path/to/dataset_a \
  --gta-path        /path/to/dataset_b \
  --backbone        resnet18 \
  --output-dir      outputs/my_run
```

The script:
1. Discovers images in both directories.
2. Creates reproducible train / val / test splits (800 / 100 / 100 per dataset by default).
3. Trains the model and saves checkpoints to `<output-dir>/checkpoints/`.
4. Evaluates the best checkpoint on the test set and writes `test_results.json`.

## All arguments

| Argument | Default | Description |
|---|---|---|
| `--cityscapes-path` | — | Path to dataset A (label 0) |
| `--gta-path` | — | Path to dataset B (label 1) |
| `--backbone` | `resnet18` | Architecture: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32` |
| `--preprocess-mode` | `crop` | How to standardise image size: `crop` (random/center crop), `resize` (stretch), `pad` (pad to square + crop if larger) |
| `--crop-size` | `224` | Crop size (used when `--preprocess-mode crop`) |
| `--resize-size` | `384` | Target size (used when `--preprocess-mode resize`) |
| `--pad-size` | `200` | Target size (used when `--preprocess-mode pad`) |
| `--batch-size` | `32` | Training batch size |
| `--epochs` | `50` | Maximum number of epochs |
| `--lr` | `1e-4` | Base learning rate (backbone gets 0.1×) |
| `--patience` | `7` | Early-stopping patience (epochs without val-loss improvement) |
| `--device` | auto | `cuda` or `cpu` |
| `--no-amp` | off | Disable automatic mixed precision (AMP) |
| `--output-dir` | `outputs` | Directory for checkpoints and result files |
| `--seed` | `42` | Random seed for reproducibility |

## Outputs

```
<output-dir>/
  config.json           — run configuration
  splits.json           — train/val/test file paths and labels
  test_results.json     — accuracy, F1, confusion matrix on test set
  checkpoints/
    best_model.pt       — best checkpoint (lowest val loss)
    checkpoint_epoch_N.pt
    training_history.json
```

## Examples

**ConvNeXt-Tiny with padding preprocessing:**
```bash
python train.py \
  --cityscapes-path data/real \
  --gta-path        data/synthetic \
  --backbone        convnext_tiny \
  --preprocess-mode pad \
  --pad-size        256 \
  --batch-size      64 \
  --epochs          100 \
  --output-dir      outputs/convnext_pad
```

**ResNet18 on CPU (no AMP):**
```bash
python train.py \
  --cityscapes-path data/real \
  --gta-path        data/synthetic \
  --backbone        resnet18 \
  --device          cpu \
  --no-amp \
  --output-dir      outputs/resnet18_cpu
```
