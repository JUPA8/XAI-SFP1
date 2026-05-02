# Preprocessing constants and transforms matching the Step 1 training run exactly.
# Source of truth: config.json from the HPC training run (backbone=resnet18, preprocess_mode=crop).

import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import torch

# Confirmed from config.py DataConfig (no CLI override available for normalization)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# Confirmed from config.json: preprocess_mode=crop, crop_size=224
CROP_SIZE = 224


def preprocess(img: Image.Image) -> torch.Tensor:
    """No augmentations — matches is_training=False in dataset.py."""
    w, h = img.size
    if w >= CROP_SIZE and h >= CROP_SIZE:
        img = TF.center_crop(img, [CROP_SIZE, CROP_SIZE])
    else:
        img = TF.resize(img, [CROP_SIZE, CROP_SIZE])

    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean=list(MEAN), std=list(STD))
    return tensor


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return preprocess(img)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse normalization for visualization."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
