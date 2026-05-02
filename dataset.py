"""
Dataset module for loading and preprocessing Cityscapes and GTA images.

This module handles:
- Image discovery and filtering
- Train/val/test splitting with proper exclusion
- Random cropping for variable-sized images
- Data augmentation
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import Config, DataConfig, TrainingConfig


class DomainDataset(Dataset):
    """
    Dataset for real vs synthetic image classification.
    
    Handles images of variable sizes by applying random crops during training
    and center crops during validation/testing.
    
    Attributes:
        image_paths: List of (path, label) tuples
        transform: Torchvision transforms to apply
        is_training: Whether this is a training dataset (enables random augmentation)
        crop_size: Size of the crop to extract
    """
    
    def __init__(
        self,
        image_paths: List[Tuple[Path, int]],
        crop_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        is_training: bool = True,
        num_crops_per_image: int = 1,
        preprocess_mode: str = "crop",
        resize_size: int = 384,
        pad_size: int = 200,
        pad_fill: Tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of (image_path, label) tuples. Label 0=real, 1=synthetic.
            crop_size: Size of random/center crop (used when preprocess_mode="crop").
            mean: Normalization mean.
            std: Normalization std.
            is_training: If True, apply random augmentations and random crops.
            num_crops_per_image: Number of crops to extract per image (only for training).
            preprocess_mode: "crop", "resize", or "pad". Determines preprocessing strategy.
            resize_size: Target size for resize mode.
            pad_size: Target size for pad mode.
            pad_fill: RGB values for padding (default white: 255, 255, 255).
        """
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.is_training = is_training
        self.num_crops_per_image = num_crops_per_image if is_training else 1
        self.preprocess_mode = preprocess_mode
        self.resize_size = resize_size
        self.pad_size = pad_size
        self.pad_fill = pad_fill
        
        # Pre-compute the effective dataset size
        self._effective_length = len(image_paths) * self.num_crops_per_image
        
        # Build transforms
        self.normalize = T.Normalize(mean=mean, std=std)
        self.to_tensor = T.ToTensor()
        
    def __len__(self) -> int:
        return self._effective_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Map effective index to actual image index
        actual_idx = idx // self.num_crops_per_image
        path, label = self.image_paths[actual_idx]
        
        # Load image
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Apply preprocessing based on mode
        if self.preprocess_mode == "resize":
            # Resize mode: resize to target size then center crop to ensure exact dimensions
            image = TF.resize(image, [self.resize_size, self.resize_size])
        elif self.preprocess_mode == "pad":
            # Pad mode: pad image to target size with white pixels
            w, h = image.size
            # Calculate padding needed
            pad_w = max(0, self.pad_size - w)
            pad_h = max(0, self.pad_size - h)
            # Pad symmetrically (left, top, right, bottom)
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            image = TF.pad(image, padding, fill=self.pad_fill)
            # If image was larger than pad_size, center crop it
            if w > self.pad_size or h > self.pad_size:
                image = TF.center_crop(image, [self.pad_size, self.pad_size])
        else:
            # Crop mode: extract crops from original images
            w, h = image.size
            if self.is_training:
                # Random crop
                if w >= self.crop_size and h >= self.crop_size:
                    i = random.randint(0, h - self.crop_size)
                    j = random.randint(0, w - self.crop_size)
                    image = TF.crop(image, i, j, self.crop_size, self.crop_size)
                else:
                    # If image is too small, resize it
                    image = TF.resize(image, [self.crop_size, self.crop_size])
            else:
                # Center crop for validation/test
                if w >= self.crop_size and h >= self.crop_size:
                    image = TF.center_crop(image, [self.crop_size, self.crop_size])
                else:
                    image = TF.resize(image, [self.crop_size, self.crop_size])
        
        # Apply augmentations (training only)
        if self.is_training:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
            
            # Random rotation (±10 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle)
            
            # Random affine (scale 0.95-1.05)
            if random.random() > 0.5:
                scale = random.uniform(0.95, 1.05)
                image = TF.affine(image, angle=0, translate=[0, 0], scale=scale, shear=0)
            
            # Random perspective distortion
            if random.random() > 0.5:
                w, h = image.size
                startpoints = [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]
                # Apply random distortion (up to 20% of image size)
                distortion = 0.2
                endpoints = [
                    [random.uniform(0, w*distortion), random.uniform(0, h*distortion)],
                    [w-1-random.uniform(0, w*distortion), random.uniform(0, h*distortion)],
                    [w-1-random.uniform(0, w*distortion), h-1-random.uniform(0, h*distortion)],
                    [random.uniform(0, w*distortion), h-1-random.uniform(0, h*distortion)]
                ]
                image = TF.perspective(image, startpoints, endpoints)
            
            # Stronger color jitter
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))
            
            # Gaussian blur
            if random.random() > 0.5:
                kernel_size = random.choice([3, 5])
                image = TF.gaussian_blur(image, kernel_size)
        
        # Convert to tensor and normalize
        image = self.to_tensor(image)
        
        # Random erasing (applied after converting to tensor, training only)
        if self.is_training and random.random() > 0.5:
            erase_transform = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))
            image = erase_transform(image)
        
        image = self.normalize(image)
        
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "path": str(path),
        }


def discover_images(
    directory: Path,
    config: DataConfig,
) -> List[Path]:
    """
    Discover all valid images in a directory.
    
    Args:
        directory: Directory to search.
        config: Data configuration with exclude_patterns and min_image_size.
        
    Returns:
        List of valid image paths.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    
    for ext in valid_extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    
    # Filter out excluded patterns
    filtered = []
    for img_path in images:
        name = img_path.name.lower()
        if not any(pattern.lower() in name for pattern in config.exclude_patterns):
            # Optionally check image size
            if config.min_image_size > 0:
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        if min(w, h) >= config.min_image_size:
                            filtered.append(img_path)
                except Exception:
                    continue
            else:
                filtered.append(img_path)
    
    return sorted(filtered)


def create_splits(
    cityscapes_images: List[Path],
    gta_images: List[Path],
    config: DataConfig,
) -> Dict[str, Dict[str, List]]:
    """
    Create train/val/test splits ensuring proper stratification.
    
    Args:
        cityscapes_images: List of Cityscapes image paths.
        gta_images: List of GTA image paths.
        config: Data configuration with split ratios and seed.
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing paths and labels.
    """
    random.seed(config.seed)
    
    # Label: 0 = real (Cityscapes), 1 = synthetic (GTA)
    cityscapes_labeled = [(p, 0) for p in cityscapes_images]
    gta_labeled = [(p, 1) for p in gta_images]
    
    # Shuffle each dataset separately
    random.shuffle(cityscapes_labeled)
    random.shuffle(gta_labeled)
    
    # Take fixed number of images from config
    train_count = config.train_count
    val_count = config.val_count
    test_count = config.test_count
    total_count = train_count + val_count + test_count
    
    # Ensure we have enough data
    if len(cityscapes_labeled) < total_count:
        print(f"Warning: Only {len(cityscapes_labeled)} Cityscapes images available, need {total_count}")
    if len(gta_labeled) < total_count:
        print(f"Warning: Only {len(gta_labeled)} GTA images available, need {total_count}")

    # Truncate to the needed amount
    cityscapes_labeled = cityscapes_labeled[:total_count]
    gta_labeled = gta_labeled[:total_count]

    # Calculate split indices
    def split_list_fixed(data: List):
        train = data[:train_count]
        val = data[train_count : train_count + val_count]
        test = data[train_count + val_count : total_count]
        return train, val, test
    
    cs_train, cs_val, cs_test = split_list_fixed(cityscapes_labeled)
    gta_train, gta_val, gta_test = split_list_fixed(gta_labeled)
    
    # Combine and shuffle each split
    train_data = cs_train + gta_train
    val_data = cs_val + gta_val
    test_data = cs_test + gta_test
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Convert to the expected format
    def to_dict(data):
        paths = [path for path, label in data]
        labels = [label for path, label in data]
        return {"paths": paths, "labels": labels}
    
    return {
        "train": to_dict(train_data),
        "val": to_dict(val_data),
        "test": to_dict(test_data),
    }


def create_dataloaders(
    splits: Dict[str, Dict[str, List]],
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from pre-computed splits.
    
    Args:
        splits: Dictionary with 'train', 'val', 'test' keys containing paths and labels.
        config: Full configuration object.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create datasets
    # Convert paths and labels to list of tuples as expected by DomainDataset
    train_image_paths = list(zip(splits["train"]["paths"], splits["train"]["labels"]))
    val_image_paths = list(zip(splits["val"]["paths"], splits["val"]["labels"]))
    test_image_paths = list(zip(splits["test"]["paths"], splits["test"]["labels"]))
    
    train_dataset = DomainDataset(
        image_paths=train_image_paths,
        crop_size=config.data.crop_size,
        mean=config.data.normalize_mean,
        std=config.data.normalize_std,
        is_training=True,
        preprocess_mode=config.data.preprocess_mode,
        resize_size=config.data.resize_size,
        pad_size=config.data.pad_size,
        pad_fill=config.data.pad_fill,
    )
    
    val_dataset = DomainDataset(
        image_paths=val_image_paths,
        crop_size=config.data.val_crop_size,
        mean=config.data.normalize_mean,
        std=config.data.normalize_std,
        is_training=False,
        preprocess_mode=config.data.preprocess_mode,
        resize_size=config.data.resize_size,
        pad_size=config.data.pad_size,
        pad_fill=config.data.pad_fill,
    )
    
    test_dataset = DomainDataset(
        image_paths=test_image_paths,
        crop_size=config.data.val_crop_size,
        mean=config.data.normalize_mean,
        std=config.data.normalize_std,
        is_training=False,
        preprocess_mode=config.data.preprocess_mode,
        resize_size=config.data.resize_size,
        pad_size=config.data.pad_size,
        pad_fill=config.data.pad_fill,
    )
    
    # Calculate weights for WeightedRandomSampler to handle class imbalance
    targets = [label for _, label in train_image_paths]
    class_counts = {}
    for t in targets:
        class_counts[t] = class_counts.get(t, 0) + 1
    
    # Calculate weight per class (inverse frequency)
    total_samples = len(targets)
    weights_per_class = {cls: total_samples / count for cls, count in class_counts.items()}
    
    # Assign weight to each sample
    sample_weights = [weights_per_class[t] for t in targets]
    
    # If using multiple crops per image during training, expand weights
    if train_dataset.num_crops_per_image > 1:
        expanded_weights = []
        for w in sample_weights:
            expanded_weights.extend([w] * train_dataset.num_crops_per_image)
        sample_weights = expanded_weights
    
    # Create sampler
    # Note: weights don't need to sum to 1, sampler handles normalization
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,  # Mutually exclusive with sampler
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    
    return train_loader, val_loader, test_loader
