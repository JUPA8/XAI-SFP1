"""
Configuration module for the domain discriminator training.

This module contains all hyperparameters and settings for training
a discriminator to classify real (Cityscapes) vs synthetic (GTA) images.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import torch


@dataclass
class DataConfig:
    """Data-related configuration."""
    # Data paths
    cityscapes_path: Path = Path("Data/driving/cityscapes/images")
    gta_path: Path = Path("Data/driving/GTA/images")
    
    # Data Split Counts (per dataset)
    train_count: int = 800
    val_count: int = 100
    test_count: int = 100
    
    # Image processing
    preprocess_mode: str = "pad"  # Options: "crop", "resize", "pad"
    crop_size: int = 512  # Random crop size for training (when mode is "crop")
    val_crop_size: int = 512  # Center crop size for validation (when mode is "crop")
    resize_size: int = 512  # Resize target size (when mode is "resize")
    pad_size: int = 256  # Padding target size (when mode is "pad")
    pad_fill: Tuple[int, int, int] = (255, 255, 255)  # White padding
    min_image_size: int = 0  # Minimum image dimension to be included
    
    # Normalization (ImageNet stats for pretrained models)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # File patterns to exclude (e.g., thumbnails)
    exclude_patterns: List[str] = field(default_factory=lambda: ["_t.jpg", "_1_2.jpg", "_2.jpg"])
    
    # Random seed for reproducible splits
    seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Backbone
    backbone: str = "convnext_tiny" # "resnet50"  # Options: resnet18, resnet34, resnet50, resnet101, vit_b_16, vit_b_32, vit_l_16, vit_l_32
    pretrained: bool = True
    
    # Classifier head
    hidden_dim: int = 512
    dropout: float = 0.3
    
    # Feature extraction (for XAI later)
    return_features: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    batch_size: int = 64
    learning_rate: float = 1e-4
    backbone_lr_multiplier: float = 0.1  # Lower LR for pretrained backbone
    weight_decay: float = 1e-4
    
    # Training duration
    max_epochs: int = 50
    
    # Early stopping
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 1e-4
    
    # Learning rate scheduling
    scheduler: str = "cosine"  # Options: cosine, step, plateau
    warmup_epochs: int = 2
    
    # Mixed precision training (good for A100)
    use_amp: bool = True
    
    # Gradient clipping
    gradient_clip_val: float = 1.0
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_top_k: int = 3
    
    # Logging
    log_every_n_steps: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seed
    seed: int = 42


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment name
    experiment_name: str = "domain_discriminator"
    
    def __post_init__(self):
        """Create necessary directories."""
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def default(cls) -> "Config":
        """Get the default configuration."""
        return cls()


def get_config() -> Config:
    """Get the default configuration."""
    return Config()
