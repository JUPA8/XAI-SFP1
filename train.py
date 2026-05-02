"""
Main entry point for training the domain discriminator.

Usage:
    python train.py [--config CONFIG_PATH]
    
Example:
    python train.py
    python train.py --config custom_config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from dataset import create_dataloaders, create_splits, discover_images
from model import create_model
from trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train domain discriminator (Real vs Synthetic)"
    )
    
    # Data paths
    parser.add_argument(
        "--cityscapes-path",
        type=str,
        default=None,
        help="Path to Cityscapes images directory",
    )
    parser.add_argument(
        "--gta-path",
        type=str,
        default=None,
        help="Path to GTA images directory",
    )
    
    # Model
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", 
                 "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
                 "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"],
        help="Backbone architecture (default: resnet50)",
    )
    
    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience (default: 7)",
    )
    
    # Image processing
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Crop size for training (default: 224)",
    )
    parser.add_argument(
        "--preprocess-mode",
        type=str,
        default="crop",
        choices=["crop", "resize", "pad"],
        help="Preprocessing mode: crop, resize, or pad (default: crop)",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=384,
        help="Resize target size when using resize mode (default: 384)",
    )
    parser.add_argument(
        "--pad-size",
        type=int,
        default=200,
        help="Padding target size when using pad mode (default: 200)",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create configuration
    config = Config.default()
    
    # Override with command line arguments
    if args.cityscapes_path:
        config.data.cityscapes_path = Path(args.cityscapes_path)
    if args.gta_path:
        config.data.gta_path = Path(args.gta_path)
    
    config.model.backbone = args.backbone
    config.data.crop_size = args.crop_size
    config.data.preprocess_mode = args.preprocess_mode
    config.data.resize_size = args.resize_size
    config.data.pad_size = args.pad_size
    
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.early_stopping_patience = args.patience
    config.training.device = args.device
    config.training.use_amp = not args.no_amp
    config.training.checkpoint_dir = Path(args.output_dir) / "checkpoints"
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "data": {
                "cityscapes_path": str(config.data.cityscapes_path),
                "gta_path": str(config.data.gta_path),
                "crop_size": config.data.crop_size,
                "preprocess_mode": config.data.preprocess_mode,
                "resize_size": config.data.resize_size,
                "pad_size": config.data.pad_size,
                "train_count": config.data.train_count,
                "val_count": config.data.val_count,
                "test_count": config.data.test_count,
            },
            "model": {
                "backbone": config.model.backbone,
                "hidden_dim": config.model.hidden_dim,
                "dropout": config.model.dropout,
            },
            "training": {
                "batch_size": config.training.batch_size,
                "max_epochs": config.training.max_epochs,
                "learning_rate": config.training.learning_rate,
                "early_stopping_patience": config.training.early_stopping_patience,
                "use_amp": config.training.use_amp,
            },
        }, f, indent=2)
    
    print("=" * 60)
    print("DOMAIN DISCRIMINATOR TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Preprocessing mode: {config.data.preprocess_mode}")
    if config.data.preprocess_mode == "crop":
        print(f"  Crop size: {config.data.crop_size}")
    elif config.data.preprocess_mode == "resize":
        print(f"  Resize size: {config.data.resize_size}")
    elif config.data.preprocess_mode == "pad":
        print(f"  Pad size: {config.data.pad_size}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Max epochs: {config.training.max_epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Early stopping patience: {config.training.early_stopping_patience}")
    print(f"  Device: {config.training.device}")
    print(f"  AMP: {config.training.use_amp}")
    print(f"  Output dir: {output_dir}")
    
    # Discover images
    print("\n" + "-" * 60)
    print("Discovering images...")
    
    real_images = discover_images(config.data.cityscapes_path, config.data)
    synthetic_images = discover_images(config.data.gta_path, config.data)
    
    print(f"  Real images (Cityscapes): {len(real_images)}")
    print(f"  Synthetic images (GTA): {len(synthetic_images)}")
    
    if len(real_images) == 0 or len(synthetic_images) == 0:
        print("ERROR: No images found. Please check the paths.")
        sys.exit(1)
    
    # Create splits
    print("\n" + "-" * 60)
    print("Creating train/val/test splits...")
    
    splits = create_splits(real_images, synthetic_images, config.data)
    
    print(f"  Training: {len(splits['train']['paths'])} images")
    print(f"  Validation: {len(splits['val']['paths'])} images")
    print(f"  Test: {len(splits['test']['paths'])} images")
    
    # Save splits
    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump({
            split_name: {
                "paths": [str(p) for p in split_data["paths"]],
                "labels": split_data["labels"],
            }
            for split_name, split_data in splits.items()
        }, f, indent=2)
    print(f"  Saved splits to {splits_path}")
    
    # Create dataloaders
    print("\n" + "-" * 60)
    print("Creating dataloaders...")
    
    train_loader, val_loader, test_loader = create_dataloaders(splits, config)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("\n" + "-" * 60)
    print("Creating model...")
    
    model = create_model(
        backbone=config.model.backbone,
        hidden_dim=config.model.hidden_dim,
        dropout_rate=config.model.dropout,
        pretrained=config.model.pretrained,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    
    # Create trainer
    print("\n" + "-" * 60)
    print("Starting training...")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    # Train
    results = trainer.train()
    
    # Final evaluation on test set
    print("\n" + "-" * 60)
    print("Evaluating on test set...")
    
    from evaluate import Evaluator
    
    # Load best model
    trainer.load_best_model()
    
    evaluator = Evaluator(
        model=trainer.model,
        test_loader=test_loader,
        device=config.training.device,
        use_amp=config.training.use_amp,
    )
    
    test_results = evaluator.evaluate()
    evaluator.print_results(test_results)
    
    # Save test results
    test_results_path = output_dir / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump({
            "metrics": test_results["metrics"],
        }, f, indent=2)
    print(f"\nTest results saved to {test_results_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest model saved to: {config.training.checkpoint_dir / 'best_model.pt'}")
    print(f"Training history saved to: {config.training.checkpoint_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
