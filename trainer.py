"""
Trainer module for training and evaluating the domain discriminator.

This module provides:
- Training loop with mixed precision support
- Validation and early stopping
- Checkpointing and model saving
- Comprehensive metrics logging
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, TrainingConfig
from model import DomainDiscriminator, ViTDiscriminator


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score.
            
        Returns:
            True if training should stop.
        """
        if self.mode == "min":
            score = -score
            
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop


class Trainer:
    """
    Trainer class for the domain discriminator.
    
    Handles training loop, validation, checkpointing, and metrics.
    """
    
    def __init__(
        self,
        model: DomainDiscriminator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The discriminator model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Full configuration object.
            device: Device to train on (default: from config).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tc = config.training  # Shorthand for training config
        
        # Device
        self.device = device or self.tc.device
        self.model = self.model.to(self.device)
        
        # Loss function (BCE with logits for numerical stability)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer with parameter groups (lower LR for backbone)
        param_groups = model.get_parameter_groups(self.tc.backbone_lr_multiplier)
        self.optimizer = AdamW(
            [
                {"params": param_groups[0]["params"], "lr": self.tc.learning_rate * param_groups[0]["lr_mult"]},
                {"params": param_groups[1]["params"], "lr": self.tc.learning_rate * param_groups[1]["lr_mult"]},
            ],
            weight_decay=self.tc.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.tc.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.tc.early_stopping_patience,
            min_delta=self.tc.early_stopping_min_delta,
            mode="min",
        )
        
        # Metrics tracking
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }
        
        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.tc.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.tc.max_epochs - self.tc.warmup_epochs,
                eta_min=self.tc.learning_rate * 0.01,
            )
        elif self.tc.scheduler == "step":
            return StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5,
            )
        elif self.tc.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.tc.scheduler}")
    
    def _warmup_lr(self, epoch: int, step: int, total_steps: int):
        """Apply linear warmup to learning rate."""
        if epoch >= self.tc.warmup_epochs:
            return
        
        warmup_steps = self.tc.warmup_epochs * len(self.train_loader)
        current_step = epoch * len(self.train_loader) + step
        
        if current_step < warmup_steps:
            warmup_factor = current_step / warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * warmup_factor / max(warmup_factor, 1e-8)
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for step, batch in enumerate(pbar):
            # Warmup
            self._warmup_lr(epoch, step, len(self.train_loader))
            
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.tc.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.tc.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.tc.gradient_clip_val,
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                
                if self.tc.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.tc.gradient_clip_val,
                    )
                
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += images.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.4f}",
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1)
            
            if self.tc.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += images.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool = False,
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            val_loss: Validation loss.
            val_acc: Validation accuracy.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": {
                "backbone": self.model._backbone_name,
                "feature_dim": self.model.feature_dim,
                "hidden_dim": self.model.hidden_dim,
                "dropout": self.model.dropout_rate,
            },
            "history": self.history,
        }
        
        # Save best checkpoint
        if is_best:
            best_path = self.tc.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Save every 5th checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = self.tc.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
    
    def train(self) -> Dict:
        """
        Full training loop.
        
        Returns:
            Dictionary with training history and best metrics.
        """
        print(f"\nStarting training for {self.tc.max_epochs} epochs")
        print(f"Training on {self.device}")
        print(f"Using AMP: {self.tc.use_amp}")
        print(f"Checkpoint dir: {self.tc.checkpoint_dir}")
        print()
        
        start_time = time.time()
        
        for epoch in range(self.tc.max_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if epoch >= self.tc.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[-1]["lr"]
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_acc, is_best)
            
            # Log progress
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.tc.max_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            if is_best:
                print(f"  ★ New best model!")
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.1f} minutes")
        print(f"Best model from epoch {self.best_epoch+1}:")
        print(f"  Val Loss: {self.best_val_loss:.4f}")
        print(f"  Val Acc:  {self.best_val_acc:.4f}")
        
        # Save training history
        history_path = self.tc.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
        }
    
    def load_best_model(self):
        """Load the best model checkpoint."""
        best_path = self.tc.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        else:
            print("No best model checkpoint found")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str = "cuda",
):
    """
    Load a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.
        
    Returns:
        Loaded model (DomainDiscriminator or ViTDiscriminator).
    """
    from model import create_model
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    model_config = checkpoint.get("config", {})
    backbone = model_config.get("backbone", "resnet50")
    
    # Try to get hidden_dim from config, otherwise infer from weights
    if "hidden_dim" in model_config:
        hidden_dim = model_config["hidden_dim"]
    else:
        # Infer hidden_dim from the shape of classifier.1.bias (first linear layer output)
        state_dict = checkpoint["model_state_dict"]
        if "classifier.1.bias" in state_dict:
            hidden_dim = state_dict["classifier.1.bias"].shape[0]
            print(f"  Inferred hidden_dim={hidden_dim} from checkpoint weights")
        else:
            hidden_dim = 256  # fallback default
    
    dropout_rate = model_config.get("dropout", 0.5)
    
    # Create model with matching architecture
    model = create_model(
        backbone=backbone,
        pretrained=False,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Backbone: {backbone}")
    print(f"  Epoch: {checkpoint['epoch']+1}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val Acc: {checkpoint['val_acc']:.4f}")
    
    return model
