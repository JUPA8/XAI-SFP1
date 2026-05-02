"""
Evaluation module for testing the domain discriminator.

This module provides:
- Test set evaluation with comprehensive metrics
- Per-sample predictions and analysis
- Confusion matrix and classification report
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import DomainDiscriminator


class Evaluator:
    """
    Evaluator for testing the domain discriminator.
    """
    
    def __init__(
        self,
        model: DomainDiscriminator,
        test_loader: DataLoader,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: The discriminator model.
            test_loader: Test data loader.
            device: Device to evaluate on.
            use_amp: Whether to use automatic mixed precision.
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.use_amp = use_amp
        self.criterion = nn.BCEWithLogitsLoss()
        
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        Evaluate the model on the test set.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_paths = []
        total_loss = 0.0
        
        pbar = tqdm(self.test_loader, desc="Evaluating")
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            paths = batch["path"]
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels.unsqueeze(1))
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels.unsqueeze(1))
            
            total_loss += loss.item() * images.size(0)
            
            # Get predictions and probabilities
            probabilities = torch.sigmoid(outputs).squeeze()
            predictions = (probabilities > 0.5).long()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_paths.extend(paths)
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, labels, probabilities)
        metrics["loss"] = total_loss / len(labels)
        
        # Per-sample results
        per_sample = {
            "paths": all_paths,
            "predictions": predictions.tolist(),
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
        }
        
        return {
            "metrics": metrics,
            "per_sample": per_sample,
        }
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict:
        """
        Calculate comprehensive metrics.
        
        Args:
            predictions: Model predictions (0 or 1).
            labels: Ground truth labels.
            probabilities: Prediction probabilities.
            
        Returns:
            Dictionary with metrics.
        """
        # Basic metrics
        correct = (predictions == labels).sum()
        total = len(labels)
        accuracy = correct / total
        
        # Per-class metrics
        # Label 0 = real (Cityscapes), Label 1 = synthetic (GTA)
        real_mask = labels == 0
        synthetic_mask = labels == 1
        
        # True positives, false positives, etc.
        # Using "synthetic" as the positive class
        tp = ((predictions == 1) & (labels == 1)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        
        # Precision, Recall, F1 for synthetic class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Per-class accuracy
        real_acc = tn / real_mask.sum() if real_mask.sum() > 0 else 0
        synthetic_acc = tp / synthetic_mask.sum() if synthetic_mask.sum() > 0 else 0
        
        # Confusion matrix
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "real_accuracy": float(real_acc),
            "synthetic_accuracy": float(synthetic_acc),
            "confusion_matrix": confusion_matrix.tolist(),
            "total_samples": int(total),
            "correct_predictions": int(correct),
        }
    
    def get_misclassified(self, results: Dict) -> Dict[str, List]:
        """
        Get paths of misclassified samples.
        
        Args:
            results: Results from evaluate().
            
        Returns:
            Dictionary with paths of false positives and false negatives.
        """
        per_sample = results["per_sample"]
        
        false_positives = []  # Real images predicted as synthetic
        false_negatives = []  # Synthetic images predicted as real
        
        for path, pred, label in zip(
            per_sample["paths"],
            per_sample["predictions"],
            per_sample["labels"],
        ):
            if pred == 1 and label == 0:
                false_positives.append(path)
            elif pred == 0 and label == 1:
                false_negatives.append(path)
        
        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
    
    def print_results(self, results: Dict):
        """
        Print evaluation results in a formatted way.
        
        Args:
            results: Results from evaluate().
        """
        metrics = results["metrics"]
        
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
        print(f"  Loss:     {metrics['loss']:.4f}")
        
        print(f"\nPer-Class Performance:")
        print(f"  Real (Cityscapes) Accuracy:      {metrics['real_accuracy']:.4f}")
        print(f"  Synthetic (GTA) Accuracy:        {metrics['synthetic_accuracy']:.4f}")
        
        print(f"\nDetailed Metrics (Synthetic as Positive):")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics["confusion_matrix"])
        print(f"                  Predicted")
        print(f"                  Real    Synthetic")
        print(f"  Actual Real     {cm[0,0]:5d}    {cm[0,1]:5d}")
        print(f"  Actual Synthetic{cm[1,0]:5d}    {cm[1,1]:5d}")
        
        print("\n" + "=" * 50)
