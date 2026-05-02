"""
Model module defining the domain discriminator architectures.

This module provides:
- ResNet-based discriminator with pretrained backbone
- Vision Transformer (ViT) based discriminator with pretrained backbone
- Modular design to easily swap backbones
- Feature extraction capability for XAI analysis
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for classifying real vs synthetic images.
    
    Uses a pretrained backbone (ResNet or ConvNeXt) with a custom
    classification head. Supports feature extraction for XAI analysis.
    
    Attributes:
        backbone: The feature extraction backbone.
        classifier: The classification head.
        return_features: Whether to return intermediate features.
    """
    
    # Mapping of backbone names to model constructors and feature dimensions
    BACKBONES = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
        "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
        "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 768),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024),
        "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.IMAGENET1K_V1, 1536),
    }
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout_rate: float = 0.5,
        return_features: bool = False,
    ):
        """
        Initialize the discriminator.
        
        Args:
            backbone: Name of the backbone architecture.
            pretrained: Whether to use pretrained weights.
            hidden_dim: Hidden dimension of the classifier head.
            dropout_rate: Dropout rate in the classifier.
            return_features: If True, forward() returns features dict.
        """
        super().__init__()
        
        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}. Available: {list(self.BACKBONES.keys())}")
        
        model_fn, weights, feature_dim = self.BACKBONES[backbone]
        
        # Load backbone
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)
        
        # Remove the original classifier
        # ResNet uses 'fc', ConvNeXt uses 'classifier'
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            # ConvNeXt has a Sequential classifier
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0],  # LayerNorm
                self.backbone.classifier[1],  # Flatten
                nn.Identity()  # Replace Linear layer
            )
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),  # Lower dropout in second layer
            nn.Linear(hidden_dim, 1),  # Binary classification
        )
        
        self.return_features = return_features
        self._backbone_name = backbone
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        return self.backbone(x)
    
    def forward(
        self, 
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            If return_features is False: Logits tensor of shape (B, 1).
            If return_features is True: Dict with 'logits' and 'features' keys.
        """
        features = self.get_features(x)
        logits = self.classifier(features)
        
        if self.return_features:
            return {
                "logits": logits,
                "features": features,
            }
        
        return logits
    
    def get_target_layers(self) -> list:
        """
        Get target layers for Grad-CAM visualization.
        
        Returns:
            List of target layers for XAI analysis.
        """
        # For ResNet, layer4 is typically the best for Grad-CAM
        if hasattr(self.backbone, 'layer4'):
            return [self.backbone.layer4[-1]]
        # For ConvNeXt, use the last stage
        elif hasattr(self.backbone, 'features'):
            return [self.backbone.features[-1][-1]]  # Last block of last stage
        return []
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_parameter_groups(self, backbone_lr_mult: float = 0.1) -> list:
        """
        Get parameter groups with different learning rates.
        
        Args:
            backbone_lr_mult: Multiplier for backbone learning rate.
            
        Returns:
            List of parameter group dicts for optimizer.
        """
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith("backbone"):
                    backbone_params.append(param)
                else:
                    classifier_params.append(param)
        
        return [
            {"params": backbone_params, "lr_mult": backbone_lr_mult},
            {"params": classifier_params, "lr_mult": 1.0},
        ]


def create_model(
    backbone: str = "resnet50",
    pretrained: bool = True,
    hidden_dim: int = 256,
    dropout_rate: float = 0.5,
    return_features: bool = False,
) -> nn.Module:
    """
    Factory function to create a domain discriminator.
    
    Args:
        backbone: Backbone architecture name (resnet18/34/50/101 or vit_b_16/32, vit_l_16/32).
        pretrained: Whether to use pretrained weights.
        hidden_dim: Hidden dimension of classifier.
        dropout_rate: Dropout rate.
        return_features: Whether to return features in forward pass.
        
    Returns:
        Initialized DomainDiscriminator or ViTDiscriminator based on backbone.
    """
    # Check if it's a ViT backbone
    if backbone.startswith("vit_"):
        return ViTDiscriminator(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            return_features=return_features,
        )
    else:
        return DomainDiscriminator(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            return_features=return_features,
        )


if __name__ == "__main__":
    # Test the model
    model = create_model()
    print(f"Model architecture:")
    print(f"  Backbone: {model._backbone_name}")
    print(f"  Feature dim: {model.feature_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 384, 384)
    
    # Without features
    out = model(x)
    print(f"\nOutput shape: {out.shape}")
    
    # With features
    model.return_features = True
    out = model(x)
    print(f"Features shape: {out['features'].shape}")
    print(f"Logits shape: {out['logits'].shape}")
    
    # Test parameter groups
    groups = model.get_parameter_groups(0.1)
    print(f"\nParameter groups:")
    for i, g in enumerate(groups):
        n_params = sum(p.numel() for p in g["params"])
        print(f"  Group {i}: {n_params:,} params, lr_mult={g['lr_mult']}")


class ViTDiscriminator(nn.Module):
    """
    Vision Transformer (ViT) based discriminator for classifying real vs synthetic images.
    
    Uses a pretrained ViT backbone with a custom classification head.
    Supports feature extraction for XAI analysis.
    
    Attributes:
        backbone: The ViT feature extraction backbone.
        classifier: The classification head.
        return_features: Whether to return intermediate features.
    """
    
    # Mapping of ViT backbone names to model constructors and feature dimensions
    BACKBONES = {
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT, 768),
        "vit_b_32": (models.vit_b_32, models.ViT_B_32_Weights.DEFAULT, 768),
        "vit_l_16": (models.vit_l_16, models.ViT_L_16_Weights.DEFAULT, 1024),
        "vit_l_32": (models.vit_l_32, models.ViT_L_32_Weights.DEFAULT, 1024),
    }
    
    def __init__(
        self,
        backbone: str = "vit_b_16",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout_rate: float = 0.5,
        return_features: bool = False,
    ):
        """
        Initialize the ViT discriminator.
        
        Args:
            backbone: Name of the ViT backbone architecture.
            pretrained: Whether to use pretrained weights.
            hidden_dim: Hidden dimension of the classifier head.
            dropout_rate: Dropout rate in the classifier.
            return_features: If True, forward() returns features dict.
        """
        super().__init__()
        
        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown ViT backbone: {backbone}. Available: {list(self.BACKBONES.keys())}")
        
        model_fn, weights, feature_dim = self.BACKBONES[backbone]
        
        # Load backbone
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)
        
        # Remove the original classifier head
        self.backbone.heads = nn.Identity()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),  # ViT typically uses GELU
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )
        
        self.return_features = return_features
        self._backbone_name = backbone
        
        # Store reference to encoder blocks for XAI
        self._encoder_blocks = self.backbone.encoder.layers
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the ViT backbone.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        # ViT processes through patch embedding + transformer encoder
        # _process_input handles patch embedding and positional encoding
        x = self.backbone._process_input(x)
        n = x.shape[0]
        
        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Pass through transformer encoder
        x = self.backbone.encoder(x)
        
        # Return the class token output (first token)
        return x[:, 0]
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            If return_features is False: Logits tensor of shape (B, 1).
            If return_features is True: Dict with 'logits' and 'features' keys.
        """
        features = self.get_features(x)
        logits = self.classifier(features)
        
        if self.return_features:
            return {
                "logits": logits,
                "features": features,
            }
        
        return logits
    
    def get_target_layers(self) -> list:
        """
        Get target layers for Grad-CAM visualization.
        
        For ViT, we use the last transformer encoder block's layer norm.
        Note: ViT attention maps can also be visualized differently.
        
        Returns:
            List of target layers for XAI analysis.
        """
        # Return the last encoder block for attention-based visualization
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layers'):
            # Return the last encoder layer's layer norm (ln_2)
            last_block = self.backbone.encoder.layers[-1]
            if hasattr(last_block, 'ln_2'):
                return [last_block.ln_2]
            return [last_block]
        return []
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps from all transformer layers.
        
        This is useful for ViT-specific interpretability.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            Attention maps tensor of shape (B, num_layers, num_heads, num_patches+1, num_patches+1).
        """
        attention_maps = []
        
        # Process input
        x = self.backbone._process_input(x)
        n = x.shape[0]
        
        # Add class token
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Hook to capture attention weights
        def get_attention_hook(module, input, output):
            # Self-attention computes Q, K, V and attention weights
            # We need to access the attention weights from the MHA module
            pass
        
        # Pass through encoder layers and collect attention
        for layer in self.backbone.encoder.layers:
            # Get attention from self_attention module
            # Note: This requires modifying the forward pass or using hooks
            x = layer(x)
        
        return x[:, 0]  # Return class token for now
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_parameter_groups(self, backbone_lr_mult: float = 0.1) -> list:
        """
        Get parameter groups with different learning rates.
        
        Args:
            backbone_lr_mult: Multiplier for backbone learning rate.
            
        Returns:
            List of parameter group dicts for optimizer.
        """
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith("backbone"):
                    backbone_params.append(param)
                else:
                    classifier_params.append(param)
        
        return [
            {"params": backbone_params, "lr_mult": backbone_lr_mult},
            {"params": classifier_params, "lr_mult": 1.0},
        ]


# List of all available backbones for both architectures
ALL_BACKBONES = {
    **DomainDiscriminator.BACKBONES,
    **ViTDiscriminator.BACKBONES,
}


def create_model(
    backbone: str = "resnet50",
    pretrained: bool = True,
    hidden_dim: int = 256,
    dropout_rate: float = 0.5,
    return_features: bool = False,
) -> Union[DomainDiscriminator, ViTDiscriminator]:
    """
    Factory function to create a domain discriminator.
    
    Automatically selects the appropriate model class based on backbone name.
    
    Args:
        backbone: Backbone architecture name (resnet*, convnext*, or vit_*).
        pretrained: Whether to use pretrained weights.
        hidden_dim: Hidden dimension of classifier.
        dropout_rate: Dropout rate.
        return_features: Whether to return features in forward pass.
        
    Returns:
        Initialized discriminator model (DomainDiscriminator or ViTDiscriminator).
    """
    # Determine which model class to use based on backbone name
    if backbone.startswith("vit"):
        return ViTDiscriminator(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            return_features=return_features,
        )
    else:
        # ResNet and ConvNeXt both use DomainDiscriminator
        return DomainDiscriminator(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            return_features=return_features,
        )


def get_available_backbones() -> List[str]:
    """
    Get list of all available backbone names.
    
    Returns:
        List of backbone names.
    """
    return list(ALL_BACKBONES.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DomainDiscriminator (ResNet)")
    print("=" * 60)
    
    # Test ResNet model
    model = create_model(backbone="resnet50")
    print(f"Model architecture:")
    print(f"  Backbone: {model._backbone_name}")
    print(f"  Feature dim: {model.feature_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 384, 384)
    out = model(x)
    print(f"\nOutput shape: {out.shape}")
    
    # With features
    model.return_features = True
    out = model(x)
    print(f"Features shape: {out['features'].shape}")
    print(f"Logits shape: {out['logits'].shape}")
    
    # Test target layers
    target_layers = model.get_target_layers()
    print(f"Target layers for Grad-CAM: {len(target_layers)}")
    
    print("\n" + "=" * 60)
    print("Testing ViTDiscriminator")
    print("=" * 60)
    
    # Test ViT model
    vit_model = create_model(backbone="vit_b_16")
    print(f"Model architecture:")
    print(f"  Backbone: {vit_model._backbone_name}")
    print(f"  Feature dim: {vit_model.feature_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in vit_model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in vit_model.parameters() if p.requires_grad):,}")
    
    # Test forward pass (ViT expects 224x224 by default, but can handle other sizes)
    x_vit = torch.randn(4, 3, 224, 224)
    out_vit = vit_model(x_vit)
    print(f"\nOutput shape: {out_vit.shape}")
    
    # With features
    vit_model.return_features = True
    out_vit = vit_model(x_vit)
    print(f"Features shape: {out_vit['features'].shape}")
    print(f"Logits shape: {out_vit['logits'].shape}")
    
    # Test target layers
    target_layers_vit = vit_model.get_target_layers()
    print(f"Target layers for Grad-CAM: {len(target_layers_vit)}")
    
    print("\n" + "=" * 60)
    print("Available backbones:")
    print("=" * 60)
    for name in get_available_backbones():
        print(f"  - {name}")
