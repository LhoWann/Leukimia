
import timm
import torch
import torch.nn as nn
import numpy as np

BACKBONE_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"

def build_backbone(pretrained: bool = True):
    backbone = timm.create_model(
        BACKBONE_NAME,
        pretrained=pretrained,
        features_only=True,
    )
    with torch.no_grad():
        dummy   = torch.zeros(1, 3, 224, 224)
        feats   = backbone(dummy)
        out_channels = feats[-1].shape[1]
    return backbone, out_channels

class SpatialAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens      = x.flatten(2).permute(0, 2, 1)
        tokens      = self.norm(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out    = self.dropout(attn_out)
        out         = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return x + out
    
class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)
    
class LeukemiaClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone, out_channels = build_backbone(pretrained=pretrained)
        self.attention = SpatialAttentionBlock(embed_dim=out_channels)
        self.head      = ClassificationHead(in_features=out_channels, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        x     = feats[-1]
        x     = self.attention(x)
        return self.head(x)
    
class GradCAM:
    def __init__(self, model: LeukemiaClassifier):
        self.model        = model
        self._gradients   = None
        self._activations = None
        model.attention.register_forward_hook(self._save_activation)
        model.attention.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam    -= cam.min()
        cam    /= cam.max() + 1e-8
        return cam
