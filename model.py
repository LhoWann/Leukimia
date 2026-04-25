
import timm
import torch
import torch.nn as nn

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