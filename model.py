
import timm
import torch

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