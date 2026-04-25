import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np


class ConvNeXtV2WithAttention(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, num_heads=8, attn_dropout=0.1):
        super().__init__()

        self.backbone = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
            pretrained=pretrained, num_classes=0, global_pool=''
        )

        with torch.no_grad():
            feat = self.backbone(torch.randn(1, 3, 224, 224))
            self.feat_dim = feat.shape[1]

        self.attention = nn.MultiheadAttention(
            embed_dim=self.feat_dim, num_heads=num_heads,
            dropout=attn_dropout, batch_first=False
        )
        self.attn_norm = nn.LayerNorm(self.feat_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        self.gradients = None
        self.activations = None
        self.attn_weights = None
        self._hooks = []

    def _register_hooks(self):
        self._remove_hooks()
        target = self.backbone.stages[-1]

        def fwd(module, inp, out):
            self.activations = out

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self._hooks = [
            target.register_forward_hook(fwd),
            target.register_full_backward_hook(bwd)
        ]

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, x):
        features = self.backbone(x)
        B, C, H, W = features.shape

        spatial = features.view(B, C, H * W).permute(2, 0, 1)
        attn_out, self.attn_weights = self.attention(spatial, spatial, spatial)
        attn_out = self.attn_norm(attn_out + spatial)
        attn_out = attn_out.permute(1, 2, 0).view(B, C, H, W)

        pooled = self.pool(attn_out).view(B, -1)
        return self.classifier(self.dropout(pooled))

    def get_gradcam(self, x, target_class=None):
        self._register_hooks()
        self.eval()

        logits = self.forward(x)

        if target_class is None:
            target_class = logits.argmax(dim=1)

        self.zero_grad()
        one_hot = torch.zeros_like(logits)
        for i in range(x.size(0)):
            one_hot[i, target_class[i]] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)

        cam_batch = []
        for i in range(cam.size(0)):
            c = cam[i, 0].detach().cpu().numpy()
            c_min, c_max = c.min(), c.max()
            if c_max - c_min > 1e-10:
                c = (c - c_min) / (c_max - c_min)
            else:
                c = np.zeros_like(c)
            cam_batch.append(c)

        self._remove_hooks()
        self.train()
        return logits, np.stack(cam_batch, axis=0)


def create_model(num_classes=3, pretrained=True):
    return ConvNeXtV2WithAttention(num_classes=num_classes, pretrained=pretrained)