import warnings
warnings.filterwarnings("ignore", message="triton not found.*", module="torch.utils.flop_counter")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import numpy as np
import lightning as L
from torchmetrics import Accuracy


class ConvNeXtV2WithAttention(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, num_heads=8, attn_dropout=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        with torch.no_grad():
            feat = self.backbone(torch.randn(1, 3, 224, 224))
            self.feat_dim = feat.shape[1]

        self.attention = nn.MultiheadAttention(
            embed_dim=self.feat_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=False,
        )
        self.attn_norm = nn.LayerNorm(self.feat_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        self.gradients = None
        self.activations = None
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
            target.register_full_backward_hook(bwd),
        ]

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, x):
        features = self.backbone(x)
        B, C, H, W = features.shape
        spatial = features.view(B, C, H * W).permute(2, 0, 1)
        attn_out, _ = self.attention(spatial, spatial, spatial)
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
            c = (c - c_min) / (c_max - c_min) if c_max - c_min > 1e-10 else np.zeros_like(c)
            cam_batch.append(c)
        self._remove_hooks()
        self.train()
        return logits, np.stack(cam_batch, axis=0)


class LeukemiaLightningModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvNeXtV2WithAttention(
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets_a, targets_b, lam = batch
        logits = self(images)
        loss = (
            lam * self.criterion(logits, targets_a)
            + (1.0 - lam) * self.criterion(logits, targets_b)
        ).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        self.val_accuracy(preds, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            },
        }
