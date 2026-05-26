"""
Training entrypoint with experiment registry.

Run a specific experiment:
    python main.py --exp baseline
    python main.py --exp focusmix_v2
    python main.py --exp focusmix_mha
    python main.py --exp focusmix_full
    python main.py --exp focusmix_cam   (uses online Grad-CAM regeneration)
"""
import warnings
warnings.filterwarnings("ignore", message="triton not found.*", module="torch.utils.flop_counter")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

import argparse
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger

from data_module import LeukemiaDataModule
from lightning_model import LeukemiaLightningModel, GradCAMExtractor

try:
    from lightning.pytorch.callbacks import RichProgressBar
    _RICH = True
except ImportError:
    _RICH = False


# Experiment registry
@dataclass
class ExperimentConfig:
    name: str
    aug_mode: str           # 'none' | 'saliency' | 'focusmix' | 'focusmix_cam'
    use_mha: bool
    mha_stage: int = 2      # 2 = after stage 3 (14x14x384), 3 = after stage 4 (7x7x768)
    aug_prob: float = 0.5
    paste_ratio: float = 0.25
    n_segments: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.05
    llrd: float = 0.75
    label_smoothing: float = 0.05
    batch_size: int = 32
    max_epochs: int = 50
    warmup_epochs: int = 5


EXPERIMENTS = {
    # A. Pure baseline — ConvNeXtV2-Tiny + standard augmentation only.
    #    Establishes floor performance. Required for honest delta comparison.
    'baseline': ExperimentConfig(
        name='baseline',
        aug_mode='none',
        use_mha=False,
    ),

    # B. Add MHA only (no mixing aug) — isolates contribution of attention.
    'mha_only': ExperimentConfig(
        name='mha_only',
        aug_mode='none',
        use_mha=True,
        mha_stage=2,
    ),

    # C. Pure SaliencyMix — rectangular saliency-guided patch.
    #    Simplest mixing aug, useful as a cheap aug baseline.
    'saliency_mix': ExperimentConfig(
        name='saliency_mix',
        aug_mode='saliency',
        use_mha=False,
        paste_ratio=0.25,
    ),

    # D. FocusAugMix V2 style — OcCaMix + SaliencyMix, no MHA.
    #    Architecture-agnostic: cleanest combination with ConvNeXtV2.
    'focusmix_v2': ExperimentConfig(
        name='focusmix_v2',
        aug_mode='focusmix',
        use_mha=False,
        paste_ratio=0.25,
        n_segments=50,
    ),

    # E. FocusAugMix V1 style adapted — OcCaMix + MHA (after stage 3).
    #    Tests whether MHA boost generalizes to ConvNeXtV2.
    'focusmix_mha': ExperimentConfig(
        name='focusmix_mha',
        aug_mode='focusmix',
        use_mha=True,
        mha_stage=2,
        paste_ratio=0.25,
    ),

    # F. FocusAugMix V4 style — OcCaMix + Saliency + MHA + Grad-CAM (online).
    #    Most comprehensive; also most expensive. Grad-CAM maps regenerated
    #    every N epochs from current model. Expect this to plateau with
    #    diminishing returns relative to V2/V1.
    'focusmix_full': ExperimentConfig(
        name='focusmix_full',
        aug_mode='focusmix_cam',
        use_mha=True,
        mha_stage=2,
        paste_ratio=0.25,
    ),

    # G. Aggressive paste ratio — for very small datasets, more aggressive
    #    mixing acts as stronger regularization. Try if focusmix_v2 saturates.
    'focusmix_aggressive': ExperimentConfig(
        name='focusmix_aggressive',
        aug_mode='focusmix',
        use_mha=True,
        mha_stage=2,
        aug_prob=0.7,
        paste_ratio=0.35,
    ),
}


# Grad-CAM regeneration callback
class GradCAMRefresher(L.Callback):
    """
    Regenerates Grad-CAM maps for the training set every `refresh_every`
    epochs. Used by 'focusmix_cam' aug mode.

    Tradeoff: refreshing every epoch is expensive (one full eval pass).
    Refreshing every 5 epochs is a good balance — Grad-CAM signal evolves
    slowly relative to training dynamics.
    """
    def __init__(self, refresh_every: int = 5, target_stage: int = 3):
        super().__init__()
        self.refresh_every = refresh_every
        self.target_stage = target_stage

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.refresh_every != 0:
            return
        # Only run if dataset is in cam-aware mode
        dm = trainer.datamodule
        if dm.aug_mode != 'focusmix_cam':
            return

        print(f"\n[GradCAM] Regenerating maps at epoch {trainer.current_epoch}")
        pl_module.eval()
        cam_maps = {}

        from torchvision import transforms
        from PIL import Image
        from torch.utils.data import DataLoader

        # Lightweight transform for cam generation (no aug)
        cam_transform = dm.val_transform
        device = pl_module.device

        with GradCAMExtractor(pl_module.model, target_stage=self.target_stage) as cam_extractor:
            for idx in range(len(dm.train_dataset)):
                img_pil, _ = dm.train_dataset.dataset[idx]  # underlying ImageFolder
                x = cam_transform(img_pil).unsqueeze(0).to(device)
                cam = cam_extractor(x)  # (1, H, W)
                cam_maps[idx] = cam[0]

        dm.train_dataset.set_gradcam_maps(cam_maps)
        pl_module.train()
        print(f"[GradCAM] Generated {len(cam_maps)} maps")


# Utils

def set_seed(seed: int = 42, strict: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # ~2x faster, slight non-determinism in kernel selection
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def run_experiment(cfg: ExperimentConfig, data_dir: str = 'dataset', seed: int = 42):
    set_seed(seed, strict=False)

    datamodule = LeukemiaDataModule(
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        num_workers=8,
        aug_mode=cfg.aug_mode,
        aug_prob=cfg.aug_prob,
        n_segments=cfg.n_segments,
        paste_ratio=cfg.paste_ratio,
    )
    datamodule.setup()

    print(f"\n{'=' * 60}")
    print(f"Experiment: {cfg.name}")
    print(f"Config: {asdict(cfg)}")
    print(f"Classes: {datamodule.classes}")
    print(f"Train: {len(datamodule.train_dataset)} | Val: {len(datamodule.val_dataset)}")
    print(f"{'=' * 60}\n")

    model = LeukemiaLightningModel(
        num_classes=datamodule.num_classes,
        pretrained=True,
        use_mha=cfg.use_mha,
        mha_stage=cfg.mha_stage,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        llrd=cfg.llrd,
        warmup_epochs=cfg.warmup_epochs,
        max_epochs=cfg.max_epochs,
        label_smoothing=cfg.label_smoothing,
    )

    ckpt_dir = Path('checkpoints') / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='{epoch:02d}-{val_acc:.4f}',
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=True),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    if cfg.aug_mode == 'focusmix_cam':
        callbacks.append(GradCAMRefresher(refresh_every=5))
    if _RICH:
        callbacks.append(RichProgressBar())

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator='auto',
        devices='auto',
        precision='bf16-mixed',           # ~1.8x speedup on modern GPUs
        callbacks=callbacks,
        logger=CSVLogger('logs', name=cfg.name),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=False,
    )

    trainer.fit(model, datamodule=datamodule)

    # Final eval on best checkpoint
    best_path = callbacks[0].best_model_path
    print(f"\nBest checkpoint: {best_path}")
    if best_path:
        trainer.validate(model, datamodule=datamodule, ckpt_path=best_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='focusmix_v2',
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--data-dir', type=str, default='dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments sequentially')
    args = parser.parse_args()

    if args.all:
        for name, cfg in EXPERIMENTS.items():
            run_experiment(cfg, args.data_dir, args.seed)
    else:
        run_experiment(EXPERIMENTS[args.exp], args.data_dir, args.seed)


if __name__ == '__main__':
    main()
