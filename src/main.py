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
import logging

warnings.filterwarnings("ignore", message="triton not found.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities._pytree").setLevel(logging.ERROR)

import argparse
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── Fix PyTorch ≥2.6 weights_only=True default ────────────────────────────────
# Checkpoints embed numpy scalars, dtype objects, and LambdaLR state.
# Allowlist all affected globals so Lightning can load/validate checkpoints.
try:
    import numpy._core.multiarray  # noqa: F401
    import numpy.dtypes            # noqa: F401
    _safe = [numpy._core.multiarray.scalar, numpy.dtype]
    _safe += [getattr(numpy.dtypes, n) for n in dir(numpy.dtypes)
              if isinstance(getattr(numpy.dtypes, n), type)]
    torch.serialization.add_safe_globals(_safe)
except Exception:
    pass  # older numpy or older torch — not needed
# ──────────────────────────────────────────────────────────────────────────────
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO


class DirectDiskCheckpointIO(TorchCheckpointIO):
    """Write checkpoint directly to disk, bypassing the BytesIO buffer used by
    _atomic_save, which causes MemoryError on large models with limited RAM."""
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        torch.save(checkpoint, str(path))

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
    max_epochs: int = 30
    warmup_epochs: int = 3
    use_robust_aug: bool = False  # ReinhardJitter + RandomRotation(180) + RandomPerspective
    stain_sigma_mean: float = 0.15  # ReinhardJitter shift magnitude
    stain_sigma_std: float = 0.10   # ReinhardJitter scale magnitude
    stain_aug_prob: float = 0.5     # probability of applying ReinhardJitter


EXPERIMENTS = {
    # A. Pure baseline — ConvNeXtV2-Tiny + standard augmentation only.
    #    Establishes floor performance. Required for honest delta comparison.
    'baseline': ExperimentConfig(
        name='baseline',
        aug_mode='none',
        use_mha=False,
    ),

    # B. Add MHA only (no mixing aug) — isolates contribution of attention.
    #    MHA needs longer warmup to stabilise before attention weights drift.
    'mha_only': ExperimentConfig(
        name='mha_only',
        aug_mode='none',
        use_mha=True,
        mha_stage=2,
        warmup_epochs=5,
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
    #    Extra warmup: MHA + augmentation convergence is slower.
    'focusmix_mha': ExperimentConfig(
        name='focusmix_mha',
        aug_mode='focusmix',
        use_mha=True,
        mha_stage=2,
        paste_ratio=0.25,
        warmup_epochs=5,
    ),

    # F. FocusAugMix V4 style — OcCaMix + Saliency + MHA + Grad-CAM (online).
    #    Most comprehensive; also most expensive. Grad-CAM maps regenerated
    #    every N epochs from current model. Expect this to plateau with
    #    diminishing returns relative to V2/V1.
    #    Bug fix: train_dataloader uses num_workers=0 for focusmix_cam so
    #    updated cam maps are visible inside __getitem__ (workers can't share
    #    main-process state when num_workers > 0).
    #    Extra warmup: MHA + augmentation convergence is slower.
    'focusmix_full': ExperimentConfig(
        name='focusmix_full',
        aug_mode='focusmix_cam',
        use_mha=True,
        mha_stage=2,
        paste_ratio=0.25,
        warmup_epochs=5,
    ),

    # G. Moderate paste ratio — tuned down from the original aggressive config
    #    (aug_prob=0.7, paste_ratio=0.35) which worsened domain-shift gap to 0.66.
    #    Current values: aug_prob=0.5, paste_ratio=0.30 as a middle ground.
    #    Extra warmup: MHA + augmentation convergence is slower.
    'focusmix_aggressive': ExperimentConfig(
        name='focusmix_aggressive',
        aug_mode='focusmix',
        use_mha=True,
        mha_stage=2,
        aug_prob=0.5,
        paste_ratio=0.30,
        warmup_epochs=5,
    ),

    # H. Cross-domain robust variant — best single-model baseline (focusmix_v2,
    #    gap=0.274) extended with three domain-generalisation improvements:
    #      1. ReinhardJitter    : random LAB-space stain perturbation (p=0.5)
    #         simulates inter-lab/inter-batch staining variation
    #      2. RandomRotation(180): full orientation invariance (cells have no
    #         canonical orientation in smears)
    #      3. RandomPerspective  : minor smear-prep deformation (distortion=0.1)
    #    No MHA — experiments show MHA increases domain-shift gap consistently.
    'focusmix_v2_robust': ExperimentConfig(
        name='focusmix_v2_robust',
        aug_mode='focusmix',
        use_mha=False,
        paste_ratio=0.25,
        n_segments=50,
        use_robust_aug=True,
    ),

    # I. Stronger stain augmentation — same as H but with increased ReinhardJitter
    #    strength and application probability to better simulate the larger stain
    #    gap between ALL-IDB (Giemsa, Italy) and C-NMC (Wright-Giemsa, India).
    #    Analysis showed the model learns stain shortcuts; stronger jitter forces
    #    it to rely on morphological features instead.
    #      sigma_mean: 0.15 → 0.25  (wider LAB mean shift range)
    #      sigma_std:  0.10 → 0.15  (wider LAB std scale range)
    #      stain_aug_prob: 0.5 → 0.7 (applied more often)
    'focusmix_v2_robust_v2': ExperimentConfig(
        name='focusmix_v2_robust_v2',
        aug_mode='focusmix',
        use_mha=False,
        paste_ratio=0.25,
        n_segments=50,
        use_robust_aug=True,
        stain_sigma_mean=0.25,
        stain_sigma_std=0.15,
        stain_aug_prob=0.7,
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
        num_workers=2,
        aug_mode=cfg.aug_mode,
        aug_prob=cfg.aug_prob,
        n_segments=cfg.n_segments,
        paste_ratio=cfg.paste_ratio,
        use_robust_aug=cfg.use_robust_aug,
        stain_sigma_mean=cfg.stain_sigma_mean,
        stain_sigma_std=cfg.stain_sigma_std,
        stain_aug_prob=cfg.stain_aug_prob,
    )
    datamodule.setup()

    # Inverse-frequency class weights to fix minority-class bias
    class_weights = datamodule.get_class_weights().tolist()

    print(f"\n{'=' * 60}")
    print(f"Experiment: {cfg.name}")
    print(f"Config: {asdict(cfg)}")
    print(f"Classes: {datamodule.classes}")
    print(f"Train: {len(datamodule.train_dataset)} | Val: {len(datamodule.val_dataset)}")
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights]}")
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
        class_weights=class_weights,
    )

    ckpt_dir = Path('checkpoints') / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Monitor val_f1 (macro): more honest than val_acc for imbalanced classes
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='{epoch:02d}-{val_f1:.4f}',
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            save_last=True,
            save_weights_only=True,  # avoids MemoryError from serializing optimizer state
        ),
        # val_loss keeps decreasing even when val_f1=1.0; patience=10/30 = 33%
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
        plugins=[DirectDiskCheckpointIO()],
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
