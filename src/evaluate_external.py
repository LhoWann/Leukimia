"""
External test evaluation on C-NMC 2019 (PKG-C-NMC).

Evaluates a model trained on ALL-IDB (Giemsa, Italy) against C-NMC 2019
(Wright-Giemsa, India) to measure cross-domain generalization.

Three conditions are reported:
  1. No normalization      → raw domain-shift gap
  2. Macenko normalization → C-NMC images mapped to ALL-IDB staining
  3. Reinhard normalization → color statistics transfer (faster, more robust)

C-NMC 2019 expected directory structure:
    <cnmc-dir>/
        all/          ← ALL (leukemia) cells
        hem/          ← HEM (normal) cells

    or equivalently:
        <cnmc-dir>/
            Abnormal/
            Normal/

Usage:
    cd src
    python evaluate_external.py \\
        --ckpt ../checkpoints/mha_only/epoch=06-val_acc=1.0000.ckpt \\
        --exp  mha_only \\
        --cnmc-dir /path/to/C-NMC_2019/test \\
        --data-dir ../dataset \\
        --output-json ../results/cnmc_eval.json
"""
from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", message="triton not found.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress torch triton logging warning (uses logging module, not warnings module)
logging.getLogger('torch.utils.flop_counter').setLevel(logging.ERROR)


def _worker_init_fn(worker_id: int) -> None:
    """Suppress torch triton warning inside DataLoader worker processes."""
    import logging as _logging
    _logging.getLogger('torch.utils.flop_counter').setLevel(_logging.ERROR)


import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as tv_datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ── UTF-8 stdout (Windows cp1252 terminals choke on box-drawing chars) ────────
import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    try:
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── Safe globals fix (PyTorch ≥ 2.6) ─────────────────────────────────────────
# Checkpoints embed numpy scalars, dtype objects, and LambdaLR scheduler state.
try:
    import numpy._core.multiarray  # noqa: F401
    import numpy.dtypes            # noqa: F401
    _safe = [numpy._core.multiarray.scalar, numpy.dtype]
    _safe += [getattr(numpy.dtypes, n) for n in dir(numpy.dtypes)
              if isinstance(getattr(numpy.dtypes, n), type)]
    torch.serialization.add_safe_globals(_safe)
except Exception:
    pass

# ── Local imports ─────────────────────────────────────────────────────────────
from stain_normalize import (
    MacenkoNormalizer,
    ReinhardNormalizer,
    compute_reference_from_dir,
)
from lightning_model import LeukemiaLightningModel

# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# C-NMC class-name → ALL-IDB class-name mapping
CNMC_CLASS_MAP: Dict[str, str] = {
    'all':      'Abnormal',
    'ALL':      'Abnormal',
    'hem':      'Normal',
    'HEM':      'Normal',
    'Abnormal': 'Abnormal',
    'Normal':   'Normal',
    'positive': 'Abnormal',  # some releases use these labels
    'negative': 'Normal',
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ExternalTestDataset(Dataset):
    """
    Flexible dataset for external test data (C-NMC 2019).

    Accepts any directory layout where class subdirectories use names
    covered by CNMC_CLASS_MAP.  Images with unrecognized class-folder names
    are silently skipped.

    Parameters
    ----------
    root_dir      : path to top-level C-NMC test directory
    normalizer    : MacenkoNormalizer | ReinhardNormalizer | None
    class_to_idx  : {'Abnormal': 0, 'Normal': 1} from training ImageFolder
    image_size    : resize target (square)
    """

    def __init__(
        self,
        root_dir: str,
        class_to_idx: Dict[str, int],
        normalizer=None,
        image_size: int = 224,
    ):
        self.root_dir     = Path(root_dir)
        self.normalizer   = normalizer
        self.class_to_idx = class_to_idx
        self.image_size   = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.samples: List[Tuple[str, int]] = []
        self._discover()

    def _discover(self) -> None:
        exts = {'.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff'}
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            canonical = CNMC_CLASS_MAP.get(class_dir.name)
            if canonical is None:
                warnings.warn(
                    f"Skipping unknown class directory: {class_dir.name!r}. "
                    f"Expected one of {list(CNMC_CLASS_MAP)}"
                )
                continue
            label = self.class_to_idx.get(canonical)
            if label is None:
                continue
            imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in exts]
            for p in sorted(imgs):
                self.samples.append((str(p), label))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {self.root_dir}. "
                f"Expected subdirectories named: {list(CNMC_CLASS_MAP)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')

        if self.normalizer is not None:
            img_np = np.array(img)
            try:
                img_np = self.normalizer.transform(img_np)
                img = Image.fromarray(img_np)
            except Exception as e:
                warnings.warn(f"Normalization failed for {path}: {e}. Using raw image.")

        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = 'Evaluating',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (predictions, ground_truths, probabilities)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc=desc, ncols=80):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> Tuple[Dict, str]:
    metrics = {
        'n_samples':        int(len(labels)),
        'accuracy':         float(accuracy_score(labels, preds)),
        'f1_macro':         float(f1_score(labels, preds, average='macro',    zero_division=0)),
        'f1_weighted':      float(f1_score(labels, preds, average='weighted', zero_division=0)),
        'precision_macro':  float(precision_score(labels, preds, average='macro', zero_division=0)),
        'recall_macro':     float(recall_score(labels, preds, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(labels, preds).tolist(),
    }
    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )
    return metrics, report


def _banner(title: str) -> str:
    bar = '─' * 60
    return f"\n{bar}\n  {title}\n{bar}"


def print_result(condition: str, metrics: Dict, report: str) -> None:
    print(_banner(condition))
    print(f"  N samples  : {metrics['n_samples']}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  F1 (macro) : {metrics['f1_macro']:.4f}")
    print(f"  Precision  : {metrics['precision_macro']:.4f}")
    print(f"  Recall     : {metrics['recall_macro']:.4f}")
    print(f"\n{report}")
    cm = np.array(metrics['confusion_matrix'])
    print(f"Confusion matrix:\n{cm}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='External evaluation on C-NMC 2019 with stain normalization'
    )
    parser.add_argument(
        '--ckpt', required=True,
        help='Path to .ckpt file (Lightning checkpoint)',
    )
    parser.add_argument(
        '--cnmc-dir', required=True,
        help='Root directory of C-NMC 2019 test set (contains all/ and hem/ subdirs)',
    )
    parser.add_argument(
        '--data-dir', default='../dataset',
        help='ALL-IDB training data directory (used to compute reference statistics)',
    )
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--num-workers',  type=int,   default=4,
                        help='DataLoader workers. Set 0 if normalization hangs.')
    parser.add_argument('--image-size',   type=int,   default=224)
    parser.add_argument('--ref-samples',  type=int,   default=100,
                        help='Number of training images used to compute reference stats')
    parser.add_argument('--device',       type=str,   default='auto')
    parser.add_argument('--no-macenko',   action='store_true',
                        help='Skip Macenko normalization (useful for quick tests)')
    parser.add_argument('--no-reinhard',  action='store_true',
                        help='Skip Reinhard normalization')
    parser.add_argument('--skip-val',     action='store_true',
                        help='Skip in-domain ALL-IDB validation')
    parser.add_argument('--output-json',  type=str,   default=None,
                        help='Save all metrics to a JSON file')
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")

    print(f"Loading : {ckpt_path}")
    lightning_model = LeukemiaLightningModel.load_from_checkpoint(
        str(ckpt_path), map_location=device
    )
    model = lightning_model.model
    model.eval().to(device)

    # ── Class mapping from training data ─────────────────────────────────────
    train_folder = tv_datasets.ImageFolder(str(Path(args.data_dir) / 'train'))
    class_to_idx = train_folder.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Classes : {class_to_idx}")

    all_results: Dict = {}

    # ════════════════════════════════════════════════════════════════════════
    # 1.  In-domain validation (ALL-IDB)
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_val:
        val_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        val_dataset = tv_datasets.ImageFolder(
            str(Path(args.data_dir) / 'val'), transform=val_transform
        )
        val_loader = build_loader(val_dataset, args.batch_size, args.num_workers)

        val_preds, val_labels, _ = run_inference(model, val_loader, device, 'ALL-IDB val')
        val_metrics, val_report  = compute_metrics(val_preds, val_labels, class_names)
        print_result('ALL-IDB Val  ·  In-Domain (same staining, same institution)', val_metrics, val_report)
        all_results['in_domain_val'] = val_metrics
    else:
        val_metrics = None

    # ════════════════════════════════════════════════════════════════════════
    # 2.  C-NMC — No normalization  (raw domain shift)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\nDiscovering C-NMC images in: {args.cnmc_dir}")
    cnmc_raw = ExternalTestDataset(
        args.cnmc_dir,
        class_to_idx=class_to_idx,
        normalizer=None,
        image_size=args.image_size,
    )
    # Count per class
    label_counts = {}
    for _, lbl in cnmc_raw.samples:
        label_counts[idx_to_class[lbl]] = label_counts.get(idx_to_class[lbl], 0) + 1
    print(f"Found {len(cnmc_raw)} images  ->  {label_counts}")

    raw_loader = build_loader(cnmc_raw, args.batch_size, args.num_workers)
    raw_preds, raw_labels, _ = run_inference(model, raw_loader, device, 'C-NMC (no norm)')
    raw_metrics, raw_report  = compute_metrics(raw_preds, raw_labels, class_names)
    print_result('C-NMC 2019  ·  No Stain Normalization', raw_metrics, raw_report)
    all_results['cnmc_no_norm'] = raw_metrics

    # ════════════════════════════════════════════════════════════════════════
    # 3 & 4.  Reference statistics from ALL-IDB training set
    # ════════════════════════════════════════════════════════════════════════
    need_ref = not args.no_macenko or not args.no_reinhard
    ref_image = None

    if need_ref:
        print(f"\nComputing stain reference from ALL-IDB training set "
              f"(n={args.ref_samples}) ...")
        try:
            ref_image, rh_mean, rh_std = compute_reference_from_dir(
                str(Path(args.data_dir) / 'train'),
                n_samples=args.ref_samples,
                image_size=args.image_size,
            )
            print(f"  Reference image shape : {ref_image.shape}")
            print(f"  Reinhard LAB mean     : {np.round(rh_mean, 2)}")
            print(f"  Reinhard LAB std      : {np.round(rh_std,  2)}")
        except Exception as exc:
            print(f"  WARNING: Could not compute reference statistics: {exc}")
            ref_image = None

    # ════════════════════════════════════════════════════════════════════════
    # 4.  Macenko normalization
    # ════════════════════════════════════════════════════════════════════════
    if not args.no_macenko and ref_image is not None:
        print(_banner('C-NMC 2019  ·  Macenko Stain Normalization'))
        try:
            macenko = MacenkoNormalizer().fit(ref_image)
            cnmc_mac = ExternalTestDataset(
                args.cnmc_dir,
                class_to_idx=class_to_idx,
                normalizer=macenko,
                image_size=args.image_size,
            )
            # num_workers=0: normalizer objects aren't picklable across worker processes
            mac_loader = build_loader(cnmc_mac, args.batch_size, num_workers=0)
            mac_preds, mac_labels, _ = run_inference(
                model, mac_loader, device, 'C-NMC (Macenko)'
            )
            mac_metrics, mac_report = compute_metrics(mac_preds, mac_labels, class_names)
            print_result('C-NMC 2019  ·  Macenko Normalization', mac_metrics, mac_report)
            all_results['cnmc_macenko'] = mac_metrics
        except Exception as exc:
            print(f"  Macenko normalization failed: {exc}")

    # ════════════════════════════════════════════════════════════════════════
    # 5.  Reinhard normalization
    # ════════════════════════════════════════════════════════════════════════
    if not args.no_reinhard and ref_image is not None:
        print(_banner('C-NMC 2019  ·  Reinhard Stain Normalization'))
        try:
            reinhard = ReinhardNormalizer().fit_from_stats(rh_mean, rh_std)
            cnmc_rh = ExternalTestDataset(
                args.cnmc_dir,
                class_to_idx=class_to_idx,
                normalizer=reinhard,
                image_size=args.image_size,
            )
            rh_loader = build_loader(cnmc_rh, args.batch_size, num_workers=0)
            rh_preds, rh_labels, _ = run_inference(
                model, rh_loader, device, 'C-NMC (Reinhard)'
            )
            rh_metrics, rh_report = compute_metrics(rh_preds, rh_labels, class_names)
            print_result('C-NMC 2019  ·  Reinhard Normalization', rh_metrics, rh_report)
            all_results['cnmc_reinhard'] = rh_metrics
        except Exception as exc:
            print(f"  Reinhard normalization failed: {exc}")

    # ════════════════════════════════════════════════════════════════════════
    # Summary table
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    print(f"  {'Condition':<40}  {'Acc':>6}  {'F1':>6}")
    print(f"  {'─'*40}  {'─'*6}  {'─'*6}")

    def _row(name, m):
        print(f"  {name:<40}  {m['accuracy']:.4f}  {m['f1_macro']:.4f}")

    if not args.skip_val:
        _row('ALL-IDB val (in-domain)', val_metrics)
    _row('C-NMC — no normalization', raw_metrics)
    if 'cnmc_macenko'  in all_results: _row('C-NMC — Macenko',  all_results['cnmc_macenko'])
    if 'cnmc_reinhard' in all_results: _row('C-NMC — Reinhard', all_results['cnmc_reinhard'])

    if not args.skip_val and val_metrics:
        gap = val_metrics['accuracy'] - raw_metrics['accuracy']
        print(f"\n  Domain-shift gap (val_acc − raw_acc) : {gap:+.4f}")
        if gap > 0.10:
            print(" Large gap — model likely relies on staining artefacts,")
            print("     not just morphology.  Normalization should reduce the gap.")
        elif gap > 0.03:
            print(" Moderate gap — partial staining dependency.")
        else:
            print(" Small gap — model generalises well across staining styles.")
        all_results['domain_shift_gap'] = float(gap)

    # ── Save JSON ──────────────────────────────────────────────────────────
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved → {out_path}")


if __name__ == '__main__':
    main()
