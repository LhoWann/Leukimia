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

Usage — single model:
    cd src
    python evaluate_external.py \\
        --ckpt checkpoints/baseline/epoch=02-val_f1=1.0000.ckpt \\
        --cnmc-dir /path/to/C-NMC_2019/test \\
        --data-dir dataset

Usage — auto (evaluates ALL experiments found in checkpoints/):
    cd src
    python evaluate_external.py \\
        --cnmc-dir /path/to/C-NMC_2019/test \\
        --data-dir dataset
"""
from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", message="triton not found.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.getLogger('torch.utils.flop_counter').setLevel(logging.ERROR)


def _worker_init_fn(worker_id: int) -> None:
    import logging as _logging
    _logging.getLogger('torch.utils.flop_counter').setLevel(_logging.ERROR)


import argparse
import json
import re
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

import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    try:
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import numpy._core.multiarray  # noqa: F401
    import numpy.dtypes            # noqa: F401
    _safe = [numpy._core.multiarray.scalar, numpy.dtype]
    _safe += [getattr(numpy.dtypes, n) for n in dir(numpy.dtypes)
              if isinstance(getattr(numpy.dtypes, n), type)]
    torch.serialization.add_safe_globals(_safe)
except Exception:
    pass

from stain_normalize import (
    MacenkoNormalizer,
    ReinhardNormalizer,
    compute_reference_from_dir,
)
from lightning_model import LeukemiaLightningModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CNMC_CLASS_MAP: Dict[str, str] = {
    'all':      'Abnormal',
    'ALL':      'Abnormal',
    'hem':      'Normal',
    'HEM':      'Normal',
    'Abnormal': 'Abnormal',
    'Normal':   'Normal',
    'positive': 'Abnormal',
    'negative': 'Normal',
}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_best_ckpt(exp_dir: Path) -> Optional[Path]:
    """
    Return the best (highest val_f1) non-last checkpoint in exp_dir.
    Falls back to any non-last .ckpt if the filename has no val_f1 score.
    Returns None if no checkpoint exists.
    """
    candidates = [p for p in exp_dir.glob('*.ckpt') if p.name != 'last.ckpt']
    if not candidates:
        return None

    def _val_f1(p: Path) -> float:
        m = re.search(r'val_f1=([0-9]+(?:\.[0-9]+)?)', p.stem)
        return float(m.group(1)) if m else 0.0

    return max(candidates, key=_val_f1)


def discover_experiments(ckpt_root: Path) -> List[Tuple[str, Path]]:
    """
    Scan ckpt_root for experiment sub-directories that contain a best checkpoint.
    Returns list of (exp_name, ckpt_path) sorted by exp_name.
    """
    found = []
    for exp_dir in sorted(ckpt_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        ckpt = find_best_ckpt(exp_dir)
        if ckpt is None:
            print(f"  [skip] {exp_dir.name}/ — no best checkpoint found")
            continue
        found.append((exp_dir.name, ckpt))
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ExternalTestDataset(Dataset):
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
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
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


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation for one checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    exp_name: str,
    ckpt_path: Path,
    cnmc_dir: str,
    data_dir: str,
    device: torch.device,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    class_names: List[str],
    ref_image: Optional[np.ndarray],
    rh_mean: Optional[np.ndarray],
    rh_std: Optional[np.ndarray],
    batch_size: int,
    num_workers: int,
    image_size: int,
    skip_val: bool,
    no_macenko: bool,
    no_reinhard: bool,
) -> Dict:
    """
    Run all evaluation conditions for one checkpoint.
    Returns the full results dict (same structure as before, keyed by condition).
    """
    print(f"\n{'═' * 60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"{'═' * 60}")

    lightning_model = LeukemiaLightningModel.load_from_checkpoint(
        str(ckpt_path), map_location=device
    )
    model = lightning_model.model
    model.eval().to(device)

    results: Dict = {'checkpoint': str(ckpt_path)}

    # ── 1. In-domain validation ───────────────────────────────────────────────
    if not skip_val:
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        val_dataset = tv_datasets.ImageFolder(
            str(Path(data_dir) / 'val'), transform=val_transform
        )
        val_loader = build_loader(val_dataset, batch_size, num_workers)
        val_preds, val_labels, _ = run_inference(model, val_loader, device, 'ALL-IDB val')
        val_metrics, val_report  = compute_metrics(val_preds, val_labels, class_names)
        print_result('ALL-IDB Val  ·  In-Domain', val_metrics, val_report)
        results['in_domain_val'] = val_metrics
    else:
        val_metrics = None

    # ── 2. C-NMC — no normalization ───────────────────────────────────────────
    cnmc_raw = ExternalTestDataset(
        cnmc_dir, class_to_idx=class_to_idx, normalizer=None, image_size=image_size
    )
    label_counts = {}
    for _, lbl in cnmc_raw.samples:
        label_counts[idx_to_class[lbl]] = label_counts.get(idx_to_class[lbl], 0) + 1
    print(f"\nC-NMC images: {len(cnmc_raw)}  →  {label_counts}")

    raw_loader  = build_loader(cnmc_raw, batch_size, num_workers)
    raw_preds, raw_labels, _ = run_inference(model, raw_loader, device, 'C-NMC (no norm)')
    raw_metrics, raw_report  = compute_metrics(raw_preds, raw_labels, class_names)
    print_result('C-NMC 2019  ·  No Stain Normalization', raw_metrics, raw_report)
    results['cnmc_no_norm'] = raw_metrics

    # ── 3. Macenko ────────────────────────────────────────────────────────────
    if not no_macenko and ref_image is not None:
        print(_banner('C-NMC 2019  ·  Macenko Stain Normalization'))
        try:
            macenko = MacenkoNormalizer().fit(ref_image)
            cnmc_mac = ExternalTestDataset(
                cnmc_dir, class_to_idx=class_to_idx,
                normalizer=macenko, image_size=image_size,
            )
            mac_loader = build_loader(cnmc_mac, batch_size, num_workers=0)
            mac_preds, mac_labels, _ = run_inference(model, mac_loader, device, 'C-NMC (Macenko)')
            mac_metrics, mac_report  = compute_metrics(mac_preds, mac_labels, class_names)
            print_result('C-NMC 2019  ·  Macenko Normalization', mac_metrics, mac_report)
            results['cnmc_macenko'] = mac_metrics
        except Exception as exc:
            print(f"  Macenko normalization failed: {exc}")

    # ── 4. Reinhard ───────────────────────────────────────────────────────────
    if not no_reinhard and ref_image is not None and rh_mean is not None:
        print(_banner('C-NMC 2019  ·  Reinhard Stain Normalization'))
        try:
            reinhard = ReinhardNormalizer().fit_from_stats(rh_mean, rh_std)
            cnmc_rh = ExternalTestDataset(
                cnmc_dir, class_to_idx=class_to_idx,
                normalizer=reinhard, image_size=image_size,
            )
            rh_loader = build_loader(cnmc_rh, batch_size, num_workers=0)
            rh_preds, rh_labels, _ = run_inference(model, rh_loader, device, 'C-NMC (Reinhard)')
            rh_metrics, rh_report  = compute_metrics(rh_preds, rh_labels, class_names)
            print_result('C-NMC 2019  ·  Reinhard Normalization', rh_metrics, rh_report)
            results['cnmc_reinhard'] = rh_metrics
        except Exception as exc:
            print(f"  Reinhard normalization failed: {exc}")

    # ── Per-experiment summary ────────────────────────────────────────────────
    print(f"\n  {'Condition':<40}  {'Acc':>6}  {'F1':>6}")
    print(f"  {'─'*40}  {'─'*6}  {'─'*6}")
    if not skip_val and val_metrics:
        print(f"  {'ALL-IDB val (in-domain)':<40}  {val_metrics['accuracy']:.4f}  {val_metrics['f1_macro']:.4f}")
    print(f"  {'C-NMC — no normalization':<40}  {raw_metrics['accuracy']:.4f}  {raw_metrics['f1_macro']:.4f}")
    if 'cnmc_macenko'  in results:
        m = results['cnmc_macenko']
        print(f"  {'C-NMC — Macenko':<40}  {m['accuracy']:.4f}  {m['f1_macro']:.4f}")
    if 'cnmc_reinhard' in results:
        m = results['cnmc_reinhard']
        print(f"  {'C-NMC — Reinhard':<40}  {m['accuracy']:.4f}  {m['f1_macro']:.4f}")

    if not skip_val and val_metrics:
        gap = val_metrics['accuracy'] - raw_metrics['accuracy']
        print(f"\n  Domain-shift gap (val_acc − raw_acc) : {gap:+.4f}")
        results['domain_shift_gap'] = float(gap)

    # Free GPU memory before loading next model
    del model, lightning_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'External evaluation on C-NMC 2019.\n'
            'If --ckpt is omitted, all experiments found in --ckpt-dir are evaluated automatically.'
        )
    )
    parser.add_argument(
        '--ckpt', default=None,
        help='Path to a single .ckpt file. Omit to auto-evaluate all experiments.',
    )
    parser.add_argument(
        '--ckpt-dir', default='checkpoints',
        help='Root folder containing per-experiment checkpoint subdirectories (default: checkpoints)',
    )
    parser.add_argument(
        '--cnmc-dir', required=True,
        help='Root directory of C-NMC 2019 test set (contains all/ and hem/ subdirs)',
    )
    parser.add_argument(
        '--data-dir', default='dataset',
        help='ALL-IDB training data directory (used for in-domain val + stain reference)',
    )
    parser.add_argument('--batch-size',   type=int, default=32)
    parser.add_argument('--num-workers',  type=int, default=2,
                        help='DataLoader workers (set 0 if normalization hangs)')
    parser.add_argument('--image-size',   type=int, default=224)
    parser.add_argument('--ref-samples',  type=int, default=100,
                        help='Training images used to compute stain reference stats')
    parser.add_argument('--device',       type=str, default='auto')
    parser.add_argument('--no-macenko',   action='store_true')
    parser.add_argument('--no-reinhard',  action='store_true')
    parser.add_argument('--skip-val',     action='store_true',
                        help='Skip in-domain ALL-IDB validation')
    # Single-model compat
    parser.add_argument('--output-json',  type=str, default=None,
                        help='[single-model mode] Save results to this JSON path')
    # Auto mode
    parser.add_argument('--results-dir',  type=str, default='results',
                        help='[auto mode] Directory to write per-experiment JSON files')
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    # ── Class mapping (shared across all experiments) ─────────────────────────
    train_folder  = tv_datasets.ImageFolder(str(Path(args.data_dir) / 'train'))
    class_to_idx  = train_folder.class_to_idx
    idx_to_class  = {v: k for k, v in class_to_idx.items()}
    class_names   = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Classes : {class_to_idx}")

    # ── Stain reference (computed once, reused for all models) ────────────────
    need_ref = not args.no_macenko or not args.no_reinhard
    ref_image = rh_mean = rh_std = None

    if need_ref:
        print(f"\nComputing stain reference from ALL-IDB training set (n={args.ref_samples}) ...")
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

    # ── Build experiment list ─────────────────────────────────────────────────
    if args.ckpt:
        # Single-model mode (backward compat)
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")
        exp_name = ckpt_path.parent.name
        experiments = [(exp_name, ckpt_path)]
    else:
        # Auto mode — scan all sub-directories
        ckpt_root = Path(args.ckpt_dir)
        if not ckpt_root.exists():
            sys.exit(f"ERROR: checkpoint directory not found: {ckpt_root}")
        print(f"\nAuto-scanning: {ckpt_root.resolve()}")
        experiments = discover_experiments(ckpt_root)
        if not experiments:
            sys.exit("No experiments with checkpoints found. Train some models first.")
        print(f"Found {len(experiments)} experiment(s): {[e for e, _ in experiments]}\n")

    # ── Evaluate each experiment ──────────────────────────────────────────────
    all_exp_results: Dict[str, Dict] = {}
    eval_kwargs = dict(
        cnmc_dir    = args.cnmc_dir,
        data_dir    = args.data_dir,
        device      = device,
        class_to_idx= class_to_idx,
        idx_to_class= idx_to_class,
        class_names = class_names,
        ref_image   = ref_image,
        rh_mean     = rh_mean,
        rh_std      = rh_std,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        image_size  = args.image_size,
        skip_val    = args.skip_val,
        no_macenko  = args.no_macenko,
        no_reinhard = args.no_reinhard,
    )

    for exp_name, ckpt_path in experiments:
        try:
            results = evaluate_checkpoint(exp_name, ckpt_path, **eval_kwargs)
            all_exp_results[exp_name] = results
        except Exception as exc:
            print(f"\n  ERROR evaluating {exp_name}: {exc}")
            all_exp_results[exp_name] = {'error': str(exc)}

    # ── Save results ──────────────────────────────────────────────────────────
    if args.ckpt and args.output_json:
        # Single-model mode: honour --output-json
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_exp_results[experiments[0][0]], f, indent=2)
        print(f"\n  Results saved → {out_path}")
    else:
        # Auto mode: one JSON per experiment + combined summary
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        for exp_name, results in all_exp_results.items():
            out_path = results_dir / f"{exp_name}.json"
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved → {out_path}")

        summary_path = results_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_exp_results, f, indent=2)
        print(f"  Saved → {summary_path}")

    # ── Combined summary table (all experiments) ──────────────────────────────
    print(f"\n\n{'═' * 76}")
    print("  COMBINED SUMMARY")
    print(f"{'═' * 76}")
    print(f"  {'Experiment':<22}  {'Val Acc':>7}  {'No-Norm':>7}  {'Macenko':>7}  {'Reinhard':>8}  {'Gap':>6}")
    print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*6}")

    for exp_name, res in all_exp_results.items():
        if 'error' in res:
            print(f"  {exp_name:<22}  ERROR: {res['error']}")
            continue
        val_acc  = res.get('in_domain_val',  {}).get('accuracy', float('nan'))
        raw_acc  = res.get('cnmc_no_norm',   {}).get('accuracy', float('nan'))
        mac_acc  = res.get('cnmc_macenko',   {}).get('accuracy', float('nan'))
        rh_acc   = res.get('cnmc_reinhard',  {}).get('accuracy', float('nan'))
        gap      = res.get('domain_shift_gap', float('nan'))

        def _fmt(v): return f"{v:.4f}" if not (v != v) else "  n/a "  # nan check

        print(f"  {exp_name:<22}  {_fmt(val_acc):>7}  {_fmt(raw_acc):>7}  {_fmt(mac_acc):>7}  {_fmt(rh_acc):>8}  {gap:+.4f}" if gap == gap else
              f"  {exp_name:<22}  {_fmt(val_acc):>7}  {_fmt(raw_acc):>7}  {_fmt(mac_acc):>7}  {_fmt(rh_acc):>8}  {'n/a':>6}")

    print()


if __name__ == '__main__':
    main()
