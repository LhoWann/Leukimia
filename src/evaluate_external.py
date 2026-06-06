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
    import numpy._core.multiarray
    import numpy.dtypes
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


def find_best_ckpt(exp_dir: Path) -> Optional[Path]:
    candidates = [p for p in exp_dir.glob('*.ckpt') if p.name != 'last.ckpt']
    if not candidates:
        return None

    def _val_f1(p: Path) -> float:
        m = re.search(r'val_f1=([0-9]+(?:\.[0-9]+)?)', p.stem)
        return float(m.group(1)) if m else 0.0

    return max(candidates, key=_val_f1)


def discover_experiments(ckpt_root: Path) -> List[Tuple[str, Path]]:
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


def _tta_views(images: torch.Tensor) -> List[torch.Tensor]:
    rot90 = torch.rot90(images, k=1, dims=(-2, -1))
    return [
        images,
        torch.flip(images, dims=[-1]),
        torch.flip(images, dims=[-2]),
        rot90,
        torch.rot90(images, k=2, dims=(-2, -1)),
        torch.rot90(images, k=3, dims=(-2, -1)),
        torch.flip(rot90, dims=[-1]),
        torch.flip(rot90, dims=[-2]),
    ]


@torch.no_grad()
def run_inference_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_tta: int = 8,
    desc: str = 'Evaluating (TTA)',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    n_views = max(1, min(n_tta, 8))

    all_preds, all_labels, all_probs = [], [], []
    for images, labels in tqdm(loader, desc=desc, ncols=80):
        images   = images.to(device, non_blocking=True)
        prob_sum = None
        for view in _tta_views(images)[:n_views]:
            logits = model(view)
            probs  = F.softmax(logits, dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs
        avg_probs = (prob_sum / n_views).cpu().numpy()
        all_preds.extend(avg_probs.argmax(axis=1))
        all_labels.extend(labels.numpy())
        all_probs.extend(avg_probs)
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


def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        preds = (probs[:, 1] >= thr).astype(int)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


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
    n_tta: int = 1,
) -> Dict:
    tta_tag = f'  [TTA×{n_tta}]' if n_tta > 1 else ''
    print(f"\n{'═' * 60}")
    print(f"  Experiment : {exp_name}{tta_tag}")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"{'═' * 60}")

    lightning_model = LeukemiaLightningModel.load_from_checkpoint(
        str(ckpt_path), map_location=device
    )
    model = lightning_model.model
    model.eval().to(device)

    def _infer(loader, desc):
        if n_tta > 1:
            return run_inference_tta(model, loader, device, n_tta, desc)
        return run_inference(model, loader, device, desc)

    results: Dict = {'checkpoint': str(ckpt_path), 'n_tta': n_tta}

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

    cnmc_raw = ExternalTestDataset(
        cnmc_dir, class_to_idx=class_to_idx, normalizer=None, image_size=image_size
    )
    label_counts = {}
    for _, lbl in cnmc_raw.samples:
        label_counts[idx_to_class[lbl]] = label_counts.get(idx_to_class[lbl], 0) + 1
    print(f"\nC-NMC images: {len(cnmc_raw)}  →  {label_counts}")

    raw_loader  = build_loader(cnmc_raw, batch_size, num_workers)
    raw_preds, raw_labels, raw_probs = _infer(raw_loader, 'C-NMC (no norm)')
    raw_metrics, raw_report  = compute_metrics(raw_preds, raw_labels, class_names)
    print_result('C-NMC 2019  ·  No Stain Normalization', raw_metrics, raw_report)
    results['cnmc_no_norm'] = raw_metrics

    _cond_probs: Dict[str, np.ndarray] = {'cnmc_no_norm': raw_probs}

    if not no_macenko and ref_image is not None:
        print(_banner('C-NMC 2019  ·  Macenko Stain Normalization'))
        try:
            macenko = MacenkoNormalizer().fit(ref_image)
            cnmc_mac = ExternalTestDataset(
                cnmc_dir, class_to_idx=class_to_idx,
                normalizer=macenko, image_size=image_size,
            )
            mac_loader = build_loader(cnmc_mac, batch_size, num_workers=0)
            mac_preds, mac_labels, mac_probs = _infer(mac_loader, 'C-NMC (Macenko)')
            mac_metrics, mac_report  = compute_metrics(mac_preds, mac_labels, class_names)
            print_result('C-NMC 2019  ·  Macenko Normalization', mac_metrics, mac_report)
            results['cnmc_macenko'] = mac_metrics
            _cond_probs['cnmc_macenko'] = mac_probs
        except Exception as exc:
            print(f"  Macenko normalization failed: {exc}")

    if not no_reinhard and ref_image is not None and rh_mean is not None:
        print(_banner('C-NMC 2019  ·  Reinhard Stain Normalization'))
        try:
            reinhard = ReinhardNormalizer().fit_from_stats(rh_mean, rh_std)
            cnmc_rh = ExternalTestDataset(
                cnmc_dir, class_to_idx=class_to_idx,
                normalizer=reinhard, image_size=image_size,
            )
            rh_loader = build_loader(cnmc_rh, batch_size, num_workers=0)
            rh_preds, rh_labels, rh_probs = _infer(rh_loader, 'C-NMC (Reinhard)')
            rh_metrics, rh_report  = compute_metrics(rh_preds, rh_labels, class_names)
            print_result('C-NMC 2019  ·  Reinhard Normalization', rh_metrics, rh_report)
            results['cnmc_reinhard'] = rh_metrics
            _cond_probs['cnmc_reinhard'] = rh_probs
        except Exception as exc:
            print(f"  Reinhard normalization failed: {exc}")

    opt_thr = find_optimal_threshold(raw_probs, raw_labels)
    results['optimal_threshold'] = opt_thr
    for key, probs in _cond_probs.items():
        cal_preds = (probs[:, 1] >= opt_thr).astype(int)
        cal_m, _  = compute_metrics(cal_preds, raw_labels, class_names)
        results[f'{key}_calibrated'] = {**cal_m, 'threshold': opt_thr}

    print(f"\n  {'Condition':<40}  {'Acc':>6}  {'F1':>6}  {'F1-cal':>7}")
    print(f"  {'─'*40}  {'─'*6}  {'─'*6}  {'─'*7}")
    if not skip_val and val_metrics:
        print(f"  {'ALL-IDB val (in-domain)':<40}  {val_metrics['accuracy']:.4f}  {val_metrics['f1_macro']:.4f}       —")
    _fmt_cal = lambda k: f"{results[k+'_calibrated']['f1_macro']:.4f}" if k+'_calibrated' in results else '     —'
    print(f"  {'C-NMC — no normalization':<40}  {raw_metrics['accuracy']:.4f}  {raw_metrics['f1_macro']:.4f}  {_fmt_cal('cnmc_no_norm')}")
    if 'cnmc_macenko'  in results:
        m = results['cnmc_macenko']
        print(f"  {'C-NMC — Macenko':<40}  {m['accuracy']:.4f}  {m['f1_macro']:.4f}  {_fmt_cal('cnmc_macenko')}")
    if 'cnmc_reinhard' in results:
        m = results['cnmc_reinhard']
        print(f"  {'C-NMC — Reinhard':<40}  {m['accuracy']:.4f}  {m['f1_macro']:.4f}  {_fmt_cal('cnmc_reinhard')}")
    print(f"\n  Optimal threshold (calibrated on no-norm, class=Normal): {opt_thr:.2f}")

    if not skip_val and val_metrics:
        gap = val_metrics['accuracy'] - raw_metrics['accuracy']
        print(f"  Domain-shift gap (val_acc − raw_acc) : {gap:+.4f}")
        results['domain_shift_gap'] = float(gap)

    del model, lightning_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return results


def evaluate_ensemble(
    exp_names: List[str],
    ckpt_root: Path,
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
    image_size: int,
    skip_val: bool,
    no_macenko: bool,
    no_reinhard: bool,
    n_tta: int = 1,
) -> Dict:
    ckpts: List[Tuple[str, Path]] = []
    for name in exp_names:
        exp_dir = ckpt_root / name
        if not exp_dir.exists():
            print(f"  [ensemble skip] {name}: directory not found")
            continue
        ckpt = find_best_ckpt(exp_dir)
        if ckpt is None:
            print(f"  [ensemble skip] {name}: no checkpoint")
            continue
        ckpts.append((name, ckpt))

    if not ckpts:
        return {'error': 'no valid checkpoints for ensemble'}

    tta_tag = f'  [TTA×{n_tta}]' if n_tta > 1 else ''
    print(f"\n{'═' * 60}")
    print(f"  ENSEMBLE ({len(ckpts)} models){tta_tag}")
    for name, ckpt in ckpts:
        print(f"    {name:<30}  {ckpt.name}")
    print(f"{'═' * 60}")

    Condition = Tuple[str, str, object]
    conditions: List[Condition] = [('no_norm', 'No Stain Normalization', None)]
    if not no_macenko and ref_image is not None:
        try:
            conditions.append(('macenko', 'Macenko Normalization', MacenkoNormalizer().fit(ref_image)))
        except Exception as exc:
            print(f"  Macenko setup failed: {exc}")
    if not no_reinhard and rh_mean is not None:
        try:
            conditions.append(('reinhard', 'Reinhard Normalization', ReinhardNormalizer().fit_from_stats(rh_mean, rh_std)))
        except Exception as exc:
            print(f"  Reinhard setup failed: {exc}")

    cnmc_datasets = {
        key: ExternalTestDataset(cnmc_dir, class_to_idx, norm, image_size)
        for key, _, norm in conditions
    }
    true_labels  = np.array([lbl for _, lbl in cnmc_datasets['no_norm'].samples])
    label_counts = {}
    for lbl in true_labels:
        label_counts[idx_to_class[lbl]] = label_counts.get(idx_to_class[lbl], 0) + 1
    print(f"\nC-NMC images: {len(true_labels)}  →  {label_counts}")

    acc_probs: Dict[str, Optional[np.ndarray]] = {key: None for key, _, _ in conditions}
    val_accs:  List[float] = []
    val_f1s:   List[float] = []

    for exp_name, ckpt_path in ckpts:
        print(f"\n  — {exp_name}  ({ckpt_path.name})")
        lm    = LeukemiaLightningModel.load_from_checkpoint(str(ckpt_path), map_location=device)
        model = lm.model.eval().to(device)

        if not skip_val:
            val_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            val_ds     = tv_datasets.ImageFolder(str(Path(data_dir) / 'val'), transform=val_transform)
            val_loader = build_loader(val_ds, batch_size, num_workers=0)
            vp, vl, _  = run_inference(model, val_loader, device, f'    val ({exp_name})')
            vm, _      = compute_metrics(vp, vl, class_names)
            val_accs.append(vm['accuracy'])
            val_f1s.append(vm['f1_macro'])
            print(f"    val_acc={vm['accuracy']:.4f}  val_f1={vm['f1_macro']:.4f}")

        for key, _, _ in conditions:
            loader = build_loader(cnmc_datasets[key], batch_size, num_workers=0)
            desc   = f'    {exp_name}/{key}'
            if n_tta > 1:
                _, _, probs = run_inference_tta(model, loader, device, n_tta, desc)
            else:
                _, _, probs = run_inference(model, loader, device, desc)
            acc_probs[key] = probs if acc_probs[key] is None else acc_probs[key] + probs

        del model, lm
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    n_models = len(ckpts)
    results: Dict = {'ensemble_members': [n for n, _ in ckpts], 'n_tta': n_tta}
    raw_acc: Optional[float] = None
    _ens_avg_probs: Dict[str, np.ndarray] = {}

    for key, display_name, _ in conditions:
        avg_probs       = acc_probs[key] / n_models
        preds           = avg_probs.argmax(axis=1)
        metrics, report = compute_metrics(preds, true_labels, class_names)
        results[f'cnmc_{key}'] = metrics
        _ens_avg_probs[key]    = avg_probs
        print_result(f'Ensemble  ·  C-NMC 2019  ·  {display_name}', metrics, report)
        if key == 'no_norm':
            raw_acc = metrics['accuracy']

    if not skip_val and val_accs:
        avg_val_acc = float(np.mean(val_accs))
        avg_val_f1  = float(np.mean(val_f1s))
        results['in_domain_val'] = {
            'accuracy': avg_val_acc, 'f1_macro': avg_val_f1,
            'note': 'mean across ensemble members',
        }
        if raw_acc is not None:
            gap = avg_val_acc - raw_acc
            results['domain_shift_gap'] = float(gap)

    if 'no_norm' in _ens_avg_probs:
        opt_thr = find_optimal_threshold(_ens_avg_probs['no_norm'], true_labels)
        results['optimal_threshold'] = opt_thr
        for key in _ens_avg_probs:
            cal_preds = (_ens_avg_probs[key][:, 1] >= opt_thr).astype(int)
            cal_m, _  = compute_metrics(cal_preds, true_labels, class_names)
            results[f'cnmc_{key}_calibrated'] = {**cal_m, 'threshold': opt_thr}
    else:
        opt_thr = 0.5

    print(f"\n  {'Condition':<40}  {'Acc':>6}  {'F1':>6}  {'F1-cal':>7}")
    print(f"  {'─'*40}  {'─'*6}  {'─'*6}  {'─'*7}")
    if not skip_val and val_accs:
        print(f"  {'ALL-IDB val (avg across members)':<40}  {results['in_domain_val']['accuracy']:.4f}  {results['in_domain_val']['f1_macro']:.4f}       —")
    _fmt_cal = lambda k: f"{results['cnmc_'+k+'_calibrated']['f1_macro']:.4f}" if 'cnmc_'+k+'_calibrated' in results else '     —'
    for key, display_name, _ in conditions:
        m = results[f'cnmc_{key}']
        print(f"  {display_name:<40}  {m['accuracy']:.4f}  {m['f1_macro']:.4f}  {_fmt_cal(key)}")
    print(f"\n  Optimal threshold (calibrated on no-norm, class=Normal): {opt_thr:.2f}")
    if 'domain_shift_gap' in results:
        print(f"  Domain-shift gap (val_acc − raw_acc) : {results['domain_shift_gap']:+.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='External evaluation on C-NMC 2019.'
    )
    parser.add_argument('--ckpt', default=None,
                        help='Path to a single .ckpt file. Omit to auto-evaluate all experiments.')
    parser.add_argument('--ckpt-dir', default='checkpoints',
                        help='Root folder containing per-experiment checkpoint subdirectories.')
    parser.add_argument('--cnmc-dir', required=True,
                        help='Root directory of C-NMC 2019 (contains all/ and hem/ subdirs).')
    parser.add_argument('--data-dir', default='dataset',
                        help='ALL-IDB training data directory.')
    parser.add_argument('--batch-size',   type=int, default=32)
    parser.add_argument('--num-workers',  type=int, default=2)
    parser.add_argument('--image-size',   type=int, default=224)
    parser.add_argument('--ref-samples',  type=int, default=100)
    parser.add_argument('--device',       type=str, default='auto')
    parser.add_argument('--no-macenko',   action='store_true')
    parser.add_argument('--no-reinhard',  action='store_true')
    parser.add_argument('--skip-val',     action='store_true')
    parser.add_argument('--output-json',  type=str, default=None)
    parser.add_argument('--results-dir',  type=str, default='results')
    parser.add_argument('--tta-n', type=int, default=1, metavar='N',
                        help='Test-Time Augmentation passes (1 = disabled, 8 = recommended)')
    parser.add_argument('--ensemble', nargs='+', default=None, metavar='EXP',
                        help='Evaluate ensemble of named experiments.')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    train_folder  = tv_datasets.ImageFolder(str(Path(args.data_dir) / 'train'))
    class_to_idx  = train_folder.class_to_idx
    idx_to_class  = {v: k for k, v in class_to_idx.items()}
    class_names   = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Classes : {class_to_idx}")

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

    shared_kwargs = dict(
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
        image_size  = args.image_size,
        skip_val    = args.skip_val,
        no_macenko  = args.no_macenko,
        no_reinhard = args.no_reinhard,
        n_tta       = args.tta_n,
    )

    if args.ensemble:
        ens_result = evaluate_ensemble(
            exp_names=args.ensemble,
            ckpt_root=Path(args.ckpt_dir),
            **shared_kwargs,
        )
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        ens_label = '+'.join(args.ensemble)
        out_path  = results_dir / f'ensemble_{ens_label}.json'
        with open(out_path, 'w') as f:
            json.dump(ens_result, f, indent=2)
        print(f"\n  Saved → {out_path}")
        return

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")
        experiments = [(ckpt_path.parent.name, ckpt_path)]
    else:
        ckpt_root = Path(args.ckpt_dir)
        if not ckpt_root.exists():
            sys.exit(f"ERROR: checkpoint directory not found: {ckpt_root}")
        print(f"\nAuto-scanning: {ckpt_root.resolve()}")
        experiments = discover_experiments(ckpt_root)
        if not experiments:
            sys.exit("No experiments with checkpoints found. Train some models first.")
        print(f"Found {len(experiments)} experiment(s): {[e for e, _ in experiments]}\n")

    all_exp_results: Dict[str, Dict] = {}
    eval_kwargs = dict(**shared_kwargs, num_workers=args.num_workers)

    for exp_name, ckpt_path in experiments:
        try:
            results = evaluate_checkpoint(exp_name, ckpt_path, **eval_kwargs)
            all_exp_results[exp_name] = results
        except Exception as exc:
            print(f"\n  ERROR evaluating {exp_name}: {exc}")
            all_exp_results[exp_name] = {'error': str(exc)}

    if args.ckpt and args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_exp_results[experiments[0][0]], f, indent=2)
        print(f"\n  Results saved → {out_path}")
    else:
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

        def _fmt(v): return f"{v:.4f}" if not (v != v) else "  n/a "

        print(f"  {exp_name:<22}  {_fmt(val_acc):>7}  {_fmt(raw_acc):>7}  {_fmt(mac_acc):>7}  {_fmt(rh_acc):>8}  {gap:+.4f}" if gap == gap else
              f"  {exp_name:<22}  {_fmt(val_acc):>7}  {_fmt(raw_acc):>7}  {_fmt(mac_acc):>7}  {_fmt(rh_acc):>8}  {'n/a':>6}")

    print()


if __name__ == '__main__':
    main()
