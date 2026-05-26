import os
import random
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import lightning as L

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Saliency (Spectral Residual, native resolution)

def compute_saliency_map(image_np: np.ndarray) -> np.ndarray:
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0

    fft = np.fft.fft2(gray)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    spectral_residual = log_amp - cv2.blur(log_amp, (3, 3))

    sal = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal, (9, 9), 2.5)

    s_min, s_max = sal.min(), sal.max()
    if s_max - s_min > 1e-8:
        sal = (sal - s_min) / (s_max - s_min)
    else:
        # Uniform image → zero saliency signal, not ones
        sal = np.zeros_like(sal)
    return sal.astype(np.float32)


# FocusAugMix variants

def focus_aug_mix(
    image_a_np: np.ndarray,           # target (HxWxC uint8)
    image_b_np: np.ndarray,           # source (HxWxC uint8)
    gradcam_map: Optional[np.ndarray] = None,
    use_saliency: bool = True,
    n_segments: int = 50,             # paper uses 50, not 100
    compactness: float = 10.0,
    paste_ratio: float = 0.25,        # fraction of segments to paste
    saliency_weight: float = 0.6,
) -> Tuple[np.ndarray, float]:
    """
    OcCaMix-style superpixel paste guided by saliency and/or Grad-CAM.

    Returns:
        mixed: HxWxC uint8 — image_a with top superpixels from image_b pasted
        lam:   fraction of pixels REMAINING from image_a (used for label mixing)

    Mechanics:
      1. SLIC on target (image_a) → defines paste regions following its contours
      2. Saliency on source (image_b) → ranks which source content is informative
      3. Top-K superpixels of target (by source saliency overlap) get replaced

    NOTE: This deviates slightly from the paper's description but follows the
    intent — paste informative SOURCE content into TARGET while respecting
    TARGET cell contours. Reversing gives unstable training (cell from B placed
    inside A's contour can break morphology).
    """
    h, w = image_a_np.shape[:2]
    if image_b_np.shape[:2] != (h, w):
        image_b_np = cv2.resize(image_b_np, (w, h), interpolation=cv2.INTER_LINEAR)

    # SLIC on TARGET (preserve target contours)
    segments = slic(
        image_a_np, n_segments=n_segments, compactness=compactness,
        start_label=0, channel_axis=2,
    )

    # Build score map: saliency from source + optional gradcam
    if use_saliency:
        score_map = compute_saliency_map(image_b_np)
    else:
        score_map = np.zeros((h, w), dtype=np.float32)

    if gradcam_map is not None:
        if gradcam_map.shape != (h, w):
            gradcam_map = cv2.resize(gradcam_map, (w, h), interpolation=cv2.INTER_LINEAR)
        if use_saliency:
            score_map = saliency_weight * score_map + (1 - saliency_weight) * gradcam_map
        else:
            score_map = gradcam_map.astype(np.float32)

    # Rank target superpixels by mean score (high score = high source info there)
    seg_ids = np.unique(segments)
    seg_scores = np.array([score_map[segments == s].mean() for s in seg_ids])
    order = np.argsort(seg_scores)[::-1]

    num_paste = max(1, int(len(seg_ids) * paste_ratio))
    paste_ids = seg_ids[order[:num_paste]]

    # Build paste mask
    paste_mask = np.isin(segments, paste_ids)
    mixed = image_a_np.copy()
    mixed[paste_mask] = image_b_np[paste_mask]

    # Lambda = fraction of pixels remaining from A (target)
    lam = 1.0 - paste_mask.mean()
    return mixed, float(lam)


def saliency_mix(
    image_a_np: np.ndarray,
    image_b_np: np.ndarray,
    patch_ratio: float = 0.25,
) -> Tuple[np.ndarray, float]:
    """
    Classic SaliencyMix (Uddin et al. 2021): rectangular patch from B → A,
    centered at peak saliency of B.
    """
    h, w = image_a_np.shape[:2]
    if image_b_np.shape[:2] != (h, w):
        image_b_np = cv2.resize(image_b_np, (w, h), interpolation=cv2.INTER_LINEAR)

    sal = compute_saliency_map(image_b_np)
    cy, cx = np.unravel_index(np.argmax(sal), sal.shape)

    ph = int(h * np.sqrt(patch_ratio))
    pw = int(w * np.sqrt(patch_ratio))
    x1 = max(0, cx - pw // 2); y1 = max(0, cy - ph // 2)
    x2 = min(w, x1 + pw);      y2 = min(h, y1 + ph)

    mixed = image_a_np.copy()
    mixed[y1:y2, x1:x2] = image_b_np[y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (h * w)
    return mixed, float(lam)


# Dataset

class FocusAugMixDataset(Dataset):
    """
    Augmentation strategies (set via `aug_mode`):
      - 'none'       : no mixing, just standard transforms
      - 'saliency'   : SaliencyMix (rectangular patch)
      - 'focusmix'   : OcCaMix + saliency (paper V2)
      - 'focusmix_cam' : OcCaMix + saliency + Grad-CAM (paper V4)
    """
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        aug_mode: str = 'focusmix',
        aug_prob: float = 0.5,
        n_segments: int = 50,
        compactness: float = 10.0,
        paste_ratio: float = 0.25,
    ):
        assert aug_mode in {'none', 'saliency', 'focusmix', 'focusmix_cam'}
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.aug_mode = aug_mode
        self.aug_prob = aug_prob
        self.n_segments = n_segments
        self.compactness = compactness
        self.paste_ratio = paste_ratio
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.gradcam_maps = {}  # {idx: HxW float32 map in [0,1]}

    def set_gradcam_maps(self, maps: dict):
        """Call this between epochs when using 'focusmix_cam' mode."""
        self.gradcam_maps = maps

    def __len__(self):
        return len(self.dataset)

    def _sample_partner(self, idx: int) -> int:
        partner = random.randint(0, len(self.dataset) - 1)
        while partner == idx:
            partner = random.randint(0, len(self.dataset) - 1)
        return partner

    def __getitem__(self, idx):
        img_a, label_a = self.dataset[idx]
        img_a_np = np.array(img_a)

        if self.aug_mode == 'none' or random.random() >= self.aug_prob:
            if self.transform:
                img_a = self.transform(img_a)
            return img_a, label_a, label_a, 1.0

        idx_b = self._sample_partner(idx)
        img_b, label_b = self.dataset[idx_b]
        img_b_np = np.array(img_b)

        if self.aug_mode == 'saliency':
            mixed_np, lam = saliency_mix(img_a_np, img_b_np, self.paste_ratio)
        else:
            cam = self.gradcam_maps.get(idx) if self.aug_mode == 'focusmix_cam' else None
            mixed_np, lam = focus_aug_mix(
                img_a_np, img_b_np,
                gradcam_map=cam,
                use_saliency=True,
                n_segments=self.n_segments,
                compactness=self.compactness,
                paste_ratio=self.paste_ratio,
            )

        mixed_pil = Image.fromarray(mixed_np)
        if self.transform:
            mixed_pil = self.transform(mixed_pil)
        return mixed_pil, label_a, label_b, lam


def focusaugmix_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets_a = torch.tensor([b[1] for b in batch], dtype=torch.long)
    targets_b = torch.tensor([b[2] for b in batch], dtype=torch.long)
    lam = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return images, targets_a, targets_b, lam


# Lightning DataModule

class LeukemiaDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = 'dataset',
        batch_size: int = 32,
        num_workers: int = 8,
        aug_mode: str = 'focusmix',
        aug_prob: float = 0.5,
        n_segments: int = 50,
        compactness: float = 10.0,
        paste_ratio: float = 0.25,
        image_size: int = 224,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_mode = aug_mode
        self.aug_prob = aug_prob
        self.n_segments = n_segments
        self.compactness = compactness
        self.paste_ratio = paste_ratio
        self.image_size = image_size
        self.save_hyperparameters()

        # Conservative photometric jitter: Giemsa stains are diagnostic
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.classes = None
        self.num_classes = None

    def setup(self, stage=None):
        self.train_dataset = FocusAugMixDataset(
            root_dir=os.path.join(self.data_dir, 'train'),
            transform=self.train_transform,
            aug_mode=self.aug_mode,
            aug_prob=self.aug_prob,
            n_segments=self.n_segments,
            compactness=self.compactness,
            paste_ratio=self.paste_ratio,
        )
        self.val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'val'),
            transform=self.val_transform,
        )
        self.classes = self.train_dataset.classes
        self.num_classes = len(self.classes)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=focusaugmix_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
