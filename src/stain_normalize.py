"""
Stain normalization for histology/cytology images.

Two methods implemented:
  • Macenko et al. (2009) — SVD-based OD decomposition, more principled for
    stain separation but sensitive to tissue coverage.
  • Reinhard et al. (2001) — LAB color-statistics transfer, faster and more
    robust when tissue coverage is inconsistent.

Typical usage (normalize C-NMC images to look like ALL-IDB):
    # Fit on a representative ALL-IDB image (or mean stats from training set)
    mac = MacenkoNormalizer().fit(reference_image_np)
    rh  = ReinhardNormalizer().fit(reference_image_np)

    # Transform each C-NMC image before inference
    normalized = mac.transform(cnmc_image_np)

Both normalizers accept and return HxWx3 uint8 RGB arrays.

References:
  Macenko M. et al. (2009) "A method for normalizing histology slides for
  quantitative analysis." ISBI 2009.

  Reinhard E. et al. (2001) "Color transfer between images." IEEE CG&A 21(5).
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helper: OD ↔ RGB conversions
# ─────────────────────────────────────────────────────────────────────────────

def _rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 (HxWx3) → optical density float64 (Nx3 or HxWx3)."""
    rgb_f = np.clip(rgb.astype(np.float64), 1, 255)
    return -np.log(rgb_f / 255.0)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    """Convert OD float64 → RGB uint8."""
    rgb = np.exp(-od) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Macenko normalizer
# ─────────────────────────────────────────────────────────────────────────────

class MacenkoNormalizer:
    """
    Macenko stain normalizer.

    Fits on a reference (target-domain) image, then transforms source images
    so their staining matches the reference.  Works in optical density space.

    Parameters
    ----------
    luminosity_threshold : float
        OD norm threshold to exclude near-white (background) pixels.
        Lower values keep more pixels; higher values are stricter.
    angular_percentile : float
        Percentile used to find the min/max stain angle vectors (default 99).
    """

    def __init__(
        self,
        luminosity_threshold: float = 0.15,
        angular_percentile: float = 99.0,
    ):
        self.luminosity_threshold = luminosity_threshold
        self.angular_percentile = angular_percentile
        self._stain_matrix_ref: Optional[np.ndarray] = None   # (3, 2)
        self._max_conc_ref: Optional[np.ndarray] = None       # (2,)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        """Return (3,2) stain matrix for image (HxWx3 uint8)."""
        od = _rgb_to_od(image).reshape(-1, 3)                 # (N, 3)
        od_norm = np.linalg.norm(od, axis=1)
        tissue = od[od_norm > self.luminosity_threshold]

        if len(tissue) < 10:
            # Degenerate: virtually all background → return identity
            return np.eye(3, 2)

        # SVD to find the 2-D stain plane
        _, _, Vt = np.linalg.svd(tissue, full_matrices=False)
        V = Vt[:2].T                                          # (3, 2)

        # Project tissue OD onto the plane
        proj = tissue @ V                                     # (N, 2)
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        lo = np.percentile(angles, 100.0 - self.angular_percentile)
        hi = np.percentile(angles, self.angular_percentile)

        v1 = V @ np.array([np.cos(lo), np.sin(lo)])
        v2 = V @ np.array([np.cos(hi), np.sin(hi)])

        # Convention: first column is the "darker" stain (larger OD[0])
        if v1[0] < v2[0]:
            v1, v2 = v2, v1

        HE = np.stack([v1, v2], axis=1)                       # (3, 2)
        # Normalise each column to unit length
        HE /= np.linalg.norm(HE, axis=0, keepdims=True) + 1e-8
        return HE

    def _get_concentrations(
        self,
        image: np.ndarray,
        stain_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Solve stain_matrix @ C ≈ OD for each pixel.
        Returns (2, N) concentration matrix.
        """
        od = _rgb_to_od(image).reshape(-1, 3).T               # (3, N)
        # Non-negative least squares is more physically meaningful, but lstsq
        # is fast and sufficient when the stain matrix is well-conditioned.
        C, _, _, _ = np.linalg.lstsq(stain_matrix, od, rcond=None)  # (2, N)
        return C

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, reference_image: np.ndarray) -> "MacenkoNormalizer":
        """
        Fit on a reference (target-domain) image.

        Parameters
        ----------
        reference_image : HxWx3 uint8 RGB array
        """
        self._stain_matrix_ref = self._get_stain_matrix(reference_image)
        C_ref = self._get_concentrations(reference_image, self._stain_matrix_ref)
        self._max_conc_ref = np.percentile(C_ref, 99, axis=1)  # (2,)
        return self

    def transform(self, source_image: np.ndarray) -> np.ndarray:
        """
        Normalize source_image staining to match reference.

        Parameters
        ----------
        source_image : HxWx3 uint8 RGB

        Returns
        -------
        HxWx3 uint8 RGB with normalized staining.
        """
        if self._stain_matrix_ref is None:
            raise RuntimeError("Call .fit(reference_image) before .transform().")

        H, W = source_image.shape[:2]
        SM_src = self._get_stain_matrix(source_image)
        C_src = self._get_concentrations(source_image, SM_src)            # (2, N)

        # Scale source concentrations to reference range
        max_src = np.percentile(C_src, 99, axis=1, keepdims=True)
        max_src = np.maximum(max_src, 1e-6)
        C_norm = C_src / max_src * self._max_conc_ref[:, np.newaxis]      # (2, N)

        # Reconstruct in OD using the reference stain matrix
        od_norm = self._stain_matrix_ref @ C_norm                          # (3, N)
        rgb_norm = _od_to_rgb(od_norm.T.reshape(H, W, 3))
        return rgb_norm

    def fit_transform(
        self,
        reference_image: np.ndarray,
        source_image: np.ndarray,
    ) -> np.ndarray:
        return self.fit(reference_image).transform(source_image)


# ─────────────────────────────────────────────────────────────────────────────
# Reinhard normalizer
# ─────────────────────────────────────────────────────────────────────────────

class ReinhardNormalizer:
    """
    Reinhard color-statistics transfer in CIELAB space.

    Simpler and faster than Macenko; robust when tissue coverage is low
    or image quality varies (e.g. artefacts, partial cells).

    Parameters
    ----------
    clip : bool
        Clip output LAB values to [0, 255] range before converting back to RGB.
    """

    def __init__(self, clip: bool = True):
        self.clip = clip
        self._target_mean: Optional[np.ndarray] = None   # (3,) float32
        self._target_std: Optional[np.ndarray] = None    # (3,) float32

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _lab_stats(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-channel mean and std in LAB space (float32)."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        mean = lab.mean(axis=(0, 1))                          # (3,)
        std  = lab.std(axis=(0, 1)) + 1e-6                   # (3,)
        return mean, std

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, reference_image: np.ndarray) -> "ReinhardNormalizer":
        """
        Compute target statistics from a single reference image.

        Parameters
        ----------
        reference_image : HxWx3 uint8 RGB
        """
        self._target_mean, self._target_std = self._lab_stats(reference_image)
        return self

    def fit_from_stats(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> "ReinhardNormalizer":
        """
        Set target statistics directly (useful when pre-computed over a dataset).

        Parameters
        ----------
        mean, std : (3,) float32 arrays of LAB channel statistics
        """
        self._target_mean = np.asarray(mean, dtype=np.float32)
        self._target_std  = np.asarray(std, dtype=np.float32)
        return self

    def transform(self, source_image: np.ndarray) -> np.ndarray:
        """
        Transfer target color statistics to source_image.

        Parameters
        ----------
        source_image : HxWx3 uint8 RGB

        Returns
        -------
        HxWx3 uint8 RGB with normalized color.
        """
        if self._target_mean is None:
            raise RuntimeError("Call .fit() or .fit_from_stats() before .transform().")

        lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        src_mean, src_std = self._lab_stats(source_image)

        # z-score normalise source, then rescale to target statistics
        lab_norm = (lab - src_mean) / src_std * self._target_std + self._target_mean

        if self.clip:
            lab_norm = np.clip(lab_norm, 0, 255)
        lab_uint8 = lab_norm.astype(np.uint8)
        return cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)

    def fit_transform(
        self,
        reference_image: np.ndarray,
        source_image: np.ndarray,
    ) -> np.ndarray:
        return self.fit(reference_image).transform(source_image)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level reference computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_reference_from_dir(
    directory: str,
    n_samples: int = 100,
    image_size: int = 224,
    seed: int = 42,
    balanced: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute aggregate stain reference statistics from a directory of images.

    Parameters
    ----------
    balanced : bool
        When True (default), sample equally from each class sub-directory so
        the reference statistics are not skewed by class imbalance.  Falls back
        to random sampling when no sub-directories are found.

    Returns
    -------
    reference_image : HxWx3 uint8 — single representative image (for Macenko).
                      Chosen as the sample whose LAB mean is closest to the
                      global mean, giving a representative rather than random
                      reference stain profile.
    reinhard_mean   : (3,) float32 — mean LAB per channel across n_samples
    reinhard_std    : (3,) float32 — std  LAB per channel across n_samples
    """
    import random as rnd
    rnd.seed(seed)

    root = Path(directory)
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    selected: list
    if balanced:
        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if len(subdirs) >= 2:
            per_class = max(1, n_samples // len(subdirs))
            selected = []
            for subdir in subdirs:
                imgs = sorted([p for p in subdir.iterdir() if p.suffix.lower() in exts])
                rnd.shuffle(imgs)
                selected.extend(imgs[:per_class])
        else:
            balanced = False  # no class sub-dirs found, fall back to random

    if not balanced:
        all_imgs = [p for p in root.rglob('*') if p.suffix.lower() in exts]
        rnd.shuffle(all_imgs)
        selected = all_imgs[:n_samples]

    lab_means, lab_stds, imgs_np = [], [], []

    for img_path in selected:
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img.resize((image_size, image_size), Image.BILINEAR))
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab_means.append(lab.mean(axis=(0, 1)))
            lab_stds.append(lab.std(axis=(0, 1)))
            imgs_np.append(img_np)
        except Exception:
            continue

    if not lab_means:
        raise RuntimeError(f"No valid images found in {directory}")

    means_arr   = np.stack(lab_means)           # (N, 3)
    mean_global = means_arr.mean(axis=0)
    std_global  = np.stack(lab_stds).mean(axis=0)

    # Reference image: sample whose LAB mean is closest to the global mean
    # (avoids staining outliers as the Macenko reference anchor)
    dists = np.linalg.norm(means_arr - mean_global, axis=1)
    reference_image = imgs_np[int(np.argmin(dists))]

    return reference_image, mean_global, std_global
