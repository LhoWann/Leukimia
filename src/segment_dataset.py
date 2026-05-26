import os
import shutil
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

RAW_DATA_DIR = "data"
IDB1_DIR = os.path.join(RAW_DATA_DIR, "ALL_IDB1")
IDB2_DIR = os.path.join(RAW_DATA_DIR, "ALL_IDB2")
OUTPUT_DIR = "dataset"
CROP_SIZE = 257
SPLIT_RATIO = 0.8
SEED = 42
CLASS_MAP = {"0": "Normal", "1": "Abnormal"}
VALID_IMG_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

# WBC detection parameters for Normal IDB1 images
WBC_HSV_LOWER = np.array([110, 30, 20])
WBC_HSV_UPPER = np.array([175, 255, 160])
WBC_AREA_MIN = 1000
WBC_AREA_MAX = 35000
WBC_CIRCULARITY_MIN = 0.35
WBC_MIN_CELL_DIST = 200     # min pixel distance between two cell centroids
WBC_MAX_CELLS_PER_IMG = 20  # cap to avoid one image dominating the class


# .xyc parsing

def parse_xyc(xyc_path):
    centroids = []
    if not os.path.exists(xyc_path):
        return centroids
    with open(xyc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    centroids.append((int(float(parts[0])), int(float(parts[1]))))
                except ValueError:
                    continue
    return centroids


# ---------------------------------------------------------------------------
# WBC detection (for Normal IDB1 images that have no .xyc annotations)
# Strategy: Giemsa-stained WBC nuclei appear dark purple/violet.
#   1. HSV threshold isolates nuclei.
#   2. Morphological close+open removes noise.
#   3. Contour circularity filter removes non-cell blobs (RBC aggregates, etc.)
#   4. Non-maximum suppression by distance removes overlapping centroids.
#   5. Hard cap prevents one image from dominating the Normal class.
# ---------------------------------------------------------------------------

def _circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 1e-5:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def _nms_centroids(centroids, min_dist):
    """Remove centroids that are closer than min_dist to a higher-area centroid."""
    centroids = sorted(centroids, key=lambda c: c[2], reverse=True)
    kept = []
    for cx, cy, area in centroids:
        too_close = any(
            (cx - kx) ** 2 + (cy - ky) ** 2 < min_dist ** 2
            for kx, ky, _ in kept
        )
        if not too_close:
            kept.append((cx, cy, area))
    return kept


def detect_wbc_centroids(img_rgb):
    """Return list of (cx, cy) for WBCs detected in a Normal blood smear."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, WBC_HSV_LOWER, WBC_HSV_UPPER)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (WBC_AREA_MIN < area < WBC_AREA_MAX):
            continue
        if _circularity(cnt) < WBC_CIRCULARITY_MIN:
            continue
        M = cv2.moments(cnt)
        if M['m00'] < 1:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        candidates.append((cx, cy, area))

    candidates = _nms_centroids(candidates, WBC_MIN_CELL_DIST)
    candidates = candidates[:WBC_MAX_CELLS_PER_IMG]
    return [(cx, cy) for cx, cy, _ in candidates]


# Cropping

def crop_around_centroid(img_np, cx, cy, size=CROP_SIZE):
    h, w = img_np.shape[:2]
    half = size // 2
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = x1 + size
    y2 = y1 + size
    if x2 > w:
        x2 = w
        x1 = max(w - size, 0)
    if y2 > h:
        y2 = h
        y1 = max(h - size, 0)
    crop = img_np[y1:y2, x1:x2]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = np.array(Image.fromarray(crop).resize((size, size), Image.LANCZOS))
    return crop


# Helpers

def get_class_label(stem):
    parts = stem.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in CLASS_MAP:
        return CLASS_MAP[parts[1]]
    return None


def save_jpg(img_np, out_path):
    Image.fromarray(img_np).save(out_path, 'JPEG', quality=95)

# Per-dataset processors

def process_idb1(output_splits):
    """
    Abnormal (_1): crop from .xyc ground-truth centroids.
    Normal   (_0): detect WBC via HSV since .xyc files are empty for healthy donors.
    """
    img_dir = os.path.join(IDB1_DIR, 'im')
    xyc_dir = os.path.join(IDB1_DIR, 'xyc')

    img_files = sorted(
        f for f in os.listdir(img_dir)
        if Path(f).suffix.lower() in VALID_IMG_EXT
    )
    random.shuffle(img_files)
    split_idx = int(len(img_files) * SPLIT_RATIO)
    splits = {'train': img_files[:split_idx], 'val': img_files[split_idx:]}

    counts = {'train': 0, 'val': 0}
    skipped = 0

    for split, files in splits.items():
        for fname in files:
            stem = Path(fname).stem
            class_name = get_class_label(stem)
            if class_name is None:
                continue

            img_np = np.array(Image.open(os.path.join(img_dir, fname)).convert('RGB'))

            if class_name == 'Abnormal':
                xyc_path = os.path.join(xyc_dir, stem + '.xyc')
                centroids = parse_xyc(xyc_path)
                if not centroids:
                    skipped += 1
                    continue
            else:
                # Normal: .xyc is always empty — detect WBC automatically
                centroids = detect_wbc_centroids(img_np)
                if not centroids:
                    skipped += 1
                    continue

            out_dir = output_splits[split][class_name]
            os.makedirs(out_dir, exist_ok=True)
            for i, (cx, cy) in enumerate(centroids):
                crop = crop_around_centroid(img_np, cx, cy)
                save_jpg(crop, os.path.join(out_dir, f"{stem}_cell{i:03d}.jpg"))
                counts[split] += 1

    if skipped:
        print(f"  IDB1: {skipped} images skipped (no centroids found)")
    return counts


def process_idb2(output_splits):
    """ALL-IDB2: already single cells — copy and resize uniformly."""
    img_dir = os.path.join(IDB2_DIR, 'img')

    img_files = sorted(
        f for f in os.listdir(img_dir)
        if Path(f).suffix.lower() in VALID_IMG_EXT
    )
    random.shuffle(img_files)
    split_idx = int(len(img_files) * SPLIT_RATIO)
    splits = {'train': img_files[:split_idx], 'val': img_files[split_idx:]}

    counts = {'train': 0, 'val': 0}
    for split, files in splits.items():
        for fname in files:
            stem = Path(fname).stem
            class_name = get_class_label(stem)
            if class_name is None:
                continue
            img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
            if img.size != (CROP_SIZE, CROP_SIZE):
                img = img.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
            out_dir = output_splits[split][class_name]
            os.makedirs(out_dir, exist_ok=True)
            img.save(os.path.join(out_dir, stem + '.jpg'), 'JPEG', quality=95)
            counts[split] += 1

    return counts

# Summary

def print_summary(output_splits):
    for split in ('train', 'val'):
        print(f"\n{split.capitalize()} set:")
        for cls in CLASS_MAP.values():
            d = output_splits[split][cls]
            n = len(os.listdir(d)) if os.path.exists(d) else 0
            print(f"  {cls}: {n} images")


def main():
    random.seed(SEED)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    class_names = list(CLASS_MAP.values())
    output_splits = {
        split: {cls: os.path.join(OUTPUT_DIR, split, cls) for cls in class_names}
        for split in ('train', 'val')
    }

    print("Processing ALL-IDB1:")
    idb1_counts = process_idb1(output_splits)
    print(f"  IDB1 -> train: {idb1_counts['train']}, val: {idb1_counts['val']}")

    print("Processing ALL-IDB2:")
    idb2_counts = process_idb2(output_splits)
    print(f"  IDB2 -> train: {idb2_counts['train']}, val: {idb2_counts['val']}")

    print_summary(output_splits)


if __name__ == '__main__':
    main()
