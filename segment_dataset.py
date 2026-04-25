import os
import shutil
import random
import hashlib
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

SOURCE_DIR = 'ALL_IDB Dataset'
PROCESSED_DIR = 'ALL_IDB_Processed'
OUTPUT_DIR = 'data'
TARGET_SIZE = 257
SPLIT_RATIO = 0.8
SEED = 42
VALID_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
CROP_PAD = 40
MIN_CELL_AREA = 2000
MAX_CELL_AREA = 80000
MIN_PURPLE_RATIO = 0.15


def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def deduplicate_files(file_list, base_dir):
    hashes = {}
    unique = []
    for fname in file_list:
        h = file_hash(os.path.join(base_dir, fname))
        if h not in hashes:
            hashes[h] = fname
            unique.append(fname)
    return unique


def detect_and_crop_cells(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower_purple = np.array([110, 80, 50])
    upper_purple = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []
    h_img, w_img = img_rgb.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CELL_AREA or area > MAX_CELL_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = w * h
        if bbox_area == 0:
            continue

        purple_in_bbox = mask[y:y+h, x:x+w]
        purple_ratio = np.count_nonzero(purple_in_bbox) / bbox_area
        if purple_ratio < MIN_PURPLE_RATIO:
            continue

        cx = x + w // 2
        cy = y + h // 2
        half = max(w, h) // 2 + CROP_PAD

        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, w_img)
        y2 = min(cy + half, h_img)

        crop = img_rgb[y1:y2, x1:x2]
        if crop.shape[0] < 50 or crop.shape[1] < 50:
            continue

        crop_resized = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
        crops.append(crop_resized)

    return crops


def process_single_file(fname, source_path, output_path):
    img_pil = Image.open(os.path.join(source_path, fname)).convert('RGB')
    img_np = np.array(img_pil)
    w, h = img_pil.size
    base_name = os.path.splitext(fname)[0]
    saved = []

    if w > TARGET_SIZE * 2 or h > TARGET_SIZE * 2:
        crops = detect_and_crop_cells(img_np)
        for i, crop in enumerate(crops):
            out_name = f"{base_name}_cell{i:03d}.jpg"
            Image.fromarray(crop).save(
                os.path.join(output_path, out_name), 'JPEG', quality=95
            )
            saved.append(out_name)
    else:
        if max(w, h) != TARGET_SIZE:
            img_pil = img_pil.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        out_name = base_name + '.jpg'
        img_pil.save(os.path.join(output_path, out_name), 'JPEG', quality=95)
        saved.append(out_name)

    return saved


def process_and_split_class(class_name, source_path, train_dir, val_dir, split_ratio):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_files = [f for f in os.listdir(source_path) if f.lower().endswith(VALID_EXT)]
    unique_files = deduplicate_files(all_files, source_path)
    removed = len(all_files) - len(unique_files)

    random.shuffle(unique_files)
    split_idx = int(len(unique_files) * split_ratio)
    train_sources = unique_files[:split_idx]
    val_sources = unique_files[split_idx:]

    train_count = 0
    for fname in train_sources:
        saved = process_single_file(fname, source_path, train_dir)
        train_count += len(saved)

    val_count = 0
    for fname in val_sources:
        saved = process_single_file(fname, source_path, val_dir)
        val_count += len(saved)

    print(f"  {class_name}: {len(unique_files)} unique sources ({removed} duplicates removed)")
    print(f"    train: {len(train_sources)} sources -> {train_count} cells")
    print(f"    val:   {len(val_sources)} sources -> {val_count} cells")


def main():
    random.seed(SEED)

    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for class_name in sorted(os.listdir(SOURCE_DIR)):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        train_dir = os.path.join(OUTPUT_DIR, 'train', class_name)
        val_dir = os.path.join(OUTPUT_DIR, 'val', class_name)
        process_and_split_class(class_name, class_path, train_dir, val_dir, SPLIT_RATIO)


if __name__ == '__main__':
    main()
