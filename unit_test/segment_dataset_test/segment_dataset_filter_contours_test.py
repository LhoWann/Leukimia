import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from segment_dataset import load_image_as_hsv, apply_hsv_threshold, apply_morphology, detect_contours, filter_contours

path = os.path.join(BASE_DIR, "ALL_IDB Dataset", "L1", "Im103_0.jpg")

img, hsv   = load_image_as_hsv(path)
mask       = apply_hsv_threshold(hsv)
mask_morph = apply_morphology(mask)
contours   = detect_contours(mask_morph)

valid = filter_contours(contours, mask_morph)

print(f"Kontur masuk  : {len(contours)}")
print(f"Kontur valid  : {len(valid)}")
print(f"Kontur dibuang: {len(contours) - len(valid)}")