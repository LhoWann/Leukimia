import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from segment_dataset import apply_hsv_threshold, load_image_as_hsv

path = os.path.join(BASE_DIR, "ALL_IDB Dataset", "L1", "Im103_0.jpg")

img, hsv = load_image_as_hsv(path)

mask = apply_hsv_threshold(hsv)
print(f"mask shape : {mask.shape}")
