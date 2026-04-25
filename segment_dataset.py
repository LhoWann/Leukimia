import hashlib
import numpy as np
import cv2

def md5_hash(filepath: str) -> str:
    # Menghitung hash MD5 untuk deduplication.
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    
def remove_duplicates(image_paths: list[str]) -> list[str]:
    # Menghapus gambar duplikat berdasarkan hash MD5.
    seen, unique = set(), []
    for p in image_paths:
        h = md5_hash(p)
        if h not in seen:
            seen.add(h)
            unique.append(p)
    return unique

def load_image_as_hsv(image_path: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, hsv
