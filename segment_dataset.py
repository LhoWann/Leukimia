import hashlib
import numpy as np
import cv2

AREA_MIN   = 2000
AREA_MAX   = 80000
PURPLE_MIN = 0.15
IMG_SIZE   = (224, 224)
PADDING    = 40

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

def apply_hsv_threshold(hsv: np.ndarray) -> np.ndarray:
    lower = np.array([110, 80,  50])
    upper = np.array([170, 255, 255])
    return cv2.inRange(hsv, lower, upper)

def apply_morphology(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

def detect_contours(mask: np.ndarray) -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)

def filter_contours(contours: list, mask: np.ndarray) -> list:
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (AREA_MIN < area < AREA_MAX):
            continue
        x, y, w, h   = cv2.boundingRect(cnt)
        roi_mask      = mask[y:y+h, x:x+w]
        purple_ratio  = roi_mask.sum() / (255 * w * h + 1e-6)
        if purple_ratio < PURPLE_MIN:
            continue
        valid.append(cnt)
    return valid

def crop_and_resize(img: np.ndarray, contours: list) -> list[np.ndarray]:
    h_img, w_img = img.shape[:2]
    crops = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x1   = max(0, x - PADDING)
        y1   = max(0, y - PADDING)
        x2   = min(w_img, x + w + PADDING)
        y2   = min(h_img, y + h + PADDING)
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, IMG_SIZE)
        crops.append(crop)
    return crops

def segment_cells_from_smear(image_path: str) -> list[np.ndarray]:
    img, hsv = load_image_as_hsv(image_path)       # Tahap 1 & 2
    if img is None:
        return []

    mask = apply_hsv_threshold(hsv)                 # Tahap 3
    mask = apply_morphology(mask)                   # Tahap 4
    contours = detect_contours(mask)                # Tahap 5
    contours = filter_contours(contours, mask)      # Tahap 6 & 7
    crops    = crop_and_resize(img, contours)       # Tahap 8 & 9

    return crops