import numpy as np
import cv2
from skimage.util import img_as_float

def spectral_residual_saliency(img_bgr: np.ndarray) -> np.ndarray:
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    f        = np.fft.fft2(gray)
    log_amp  = np.log(np.abs(f) + 1e-8)
    smooth   = cv2.blur(log_amp, (3, 3))
    residual = log_amp - smooth
    sr       = np.exp(residual + 1j * np.angle(f))
    saliency = np.abs(np.fft.ifft2(sr)) ** 2
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)
    saliency -= saliency.min()
    saliency /= saliency.max() + 1e-8
    return saliency.astype(np.float32)


def focus_augmix(img_bgr: np.ndarray, n_segments: int = 50, alpha: float = 0.5) -> np.ndarray:
    saliency  = spectral_residual_saliency(img_bgr)
    img_float = img_as_float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    segments  = slic(img_float, n_segments=n_segments, compactness=10, start_label=1)
    result    = img_bgr.copy().astype(np.float32)
    for seg_id in np.unique(segments):
        mask    = segments == seg_id
        sal_val = saliency[mask].mean()
        if sal_val > 0.5:
            noise         = np.random.normal(0, 15, result[mask].shape)
            result[mask]  = np.clip(result[mask] + noise * alpha, 0, 255)
    return result.astype(np.uint8)