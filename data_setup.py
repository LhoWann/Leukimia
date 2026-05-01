import numpy as np
import cv2

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