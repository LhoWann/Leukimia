import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

import numpy as np

from data_setup import spectral_residual_saliency

img      = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
saliency = spectral_residual_saliency(img)

print(f"  input shape    : {img.shape}")
print(f"  saliency shape : {saliency.shape}")