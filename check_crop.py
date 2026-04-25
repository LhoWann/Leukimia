import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

random.seed(42)

DATA_DIR = 'data'
SAMPLES_PER_CLASS = 5

fig, axes = plt.subplots(3, SAMPLES_PER_CLASS, figsize=(SAMPLES_PER_CLASS * 3, 9))

for row, cls in enumerate(sorted(os.listdir(os.path.join(DATA_DIR, 'train')))):
    cls_dir = os.path.join(DATA_DIR, 'train', cls)
    files = sorted(os.listdir(cls_dir))
    samples = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))

    for col, fname in enumerate(samples):
        img = Image.open(os.path.join(cls_dir, fname))
        axes[row][col].imshow(img)
        axes[row][col].set_title(f"{cls} | {img.size[0]}x{img.size[1]}", fontsize=9)
        axes[row][col].axis('off')

plt.suptitle("Cropped Image Samples per Class", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('crop_check.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved to crop_check.png")
