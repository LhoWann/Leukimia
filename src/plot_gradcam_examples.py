import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision import transforms

from lightning_model import LeukemiaLightningModel, GradCAMExtractor
from stain_normalize import ReinhardNormalizer, compute_reference_from_dir

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def overlay_cam(img_np, cam, alpha=0.45):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8(img_np * (1 - alpha) + heat * alpha)


def cam_agreement(cam_a, cam_b, thr=0.5) -> float:
    # IoU antara CAM raw vs CAM hasil stain-normalization.
    # Tinggi = model konsisten fokus ke morfologi sel, bukan artefak warna.
    mask_a, mask_b = cam_a >= thr, cam_b >= thr
    union = np.logical_or(mask_a, mask_b).sum()
    return float(np.logical_and(mask_a, mask_b).sum() / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser(
        description='Visualisasi Grad-CAM dan robustness-nya terhadap stain normalization.'
    )
    ap.add_argument('--ckpt', required=True, help='Checkpoint model (.ckpt)')
    ap.add_argument('--images', nargs='+', required=True,
                    help='Path gambar sel individual (hasil segmentasi)')
    ap.add_argument('--ref-dir', default=None,
                    help='Folder training (mis. dataset/train) untuk fit Reinhard normalizer '
                         'sebagai pembanding stain. Jika diisi, baris ketiga Grad-CAM '
                         'ter-normalisasi + skor IoU agreement ikut ditampilkan.')
    ap.add_argument('--target-stage', type=int, default=3)
    ap.add_argument('--out', default='figures/gradcam_examples.png')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm = LeukemiaLightningModel.load_from_checkpoint(args.ckpt, map_location=device)
    model = lm.model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    normalizer = None
    if args.ref_dir:
        ref_image, rh_mean, rh_std = compute_reference_from_dir(args.ref_dir, n_samples=50)
        normalizer = ReinhardNormalizer().fit_from_stats(rh_mean, rh_std)

    n = len(args.images)
    n_rows = 3 if normalizer is not None else 2
    fig, axes = plt.subplots(n_rows, n, figsize=(2.8 * n, 2.8 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(n_rows, n)

    agreements = []
    with GradCAMExtractor(model, target_stage=args.target_stage) as cam_extractor:
        for i, img_path in enumerate(args.images):
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img.resize((224, 224)))
            x = transform(img).unsqueeze(0).to(device)
            cam = cam_extractor(x)[0]
            overlay = overlay_cam(img_np, cam)

            axes[0, i].imshow(img_np)
            axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
            axes[0, i].set_title(Path(img_path).stem, fontsize=8)

            axes[1, i].imshow(overlay)
            axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

            if normalizer is not None:
                norm_np = normalizer.transform(img_np)
                x_norm = transform(Image.fromarray(norm_np)).unsqueeze(0).to(device)
                cam_norm = cam_extractor(x_norm)[0]
                overlay_norm = overlay_cam(norm_np, cam_norm)

                axes[2, i].imshow(overlay_norm)
                axes[2, i].set_xticks([]); axes[2, i].set_yticks([])

                agree = cam_agreement(cam, cam_norm)
                agreements.append(agree)
                axes[2, i].set_title(f'IoU={agree:.2f}', fontsize=8)

    axes[0, 0].set_ylabel('Original', fontsize=8)
    axes[1, 0].set_ylabel('Grad-CAM (raw)', fontsize=8)
    if normalizer is not None:
        axes[2, 0].set_ylabel('Grad-CAM (Reinhard)', fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Saved -> {args.out}')

    if agreements:
        print(f'Mean CAM agreement (raw vs Reinhard) IoU: {np.mean(agreements):.4f}')


if __name__ == '__main__':
    main()
