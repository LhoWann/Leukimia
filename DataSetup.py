import os
import random
import numpy as np
import cv2
from PIL import Image
from skimage.segmentation import slic
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_saliency_map(image_np):
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (64, 64)).astype(np.float64) / 255.0

    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.maximum(np.abs(f_shift), 1e-10)
    log_amplitude = np.log(magnitude)
    phase = np.angle(f_shift)

    spectral_residual = log_amplitude - cv2.blur(log_amplitude, (3, 3))

    saliency = np.abs(
        np.fft.ifft2(np.fft.ifftshift(np.exp(spectral_residual + 1j * phase)))
    ) ** 2
    saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)

    sal_min, sal_max = saliency.min(), saliency.max()
    if sal_max - sal_min > 1e-10:
        saliency = (saliency - sal_min) / (sal_max - sal_min)
    else:
        saliency = np.ones_like(saliency)

    h, w = image_np.shape[:2]
    return cv2.resize(saliency, (w, h)).astype(np.float32)


def focus_aug_mix(image_a_np, image_b_np, gradcam_map=None, n_segments=100, compactness=10):
    h, w = image_a_np.shape[:2]
    image_b_resized = cv2.resize(image_b_np, (w, h))

    segments = slic(
        image_a_np, n_segments=n_segments,
        compactness=compactness, start_label=0, channel_axis=2
    )

    saliency = compute_saliency_map(image_b_resized)

    if gradcam_map is not None:
        gradcam_resized = gradcam_map
        if gradcam_map.shape != saliency.shape:
            gradcam_resized = cv2.resize(gradcam_map, (w, h))
        saliency = 0.6 * saliency + 0.4 * gradcam_resized

    segment_scores = {
        seg_id: saliency[segments == seg_id].mean()
        for seg_id in np.unique(segments)
    }
    sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
    num_to_mix = max(1, len(sorted_segments) // 3)

    mixed = image_a_np.copy().astype(np.float32)
    mixed_pixels = 0

    for seg_id, _ in sorted_segments[:num_to_mix]:
        mask = segments == seg_id
        mixed[mask] = image_b_resized[mask].astype(np.float32)
        mixed_pixels += mask.sum()

    lam = 1.0 - (mixed_pixels / (h * w))
    return np.clip(mixed, 0, 255).astype(np.uint8), lam


class FocusAugMixDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=True, n_segments=100, compactness=10):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.augment = augment
        self.n_segments = n_segments
        self.compactness = compactness
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.gradcam_maps = {}

    def set_gradcam_maps(self, maps_dict):
        self.gradcam_maps = maps_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_a, label_a = self.dataset[idx]
        img_a_np = np.array(img_a)

        if self.augment and random.random() < 0.5:
            idx_b = random.randint(0, len(self.dataset) - 1)
            img_b, label_b = self.dataset[idx_b]

            mixed_np, lam = focus_aug_mix(
                img_a_np, np.array(img_b),
                gradcam_map=self.gradcam_maps.get(idx),
                n_segments=self.n_segments,
                compactness=self.compactness
            )

            mixed_img = Image.fromarray(mixed_np)
            if self.transform:
                mixed_img = self.transform(mixed_img)
            return mixed_img, label_a, label_b, lam

        if self.transform:
            img_a = self.transform(img_a)
        return img_a, label_a, label_a, 1.0


def focusaugmix_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets_a = torch.tensor([item[1] for item in batch], dtype=torch.long)
    targets_b = torch.tensor([item[2] for item in batch], dtype=torch.long)
    lam = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    return images, targets_a, targets_b, lam


def create_dataloaders(data_dir, batch_size=32, num_workers=2, n_segments=100, compactness=10):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    train_dataset = FocusAugMixDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform, augment=True,
        n_segments=n_segments, compactness=compactness
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'), transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=focusaugmix_collate_fn, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = train_dataset.classes

    return dataloaders, dataset_sizes, class_names