import os
import shutil
import random

def split_dataset(source_dir, dest_dir, split_ratio=0.8):
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')

    for folder in [train_dir, val_dir]:
        os.makedirs(folder, exist_ok=True)

    classes = [c for c in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, c))]

    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        cls_dir = os.path.join(source_dir, cls)
        images = os.listdir(cls_dir)
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        for img in train_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(train_dir, cls, img))

        for img in val_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(val_dir, cls, img))

source_directory = 'ALL_IDB Dataset'
destination_directory = 'data'
split_dataset(source_directory, destination_directory)