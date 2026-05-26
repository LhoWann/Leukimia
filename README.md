# Leukemia Classification — FocusAugMix & ConvNeXt V2

Deteksi **Acute Lymphoblastic Leukemia (ALL)** dari citra mikroskopis sel darah menggunakan arsitektur **FocusAugMix** yang dipadukan dengan **ConvNeXt V2 Tiny + Multi-Head Attention**. Pipeline dibangun di atas **PyTorch Lightning** untuk training yang bersih, modular, dan mudah di-debug.

---

## Daftar Isi

- [Dataset](#dataset)
- [Arsitektur](#arsitektur)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Konfigurasi](#konfigurasi)
- [Output & Checkpoint](#output--checkpoint)
- [Requirements](#requirements)

---

## Dataset

Proyek ini menggunakan **ALL-IDB (Acute Lymphoblastic Leukemia Image Database)**:

| Dataset   | Format                      | Lokasi             | Keterangan                                                  |
|-----------|-----------------------------|--------------------|-------------------------------------------------------------|
| ALL-IDB1  | Full blood smear (`.jpg`)   | `data/ALL_IDB1/im/` | Gambar smear lengkap + file `.xyc` berisi koordinat sentroid sel blast |
| ALL-IDB2  | Single cell (`.tif`/`.jpg`) | `data/ALL_IDB2/img/`| Gambar sel tunggal yang sudah di-crop                       |

### Konvensi Penamaan File

Nama file mengikuti pola `ImXXX_Y` dimana:
- `Y = 1` → kelas **Abnormal** (sel blast / ALL positif)
- `Y = 0` → kelas **Normal**

### Format File `.xyc`

File koordinat untuk ALL-IDB1 berformat teks, satu koordinat per baris:

```
X1 Y1
X2 Y2
...
```

Setiap baris merepresentasikan sentroid satu sel blast dalam gambar smear.

---

## Arsitektur

### Preprocessing Pipeline (`segment_dataset.py`)

```
ALL-IDB1 (full smear)
  └── Baca .xyc → ambil (cx, cy) tiap sel blast
      └── Crop 257×257 di sekitar sentroid → simpan ke Abnormal/
  └── Gambar Normal (Y=0, tanpa .xyc) → center crop 257×257 → simpan ke Normal/

ALL-IDB2 (single cell)
  └── Copy langsung, resize ke 257×257 jika perlu
      └── Label dari suffix _Y → Abnormal/ atau Normal/

Split 80:20 dilakukan di level source image → tidak ada data leakage
```

### FocusAugMix (`data_module.py`)

Augmentasi spasial adaptif yang sadar konten medis:

1. **SLIC Superpixels** — membagi gambar A menjadi segmen-segmen superpixel
2. **Spectral Residual Saliency (FFT)** — mengidentifikasi area paling informatif dari gambar B menggunakan log-spectrum FFT
3. **Grad-CAM Fusion** — jika tersedia, peta Grad-CAM dari epoch sebelumnya digabungkan dengan saliency map (bobot 0.6 saliency + 0.4 Grad-CAM)
4. **Saliency-Guided Mixing** — sepertiga superpixel paling salient dari gambar B ditempel ke gambar A
5. **Soft Label Output** — dataset me-return `(image, target_a, target_b, λ)` untuk mixup loss

### Model (`lightning_model.py`)

```
Input (B, 3, 224, 224)
  │
  ▼
ConvNeXt V2 Tiny (pretrained: fcmae_ft_in22k_in1k)
  │  global_pool='' → feature map (B, C, H, W)
  │
  ▼
Spatial Reshape → (H×W, B, C)
  │
  ▼
Multi-Head Self-Attention (8 heads) + LayerNorm + Residual
  │
  ▼
AdaptiveAvgPool2d → (B, C)
  │
  ▼
Dropout(0.3) → Linear → Logits (B, num_classes)
```

**Grad-CAM** di-attach via `register_forward_hook` dan `register_full_backward_hook` pada stage terakhir backbone.

### Training

- **Loss**: `Σ [λᵢ · CE(pred, target_a) + (1-λᵢ) · CE(pred, target_b)]` (per-sample, lalu `.mean()`)
- **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- **Scheduler**: ReduceLROnPlateau (`factor=0.5`, `patience=3`, monitor `val_loss`)
- **Gradient Clipping**: `max_norm=1.0` (dihandle otomatis oleh Lightning Trainer)

---

## Struktur Proyek

```
.
├── src/
│   ├── segment_dataset.py   # Preprocessing: .xyc cropping + IDB2 copy + split
│   ├── data_module.py       # LightningDataModule + FocusAugMixDataset
│   ├── lightning_model.py   # LightningModule: ConvNeXtV2 + Attention + Grad-CAM
│   └── main.py              # Entry point: Trainer.fit()
│
├── data/
│   ├── ALL_IDB1/
│   │   ├── im/              # Full blood smear images (ImXXX_Y.jpg)
│   │   └── xyc/             # Centroid coordinates (ImXXX_Y.xyc)
│   └── ALL_IDB2/
│       └── img/             # Pre-cropped single cells (ImXXX_Y.tif/.jpg)
│
├── dataset/                 # Dibuat otomatis oleh segment_dataset.py
│   ├── train/
│   │   ├── Abnormal/
│   │   └── Normal/
│   └── val/
│       ├── Abnormal/
│       └── Normal/
│
├── checkpoints/             # Dibuat otomatis saat training
│   ├── leukemia-XX-X.XXXX.ckpt   # Best checkpoint
│   └── last.ckpt
│
├── requirements.txt
└── README.md
```

---

## Instalasi

### 1. Clone & Buat Virtual Environment

```bash
git clone <repo-url>
cd LEUKIMIA

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Siapkan Dataset

Letakkan dataset sehingga strukturnya seperti berikut:

```
data/
├── ALL_IDB1/
│   ├── im/     ← berisi Im001_1.jpg, Im002_0.jpg, dst.
│   └── xyc/    ← berisi Im001_1.xyc, Im002_1.xyc, dst.
└── ALL_IDB2/
    └── img/    ← berisi Im001_1.tif, Im002_0.tif, dst.
```

---

## Cara Penggunaan

Semua script dijalankan dari folder `src/`:

```bash
cd src
```

### Langkah 1 — Segmentasi & Persiapan Dataset

Script ini membaca koordinat `.xyc`, men-crop sel dari ALL-IDB1, menyalin ALL-IDB2, dan membagi data ke `dataset/train` dan `dataset/val`.

```bash
python segment_dataset.py
```

Output yang diharapkan:

```
Processing ALL-IDB1
  IDB1 -> train: 89, val: 22

Processing ALL-IDB2
  IDB2 -> train: 180, val: 45

Train set:
  Normal: 134 images
  Abnormal: 135 images

Val set:
  Normal: 33 images
  Abnormal: 34 images
```

> **Catatan:** Folder `dataset/` akan dihapus dan dibuat ulang setiap kali script ini dijalankan.

### Langkah 2 — Training

```bash
python main.py
```

Lightning Trainer akan otomatis:
- Mendeteksi GPU/CPU yang tersedia
- Menampilkan progress bar per epoch
- Menyimpan checkpoint terbaik berdasarkan `val_loss` ke folder `checkpoints/`

Output log per epoch:

```
Epoch 5: train_loss=0.312 val_loss=0.287 val_acc=0.891
```

### (Opsional) Melanjutkan Training dari Checkpoint

Tambahkan argumen `ckpt_path` ke `trainer.fit()` di `main.py`:

```python
trainer.fit(model, datamodule=datamodule, ckpt_path='checkpoints/last.ckpt')
```

### (Opsional) Menjalankan Validasi Saja

```python
trainer.validate(model, datamodule=datamodule, ckpt_path='checkpoints/leukemia-XX-X.XXXX.ckpt')
```

---

## Konfigurasi

Semua hyperparameter utama berada di `main.py` dan dapat disesuaikan langsung:

| Parameter         | Default     | Lokasi                  | Keterangan                                      |
|-------------------|-------------|-------------------------|-------------------------------------------------|
| `data_dir`        | `'dataset'` | `LeukemiaDataModule`    | Path ke dataset hasil preprocessing             |
| `batch_size`      | `16`        | `LeukemiaDataModule`    | Sesuaikan dengan VRAM GPU                       |
| `num_workers`     | `2`         | `LeukemiaDataModule`    | Jumlah worker DataLoader                        |
| `n_segments`      | `100`       | `LeukemiaDataModule`    | Jumlah superpixel SLIC                          |
| `compactness`     | `10`        | `LeukemiaDataModule`    | Kompaksi superpixel SLIC                        |
| `lr`              | `1e-4`      | `LeukemiaLightningModel`| Learning rate AdamW                             |
| `weight_decay`    | `1e-4`      | `LeukemiaLightningModel`| Weight decay AdamW                              |
| `max_epochs`      | `30`        | `Trainer`               | Jumlah epoch maksimum                           |
| `gradient_clip_val` | `1.0`     | `Trainer`               | Nilai clipping gradient                         |
| `CROP_SIZE`       | `257`       | `segment_dataset.py`    | Ukuran crop sel dalam piksel                    |
| `SPLIT_RATIO`     | `0.8`       | `segment_dataset.py`    | Rasio train/val split                           |

---

## Output & Checkpoint

Setelah training selesai, file berikut akan tersedia:

```
checkpoints/
├── leukemia-{epoch:02d}-{val_loss:.4f}.ckpt   # Model terbaik (val_loss terendah)
└── last.ckpt                                   # Checkpoint epoch terakhir
```

Untuk memuat model dari checkpoint:

```python
from lightning_model import LeukemiaLightningModel

model = LeukemiaLightningModel.load_from_checkpoint('checkpoints/leukemia-XX-X.XXXX.ckpt')
model.eval()
```

Untuk inferensi Grad-CAM pada gambar tunggal:

```python
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = transform(Image.open('path/to/cell.jpg').convert('RGB')).unsqueeze(0)
logits, cam = model.model.get_gradcam(img)
pred_class = logits.argmax(dim=1).item()
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning 2.0+ (`lightning`)
- timm
- torchmetrics
- scikit-image
- opencv-python
- Pillow
- numpy
