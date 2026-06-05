# Leukemia Classification: FocusAugMix + ConvNeXt V2

Deteksi Otomatis Acute Lymphoblastic Leukemia (ALL) dari Citra Apusan Darah Tepi

Model dilatih pada **ALL-IDB** (Giemsa stain, Italia) dan dievaluasi secara eksternal pada
**C-NMC 2019** (Wright-Giemsa, India) dengan stain normalization untuk mengukur generalisasi
lintas-domain.

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Arsitektur &amp; Metodologi](#arsitektur--metodologi)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Alur Kerja Lengkap](#alur-kerja-lengkap)
  - [1. Persiapan Data](#1-persiapan-data)
  - [2. Training](#2-training)
  - [3. Monitoring](#3-monitoring)
  - [4. Evaluasi In-Domain](#4-evaluasi-in-domain)
  - [5. Evaluasi Eksternal C-NMC 2019](#5-evaluasi-eksternal-c-nmc-2019)
- [Stain Normalization](#stain-normalization)
- [Eksperimen &amp; Konfigurasi](#eksperimen--konfigurasi)
- [Output &amp; Checkpoint](#output--checkpoint)
- [Troubleshooting](#troubleshooting)
- [Referensi](#referensi)

---

## Latar Belakang

ALL adalah jenis kanker darah paling umum pada anak-anak. Diagnosis awal memerlukan identifikasi
*blast cells* (sel leukemia) dari apusan darah secara mikroskopis — proses yang memakan waktu
dan bergantung pada keahlian patologis.

Proyek ini mengimplementasikan sistem klasifikasi otomatis berbasis:

- **ConvNeXt V2 Tiny** — backbone pretrained FCMAE + IN22k + IN1k (28.5 M params)
- **Multi-Head Self-Attention** — injeksi perhatian spasial setelah stage backbone
- **FocusAugMix** — mixing augmentation berbasis SLIC superpixel + saliency + Grad-CAM
- **Stain Normalization** — Macenko & Reinhard untuk evaluasi lintas-domain

---

## Dataset

### ALL-IDB (Training & Validation)

| Versi    | Format                          | Deskripsi                                       | Jumlah      |
| -------- | ------------------------------- | ----------------------------------------------- | ----------- |
| ALL-IDB1 | `.jpg` full smear             | Gambar penuh + koordinat blast di file `.xyc` | ~108 gambar |
| ALL-IDB2 | `.tif` / `.jpg` single cell | Sel individual yang sudah di-crop               | ~260 gambar |

Split: **80% train / 20% val** berbasis *image-level* — semua crop dari gambar yang sama
masuk ke satu split (tidak ada data leakage).

### C-NMC 2019 (External Test)

Dataset PKG-C-NMC 2019 dari ISBI Challenge, dikumpulkan di India dengan protokol staining
berbeda (Wright-Giemsa vs Giemsa ALL-IDB). Digunakan sebagai *external test set* untuk
mengukur generalisasi lintas-domain.

| Label          | Folder C-NMC | Setara ALL-IDB |
| -------------- | ------------ | -------------- |
| ALL (leukemia) | `all/`     | `Abnormal/`  |
| Normal (HEM)   | `hem/`     | `Normal/`    |

---

## Arsitektur & Metodologi

### Pipeline Augmentasi: FocusAugMix

```text
Gambar A (target) + Gambar B (source)
         |
         v
  SLIC Superpixels (n=50)   <- pada Gambar A, preserve kontur sel
         |
         v
  Saliency Map (Spectral Residual) <- pada Gambar B
         |
  + Grad-CAM (opsional, diupdate setiap 5 epoch)
         |
         v
  Rank superpixels by score -> paste top-K dari B ke A
         |
         v
  Mixed Image + lambda (label mixing weight)
```

Mode augmentasi: `none` | `saliency` | `focusmix` | `focusmix_cam`

### Arsitektur Model

```text
Input (B, 3, 224, 224)
   |
   v
ConvNeXt V2 Tiny — pretrained fcmae_ft_in22k_in1k
   +-- Stage 0 -> (B,  96, 56, 56)
   +-- Stage 1 -> (B, 192, 28, 28)
   +-- Stage 2 -> (B, 384, 14, 14)  [MHA disisipkan di sini (default)]
   |                  |
   |            tokens: (B, 196, 384)
   |            MultiheadAttention (8 heads) + Residual + LayerNorm
   |
   +-- Stage 3 -> (B, 768,  7,  7)
   |
   v
AdaptiveAvgPool2d(1) -> Flatten -> Dropout(0.3) -> Linear(num_classes)
```

**Layer-wise Learning Rate Decay (LLRD, factor 0.75):**

| Layer      | LR multiplier  |
| ---------- | -------------- |
| Head / MHA | 1.00 x base_lr |
| Stage 3    | 0.75 x base_lr |
| Stage 2    | 0.56 x base_lr |
| Stage 1    | 0.42 x base_lr |
| Stage 0    | 0.32 x base_lr |
| Stem       | 0.24 x base_lr |

**Optimizer:** AdamW + Linear Warmup (5 epoch) + Cosine Decay

**Trainer settings:**

| Setting             | Nilai         | Keterangan                                   |
| ------------------- | ------------- | -------------------------------------------- |
| `precision`       | `bf16-mixed`  | ~1.8x speedup pada GPU Ampere+               |
| `gradient_clip_val` | 1.0         | Mencegah exploding gradient saat fine-tuning |
| `log_every_n_steps` | 10          | Frekuensi logging ke CSV                     |

**Class index mapping** (ditentukan oleh urutan alfabet folder ImageFolder):

| Index | Nama folder | Label biologis          |
| ----- | ----------- | ----------------------- |
| 0     | `Abnormal`  | Sel ALL / blast cell    |
| 1     | `Normal`    | Sel WBC normal          |

---

## Struktur Proyek

```text
LEUKIMIA/
+-- src/
|   +-- segment_dataset.py      # Preprocessing ALL-IDB raw -> dataset/
|   +-- data_module.py          # Dataset, augmentasi, DataModule, external loader
|   +-- lightning_model.py      # ConvNeXtV2Classifier, GradCAMExtractor, LightningModule
|   +-- stain_normalize.py      # MacenkoNormalizer, ReinhardNormalizer
|   +-- evaluate_external.py    # Evaluasi C-NMC 2019 dengan stain normalization
|   +-- main.py                 # CLI training, experiment registry
|
+-- data/                       # Raw dataset (isi manual)
|   +-- ALL_IDB1/
|   |   +-- im/                 # Im001_1.jpg, Im002_0.jpg, ...
|   |   +-- xyc/                # Im001_1.xyc (koordinat blast)
|   +-- ALL_IDB2/
|       +-- img/                # Im001_1.tif, Im002_0.tif, ...
|
+-- dataset/                    # Di-generate oleh segment_dataset.py
|   +-- train/
|   |   +-- Abnormal/           # ~600+ sel leukemia
|   |   +-- Normal/             # ~200+ sel normal
|   +-- val/
|       +-- Abnormal/           # ~111 sel leukemia
|       +-- Normal/             # ~93 sel normal
|
+-- checkpoints/                # Tersimpan otomatis saat training
|   +-- baseline/
|   |   +-- epoch=XX-val_acc=1.0000.ckpt
|   |   +-- last.ckpt
|   +-- mha_only/ ...
|
+-- logs/                       # CSV metrics
|   +-- baseline/version_0/metrics.csv
|   +-- mha_only/version_0/metrics.csv
|
+-- result/                     # JSON output evaluasi eksternal
+-- requirements.txt
+-- README.md
```

---

## Instalasi

### Prasyarat

- Python 3.9–3.11
- GPU dengan CUDA 11.8+ (direkomendasikan; CPU fallback tersedia)
- Git

### Langkah 1 — Clone Repositori

```bash
git clone <repo-url>
cd LEUKIMIA
```

### Langkah 2 — Buat Virtual Environment

```bash
# venv (Windows)
python -m venv .venv
.venv\Scripts\activate

# venv (Linux / macOS)
python -m venv .venv
source .venv/bin/activate

# conda (alternatif)
conda create -n leukemia python=3.11 -y
conda activate leukemia
```

### Langkah 3 — Install Dependencies

```bash
pip install -r requirements.txt
pip install torchmetrics>=1.0.0   # tidak tercantum di requirements.txt, diperlukan oleh lightning_model.py
```

Untuk CUDA support (jika PyTorch terinstall tanpa CUDA):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Langkah 4 — Verifikasi

```bash
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import lightning; print('Lightning', lightning.__version__)"
python -c "import timm; print('timm', timm.__version__)"
```

---

## Alur Kerja Lengkap

### 1. Persiapan Data

#### Download ALL-IDB

- https://scotti.di.unimi.it/all/

```bash
mkdir -p data/ALL_IDB1/im data/ALL_IDB1/xyc
mkdir -p data/ALL_IDB2/img

cp /path/to/ALL_IDB1/im/*.jpg   data/ALL_IDB1/im/
cp /path/to/ALL_IDB1/xyc/*.xyc  data/ALL_IDB1/xyc/
cp /path/to/ALL_IDB2/img/*.tif  data/ALL_IDB2/img/
```

#### Jalankan Segmentasi

```bash
cd src
python segment_dataset.py
```

Script akan:

1. Membaca ALL-IDB1 — crop sel abnormal (dari koordinat `.xyc`) + deteksi normal (HSV thresholding)
2. Membaca ALL-IDB2 — salin sel individu ke folder yang sesuai
3. Split 80/20 per-gambar (tidak ada leakage)
4. Output ke `../dataset/`

Output yang diharapkan:

```text
=== ALL-IDB1 ===
  Splitting 108 images: 86 train / 22 val
  Abnormal train: 320 cells  |  val: 80 cells
  Normal   train:  80 cells  |  val: 20 cells

=== ALL-IDB2 ===
  Abnormal train:  52 cells  |  val: 13 cells
  Normal   train: 104 cells  |  val: 26 cells

=== Summary ===
  Train : 556 images  (Abnormal: 372, Normal: 184)
  Val   : 139 images  (Abnormal:  93, Normal:  46)
```

---

### 2. Training

Semua command dijalankan dari direktori `src/`.

```bash
cd src
```

#### Jalankan Satu Eksperimen

```bash
# Eksperimen yang direkomendasikan
python main.py --exp focusmix_mha

# Eksperimen lain
python main.py --exp baseline
python main.py --exp mha_only
python main.py --exp saliency_mix
python main.py --exp focusmix_v2
python main.py --exp focusmix_full
python main.py --exp focusmix_aggressive

# Dengan seed tertentu dan direktori data kustom
python main.py --exp focusmix_mha --seed 123 --data-dir ../dataset
```

#### Jalankan Semua Eksperimen Berurutan

```bash
python main.py --all
```

Progress dan metrics ditampilkan real-time di terminal. Checkpoint terbaik (berdasarkan `val_acc`) disimpan otomatis.

---

### 3. Monitoring

#### CSV Logs (Default)

File metrics tersimpan di `logs/<exp_name>/version_0/metrics.csv`.

Kolom yang tersedia di CSV:

| Kolom             | Keterangan                         |
| ----------------- | ---------------------------------- |
| `train_loss`      | Loss per step                      |
| `train_loss_epoch`| Loss rata-rata per epoch           |
| `val_loss`        | Validation loss                    |
| `val_acc`         | Validation accuracy                |
| `val_f1`          | F1 macro                           |
| `val_precision`   | Precision macro                    |
| `val_recall`      | Recall macro                       |
| `lr-AdamW`        | Learning rate head/MHA per epoch   |

Plot training curve:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../logs/mha_only/version_0/metrics.csv')

train = df.dropna(subset=['train_loss_epoch'])
val   = df.dropna(subset=['val_loss'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train['epoch'], train['train_loss_epoch'], label='train loss')
axes[0].plot(val['epoch'],   val['val_loss'],           label='val loss')
axes[0].set_title('Loss'); axes[0].legend()

axes[1].plot(val['epoch'], val['val_acc'], color='green', label='val acc')
axes[1].set_title('Val Accuracy'); axes[1].legend()

plt.tight_layout()
plt.savefig('training_curve.png', dpi=150)
plt.show()
```

#### TensorBoard (Opsional)

Edit satu baris di `src/main.py`:

```python
# Ganti:
logger = CSVLogger('logs', name=cfg.name)

# Dengan:
from lightning.pytorch.loggers import TensorBoardLogger
logger = TensorBoardLogger('logs', name=cfg.name)
```

Kemudian:

```bash
tensorboard --logdir=../logs
# Buka: http://localhost:6006
```

---

### 4. Evaluasi In-Domain

Evaluasi pada validation set ALL-IDB menggunakan checkpoint terbaik.
Ini dijalankan otomatis di akhir setiap training, bisa juga dijalankan manual:

```bash
cd src

python - << 'EOF'
import torch
import numpy._core.multiarray
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

import lightning as L
from lightning_model import LeukemiaLightningModel
from data_module import LeukemiaDataModule

ckpt = '../checkpoints/mha_only/epoch=06-val_acc=1.0000.ckpt'
model = LeukemiaLightningModel.load_from_checkpoint(ckpt)

dm = LeukemiaDataModule(data_dir='../dataset', batch_size=32)
dm.setup()

trainer = L.Trainer(
    accelerator='auto', devices=1,
    logger=False, enable_checkpointing=False
)
trainer.validate(model, datamodule=dm)
EOF
```

---

### 5. Evaluasi Eksternal C-NMC 2019

#### Download C-NMC 2019

- https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjczNCIsInBhc3Njb2RlIjoiNDM3ZmMzM2RkMzQ1ZmMzZjNjM2FlY2JmZWQ0MThlY2NjYTkzM2RmMiIsInBhY2thZ2VfaWQiOiI3MzQiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0=&redirected=true&authenticated=true

#### Struktur Direktori C-NMC

Gunakan `C-NMC_train_merged` — satu-satunya split yang memiliki label kelas publik.
Split test (prelim & final) berisi file flat tanpa label sehingga tidak dapat dievaluasi.

```text
PKG_C_NMC 2019/
+-- C-NMC_train_merged/
|   +-- all/          <- sel ALL (leukemia)  [pemetaan: Abnormal]
|   +-- hem/          <- sel HEM (normal)    [pemetaan: Normal]
+-- C-NMC_training_data/
|   +-- fold_0/all/ hem/
|   +-- fold_1/all/ hem/
|   +-- fold_2/all/ hem/
+-- C-NMC_test_prelim_phase_data/   <- flat files, tanpa label (tidak bisa dievaluasi)
+-- C-NMC_test_final_phase_data/    <- flat files, tanpa label (tidak bisa dievaluasi)
```

Format gambar yang didukung: `.jpg`, `.bmp`, `.png`, `.tif`

#### Jalankan Evaluasi — Auto Mode (semua eksperimen sekaligus)

```bash
# dari root direktori proyek
python src/evaluate_external.py \
    --cnmc-dir "PKG_C_NMC 2019/C-NMC_train_merged" \
    --data-dir dataset
```

#### Jalankan Evaluasi — Single Model

```bash
cd src

python evaluate_external.py \
    --ckpt      ../checkpoints/mha_only/epoch=07-val_f1=1.0000.ckpt \
    --cnmc-dir  "../PKG_C_NMC 2019/C-NMC_train_merged" \
    --data-dir  ../dataset \
    --output-json ../results/cnmc_eval_mha_only.json
```

Script akan menjalankan **4 kondisi** secara berurutan:

1. ALL-IDB val (in-domain, sebagai baseline)
2. C-NMC tanpa normalisasi (raw domain shift)
3. C-NMC + Macenko normalization
4. C-NMC + Reinhard normalization

#### Argumen Lengkap

```text
--ckpt            Path ke file .ckpt  (wajib)
--cnmc-dir        Direktori C-NMC berlabel, berisi all/ dan hem/  (wajib)
--data-dir        Direktori dataset ALL-IDB  (default: ../dataset)
--batch-size      Batch size inference  (default: 32)
--num-workers     Worker DataLoader  (default: 4; gunakan 0 jika hang)
--image-size      Ukuran resize gambar  (default: 224)
--ref-samples     Gambar training untuk hitung referensi stain  (default: 100)
--device          auto / cpu / cuda  (default: auto)
--no-macenko      Skip Macenko normalization
--no-reinhard     Skip Reinhard normalization
--skip-val        Skip evaluasi ALL-IDB val
--output-json     Simpan semua metrics ke JSON
```

#### Contoh Cepat (Tanpa Normalisasi)

```bash
cd src

python evaluate_external.py \
    --ckpt     ../checkpoints/mha_only/epoch=07-val_f1=1.0000.ckpt \
    --cnmc-dir "../PKG_C_NMC 2019/C-NMC_train_merged" \
    --no-macenko --no-reinhard
```

#### Contoh Output

Output berikut adalah hasil nyata dari eksperimen `baseline` pada `C-NMC_train_merged`
(10.661 sel). Lihat analisis lengkap di [experiment_results.md](experiment_results.md).

```text
────────────────────────────────────────────────────────────
  ALL-IDB Val  .  In-Domain
────────────────────────────────────────────────────────────
  N samples  : 204
  Accuracy   : 1.0000  (100.0%)
  F1 (macro) : 1.0000

────────────────────────────────────────────────────────────
  C-NMC 2019  .  No Stain Normalization
────────────────────────────────────────────────────────────
  N samples  : 10661
  Accuracy   : 0.6904  (69.0%)
  F1 (macro) : 0.5289

────────────────────────────────────────────────────────────
  C-NMC 2019  .  Macenko Normalization
────────────────────────────────────────────────────────────
  N samples  : 10661
  Accuracy   : 0.6537  (65.4%)
  F1 (macro) : 0.4864

────────────────────────────────────────────────────────────
  C-NMC 2019  .  Reinhard Normalization
────────────────────────────────────────────────────────────
  N samples  : 10661
  Accuracy   : 0.7035  (70.4%)
  F1 (macro) : 0.5101

════════════════════════════════════════════════════════════
  SUMMARY
════════════════════════════════════════════════════════════
  Condition                                 Acc    F1
  ──────────────────────────────────────── ────── ──────
  ALL-IDB val (in-domain)                 1.0000 1.0000
  C-NMC -- no normalization               0.6904 0.5289
  C-NMC -- Macenko                        0.6537 0.4864
  C-NMC -- Reinhard                       0.7035 0.5101

  Domain-shift gap (val_acc - raw_acc) : +0.3096
  WARNING: Large gap -- model likely relies on staining artefacts.
```

#### Interpretasi Domain-Shift Gap

| Gap       | Interpretasi                                                            |
| --------- | ----------------------------------------------------------------------- |
| < 0.05    | Model robust — belajar morfologi sel, tidak bergantung staining        |
| 0.05-0.10 | Ketergantungan staining moderat                                         |
| > 0.10    | Ketergantungan staining signifikan; normalisasi sangat direkomendasikan |

Jika Macenko / Reinhard menutup sebagian besar gap, maka staining adalah faktor utama penurunan performa — bukan kualitas arsitektur.

#### Gap Per-Kelas: Recall Abnormal vs Normal

Selain gap keseluruhan, penting untuk melihat **asimetri recall per kelas**. Confusion matrix
C-NMC menunjukkan dua pola berlawanan tergantung konfigurasi eksperimen:

| Pola              | Eksperimen                               | Recall Abnormal | Recall Normal |
| ----------------- | ---------------------------------------- | :-------------: | :-----------: |
| Bias → Abnormal   | `baseline`, `mha_only`, `saliency_mix` | 93–100%         | 1–17%         |
| Bias → Normal     | `focusmix_mha`, `focusmix_full`        | 7–28%           | 87–95%        |
| Relatif seimbang  | `focusmix_v2`                           | 83%             | 50%           |

**Penyebab bias ke Abnormal:**

- Imbalance kelas training ALL-IDB (~2:1 Abnormal:Normal) mendorong model memilih Abnormal secara statistik.
- Sel blast (Abnormal) memiliki ciri morfologi persisten lintas domain: inti besar, kromatin kasar,
  rasio nukleus-sitoplasma tinggi — ciri ini relatif bertahan walau protokol pewarnaan berbeda.
- Sel Normal (HEM/limfosit) lebih sensitif terhadap perubahan staining; fitur warna yang dipelajari
  dari Giemsa ALL-IDB tidak cocok dengan tampilan HEM di Wright-Giemsa C-NMC.
- Distribusi C-NMC yang tidak seimbang (68% ALL, 32% HEM) membuat model bias Abnormal tetap
  terlihat "wajar" dari sisi accuracy keseluruhan meskipun recall Normal nyaris nol.

**Penyebab bias ke Normal (eksperimen FocusAugMix+MHA):**

- FocusAugMix mem-paste potongan superpixel antar gambar, menciptakan pola "tambal sulam" yang
  tidak lazim. Model belajar mengasosiasikan penampilan tidak seragam itu dengan Abnormal —
  tetapi di C-NMC, sel blast terlihat uniform dan bersih, sehingga salah diprediksi Normal.
- MHA memperkuat *spatial attention patterns* yang sangat spesifik terhadap distribusi warna
  Giemsa ALL-IDB. Ketika domain bergeser, pola atensi ini tidak lagi menemukan fitur yang
  diharapkan, lalu kolaps ke prediksi Normal secara masif.

**Catatan klinis:** Dalam konteks medis, False Negative Abnormal (sel leukemia yang diprediksi
Normal) jauh lebih berbahaya dari False Positive. Akurasi keseluruhan yang terlihat baik bisa
menyembunyikan recall Abnormal yang sangat rendah — selalu periksa confusion matrix dan F1 per
kelas, bukan hanya accuracy agregat.

---

## Stain Normalization

### Konsep

Model yang dilatih di ALL-IDB (Giemsa, Italia) dan ditest di C-NMC (Wright-Giemsa, India)
mengalami penurunan performa karena **distribusi warna berbeda**, bukan morfologi sel berbeda.
Stain normalization memetakan C-NMC agar terlihat seperti ALL-IDB sebelum inference.

```text
C-NMC image (Wright-Giemsa)
    |
    v   MacenkoNormalizer.transform()
Normalized (distribusi warna mendekati Giemsa ALL-IDB)
    |
    v   Model inference
Prediction
```

### Macenko vs Reinhard

| Aspek            | Macenko                      | Reinhard                         |
| ---------------- | ---------------------------- | -------------------------------- |
| Prinsip kerja    | SVD di Optical Density space | Color statistics transfer (LAB)  |
| Kualitas         | Lebih akurat, stain-aware    | Lebih sederhana, global          |
| Kecepatan        | Lambat (SVD per gambar)      | Sangat cepat                     |
| Robustness       | Sensitif gambar tanpa tissue | Robust terhadap gambar partial   |
| Direkomendasikan | Gambar berkualitas tinggi    | Dataset besar / batch processing |

### API Python

```python
import numpy as np
from PIL import Image
from stain_normalize import (
    MacenkoNormalizer,
    ReinhardNormalizer,
    compute_reference_from_dir,
)

# Hitung referensi dari training set ALL-IDB (sekali saja)
ref_image, rh_mean, rh_std = compute_reference_from_dir(
    directory='../dataset/train',
    n_samples=100,       # jumlah gambar yang disampling
    image_size=224,
)

# --- Macenko ---
mac = MacenkoNormalizer(
    luminosity_threshold=0.15,   # ambang batas untuk hapus background
    angular_percentile=99,       # persentil untuk estimasi sudut stain
)
mac.fit(ref_image)               # fit ke satu gambar referensi ALL-IDB

cnmc_np = np.array(Image.open('cnmc_cell.jpg').convert('RGB'))
normalized_mac = mac.transform(cnmc_np)   # HxWx3 uint8

# --- Reinhard ---
rh = ReinhardNormalizer()
rh.fit_from_stats(rh_mean, rh_std)        # fit dari statistik agregat dataset

normalized_rh = rh.transform(cnmc_np)    # HxWx3 uint8

# --- Atau fit dari satu gambar ---
rh2 = ReinhardNormalizer().fit(ref_image)
normalized_rh2 = rh2.transform(cnmc_np)
```

---

> **Hasil eksperimen lengkap, analisis val accuracy 100%, dan perbandingan lintas-domain**
> tersedia di [experiment_results.md](experiment_results.md).

---

## Eksperimen & Konfigurasi

### Daftar Eksperimen

| Eksperimen              | MHA | aug_mode         | aug_prob | paste_ratio | Tujuan                     |
| ----------------------- | --- | ---------------- | -------- | ----------- | -------------------------- |
| `baseline`            | No  | `none`         | 0.5      | 0.25        | Batas bawah performa       |
| `mha_only`            | Yes | `none`         | 0.5      | 0.25        | Isolasi kontribusi MHA     |
| `saliency_mix`        | No  | `saliency`     | 0.5      | 0.25        | SaliencyMix murni          |
| `focusmix_v2`         | No  | `focusmix`     | 0.5      | 0.25        | OcCaMix + Saliency         |
| `focusmix_mha`        | Yes | `focusmix`     | 0.5      | 0.25        | **Direkomendasikan** |
| `focusmix_full`       | Yes | `focusmix_cam` | 0.5      | 0.25        | + Grad-CAM online          |
| `focusmix_aggressive` | Yes | `focusmix`     | 0.7      | 0.35        | Dataset sangat kecil       |

### Hyperparameter Lengkap

| Parameter           | Default | Deskripsi                                  |
| ------------------- | ------- | ------------------------------------------ |
| `batch_size`      | 32      | Turunkan ke 16 jika GPU OOM                |
| `lr`              | 1e-4    | Base learning rate untuk head / MHA        |
| `weight_decay`    | 0.05    | AdamW weight decay                         |
| `llrd`            | 0.75    | Layer-wise LR decay factor per stage       |
| `label_smoothing` | 0.05    | Label smoothing di CrossEntropy            |
| `max_epochs`      | 7       | Maksimum epoch (early stopping aktif; cukup karena fine-tuning pretrained) |
| `warmup_epochs`   | 5       | Epoch linear warmup sebelum cosine         |
| `aug_prob`        | 0.5     | Probabilitas augmentasi per sampel         |
| `paste_ratio`     | 0.25    | Fraksi superpixel yang di-paste            |
| `n_segments`      | 50      | Jumlah superpixel SLIC                     |
| `mha_stage`       | 2       | Stage backbone tempat MHA disisipkan (0-3) |

### Menambah Eksperimen Baru

Edit `EXPERIMENTS` di `src/main.py`:

```python
EXPERIMENTS['my_exp'] = ExperimentConfig(
    name='my_exp',
    aug_mode='focusmix',
    use_mha=True,
    mha_stage=3,       # coba MHA di stage terakhir (7x7 tokens)
    paste_ratio=0.30,
    lr=5e-5,
)
```

```bash
python main.py --exp my_exp
```

---

## Output & Checkpoint

### Load Checkpoint untuk Inference

```python
import torch
import numpy._core.multiarray
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

from lightning_model import LeukemiaLightningModel
from torchvision import transforms
from PIL import Image

# Load model
model = LeukemiaLightningModel.load_from_checkpoint(
    '../checkpoints/mha_only/epoch=06-val_acc=1.0000.ckpt',
    map_location='cuda',
)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Inference
img = transform(Image.open('cell.jpg').convert('RGB')).unsqueeze(0).cuda()
with torch.no_grad():
    pred = model(img).argmax(dim=1).item()

print({0: 'Abnormal (ALL)', 1: 'Normal'}[pred])
```

### Inference dengan Stain Normalization

```python
import numpy as np
from PIL import Image
from stain_normalize import MacenkoNormalizer, compute_reference_from_dir

# Fit normalizer ke training set ALL-IDB (sekali, simpan ke pickle jika perlu)
ref_img, _, _ = compute_reference_from_dir('../dataset/train', n_samples=100)
mac = MacenkoNormalizer().fit(ref_img)

# Normalize gambar C-NMC sebelum inference
cnmc_img = np.array(Image.open('cnmc_cell.bmp').convert('RGB'))
normalized = mac.transform(cnmc_img)

img_tensor = transform(Image.fromarray(normalized)).unsqueeze(0).cuda()
with torch.no_grad():
    pred = model(img_tensor).argmax(dim=1).item()
```

### Resume Training

Edit `trainer.fit()` di `src/main.py`:

```python
trainer.fit(
    model,
    datamodule=datamodule,
    ckpt_path='../checkpoints/mha_only/last.ckpt',  # tambahkan ini
)
```

### Export ke TorchScript

```python
model = LeukemiaLightningModel.load_from_checkpoint('best.ckpt').model
scripted = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
torch.jit.save(scripted, 'leukemia_classifier.pt')
```

---

## Troubleshooting

### `_pickle.UnpicklingError` saat load checkpoint (PyTorch >= 2.6)

```text
Weights only load failed.
GLOBAL numpy._core.multiarray.scalar was not an allowed global by default.
```

**Penyebab:** PyTorch 2.6 mengubah default `weights_only=True`. Checkpoint menyimpan numpy
scalar yang tidak ada di safe globals list.

**Status:** Sudah diperbaiki di `main.py` dan `evaluate_external.py`.
Jika error di script lain, tambahkan di bagian atas sebelum load:

```python
import torch
import numpy._core.multiarray
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
```

### CUDA Out of Memory

```python
# Di src/main.py, ubah ExperimentConfig:
batch_size=16   # turunkan dari 32
```

Atau aktifkan gradient checkpointing:

```python
# Di lightning_model.py dalam __init__:
self.model.backbone.set_grad_checkpointing(True)
```

### Training Sangat Lambat

```bash
# Kurangi DataLoader workers jika CPU bottleneck
# Di LeukemiaDataModule: num_workers=8 (default)

# Hindari focusmix_full (Grad-CAM online paling lambat)
python main.py --exp focusmix_mha   # lebih cepat dari focusmix_full

# Matikan SLIC untuk cek overhead augmentasi
python main.py --exp mha_only
```

### Segmentation Error `*.xyc not found`

```bash
ls data/ALL_IDB1/xyc/ | head -5
# Harus ada: Im001_1.xyc  Im002_1.xyc  Im003_1.xyc ...
```

### Macenko Hang / RuntimeError

```bash
# Jalankan dengan num_workers=0 (normalizer tidak picklable lintas proses)
python evaluate_external.py \
    --ckpt ../checkpoints/mha_only/epoch=07-val_f1=1.0000.ckpt \
    --cnmc-dir "../PKG_C_NMC 2019/C-NMC_train_merged" \
    --num-workers 0

# Jika masih crash, skip Macenko dan pakai Reinhard saja
python evaluate_external.py ... --no-macenko
```

### `No images found` pada C-NMC

Penyebab paling umum: menggunakan split test (prelim/final) yang berisi flat files tanpa
subdirektori kelas. Gunakan `C-NMC_train_merged` atau salah satu fold di `C-NMC_training_data`.

```bash
# Benar — ada subdirektori all/ dan hem/
ls "PKG_C_NMC 2019/C-NMC_train_merged/"
# Output: all/  hem/

# Salah — flat files tanpa label
ls "PKG_C_NMC 2019/C-NMC_test_prelim_phase_data/"
# Output: 1.bmp  2.bmp  3.bmp  ...  (tidak bisa dievaluasi)
```

Script mendukung layout `all/hem/`, `Abnormal/Normal/`, `ALL/HEM/`, `positive/negative/`.

### `ModuleNotFoundError: stain_normalize`

```bash
# Selalu jalankan dari direktori src/
cd src
python evaluate_external.py ...   # BENAR

# Bukan dari root:
python src/evaluate_external.py ...  # salah (import akan gagal)
```

### Early Stopping Terlalu Cepat

Edit patience di `src/main.py`:

```python
EarlyStopping(monitor='val_loss', mode='min', patience=15),
#                                                       ^^ naikkan dari 10
```

---

## Referensi

### Paper Utama

- **FocusAugMix**: Mustaqim T., Fatichah C., Suciati N., Obi T., Lee J. (2025).
  *FocusAugMix: A data augmentation method for enhancing Acute Lymphoblastic Leukemia classification.*
  Intelligent Systems With Applications, 26, 200512.
  [https://doi.org/10.1016/j.iswa.2025.200512](https://doi.org/10.1016/j.iswa.2025.200512)
- **ConvNeXt V2**: Woo S., et al. (2023).
  *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders.*
  CVPR 2023.
- **SaliencyMix**: Uddin A.F.M.S., et al. (2021).
  *SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization.*
  ICLR 2021.
- **Macenko Stain Normalization**: Macenko M., et al. (2009).
  *A method for normalizing histology slides for quantitative analysis.*
  ISBI 2009.
- **Reinhard Color Transfer**: Reinhard E., et al. (2001).
  *Color transfer between images.*
  IEEE Computer Graphics and Applications, 21(5), 34-41.

### Dataset

```bibtex
@article{labati2011allidb,
  title   = {ALL-IDB: The Acute Lymphoblastic Leukemia Image Database for Image Processing},
  author  = {Labati, R. D. and Piuri, V. and Scotti, F.},
  journal = {Proc. IEEE ICIP},
  year    = {2011}
}

@article{gupta2019cnmc,
  title   = {Preparation of a comprehensive leukocyte dataset},
  author  = {Gupta, A. and Gupta, R.},
  journal = {Scientific Data},
  year    = {2019},
  doi     = {10.1038/s41597-019-0054-7}
}
```

---

Last updated: 2026-06-05
