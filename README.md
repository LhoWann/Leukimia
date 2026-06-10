# Leukemia Classification: FocusAugMix + ConvNeXt V2

Deteksi Otomatis Acute Lymphoblastic Leukemia (ALL) dari Citra Apusan Darah Tepi

Model dilatih pada **ALL-IDB** (Giemsa stain, Italia) dan dievaluasi secara eksternal pada
**C-NMC 2019** (Wright-Giemsa, India) dengan stain normalization untuk mengukur generalisasi
lintas-domain. Repositori ini menyediakan pipeline **end-to-end yang dapat direproduksi**: dari
preprocessing dataset mentah, training multi-seed, baseline pembanding (CoAtNet-0), evaluasi
lintas-domain, uji signifikansi statistik, benchmark kompleksitas, sampai pembuatan figur paper.

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Arsitektur &amp; Metodologi](#arsitektur--metodologi)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Reproduksi Lengkap (TL;DR)](#reproduksi-lengkap-tldr)
- [Alur Kerja Detail](#alur-kerja-detail)
  - [1. Persiapan Data](#1-persiapan-data)
  - [2. Training Satu Eksperimen](#2-training-satu-eksperimen)
  - [3. Daftar Eksperimen](#3-daftar-eksperimen)
  - [4. Validasi Multi-Seed](#4-validasi-multi-seed-run_multiseedpy)
  - [5. Baseline CoAtNet-0](#5-baseline-coatnet-0)
  - [6. Agregasi Hasil](#6-agregasi-hasil-aggregate_seedspy)
  - [7. Uji Signifikansi Statistik](#7-uji-signifikansi-statistik-significance_testpy)
  - [8. Benchmark Kompleksitas](#8-benchmark-kompleksitas-complexity_benchmarkpy)
  - [9. Figur Confusion Matrix](#9-figur-confusion-matrix-plot_confusion_matricespy)
  - [10. Monitoring Training](#10-monitoring-training)
  - [11. Evaluasi In-Domain](#11-evaluasi-in-domain)
  - [12. Evaluasi Eksternal C-NMC 2019](#12-evaluasi-eksternal-c-nmc-2019-evaluate_externalpy)
- [Stain Normalization](#stain-normalization)
- [Output &amp; Checkpoint](#output--checkpoint)
- [Hyperparameter Lengkap](#hyperparameter-lengkap)
- [Troubleshooting](#troubleshooting)
- [Referensi](#referensi)

---

## Latar Belakang

ALL adalah jenis kanker darah paling umum pada anak-anak. Diagnosis awal memerlukan identifikasi
*blast cells* (sel leukemia) dari apusan darah secara mikroskopis — proses yang memakan waktu
dan bergantung pada keahlian patologis.

Proyek ini mengimplementasikan sistem klasifikasi otomatis berbasis:

- **ConvNeXt V2 Tiny** — backbone pretrained FCMAE + IN22k + IN1k (~28.5 M params)
- **Multi-Head Self-Attention (MHA)** — injeksi perhatian spasial setelah stage backbone (opsional; terbukti
  memperburuk generalisasi lintas-domain pada dataset kecil ini, lihat [experiment_results.md](experiment_results.md))
- **FocusAugMix** — mixing augmentation berbasis SLIC superpixel + saliency + Grad-CAM
- **ReinhardJitter (train-time stain augmentation)** — randomisasi statistik warna LAB saat training;
  **kontribusi utama** yang menghasilkan model paling robust lintas-domain (`focusmix_stain`)
- **Weighted Focal Loss** — Focal Loss (γ=2.0) + inverse-frequency class weights untuk menangani imbalance
- **Stain Normalization** — Macenko & Reinhard untuk evaluasi lintas-domain (test-time)
- **Baseline CoAtNet-0** — backbone hybrid conv+attention + CutMix/Mixup, untuk perbandingan *fair*
  dengan protokol training identik (lihat [experiment_results.md](experiment_results.md))

> **Ringkasan temuan (divalidasi 3 seed):** kontribusi utama adalah **`focusmix_stain`** (FocusAugMix +
> ReinhardJitter σ=0.15, tanpa MHA). Pada single-seed ia mencapai F1 lintas-domain 0.635, tetapi setelah
> validasi **3 seed (42/123/2025)** rata-ratanya **0.5535 ± 0.1189** — secara statistik setara dengan
> baseline `no_mix` (0.5636 ± 0.0817). Pembeda nyata stain aug adalah **keseimbangan recall** (Abn/Norm
> 48%/76% vs `focusmix` murni yang kolaps 13%/95%), bukan F1 absolut, dan dicapai **tanpa normalisasi
> test-time**. Detail lengkap + ablation TTA-8 di [experiment_results.md](experiment_results.md).

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
mengukur generalisasi lintas-domain (10.661 sel di `C-NMC_train_merged`).

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

Mode augmentasi: `none` | `saliency` | `focusmix` | `focusmix_cam` (di `src/data_module.py`).

### Train-Time Stain Augmentation: ReinhardJitter

Selain mixing, tersedia augmentasi warna khusus untuk mensimulasikan variasi protokol pewarnaan
antar-laboratorium. `ReinhardJitter` (di `src/data_module.py`) bekerja di ruang CIELAB: per channel,
nilai di-z-score lalu di-rescale dengan mean & std yang di-perturbasi secara acak.

```text
Untuk tiap channel L, A, B:
  z        = (pixel − μ) / σ
  μ_baru   = μ + N(0, σ_mean · σ)          # geser mean warna
  σ_baru   = σ · exp(N(0, σ_std))          # skala kontras warna
  pixel    = z · σ_baru + μ_baru
```

Diaktifkan via `use_robust_aug=True` dengan parameter `stain_sigma_mean`, `stain_sigma_std`, dan
`stain_aug_prob`. Intensitas σ_mean=0.15 (moderat) terbukti optimal; lebih agresif justru menurunkan
robustness (hubungan non-monoton — lihat [experiment_results.md](experiment_results.md)).

### Loss Function: Weighted Focal Loss

Kedua kelas ditangani dengan **Weighted Focal Loss** (`use_focal_loss=True`, default):

```text
FL = (1 − p_t)^γ · CE(logits, target, weight=class_weights)
γ  = 2.0
class_weights = inverse-frequency dari train set (otomatis, src/data_module.py:get_class_weights)
```

Loss diterapkan kompatibel dengan label mixing FocusAugMix:
`loss = λ·FL(target_a) + (1−λ)·FL(target_b)`, di mana λ adalah bobot mixing per-sampel.

### Arsitektur Model (Proposal)

```text
Input (B, 3, 224, 224)
   |
   v
ConvNeXt V2 Tiny — pretrained fcmae_ft_in22k_in1k
   +-- Stage 0 -> (B,  96, 56, 56)
   +-- Stage 1 -> (B, 192, 28, 28)
   +-- Stage 2 -> (B, 384, 14, 14)  [MHA disisipkan di sini jika use_mha=True]
   |                  |
   |            tokens: (B, 196, 384)
   |            MultiheadAttention (8 heads) + Residual + LayerNorm
   |
   +-- Stage 3 -> (B, 768,  7,  7)
   |
   v
AdaptiveAvgPool2d(1) -> Flatten -> Dropout(0.3) -> Linear(num_classes)
```

### Arsitektur Baseline (Pembanding)

```text
Input (B, 3, 224, 224)
   |
   v
CoAtNet-0 (timm: coatnet_0_rw_224.sw_in1k, pretrained IN1k, global_pool='avg')
   |
   v
Dropout(0.3) -> Linear(num_classes)
```

Baseline memakai **uniform LR fine-tuning** (tanpa LLRD) + **CutMix/Mixup level-batch**, tanpa
ReinhardJitter dan tanpa MHA. Selain itu protokol training identik (epoch / wd / focal / warmup /
clip / bf16). Implementasi: `CoAtNetClassifier` di `src/lightning_model.py`.

**Layer-wise Learning Rate Decay (LLRD, factor 0.75 — hanya ConvNeXtV2):**

| Layer      | LR multiplier  |
| ---------- | -------------- |
| Head / MHA | 1.00 x base_lr |
| Stage 3    | 0.75 x base_lr |
| Stage 2    | 0.56 x base_lr |
| Stage 1    | 0.42 x base_lr |
| Stage 0    | 0.32 x base_lr |
| Stem       | 0.24 x base_lr |

**Optimizer:** AdamW + Linear Warmup (3 epoch default; 5 untuk eksperimen MHA) + Cosine Decay
sepanjang `max_epochs=30`.

**Trainer settings:**

| Setting             | Nilai         | Keterangan                                       |
| ------------------- | ------------- | ------------------------------------------------ |
| `precision`         | `bf16-mixed`  | ~1.8x speedup pada GPU Ampere+                   |
| `gradient_clip_val` | 1.0           | Mencegah exploding gradient saat fine-tuning     |
| `log_every_n_steps` | 10            | Frekuensi logging ke CSV                          |
| `max_epochs`        | 30            | Early stopping `val_loss` aktif (patience 10)    |
| Checkpoint monitor  | `val_f1`      | Bukan `val_acc` — val_acc jenuh 100% sejak awal  |

**Class index mapping** (ditentukan oleh urutan alfabet folder ImageFolder):

| Index | Nama folder | Label biologis          |
| ----- | ----------- | ----------------------- |
| 0     | `Abnormal`  | Sel ALL / blast cell    |
| 1     | `Normal`    | Sel WBC normal          |

---

## Struktur Proyek

```text
LEUKIMIA/
├── src/
│   ├── segment_dataset.py         # Preprocessing ALL-IDB raw -> dataset/
│   ├── data_module.py             # Dataset, FocusAugMix, ReinhardJitter, DataModule
│   ├── lightning_model.py         # ConvNeXtV2Classifier, CoAtNetClassifier, CutMix/Mixup, GradCAM
│   ├── stain_normalize.py         # MacenkoNormalizer, ReinhardNormalizer
│   ├── main.py                    # CLI training + registry EXPERIMENTS
│   ├── run_multiseed.py           # Training + evaluasi multi-seed (3 exp kunci × 3 seed)
│   ├── aggregate_seeds.py         # Agregasi mean ± std lintas seed -> aggregate.{json,md}
│   ├── significance_test.py       # Paired t-test + Wilcoxon + Cohen's d antar model
│   ├── complexity_benchmark.py    # Params / FLOPs / latensi (ConvNeXtV2 vs CoAtNet-0)
│   ├── plot_confusion_matrices.py # Figur CM kualitas-paper (PDF + PNG)
│   └── evaluate_external.py       # Evaluasi C-NMC 2019 + stain normalization (+ TTA, ensemble)
│
├── data/                          # Raw dataset (isi manual)
│   ├── ALL_IDB1/
│   │   ├── im/                    # Im001_1.jpg, Im002_0.jpg, ...
│   │   └── xyc/                   # Im001_1.xyc (koordinat blast)
│   └── ALL_IDB2/
│       └── img/                   # Im001_1.tif, Im002_0.tif, ...
│
├── dataset/                       # Di-generate oleh segment_dataset.py
│   ├── train/{Abnormal,Normal}/
│   └── val/{Abnormal,Normal}/
│
├── PKG_C_NMC 2019/                # External test (isi manual)
│   └── C-NMC_train_merged/{all,hem}/
│
├── checkpoints/                   # Training single-run (main.py)
├── checkpoints_multiseed/         # Training multi-seed ConvNeXtV2 (run_multiseed.py)
├── checkpoints_coatnet/           # Training multi-seed baseline CoAtNet-0
├── logs/  logs_multiseed/  logs_coatnet/         # CSV metrics per run
├── results/                       # JSON evaluasi single-run
├── results_multiseed/             # JSON + aggregate.{json,md} ConvNeXtV2 (no-TTA)
├── results_multiseed_tta8/        # JSON + aggregate ablation TTA-8
├── results_coatnet/               # JSON + aggregate baseline CoAtNet-0
├── results_comparison/            # aggregate gabungan + significance.md + complexity.md
├── figures/                       # cm_*.pdf / cm_*.png untuk paper
├── experiment_results.md          # Hasil, analisis, protokol fair-comparison, temuan (doc tunggal)
├── requirements.txt
└── README.md
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
# venv (Windows / PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

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
pip install torchmetrics>=1.0.0   # diperlukan oleh lightning_model.py
```

Untuk CUDA support (jika PyTorch terinstall tanpa CUDA):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Paket inti yang dipakai: `torch`, `torchvision`, `lightning`, `timm`, `torchmetrics`,
`scikit-image` (SLIC), `opencv-python` (warna/HSV), `scikit-learn` (metrik), `scipy` (uji statistik),
`matplotlib` (figur), `tqdm`, `Pillow`, `numpy`.

### Langkah 4 — Verifikasi

```bash
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import lightning, timm, torchmetrics; print('lightning', lightning.__version__, '| timm', timm.__version__)"
python -c "import cv2, skimage, sklearn, scipy, matplotlib; print('cv/skimage/sklearn/scipy/mpl OK')"
```

---

## Reproduksi Lengkap (TL;DR)

Urutan command minimal untuk mereproduksi **seluruh** hasil paper, dijalankan **dari root proyek**.
Skrip multi-seed/agregasi/figur otomatis `chdir` ke root, jadi aman dipanggil sebagai `python src/...`.

```bash
# 0. Preprocessing ALL-IDB mentah -> dataset/train, dataset/val
python src/segment_dataset.py        # (dijalankan dari src/, lihat catatan di bawah)

# 1. Proposal + baseline ConvNeXtV2: 3 eksperimen kunci × 3 seed (training + eval C-NMC, no-TTA)
python src/run_multiseed.py

# 2. Baseline CoAtNet-0 × 3 seed (artefak DIPISAH agar tidak tercampur ConvNeXtV2)
python src/run_multiseed.py --exps coatnet_0 \
    --ckpt-root checkpoints_coatnet --log-root logs_coatnet --results-root results_coatnet

# 3. (Opsional) Ablation TTA-8 di checkpoint yang sudah ada — tanpa latih ulang
python src/run_multiseed.py --tta-n 8 --no-train --results-root results_multiseed_tta8

# 4. Agregasi mean ± std. Gabungkan ConvNeXtV2 + CoAtNet ke satu tabel perbandingan
python src/aggregate_seeds.py --results-dir results_multiseed
python src/aggregate_seeds.py --results-dir results_multiseed results_coatnet --out-dir results_comparison
python src/aggregate_seeds.py --results-dir results_multiseed_tta8     # tabel ablation TTA

# 5. Uji signifikansi statistik antar model (paired t-test + Wilcoxon + Cohen's d)
python src/significance_test.py

# 6. Benchmark kompleksitas (params / FLOPs / latensi)
python src/complexity_benchmark.py

# 7. Figur confusion matrix untuk paper (PDF + PNG)
python src/plot_confusion_matrices.py
```

Artefak akhir yang dihasilkan:

| Artefak                                          | Dihasilkan oleh                |
| ------------------------------------------------ | ------------------------------ |
| `results_multiseed/aggregate.{json,md}`          | `aggregate_seeds.py`           |
| `results_coatnet/aggregate.{json,md}`            | `aggregate_seeds.py`           |
| `results_comparison/aggregate.{json,md}`         | `aggregate_seeds.py` (gabung)  |
| `results_comparison/significance.md`             | `significance_test.py`         |
| `results_comparison/complexity.md`               | `complexity_benchmark.py`      |
| `figures/cm_no_norm_3models.{pdf,png}`           | `plot_confusion_matrices.py`   |
| `figures/cm_focusmix_stain_conditions.{pdf,png}` | `plot_confusion_matrices.py`   |

---

## Alur Kerja Detail

### 1. Persiapan Data

#### Download ALL-IDB

- https://scotti.di.unimi.it/all/

```bash
mkdir -p data/ALL_IDB1/im data/ALL_IDB1/xyc data/ALL_IDB2/img

cp /path/to/ALL_IDB1/im/*.jpg   data/ALL_IDB1/im/
cp /path/to/ALL_IDB1/xyc/*.xyc  data/ALL_IDB1/xyc/
cp /path/to/ALL_IDB2/img/*.tif  data/ALL_IDB2/img/
```

#### Jalankan Segmentasi

`segment_dataset.py` memakai path relatif (`data/`, `dataset/`) sehingga dijalankan dari `src/`:

```bash
cd src
python segment_dataset.py
cd ..
```

Script akan:

1. Membaca ALL-IDB1 — crop sel abnormal (dari koordinat `.xyc`) + deteksi normal (HSV thresholding
   karena `.xyc` healthy donor kosong)
2. Membaca ALL-IDB2 — salin sel individu, resize ke 257×257
3. Split 80/20 per-gambar (tidak ada leakage)
4. Output ke `dataset/train` & `dataset/val`

> **Catatan:** script meng-`rmtree` `dataset/` di awal lalu membangun ulang. Jangan letakkan data lain di sana.

---

### 2. Training Satu Eksperimen

`main.py` dijalankan dari `src/` (path data relatif default `dataset`).

```bash
cd src

# Proposal utama
python main.py --exp focusmix_stain

# Seed & data-dir kustom
python main.py --exp focusmix_stain --seed 123 --data-dir ../dataset

# Semua eksperimen berurutan (termasuk baseline coatnet_0)
python main.py --all
```

Argumen `main.py`:

| Argumen      | Default      | Keterangan                                         |
| ------------ | ------------ | -------------------------------------------------- |
| `--exp`      | `focusmix`   | Nama eksperimen dari registry `EXPERIMENTS`        |
| `--data-dir` | `dataset`    | Folder berisi `train/` dan `val/`                  |
| `--seed`     | `42`         | Random seed                                        |
| `--all`      | (flag)       | Jalankan semua eksperimen berurutan                |

Checkpoint terbaik (monitor `val_f1`) + `last.ckpt` disimpan ke `checkpoints/<exp>/`.
Di akhir training otomatis dijalankan `trainer.validate` pada checkpoint terbaik.

---

### 3. Daftar Eksperimen

Sebelas eksperimen terdaftar di `EXPERIMENTS` (`src/main.py`). **`focusmix_stain` adalah proposal utama**;
`coatnet_0` adalah **baseline pembanding fair**.

| Eksperimen                | Backbone     | MHA | aug_mode       | Mixing batch  | Stain aug (σ_mean/prob) | F1 lintas-domain¹ | Tujuan                          |
| ------------------------- | ------------ | --- | -------------- | ------------- | ----------------------- | :---------------: | ------------------------------- |
| `no_mix`                  | ConvNeXtV2   | No  | `none`         | –             | –                       | 0.540             | Baseline (augmentasi dasar)     |
| `no_mix_mha`              | ConvNeXtV2   | Yes | `none`         | –             | –                       | 0.266             | Isolasi kontribusi MHA          |
| `saliency`                | ConvNeXtV2   | No  | `saliency`     | –             | –                       | 0.546             | SaliencyMix murni               |
| `focusmix`                | ConvNeXtV2   | No  | `focusmix`     | –             | –                       | 0.424             | FocusAugMix murni               |
| `focusmix_mha`            | ConvNeXtV2   | Yes | `focusmix`     | –             | –                       | 0.338             | FocusAugMix + MHA               |
| `focusmix_mha_strong`     | ConvNeXtV2   | Yes | `focusmix`     | –             | – (paste 0.30)          | 0.434             | FocusAugMix + MHA, paste besar  |
| `focusmix_cam`            | ConvNeXtV2   | Yes | `focusmix_cam` | –             | –                       | 0.433             | + Grad-CAM online               |
| **`focusmix_stain`**      | ConvNeXtV2   | No  | `focusmix`     | –             | 0.15 / 0.5              | **0.635**         | **Proposal — terbaik**          |
| `focusmix_stain_strong`   | ConvNeXtV2   | No  | `focusmix`     | –             | 0.25 / 0.7              | 0.413             | Stain aug kuat                  |
| `focusmix_stain_max`      | ConvNeXtV2   | No  | `focusmix`     | –             | 0.35 / 0.8              | 0.506             | Stain aug maksimal              |
| `coatnet_0`               | CoAtNet-0    | No  | `none`         | CutMix/Mixup  | –                       | (3-seed)          | **Baseline pembanding**         |

> ¹ F1 macro pada C-NMC no-norm, threshold 0.5, **single-seed (42)** — kolom ini adalah ablation lengkap.
> **Angka headline = mean ± std atas 3 seed** untuk tiga eksperimen kunci (`no_mix` 0.5636 ± 0.0817,
> `focusmix_stain` 0.5535 ± 0.1189, `focusmix` 0.3486 ± 0.1405) — lihat
> [experiment_results.md](experiment_results.md).

#### Menambah Eksperimen Baru

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

### 4. Validasi Multi-Seed (`run_multiseed.py`)

Melatih tiap (eksperimen, seed) ke direktori terpisah, lalu menjalankan evaluasi C-NMC dan menyimpan
**satu JSON per (eksperimen, seed)** dengan field `seed`/`exp` yang siap diagregasi.

Default: `KEY_EXPERIMENTS = ['focusmix_stain', 'no_mix', 'focusmix']` × `SEEDS = [42, 123, 2025]`,
TTA-1 (no-TTA, headline).

```bash
# Dari root proyek — 3 exp kunci × 3 seed
python src/run_multiseed.py

# Satu eksperimen saja
python src/run_multiseed.py --exps focusmix_stain

# Subset seed
python src/run_multiseed.py --seeds 42 123

# Lewati run yang checkpoint/JSON-nya sudah ada
python src/run_multiseed.py --skip-existing

# Hanya evaluasi checkpoint yang ada (tanpa latih ulang)
python src/run_multiseed.py --no-train

# Ablation TTA-8 di checkpoint yang ada -> folder terpisah agar tidak menimpa headline
python src/run_multiseed.py --tta-n 8 --no-train --results-root results_multiseed_tta8
```

Argumen lengkap:

| Argumen           | Default                                | Keterangan                                              |
| ----------------- | -------------------------------------- | ------------------------------------------------------- |
| `--exps`          | `focusmix_stain no_mix focusmix`       | Daftar eksperimen (dari registry)                       |
| `--seeds`         | `42 123 2025`                          | Daftar seed                                             |
| `--data-dir`      | `dataset`                              | Folder ALL-IDB                                          |
| `--cnmc-dir`      | `PKG_C_NMC 2019/C-NMC_train_merged`    | Folder C-NMC berlabel                                   |
| `--tta-n`         | `1`                                    | 1 = no-TTA (headline); 8 = ablation                     |
| `--results-root`  | `results_multiseed`                    | Folder output JSON                                      |
| `--ckpt-root`     | `checkpoints_multiseed`                | Folder checkpoint per-run                               |
| `--log-root`      | `logs_multiseed`                       | Folder CSV log per-run                                  |
| `--no-train`      | (flag)                                 | Lewati training                                         |
| `--no-eval`       | (flag)                                 | Lewati evaluasi                                         |
| `--skip-existing` | (flag)                                 | Lewati run yang checkpoint/JSON-nya sudah ada           |

Output: `checkpoints_multiseed/<exp>_seed<seed>/`, `logs_multiseed/<exp>_seed<seed>/`,
`results_multiseed/<exp>_seed<seed>.json`, dan `results_multiseed/manifest.json`.

---

### 5. Baseline CoAtNet-0

Untuk perbandingan **fair**, baseline dilatih dengan skrip yang sama tetapi artefaknya **dipisah**
ke folder tersendiri (agar tidak tercampur dengan ConvNeXtV2):

```bash
python src/run_multiseed.py --exps coatnet_0 \
    --ckpt-root checkpoints_coatnet \
    --log-root  logs_coatnet \
    --results-root results_coatnet
```

Ini menghasilkan `results_coatnet/coatnet_0_seed{42,123,2025}.json` yang nanti ikut teragregasi oleh
`aggregate_seeds.py` saat folder ini disertakan. Detail protokol: [experiment_results.md](experiment_results.md).

---

### 6. Agregasi Hasil (`aggregate_seeds.py`)

Membaca semua `<exp>_seed<seed>.json`, menghitung **mean ± std** F1 macro & accuracy lintas seed per
kondisi (no-norm / Macenko / Reinhard), menurunkan recall per-kelas dari confusion matrix, lalu menulis
`aggregate.json` + `aggregate.md`.

```bash
# Hanya ConvNeXtV2 (no-TTA)
python src/aggregate_seeds.py --results-dir results_multiseed

# Gabungkan ConvNeXtV2 + CoAtNet ke satu tabel perbandingan
python src/aggregate_seeds.py --results-dir results_multiseed results_coatnet \
    --out-dir results_comparison

# Tabel ablation TTA-8
python src/aggregate_seeds.py --results-dir results_multiseed_tta8
```

| Argumen        | Default               | Keterangan                                                |
| -------------- | --------------------- | --------------------------------------------------------- |
| `--results-dir`| `results_multiseed`   | Satu/lebih folder hasil per-seed untuk digabung           |
| `--out-dir`    | folder pertama        | Folder output `aggregate.{json,md}`                       |

---

### 7. Uji Signifikansi Statistik (`significance_test.py`)

Paired t-test + Wilcoxon signed-rank + Cohen's d berpasangan atas F1 macro per-seed (42/123/2025),
membaca JSON yang sama dengan agregasi. Membandingkan proposal vs baseline (CoAtNet-0, `no_mix`, `focusmix`).

```bash
python src/significance_test.py                          # kondisi cnmc_no_norm (default)
python src/significance_test.py --condition cnmc_reinhard
python src/significance_test.py --condition cnmc_macenko
```

| Argumen       | Default                                  | Keterangan                                        |
| ------------- | ---------------------------------------- | ------------------------------------------------- |
| `--condition` | `cnmc_no_norm`                           | `cnmc_no_norm` / `cnmc_macenko` / `cnmc_reinhard` |
| `--out`       | `results_comparison/significance.md`     | Tabel markdown output                             |

> **Catatan:** n=3 seed → daya uji rendah. Wilcoxon n=3 tak pernah mencapai p<0.25. Baca paired t-test +
> Cohen's d (d>0.8 = efek besar) sebagai indikator utama, bukan p Wilcoxon. Memerlukan `results_coatnet/`
> sudah terisi (langkah 5).

---

### 8. Benchmark Kompleksitas (`complexity_benchmark.py`)

Membandingkan **params / FLOPs / GMACs / latensi** ConvNeXtV2-Tiny (proposal, tanpa MHA) vs CoAtNet-0
(baseline) pada input 224×224. Tidak butuh checkpoint (pakai bobot acak; metrik tak bergantung nilai bobot).

```bash
python src/complexity_benchmark.py
python src/complexity_benchmark.py --iters 100 --device cuda
```

| Argumen    | Default                              | Keterangan                                   |
| ---------- | ------------------------------------ | -------------------------------------------- |
| `--iters`  | `50`                                 | Iterasi pengukuran latensi (batch=1)         |
| `--device` | `auto`                               | `auto` / `cpu` / `cuda`                      |
| `--out`    | `results_comparison/complexity.md`   | Tabel markdown output                        |

---

### 9. Figur Confusion Matrix (`plot_confusion_matrices.py`)

Merata-ratakan confusion matrix lintas seed (C-NMC = sel yang sama tiap seed) lalu memplot heatmap
row-normalized (recall) kualitas-paper. Menghasilkan PDF (LaTeX) + PNG (pratinjau).

```bash
python src/plot_confusion_matrices.py
```

Output:

- `figures/cm_no_norm_3models.{pdf,png}` — CoAtNet-0 vs ConvNeXtV2 `no_mix` vs `focusmix_stain` (no-norm)
- `figures/cm_focusmix_stain_conditions.{pdf,png}` — proposal pada 3 kondisi (No-norm / Macenko / Reinhard)

> Memerlukan `results_multiseed/` dan `results_coatnet/` sudah terisi (langkah 1, 2, 5).

---

### 10. Monitoring Training

#### CSV Logs (Default)

File metrics tersimpan di `logs/<exp>/version_0/metrics.csv` (atau `logs_multiseed/...` untuk multi-seed).

| Kolom              | Keterangan                         |
| ------------------ | ---------------------------------- |
| `train_loss`       | Loss per step                      |
| `train_loss_epoch` | Loss rata-rata per epoch           |
| `val_loss`         | Validation loss                    |
| `val_acc`          | Validation accuracy                |
| `val_f1`           | F1 macro                           |
| `val_precision`    | Precision macro                    |
| `val_recall`       | Recall macro                       |
| `lr-AdamW`         | Learning rate head/MHA per epoch   |

Plot training curve:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs/focusmix_stain/version_0/metrics.csv')
train = df.dropna(subset=['train_loss_epoch'])
val   = df.dropna(subset=['val_loss'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train['epoch'], train['train_loss_epoch'], label='train loss')
axes[0].plot(val['epoch'],   val['val_loss'],           label='val loss')
axes[0].set_title('Loss'); axes[0].legend()
axes[1].plot(val['epoch'], val['val_acc'], color='green', label='val acc')
axes[1].set_title('Val Accuracy'); axes[1].legend()
plt.tight_layout(); plt.savefig('training_curve.png', dpi=150)
```

#### TensorBoard (Opsional)

Ganti logger di `src/main.py`:

```python
from lightning.pytorch.loggers import TensorBoardLogger
logger = TensorBoardLogger('logs', name=run_name)
```

```bash
tensorboard --logdir=logs   # buka http://localhost:6006
```

---

### 11. Evaluasi In-Domain

Evaluasi pada validation set ALL-IDB. Otomatis berjalan di akhir training; bisa juga manual:

```bash
cd src
python - << 'EOF'
import torch, numpy._core.multiarray
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

import lightning as L
from lightning_model import LeukemiaLightningModel
from data_module import LeukemiaDataModule

ckpt = '../checkpoints/focusmix_stain/epoch=06-val_f1=1.0000.ckpt'
model = LeukemiaLightningModel.load_from_checkpoint(ckpt)

dm = LeukemiaDataModule(data_dir='../dataset', batch_size=32); dm.setup()
trainer = L.Trainer(accelerator='auto', devices=1, logger=False, enable_checkpointing=False)
trainer.validate(model, datamodule=dm)
EOF
```

---

### 12. Evaluasi Eksternal C-NMC 2019 (`evaluate_external.py`)

#### Download C-NMC 2019

- https://www.cancerimagingarchive.net/collection/c-nmc-2019/

#### Struktur Direktori C-NMC

Gunakan `C-NMC_train_merged` — satu-satunya split dengan label kelas publik. Split test (prelim & final)
berisi file flat tanpa label sehingga tidak dapat dievaluasi.

```text
PKG_C_NMC 2019/
├── C-NMC_train_merged/
│   ├── all/          <- sel ALL (leukemia)  [pemetaan: Abnormal]
│   └── hem/          <- sel HEM (normal)    [pemetaan: Normal]
├── C-NMC_training_data/{fold_0,fold_1,fold_2}/{all,hem}/
├── C-NMC_test_prelim_phase_data/   <- flat files, tanpa label (tidak bisa dievaluasi)
└── C-NMC_test_final_phase_data/    <- flat files, tanpa label (tidak bisa dievaluasi)
```

Format gambar didukung: `.jpg`, `.jpeg`, `.bmp`, `.png`, `.tif`, `.tiff`.
Layout kelas yang dikenali: `all/hem`, `Abnormal/Normal`, `ALL/HEM`, `positive/negative`.

#### Mode Auto (semua eksperimen di `checkpoints/` sekaligus)

```bash
python src/evaluate_external.py \
    --cnmc-dir "PKG_C_NMC 2019/C-NMC_train_merged" \
    --data-dir dataset
```

#### Mode Single Model

```bash
cd src
python evaluate_external.py \
    --ckpt        ../checkpoints/focusmix_stain/epoch=06-val_f1=1.0000.ckpt \
    --cnmc-dir    "../PKG_C_NMC 2019/C-NMC_train_merged" \
    --data-dir    ../dataset \
    --output-json ../results/cnmc_eval_focusmix_stain.json
```

#### Mode Ensemble & TTA

```bash
# Ensemble beberapa eksperimen
python evaluate_external.py --ensemble focusmix_stain no_mix focusmix \
    --cnmc-dir "../PKG_C_NMC 2019/C-NMC_train_merged"

# Test-Time Augmentation 8 view
python evaluate_external.py --ckpt <ckpt> --cnmc-dir <dir> --tta-n 8
```

Tiap evaluasi menjalankan hingga **4 kondisi**: ALL-IDB val (in-domain), C-NMC no-norm, C-NMC + Macenko,
C-NMC + Reinhard, plus threshold terkalibrasi.

#### Argumen Lengkap

| Argumen          | Default        | Keterangan                                                    |
| ---------------- | -------------- | ------------------------------------------------------------- |
| `--ckpt`         | `None`         | Path satu `.ckpt`. Kosongkan untuk auto-scan `--ckpt-dir`     |
| `--ckpt-dir`     | `checkpoints`  | Root folder berisi subdir per-eksperimen                      |
| `--cnmc-dir`     | **(wajib)**    | Folder C-NMC berlabel (berisi `all/` dan `hem/`)              |
| `--data-dir`     | `dataset`      | Folder ALL-IDB (untuk val + referensi stain)                 |
| `--batch-size`   | `32`           | Batch inference                                               |
| `--num-workers`  | `2`            | Worker DataLoader (gunakan `0` jika hang)                     |
| `--image-size`   | `224`          | Ukuran resize                                                 |
| `--ref-samples`  | `100`          | Gambar training untuk referensi stain                        |
| `--device`       | `auto`         | `auto` / `cpu` / `cuda`                                       |
| `--no-macenko`   | (flag)         | Skip Macenko                                                  |
| `--no-reinhard`  | (flag)         | Skip Reinhard                                                 |
| `--skip-val`     | (flag)         | Skip evaluasi ALL-IDB val                                     |
| `--tta-n`        | `1`            | TTA passes (1 = nonaktif, 8 = direkomendasikan)              |
| `--ensemble`     | `None`         | Daftar eksperimen untuk evaluasi ensemble                     |
| `--output-json`  | `None`         | Simpan metrics single-model ke JSON                           |
| `--results-dir`  | `results`      | Folder output JSON (mode auto)                                |

#### Contoh Output

Hasil nyata `focusmix_stain` pada `C-NMC_train_merged` (10.661 sel):

```text
────────────────────────────────────────────────────────────
  C-NMC 2019  .  No Stain Normalization
────────────────────────────────────────────────────────────
  N samples  : 10661
  Accuracy   : 0.6581  (65.8%)
  F1 (macro) : 0.6351

════════════════════════════════════════════════════════════
  SUMMARY
════════════════════════════════════════════════════════════
  Condition                                 Acc    F1
  ──────────────────────────────────────── ────── ──────
  ALL-IDB val (in-domain)                 1.0000 1.0000
  C-NMC -- no normalization               0.6581 0.6351
  C-NMC -- Macenko                        0.7057 0.6225
  C-NMC -- Reinhard                       0.5606 0.5559

  Optimal threshold (calibrated on no-norm, class=Normal): 0.55
  Domain-shift gap (val_acc - raw_acc) : +0.3419
```

#### Interpretasi Domain-Shift Gap

| Gap       | Interpretasi                                                            |
| --------- | ----------------------------------------------------------------------- |
| < 0.05    | Model robust — belajar morfologi sel, tidak bergantung staining        |
| 0.05-0.10 | Ketergantungan staining moderat                                         |
| > 0.10    | Ketergantungan staining signifikan; normalisasi sangat direkomendasikan |

> **Catatan penting:** gap sendiri bisa menyesatkan pada data tidak seimbang. Selalu baca gap **bersama**
> F1 macro dan recall per-kelas. Dalam konteks klinis, **False Negative Abnormal** (sel leukemia diprediksi
> Normal) jauh lebih berbahaya dari False Positive — periksa confusion matrix, bukan hanya accuracy agregat.

---

## Stain Normalization

### Konsep

Model dilatih di ALL-IDB (Giemsa, Italia), ditest di C-NMC (Wright-Giemsa, India) → penurunan performa
karena **distribusi warna berbeda**, bukan morfologi sel. Stain normalization memetakan C-NMC agar
terlihat seperti ALL-IDB sebelum inference.

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
from stain_normalize import MacenkoNormalizer, ReinhardNormalizer, compute_reference_from_dir

# Hitung referensi dari training set ALL-IDB (sekali saja)
ref_image, rh_mean, rh_std = compute_reference_from_dir('../dataset/train', n_samples=100, image_size=224)

# Macenko
mac = MacenkoNormalizer(luminosity_threshold=0.15, angular_percentile=99).fit(ref_image)
cnmc_np = np.array(Image.open('cnmc_cell.jpg').convert('RGB'))
normalized_mac = mac.transform(cnmc_np)          # HxWx3 uint8

# Reinhard (dari statistik agregat dataset)
rh = ReinhardNormalizer().fit_from_stats(rh_mean, rh_std)
normalized_rh = rh.transform(cnmc_np)            # HxWx3 uint8
```

---

## Output & Checkpoint

### Load Checkpoint untuk Inference

```python
import torch, numpy._core.multiarray
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

from lightning_model import LeukemiaLightningModel
from torchvision import transforms
from PIL import Image

model = LeukemiaLightningModel.load_from_checkpoint(
    '../checkpoints/focusmix_stain/epoch=06-val_f1=1.0000.ckpt', map_location='cuda',
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = transform(Image.open('cell.jpg').convert('RGB')).unsqueeze(0).cuda()
with torch.no_grad():
    pred = model(img).argmax(dim=1).item()
print({0: 'Abnormal (ALL)', 1: 'Normal'}[pred])
```

### Resume Training

Tambahkan `ckpt_path` di `trainer.fit()` (`src/main.py`):

```python
trainer.fit(model, datamodule=datamodule, ckpt_path='../checkpoints/focusmix_stain/last.ckpt')
```

### Export ke TorchScript

```python
model = LeukemiaLightningModel.load_from_checkpoint('best.ckpt').model
scripted = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
torch.jit.save(scripted, 'leukemia_classifier.pt')
```

---

## Hyperparameter Lengkap

Semua dapat di-override per-eksperimen di `ExperimentConfig` (`src/main.py`).

| Parameter            | Default | Deskripsi                                            |
| -------------------- | ------- | ---------------------------------------------------- |
| `batch_size`         | 32      | Turunkan ke 16 jika GPU OOM                          |
| `lr`                 | 1e-4    | Base learning rate untuk head / MHA                  |
| `weight_decay`       | 0.05    | AdamW weight decay                                   |
| `llrd`               | 0.75    | Layer-wise LR decay factor (ConvNeXtV2 saja)         |
| `label_smoothing`    | 0.0     | Label smoothing di CrossEntropy (default nonaktif)   |
| `use_focal_loss`     | True    | Aktifkan Weighted Focal Loss                         |
| `focal_gamma`        | 2.0     | Faktor fokus Focal Loss                              |
| `max_epochs`         | 30      | Maksimum epoch (early stopping `val_loss` patience 10)|
| `warmup_epochs`      | 3       | Epoch linear warmup (5 untuk eksperimen MHA)         |
| `aug_prob`           | 0.5     | Probabilitas augmentasi mixing per sampel            |
| `paste_ratio`        | 0.25    | Fraksi superpixel yang di-paste                      |
| `n_segments`         | 50      | Jumlah superpixel SLIC                               |
| `mha_stage`          | 2       | Stage backbone tempat MHA disisipkan (0-3)           |
| `use_robust_aug`     | False   | Aktifkan ReinhardJitter + augmentasi geometrik kuat  |
| `stain_sigma_mean`   | 0.15    | Std perturbasi mean warna LAB (ReinhardJitter)       |
| `stain_sigma_std`    | 0.10    | Std perturbasi kontras warna LAB                     |
| `stain_aug_prob`     | 0.5     | Probabilitas penerapan ReinhardJitter per sampel     |
| `backbone`           | convnextv2 | `convnextv2` atau `coatnet`                       |
| `mixing`             | none    | `none` atau `cutmix_mixup` (level-batch)             |
| `cutmix_alpha`       | 1.0     | Beta(α,α) untuk CutMix                               |
| `mixup_alpha`        | 0.2     | Beta(α,α) untuk Mixup                                |
| `mix_prob`           | 0.5     | Probabilitas menerapkan CutMix/Mixup per batch       |

---

## Troubleshooting

### `_pickle.UnpicklingError` saat load checkpoint (PyTorch ≥ 2.6)

```text
Weights only load failed. GLOBAL numpy._core.multiarray.scalar was not an allowed global by default.
```

**Penyebab:** PyTorch 2.6 mengubah default `weights_only=True`. Sudah ditangani di `main.py` dan
`evaluate_external.py`. Untuk script lain, tambahkan sebelum load:

```python
import torch, numpy._core.multiarray
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
```

### CUDA Out of Memory

Turunkan `batch_size=16` di `ExperimentConfig`, atau aktifkan gradient checkpointing di
`lightning_model.py` (`__init__`): `self.model.backbone.set_grad_checkpointing(True)`.

### Training Sangat Lambat

`focusmix_cam` (Grad-CAM online) paling lambat dan memaksa `num_workers=0`. Untuk kecepatan gunakan
`focusmix_stain` atau `no_mix`.

### Macenko Hang / RuntimeError

Normalizer tidak picklable lintas proses → jalankan dengan `--num-workers 0`. Jika masih crash, pakai
`--no-macenko` (Reinhard saja).

### `No images found` pada C-NMC

Pastikan memakai `C-NMC_train_merged` (ada subdir `all/` dan `hem/`), bukan split test (flat files).

```bash
ls "PKG_C_NMC 2019/C-NMC_train_merged/"   # harus: all/  hem/
```

### `ModuleNotFoundError: stain_normalize` / `main`

`main.py` dan `evaluate_external.py` single-mode mengandalkan import relatif → jalankan dari `src/`.
Skrip `run_multiseed.py`, `aggregate_seeds.py`, `significance_test.py`, `complexity_benchmark.py`,
`plot_confusion_matrices.py` otomatis `chdir` ke root dan menambah `src/` ke `sys.path` → aman dipanggil
sebagai `python src/<script>.py` dari root.

### Early Stopping Terlalu Cepat

Naikkan patience di `src/main.py`: `EarlyStopping(monitor='val_loss', mode='min', patience=15)`.

### `significance_test.py` / `plot_confusion_matrices.py` kosong atau error

Keduanya membutuhkan hasil baseline. Jalankan langkah 5 (CoAtNet-0) sehingga `results_coatnet/` terisi
sebelum menjalankan uji signifikansi & figur.

---

## Referensi

### Paper Utama

- **FocusAugMix**: Mustaqim T., Fatichah C., Suciati N., Obi T., Lee J. (2025).
  *FocusAugMix: A data augmentation method for enhancing Acute Lymphoblastic Leukemia classification.*
  Intelligent Systems With Applications, 26, 200512.
  [https://doi.org/10.1016/j.iswa.2025.200512](https://doi.org/10.1016/j.iswa.2025.200512)
- **ConvNeXt V2**: Woo S., et al. (2023). *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders.* CVPR 2023.
- **CoAtNet**: Dai Z., et al. (2021). *CoAtNet: Marrying Convolution and Attention for All Data Sizes.* NeurIPS 2021.
- **SaliencyMix**: Uddin A.F.M.S., et al. (2021). *SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization.* ICLR 2021.
- **CutMix**: Yun S., et al. (2019). *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019.
- **Macenko Stain Normalization**: Macenko M., et al. (2009). *A method for normalizing histology slides for quantitative analysis.* ISBI 2009.
- **Reinhard Color Transfer**: Reinhard E., et al. (2001). *Color transfer between images.* IEEE CG&A, 21(5), 34-41.

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

Last updated: 2026-06-08
