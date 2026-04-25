# Leukemia Classification using FocusAugMix & ConvNeXt V2

Klasifikasi citra leukemia limfoblastik akut (ALL) ke dalam subtipe L1, L2, dan L3 menggunakan pipeline modern yang mengimplementasikan arsitektur **FocusAugMix** (paper 2025) dipadukan dengan **ConvNeXt V2 Tiny** yang dilengkapi dengan Multi-Head Attention dan Grad-CAM hooks.

## Dataset & Preprocessing

Dataset yang digunakan adalah **ALL-IDB (Acute Lymphoblastic Leukemia Image Database)** yang berisi citra mikroskopis sel darah. Data L1 dan L2 awalnya berupa *full blood smear* resolusi tinggi dengan banyak sel, sedangkan L3 adalah *single cell*.

### Segmentasi Sel Mandiri (`segment_dataset.py`)
Mengingat perbedaan format raw image, sebuah pipeline preprocessing khusus telah dibangun:
1. **Deduplikasi Hash**: Mendeteksi dan menghapus duplikasi *source image* menggunakan MD5 hash untuk mencegah *data leakage*.
2. **HSV Color Segmentation**: Mengisolasi *blast cell* yang berwarna ungu pekat dari *red blood cells* (RBC) pada smear L1/L2.
3. **Cropping & Resizing**: Sel yang terdeteksi di-crop ke *bounding box*-nya dengan padding, filter noise (minimal luas dan rasio warna), lalu di-resize seragam menjadi **257x257**.
4. **Data Splitting**: Splitting 80:20 (Train/Val) dilakukan *berdasarkan source image*, sehingga crop cell dari gambar pasien/source yang sama tidak akan bocor ke set validasi.

## FocusAugMix Data Augmentation

Alih-alih menggunakan augmentasi standar, model ini menggunakan **FocusAugMix** (`DataSetup.py`), sebuah metode augmentasi spasial mutakhir untuk citra medis:
- **SLIC Superpixels**: Memecah gambar menjadi segmen-segmen *superpixel*.
- **Spectral Residual Saliency**: Menggunakan FFT (Fast Fourier Transform) log-spectrum untuk menghasilkan peta *saliency*, mendeteksi area gambar paling informatif (nukleus/sitoplasma sel).
- **Saliency-Guided Mix**: Mencampur (*mix*) segmen paling salient dari gambar B ke gambar A. Jika model mengekspor Grad-CAM, peta saliency akan dipadukan dengan Grad-CAM.
- Dataset me-return label *mixed* `(image, target_a, target_b, lambda)`.

## Arsitektur Model

- **Backbone**: ConvNeXt V2 Tiny (`convnextv2_tiny.fcmae_ft_in22k_in1k`) via `timm`.
- **Multi-Head Attention**: Menambahkan layer self-attention di atas fitur spasial terakhir sebelum pooling.
- **Grad-CAM System**: Model telah dibungkus dengan hook maju-mundur (*forward-backward hooks*) untuk dapat me-return *class activation heatmaps* (untuk guidance FocusAugMix lanjutan).

## Konfigurasi Training

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Loss Function**: Custom `mixup_criterion` untuk CrossEntropy (karena label bersifat kontinyu akibat FocusAugMix)
- **Training Loops**: Gradient clipping (max_norm=1.0), best model checkpointing (`best_model.pth`), 20 Epochs.

## Struktur Proyek

```
.
├── main.py                 # Entry point, orchestrasi training dan evaluasi
├── model.py                # Definisi ConvNeXtV2WithAttention & GradCAM hook
├── DataSetup.py            # FocusAugMixDataset, SLIC, dan FFT Saliency
├── engine.py               # Training loop khusus dengan mixup_criterion
├── evaluate.py             # Evaluasi model pada set validasi
├── segment_dataset.py      # Pipeline deteksi sel, deduplikasi, dan splitting
├── check_crop.py           # Script utility untuk visualisasi hasil cropping
├── requirements.txt        # Dependensi Python lengkap
├── ALL_IDB Dataset/        # Dataset raw (full smears & single cells)
├── data/                   # Dataset hasil proses (dibentuk oleh segment_dataset.py)
│   ├── train/
│   └── val/
```

## Cara Menjalankan

### 1. Setup Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Segmentasi Data dan Splitting

Extract `archive (1).zip` sehingga membentuk folder `ALL_IDB Dataset/` dengan subfolder `L1`, `L2`, dan `L3`. Kemudian jalankan:

```bash
python segment_dataset.py
```
*(Proses ini akan mengidentifikasi sel pada raw image, men-deduplikasi data, membagi train/val secara aman, dan menyimpannya ke folder `data/`)*

(Opsional) Cek hasil deteksi sel:
```bash
python check_crop.py
```

### 3. Training Pipeline

```bash
python main.py
```
*(Script ini akan men-train model selama 20 epoch menggunakan augmentasi FocusAugMix, menyimpan checkpoint terbaik, dan mencetak Classification Report / Confusion Matrix di akhir).*

## Requirements

- Python 3.10+
- PyTorch 2.10.0
- torchvision
- timm
- scikit-image (untuk SLIC superpixels)
- opencv-python (untuk FFT & HSV color segmentation)
- scikit-learn
- tqdm
- numpy, pillow, scipy, matplotlib
