# Tugas: Implementasi Baseline CoAtNet-0

Dokumen ini adalah panduan implementasi baseline **CoAtNet-0** untuk keperluan
perbandingan dengan model proposal **ConvNeXtV2-Tiny + FocusAugMix** pada conference.

Pembagian tugas:

- Implementasi CoAtNet-0 sesuai spesifikasi di dokumen ini
- Implementasi ConvNeXtV2-Tiny + FocusAugMix

Agar hasil dua model ini bisa dibandingkan secara adil (*fair comparison*),
ada sejumlah komponen yang **harus sama persis** di kedua implementasi,
dan ada komponen yang **boleh berbeda** karena itulah kontribusi masing-masing.

---

## ⚠️ Update Protokol (2026-06-06) — WAJIB DIBACA

Ada **dua perubahan protokol** setelah evaluasi 10 eksperimen ConvNeXtV2 (`results/summary.json`).
Mohon disesuaikan di implementasi CoAtNet:

**1. TTA bukan lagi setting utama — angka headline = TANPA TTA.**

- **Angka utama (headline) yang masuk tabel perbandingan = no-TTA.** Ini *selling point* penelitian:
  model bekerja baik **tanpa preprocessing/TTA tambahan** (relevan untuk deployment klinis, karena
  TTA-8 = 8× biaya inference per sel).
- **TTA-8 tetap dijalankan**, tetapi hanya sebagai **ablation "+gain dengan TTA"**, bukan angka utama.
  Jangan dihapus — cukup turun pangkat.
- **Syarat fair comparison tetap berlaku:** setting TTA pada tabel perbandingan **harus sama** untuk
  kedua model. Jika headline no-TTA, maka CoAtNet **dan** ConvNeXtV2 dilaporkan no-TTA berdampingan.

**2. Wajib multi-seed (3 seed) untuk angka yang masuk klaim utama.**

- Latih dengan **3 seed: 42 / 123 / 2025**, lalu laporkan **mean ± std** F1 macro.
- Single-seed adalah kelemahan yang paling mungkin dipersoalkan reviewer — lebih penting dari TTA.
- Untuk CoAtNet: jalankan **baseline CoAtNet ×3 seed**. Eksperimen ablation boleh single-seed
  (sebut "single run" di tabelnya).

**Konsekuensi praktis untuk CoAtNet:** hasil minimal yang harus diserahkan =
**no-TTA × 3 seed** (mean ± std), pada kondisi `no_norm` dan `reinhard`. TTA-8 menyusul sebagai
ablation opsional.

---

## 1. Pembagian

| Komponen                   | CoAtNet-0                                 | ConvNeXtV2-Tiny     | Keterangan             |
| -------------------------- | ----------------------------------------- | ------------------- | ---------------------- |
| Backbone                   | **CoAtNet-0**                       | ConvNeXtV2-Tiny     | Dibedakan (kontribusi) |
| Augmentasi Mixing          | **CutMix + Mixup**                  | FocusAugMix         | Dibedakan (kontribusi) |
| Augmentasi Stain Training  | Tidak ada                                 | ReinhardJitter      | Dibedakan (kontribusi) |
| Augmentasi Dasar           | RandomFlip + RandomRotation + ColorJitter | Sama                | **Harus sama**   |
| Loss Function              | **Weighted Focal Loss**             | Weighted Focal Loss | **Harus sama**   |
| Optimizer                  | AdamW                                     | AdamW               | **Harus sama**   |
| Scheduler                  | Cosine Annealing                          | Cosine Annealing    | **Harus sama**   |
| Normalisasi Test           | no_norm + Reinhard (lapor keduanya)       | Sama                | **Harus sama**   |
| TTA Inference              | Headline no-TTA; TTA-8 = ablation         | Sama                | **Harus sama**   |
| Multi-seed                 | 3 seed (42/123/2025), lapor mean ± std    | Sama                | **Harus sama**   |
| Dataset Training           | ALL-IDB                                   | ALL-IDB             | **Harus sama**   |
| Dataset Evaluasi Eksternal | C-NMC 2019                                | C-NMC 2019          | **Harus sama**   |

---

## 2. Protokol Shared (Harus Sama di Kedua Model)

### 2.1 Dataset

| Split         | Sumber                                              | Jumlah     |
| ------------- | --------------------------------------------------- | ---------- |
| Training      | ALL-IDB1 + ALL-IDB2 (Giemsa, Italia)                | ~556 sel   |
| Validasi      | ALL-IDB (image-level split, tidak overlap training) | 204 sel    |
| External Test | C-NMC 2019 train-merged (Wright-Giemsa, India)      | 10.661 sel |

> **Penting:** C-NMC hanya dipakai untuk evaluasi, **bukan** training.
> External test terdiri dari 7.272 sel Abnormal (ALL) dan 3.389 sel Normal (HEM).

### 2.2 Augmentasi Dasar Training (Harus Identik)

```python
transforms.Resize((224, 224))
transforms.RandomHorizontalFlip()
transforms.RandomVerticalFlip()
transforms.RandomRotation(degrees=20)
transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08)
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### 2.3 Hyperparameter Training (Harus Sama)

| Parameter     | Nilai            | Catatan                                           |
| ------------- | ---------------- | ------------------------------------------------- |
| Image size    | 224 × 224       |                                                   |
| Batch size    | 32               |                                                   |
| Max epochs    | 30               |                                                   |
| Optimizer     | AdamW            |                                                   |
| Learning rate | 1e-4             |                                                   |
| Weight decay  | 0.05             |                                                   |
| LR scheduler  | Cosine Annealing | dari epoch 1 sampai epoch 30                      |
| Warmup        | 3 epoch linear   | LR naik dari 0 ke LR penuh selama 3 epoch pertama |
| Gradient clip | 1.0              |                                                   |
| Precision     | bf16-mixed       | jika GPU support; fallback ke fp32                |
| Seed          | 42               | untuk reproducibility                             |

### 2.4 Loss Function: Weighted Focal Loss

Kedua model menggunakan **Weighted Focal Loss** untuk menangani class imbalance
(ALL-IDB memiliki ~2:1 rasio Abnormal:Normal).

```python
# Class weights: inverse-frequency dari training set
# Hitung otomatis dari label di folder train/
counts  = bincount(training_labels)           # [count_Abnormal, count_Normal]
weights = total_samples / (num_classes * counts)

# Focal Loss dengan class weights
# gamma = 2.0 (standar), alpha = class_weights
focal_loss(logits, targets, gamma=2.0, weight=class_weights)
```

> Formula: `FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)`
> Gunakan implementasi dari `torchvision.ops.sigmoid_focal_loss` atau tulis manual.

### 2.5 Normalisasi Stain saat Evaluasi Eksternal (Test-Time)

Evaluasi C-NMC dilakukan di **dua kondisi** — laporkan keduanya:

| Kondisi      | Keterangan                                                                       |
| ------------ | -------------------------------------------------------------------------------- |
| `no_norm`  | Tidak ada normalisasi stain — gambar C-NMC dipakai apa adanya                   |
| `reinhard` | Reinhard stain normalization diterapkan ke setiap gambar C-NMC sebelum inference |

Normalisasi ini diterapkan ke gambar **sebelum** masuk model, bukan sebagai augmentasi.
Referensi target normalizer: fit dari training set ALL-IDB.

### 2.6 Test-Time Augmentation (TTA) — ablation opsional, bukan headline

> **Status (lihat Update Protokol di atas):** angka utama = **no-TTA**. TTA-8 dijalankan hanya sebagai
> ablation tambahan untuk menunjukkan "+gain dengan TTA". Tetap deterministik & identik di kedua model.

Bila menjalankan TTA, gunakan 8 augmentasi deterministik berikut, lalu rata-rata probabilitas softmax-nya:

| # | Augmentasi                    |
| - | ----------------------------- |
| 1 | Original (tidak diubah)       |
| 2 | Horizontal flip               |
| 3 | Vertical flip                 |
| 4 | Rotasi 90°                   |
| 5 | Rotasi 180°                  |
| 6 | Rotasi 270°                  |
| 7 | Horizontal flip + Rotasi 90° |
| 8 | Vertical flip + Rotasi 90°   |

```python
# Pseudocode TTA
all_probs = []
for aug in tta_transforms:
    logits = model(aug(images))
    all_probs.append(softmax(logits))
final_prob = mean(all_probs, axis=0)
prediction = argmax(final_prob)
```

### 2.7 Metrik Evaluasi (Harus Sama)

Laporkan semua metrik berikut untuk kondisi `no_norm` dan `reinhard`:

| Metrik           | Keterangan                                                          |
| ---------------- | ------------------------------------------------------------------- |
| Accuracy         | Overall                                                             |
| F1 Macro         | Rata-rata F1 kedua kelas, tidak terboboti —**metrik utama**  |
| Precision Macro  |                                                                     |
| Recall Macro     |                                                                     |
| Recall Abnormal  | Sensitivity — seberapa banyak sel leukemia terdeteksi              |
| Recall Normal    | Specificity proxy — seberapa banyak sel sehat tidak salah diagnosa |
| AUC-ROC          | Pakai probabilitas kelas Abnormal                                   |
| Domain-shift gap | `val_acc − cnmc_no_norm_acc` — ukuran robustness lintas-domain  |
| Confusion matrix | `[[TP_Abn, FN_Abn], [FP_Abn, TN_Abn]]`                            |

---

## 3. Spesifikasi CoAtNet-0

### 3.1 Backbone

```
Model  : CoAtNet-0
Source : timm  →  timm.create_model('coatnet_0_rw_224', pretrained=True)
         atau model serupa dari timm dengan pretrained ImageNet-1k
Params : ~25M
Input  : 224 × 224 × 3
Output : feature vector → Linear(feat_dim, 2)
```

> Cek di `timm.list_models('coatnet*')` untuk nama model yang tersedia.

### 3.2 Augmentasi Mixing: CutMix + Mixup

Terapkan **salah satu** secara acak per batch dengan probabilitas sama:

```python
# Pseudocode — pilih CutMix atau Mixup per batch
if random() < 0.5:
    images, labels_a, labels_b, lam = cutmix(images, labels, alpha=1.0)
else:
    images, labels_a, labels_b, lam = mixup(images, labels, alpha=0.2)

loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
```

Parameter yang disarankan:

- CutMix alpha: `1.0`
- Mixup alpha: `0.2`
- Probabilitas mixing diterapkan: `0.5` per sampel (sama dengan FocusAugMix-ku)

### 3.3 Checkpoint

Simpan checkpoint terbaik berdasarkan `val_f1` (F1 macro di validasi), bukan `val_acc`.
Alasan: val set hanya 204 sampel dan binary task — `val_acc` bisa 100% dari epoch awal,
tidak informatif untuk monitoring.

---

## 4. Format Output yang Dibutuhkan

Agar hasil bisa digabung ke satu tabel perbandingan, simpan hasil evaluasi eksternal
dalam format JSON seperti berikut. **Buat satu file JSON per seed** (mis. `coatnet_0_seed42.json`),
lalu agregasi mean ± std antar-seed untuk angka final. Sertakan field `seed` dan `n_tta`:

```json
{
  "model": "coatnet_0",
  "seed": 42,
  "n_tta": 1,
  "val_acc": 0.0000,
  "val_f1": 0.0000,
  "cnmc_no_norm": {
    "accuracy": 0.0000,
    "f1_macro": 0.0000,
    "f1_weighted": 0.0000,
    "precision_macro": 0.0000,
    "recall_macro": 0.0000,
    "recall_abnormal": 0.0000,
    "recall_normal": 0.0000,
    "auc_roc": 0.0000,
    "confusion_matrix": [[0, 0], [0, 0]]
  },
  "cnmc_reinhard": {
    "accuracy": 0.0000,
    "f1_macro": 0.0000,
    "f1_weighted": 0.0000,
    "precision_macro": 0.0000,
    "recall_macro": 0.0000,
    "recall_abnormal": 0.0000,
    "recall_normal": 0.0000,
    "auc_roc": 0.0000,
    "confusion_matrix": [[0, 0], [0, 0]]
  },
  "domain_shift_gap": 0.0000
}
```

Untuk angka final (gabungan 3 seed), agregasikan menjadi format ringkas seperti ini —
`n_tta=1` (no-TTA) sebagai headline:

```json
{
  "model": "coatnet_0",
  "n_tta": 1,
  "seeds": [42, 123, 2025],
  "cnmc_no_norm": {
    "f1_macro_mean": 0.0000, "f1_macro_std": 0.0000,
    "accuracy_mean": 0.0000, "accuracy_std": 0.0000,
    "recall_abnormal_mean": 0.0000, "recall_normal_mean": 0.0000
  },
  "cnmc_reinhard": {
    "f1_macro_mean": 0.0000, "f1_macro_std": 0.0000,
    "accuracy_mean": 0.0000, "accuracy_std": 0.0000
  }
}
```

> **Tooling (sisi ConvNeXtV2).** Multi-seed di-otomatisasi dengan `src/run_multiseed.py` (train + eval
> per seed) dan `src/aggregate_seeds.py` (hitung mean ± std + tabel markdown). `aggregate_seeds.py`
> bersifat umum: ia akan ikut mengagregasi JSON CoAtNet **asalkan** file per-seed Anda diberi nama
> `<model>_seed<seed>.json`, ditaruh di `results_multiseed/`, dan mengikuti skema per-seed di atas
> (punya `cnmc_no_norm` / `cnmc_reinhard` dengan `f1_macro`, `accuracy`, `confusion_matrix`, plus
> `n_tta` dan `seed`). Recall per-kelas & gap diturunkan otomatis dari `confusion_matrix`, jadi tidak
> perlu Anda hitung manual.

---

## 5. Note

| Item                        | Status | Catatan                                         |
| --------------------------- | :----: | ----------------------------------------------- |
| Nilai `max_epochs`          |        | 30                                              |
| Nilai `lr` awal             |        | 1e-4                                            |
| Focal Loss gamma            |        | 2.0 (standar)                                   |
| CutMix/Mixup alpha          |        | 1.0 / 0.2                                       |
| Sumber pretrained CoAtNet-0 |        | timm atau lain?                                 |
| **Headline TTA**            |        | **no-TTA** (`n_tta=1`); TTA-8 hanya ablation    |
| **Multi-seed**              |        | **42 / 123 / 2025**, lapor mean ± std           |
| Kondisi normalisasi         |        | lapor `no_norm` **dan** `reinhard`              |

---

## 6. Struktur Tabel Perbandingan Akhir

Ini target tabel yang akan masuk ke paper. **Semua angka = no-TTA, mean ± std atas 3 seed**
(42/123/2025). F1 Macro adalah metrik utama; tulis sebagai `mean ± std`.

| Model                                | Val F1 | No-Norm F1 (mean±std) | Reinhard F1 (mean±std) | Recall Abn | Recall Norm | Gap           |
| ------------------------------------ | :----: | :-------------------: | :--------------------: | :--------: | :---------: | :-----------: |
| CoAtNet-0 + CutMix+Mixup (Baseline)  | _TBD_  | _TBD_                 | _TBD_                  | _TBD_      | _TBD_       | _TBD_         |
| ConvNeXtV2-Tiny + FocusAugMix (Ours) | 1.000  | **0.5535 ± 0.1189**   | 0.5187 ± 0.0500        | 48.4%      | 75.5%       | 0.430 ± 0.130 |

> **Ours = `focusmix_stain`** (FocusAugMix + ReinhardJitter σ=0.15, tanpa MHA), **3 seed (42/123/2025),
> no-TTA**, sumber `results_multiseed/aggregate.json`. Recall Abn/Norm pada kondisi headline **no_norm**.
> Pembanding baseline ConvNeXtV2 tanpa stain aug = `no_mix` (no-norm F1 **0.5636 ± 0.0817**) — keduanya
> setara secara statistik; pembeda `focusmix_stain` ada di keseimbangan recall (lihat catatan di bawah).

Aturan tabel utama:

- **Kondisi: no-TTA.** TTA-8 ditaruh di tabel ablation terpisah (lihat di bawah).
- **Laporkan kedua kondisi normalisasi** (`no_norm` dan `reinhard`) untuk **kedua** model, karena
  kondisi terbaik berbeda antar model (lihat catatan). Pilih kondisi terbaik per model sebagai
  angka headline, tapi tampilkan keduanya agar transparan.
- **Setting identik untuk kedua model** — kalau Ours dilaporkan no-TTA, CoAtNet juga no-TTA.

### Tabel Ablation TTA (terpisah, hanya model terbaik)

Untuk menunjukkan TTA bukan faktor utama, sajikan ablation pada **model terbaik masing-masing** saja:

| Config | No-Norm F1 | Catatan                          |
| ------ | :--------: | -------------------------------- |
| no-TTA |            | angka headline                   |
| TTA-4  |            | original + 3 flip/rotasi         |
| TTA-8  |            | full dihedral (8 views)          |

### Angka Referensi Model "Ours" (FINAL — multi-seed)

Model proposal **`focusmix_stain`** (FocusAugMix + ReinhardJitter σ=0.15, tanpa MHA), **3 seed
(42/123/2025), no-TTA** — sumber `results_multiseed/aggregate.json`. **Ini angka headline final Ours**
(bukan lagi single-seed) dan jadi target yang harus dilampaui CoAtNet baseline:

| Kondisi   | Acc (mean±std)  | F1 Macro (mean±std) | Recall Abn | Recall Norm |
| --------- | :-------------: | :-----------------: | :--------: | :---------: |
| No Norm   | 0.5703 ± 0.1300 | **0.5535 ± 0.1189** | 48.4%      | 75.5%       |
| Macenko   | 0.6328 ± 0.0335 | 0.4364 ± 0.0219     | 89.5%      | 6.9%        |
| Reinhard  | 0.6205 ± 0.0412 | 0.5187 ± 0.0500     | 75.0%      | 34.2%       |

> **Catatan revisi (penting).** Angka single-seed lama (No-Norm F1 0.6351) **tidak bertahan lintas seed**;
> rata-rata 3 seed = 0.5535 ± 0.1189, setara dengan baseline ConvNeXtV2 `no_mix` (0.5636 ± 0.0817). Untuk
> klaim paper, posisikan kontribusi `focusmix_stain` sebagai **keseimbangan recall lintas-domain + tanpa
> normalisasi test-time**, bukan superioritas F1 absolut. **Ablation TTA-8** (`results_multiseed_tta8/`)
> hanya menggeser F1 ≤ 0.005 → bukti TTA bukan faktor utama.

> **Temuan penting untuk fair comparison.** Berbeda dari asumsi di Bagian 2.5, model proposal yang
> sudah dilatih dengan train-time stain augmentation **paling baik pada kondisi `no_norm`** — Reinhard
> test-time justru **menurunkan** F1 (0.635 → 0.556) karena terjadi koreksi warna ganda. Sebaliknya,
> baseline CoAtNet (tanpa stain augmentation training) kemungkinan besar **butuh** Reinhard untuk
> menyeimbangkan prediksi. Karena itu **laporkan kedua kondisi untuk kedua model** dan pilih kondisi
> terbaik per model sebagai angka headline; kondisi terbaik yang berbeda inilah salah satu kontribusi
> yang ditonjolkan (train-time vs test-time stain handling). Angka di atas no-TTA — TTA-8 hanya
> diharapkan menaikkan sedikit dan masuk tabel ablation, bukan headline.

---

Last updated: 2026-06-07
