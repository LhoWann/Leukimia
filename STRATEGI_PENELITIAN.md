# Strategi Penelitian: Klasifikasi Leukemia Lintas-Domain

Dokumen ini merangkum rencana kerja dari kondisi saat ini hingga penggunaan data primer (rumah sakit).

---

## Desain Penelitian

| Dataset | Peran | Status |
|---------|-------|--------|
| **ALL-IDB** (Giemsa, Italia) | Data sekunder — pre-training (10 eksperimen) | Selesai |
| **Data Rumah Sakit** | Data primer — fine-tuning + evaluasi akhir | Belum tersedia |
| C-NMC 2019 (Wright-Giemsa, India) | Evaluasi lintas-domain saja | Hanya untuk test |

---

## Fase 1 — Persiapan (Sekarang, Sebelum Data Primer)

### 1.1 Perkuat Model Pre-Training

Model proposal utama: **`focusmix_stain`** (FocusAugMix + ReinhardJitter σ=0.15, tanpa MHA). Pada
single-seed (42) F1 macro lintas-domain 0.635; setelah validasi **3 seed (42/123/2025)** rata-ratanya
**0.5535 ± 0.1189** — secara statistik setara dengan baseline `no_mix` (0.5636 ± 0.0817). Kontribusi
nyata stain aug yang bertahan lintas seed adalah **keseimbangan recall** (Abnormal/Normal 48%/76%),
bukan F1 absolut. Lihat [experiment_results.md](experiment_results.md).

> **Implikasi strategi:** karena pada dataset sekunder kecil (ALL-IDB) keunggulan F1 belum signifikan
> lintas seed, **fine-tuning pada data primer RS menjadi langkah penentu**, bukan opsional. Pre-training
> stain-robust hanya menyediakan titik awal yang seimbang & stabil untuk adaptasi domain RS.

> **Temuan penting yang mengoreksi asumsi awal:** intensitas stain augmentation **tidak monoton**.
> Eksperimen `focusmix_stain` (σ=0.15) jauh mengungguli `focusmix_stain_strong` (σ=0.25, F1 0.413)
> dan `focusmix_stain_max` (σ=0.35, F1 0.506). Jadi **"makin agresif" justru memperburuk** —
> distorsi warna berlebihan menghancurkan sinyal warna yang masih valid. Sweet spot ada di σ≈0.15.

Untuk mempersiapkan variasi staining RS Indonesia yang belum diketahui, strategi yang tepat **bukan**
menaikkan σ membabi buta, melainkan **sweep terkontrol** di sekitar optimum saat ini (σ ∈ {0.10, 0.15,
0.20}) dan memilih berdasarkan F1 macro lintas-domain — bukan menebak nilai ekstrem:

```python
# Tambahkan ke EXPERIMENTS di src/main.py — sweep di sekitar optimum
'pretrain_hospital_ready': ExperimentConfig(
    name='pretrain_hospital_ready',
    aug_mode='focusmix',
    use_mha=False,
    use_robust_aug=True,
    stain_sigma_mean=0.15,   # optimum terbukti; jangan langsung ekstrem
    stain_sigma_std=0.10,
    stain_aug_prob=0.6,      # sedikit di atas default 0.5
    paste_ratio=0.25,
    max_epochs=30,
    warmup_epochs=3,
)
```

> Catatan: `focusmix_stain_max` di kode (σ=0.35, prob=0.8) sudah menguji ujung agresif dan hasilnya
> lebih buruk — tidak perlu diulang. Fokuskan eksplorasi pada σ ≤ 0.20.

---

### 1.2 Siapkan Infrastruktur Kode

File-file berikut perlu ditulis sekarang agar saat data RS tiba langsung bisa dipakai.

#### `src/split_hospital_data.py`

Split per-pasien — bukan per-gambar/sel. Semua sel dari satu pasien harus masuk
ke satu split saja untuk menghindari data leakage biologis.

```
Input : folder gambar dengan ID pasien di nama file
Output: train/ val/ test/ dengan rasio ~70/15/15
Rule  : pasien A → hanya masuk train ATAU val ATAU test, tidak keduanya
```

#### `src/finetune_hospital.py`

Fine-tuning bertahap dari checkpoint pre-training:

```
Fase 1 (epoch 1–3)   : Freeze backbone, train head saja
                       → adaptasi cepat ke distribusi warna RS
Fase 2 (epoch 4–10)  : Unfreeze stage 3 + head, LLRD aktif
                       → fine-tune fitur high-level
Fase 3 (opsional)    : Unfreeze semua jika data RS cukup (>200 pasien)
```

#### `src/calibrate_threshold.py`

Cari threshold klasifikasi optimal di val set RS.
Default 0.5 tidak optimal — konteks klinis memerlukan sensitivity tinggi.

```
Target : recall Abnormal (sensitivity) >= 0.95
Input  : model + val set RS berlabel
Output : threshold optimal + kurva ROC
```

#### `src/patient_diagnosis.py`

Agregasi prediksi per-sel ke diagnosis per-pasien.

```python
def aggregate_cells_to_patient(cell_probs_abnormal, threshold_cell,
                                min_blast_fraction=0.20):
    """
    Referensi WHO: >= 20% blast cells = diagnosis ALL positif.
    cell_probs_abnormal : list probabilitas tiap sel menjadi Abnormal
    min_blast_fraction  : fraksi minimum sel blast untuk diagnosis positif
    """
    blast_cells = sum(1 for p in cell_probs_abnormal if p >= threshold_cell)
    blast_fraction = blast_cells / len(cell_probs_abnormal)
    is_positive = blast_fraction >= min_blast_fraction
    return is_positive, blast_fraction
```

#### `src/quality_check.py`

Filter gambar berkualitas buruk sebelum inference.

```python
def check_image_quality(img_np):
    """Deteksi gambar buram, over-exposed, atau under-exposed."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    return {
        'is_blurry'      : blur_score < 100,
        'is_overexposed' : brightness > 230,
        'is_underexposed': brightness < 30,
        'blur_score'     : blur_score,
    }
```

---

### 1.3 Checklist Protokol Pengumpulan Data RS

Hal-hal yang perlu disepakati dengan RS **sebelum** pengumpulan data:

| Item | Detail | Catatan |
|------|--------|---------|
| Protokol pewarnaan | Giemsa / Wright / May-Grünwald / lainnya? | Menentukan besarnya domain shift |
| Perbesaran objektif | 100× (konsisten dengan ALL-IDB) | Jangan campur 40× dan 100× |
| Format file | JPG (quality ≥ 95) / BMP / TIFF | Hindari kompresi lossy berlebihan |
| Jumlah pasien | ≥ 30 positif (ALL) + ≥ 30 negatif (HEM) | Minimum untuk fine-tuning stabil |
| Jumlah sel per pasien | Minimal 50 sel per pasien | Cukup untuk agregasi yang valid |
| Pemberi label | SpPA / dokter spesialis | Ground truth harus dari ahli |
| Penanganan disagreement | Diskusi atau konsensus 2 ahli | Definisikan sebelum labeling |
| Anonimisasi | Tidak ada nama/ID pasien di nama file | Pakai kode: `RS001_P001_sel001.jpg` |
| Periode pengambilan | Catat tanggal untuk tracking batch | Deteksi batch effect jika ada |

---

## Fase 2 — Saat Data Primer Tiba

### Alur Kerja Lengkap

```
Data RS masuk (gambar apusan darah berlabel)
        │
        ▼
[Quality Check]
  → Tandai gambar buram/over-exposed
  → Eksklusi atau re-akuisisi
        │
        ▼
[Segmentasi Sel]
  → Jalankan segment_dataset.py
  → VALIDASI DULU: apakah segmentasi benar untuk mikroskop RS?
  → Pipeline HSV thresholding mungkin perlu tuning ulang
        │
        ▼
[Split Per-Pasien]
  → split_hospital_data.py
  → Rasio: 70% train / 15% val / 15% test
  → Pastikan tidak ada satu pasien di dua split
        │
        ▼
[Fit Stain Normalizer ke Domain RS]
  → Hitung referensi dari training set RS (bukan ALL-IDB)
  → Simpan normalizer untuk inference production
        │
        ▼
[Fine-Tuning]
  → Starting point: checkpoint focusmix_stain atau pretrain_hospital_ready
  → finetune_hospital.py (3 fase bertahap)
  → Monitor val F1 macro (bukan val accuracy)
        │
        ▼
[Kalibrasi Threshold]
  → calibrate_threshold.py di val set RS
  → Target: sensitivity Abnormal >= 0.95
        │
        ▼
[Evaluasi Final]
  → Test set RS: per-sel + per-pasien
  → Bandingkan: model pre-train saja vs setelah fine-tuning
  → Metrik utama: sensitivity, specificity, F1 macro, AUC-ROC
  → Laporan confusion matrix per kelas (jangan hanya accuracy)
```

---

## Metrik Evaluasi Final

| Metrik | Target | Alasan |
|--------|--------|--------|
| Sensitivity (Recall Abnormal) | ≥ 0.95 | False negative leukemia berbahaya secara klinis |
| Specificity (Recall Normal) | ≥ 0.80 | Mengurangi beban pemeriksaan lanjutan |
| F1 Macro | ≥ 0.85 | Seimbangkan performa kedua kelas |
| AUC-ROC | ≥ 0.90 | Evaluasi performa lintas threshold |
| Blast fraction accuracy | — | Seberapa akurat estimasi % sel blast per pasien |

---

## Risiko dan Mitigasi

| Risiko | Mitigasi |
|--------|----------|
| Data RS terlalu sedikit (<30 pasien/kelas) | Gunakan fine-tuning Fase 1 saja (head only), tambah augmentasi |
| Domain shift sangat besar (staining sangat berbeda) | Re-fit stain normalizer, pertimbangkan Reinhard normalization di preprocessing |
| Segmentasi sel gagal untuk mikroskop RS | Validasi manual sample kecil dulu, tuning parameter HSV jika perlu |
| Label tidak konsisten antar labeler | Hitung inter-rater agreement (Cohen's Kappa) sebelum training |
| Kelas tidak seimbang di data RS | Inverse-frequency class weights sudah ada di kode, aktif secara otomatis |

---

## Status Implementasi

| Komponen | Status |
|----------|--------|
| Pre-training ALL-IDB (10 eksperimen ablation, single-seed) | Selesai |
| Evaluasi lintas-domain C-NMC | Selesai |
| Validasi multi-seed (3 eksperimen kunci × 3 seed, no-TTA) | Selesai |
| Ablation TTA-8 (3 eksperimen kunci × 3 seed) | Selesai |
| Stain normalization (Macenko + Reinhard) | Selesai |
| Train-time stain augmentation (`focusmix_stain` / `_strong` / `_max`) | Selesai |
| Tooling multi-seed (`run_multiseed.py` + `aggregate_seeds.py`) | Selesai |
| Eksperimen `pretrain_hospital_ready` (sweep σ ≤ 0.20) | Belum |
| `split_hospital_data.py` | Belum |
| `finetune_hospital.py` | Belum |
| `calibrate_threshold.py` | Belum |
| `patient_diagnosis.py` | Belum |
| `quality_check.py` | Belum |
| Data primer RS | Belum tersedia |

---

Last updated: 2026-06-07
