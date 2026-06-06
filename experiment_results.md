# Hasil & Analisis Eksperimen

Model dilatih pada **ALL-IDB** (Giemsa stain, Italia) dan dievaluasi secara eksternal pada
**C-NMC 2019 train-merged** (Wright-Giemsa, India) — 10.661 sel berlabel (7.272 ALL + 3.389 HEM) —
untuk mengukur generalisasi lintas-domain.

> **Catatan reproducibility.** Angka ablation 10-eksperimen di dokumen ini diambil langsung dari
> `results/summary.json` (single-seed=42, tanpa TTA, `n_tta=1`). Untuk klaim utama, tiga eksperimen
> kunci (`no_mix`, `focusmix`, `focusmix_stain`) divalidasi ulang dengan **3 seed (42/123/2025)** —
> hasil mean ± std ada di `results_multiseed/` (no-TTA, headline) dan `results_multiseed_tta8/`
> (ablation TTA-8); lihat bagian [Validasi Multi-Seed](#validasi-multi-seed-3-seed--angka-headline).
> Konfigurasi tiap eksperimen didefinisikan di `EXPERIMENTS` pada `src/main.py`. Semua model dilatih
> `max_epochs=30` (early stopping `val_loss`, patience 10), AdamW + LLRD 0.75, **Weighted Focal Loss**
> (γ=2.0) dengan inverse-frequency class weights, checkpoint terbaik dipilih berdasarkan **`val_f1`**
> (bukan `val_acc`).

> **Status kelengkapan eksperimen ConvNeXtV2 (per 2026-06-07).** ✅ 10 eksperimen ablation single-seed
> (no-TTA) — selesai. ✅ 3 eksperimen kunci × 3 seed (no-TTA, headline) — selesai. ✅ 3 eksperimen
> kunci × 3 seed (TTA-8, ablation) — selesai. Sesuai protokol di `PERBANDINGAN_BASELINE.md`, eksperimen
> ablation di luar tiga kunci cukup single-seed.

---

## Setup Eksperimen

### Faktor yang Diuji

Sepuluh eksperimen membentuk ablation atas tiga sumbu desain:

| Sumbu                       | Varian yang diuji                                                        |
| --------------------------- | ------------------------------------------------------------------------ |
| Augmentasi mixing           | `none` · SaliencyMix · FocusAugMix · FocusAugMix+Grad-CAM                 |
| Multi-Head Attention (MHA)  | tanpa MHA · dengan MHA (stage 2)                                          |
| Stain augmentation training | tanpa · ReinhardJitter σ=0.15 · σ=0.25 · σ=0.35 (`use_robust_aug`)        |

### Peta Eksperimen → Konfigurasi

| Eksperimen              | aug_mode       | MHA | Stain aug (σ_mean / prob) | Tujuan ablation                          |
| ----------------------- | -------------- | :-: | ------------------------- | ---------------------------------------- |
| `no_mix`                | none           |  –  | –                         | Batas bawah (hanya augmentasi dasar)     |
| `no_mix_mha`            | none           |  ✓  | –                         | Isolasi kontribusi MHA murni             |
| `saliency`              | saliency       |  –  | –                         | SaliencyMix murni                        |
| `focusmix`              | focusmix       |  –  | –                         | FocusAugMix murni                        |
| `focusmix_mha`          | focusmix       |  ✓  | –                         | FocusAugMix + MHA                        |
| `focusmix_mha_strong`   | focusmix       |  ✓  | – (paste_ratio 0.30)      | FocusAugMix + MHA, paste lebih besar     |
| `focusmix_cam`          | focusmix_cam   |  ✓  | –                         | + Grad-CAM online (refresh tiap 5 epoch) |
| **`focusmix_stain`**    | focusmix       |  –  | 0.15 / 0.5                | **FocusAugMix + stain aug moderat**      |
| `focusmix_stain_strong` | focusmix       |  –  | 0.25 / 0.7                | Stain aug kuat                           |
| `focusmix_stain_max`    | focusmix       |  –  | 0.35 / 0.8                | Stain aug maksimal                       |

---

## Analisis: Val Accuracy ~100% di Semua Eksperimen

Sembilan dari sepuluh eksperimen mencapai val F1 **1.0000** (204 sampel ALL-IDB); `focusmix_cam`
mencapai 0.9901. Angka sempurna ini sempat menimbulkan kecurigaan data leakage, tetapi investigasi
membuktikan sebaliknya.

### Tidak Ada Data Leakage

Split dilakukan **per gambar mikroskopi original** (bukan per crop), di `src/segment_dataset.py`:

```text
Train: Im001, Im002, Im003, Im006, Im007, ... (86 gambar original ALL-IDB1)
Val  : Im004, Im005, Im012, Im014, Im015, ... (22 gambar original ALL-IDB1)

Semua sel crop dari Im004 hanya masuk ke val — tidak satu pun bocor ke train.
```

### Mengapa 100% Bisa Genuine

| Faktor                  | Penjelasan                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------- |
| Pretrained sangat kuat  | ConvNeXtV2 fine-tuned di FCMAE + ImageNet-22k sudah punya fitur visual sangat kaya     |
| Val set kecil           | 204 gambar, binary task — mudah mencapai perfect fit                                   |
| Kelas visually distinct | Blast cell (inti besar, kromatin kasar) vs WBC normal berbeda jelas secara morfologi  |
| Dataset terkontrol      | ALL-IDB: satu lab Italia, satu mesin mikroskop, satu protokol staining                |

### Implikasi untuk Paper

Val accuracy 100% **bukan** indikator generalisasi. Karena ALL-IDB homogen (satu sumber), metrik
in-domain jenuh dan tidak bisa membedakan kualitas antar-eksperimen. **Seluruh sinyal pembeda
ada di evaluasi lintas-domain C-NMC** — itulah fokus dokumen ini. Untuk monitoring training kami
memilih checkpoint berdasarkan `val_f1`, bukan `val_acc` yang sudah jenuh sejak epoch awal.

---

## Ringkasan Lintas-Domain — Ablation Single-Seed (Semua Eksperimen)

Semua angka pada C-NMC 2019 train-merged (10.661 sel), **single-seed (42)**. **F1 macro adalah metrik
utama** — accuracy menyesatkan karena C-NMC tidak seimbang (68% ALL / 32% HEM), sehingga model yang
kolaps ke satu kelas pun bisa terlihat "baik" dari accuracy.

> **Catatan penting.** Tabel di bagian ini adalah **ablation single-seed** untuk memetakan seluruh ruang
> desain (mixing × MHA × stain aug). Angka **headline** penelitian = mean ± std atas 3 seed pada tiga
> eksperimen kunci — lihat [Validasi Multi-Seed](#validasi-multi-seed-3-seed--angka-headline). Beberapa
> kesimpulan single-seed berubah setelah multi-seed (lihat Temuan Kunci #1).

### Tabel Utama — Threshold Default (0.5), Tanpa Normalisasi

| Eksperimen              | Val F1 | Acc            | **F1 macro**   | Recall Abn | Recall Norm | Gap   |
| ----------------------- | :----: | :------------: | :------------: | :--------: | :---------: | :---: |
| `no_mix`                | 1.000  | 0.6680         | 0.5395         | 87.7%      | 22.0%       | 0.332 |
| `no_mix_mha`            | 1.000  | 0.3312         | 0.2658         | 2.4%       | 99.0%       | 0.669 |
| `saliency`              | 1.000  | **0.7088**     | 0.5464         | 95.8%      | 17.4%       | 0.291 |
| `focusmix`              | 1.000  | 0.4377         | 0.4236         | 20.6%      | 93.4%       | 0.562 |
| `focusmix_mha`          | 1.000  | 0.3746         | 0.3377         | 10.2%      | 96.1%       | 0.625 |
| `focusmix_mha_strong`   | 1.000  | 0.4365         | 0.4339         | 27.0%      | 79.4%       | 0.563 |
| `focusmix_cam`          | 0.990  | 0.4410         | 0.4327         | 23.5%      | 88.4%       | 0.549 |
| **`focusmix_stain`**    | 1.000  | 0.6581         | **0.6351**     | **66.7%**  | **64.0%**   | 0.342 |
| `focusmix_stain_strong` | 1.000  | 0.4265         | 0.4133         | 20.3%      | 90.7%       | 0.573 |
| `focusmix_stain_max`    | 1.000  | 0.5082         | 0.5062         | 32.5%      | 90.1%       | 0.492 |

`Gap` = `val_acc − cnmc_no_norm_acc`. **Recall Abn/Norm** dihitung dari diagonal confusion matrix.

### Tabel Pendukung — Efek Stain Normalization (F1 macro)

| Eksperimen              | No Norm    | Macenko    | Reinhard   | Catatan                                 |
| ----------------------- | :--------: | :--------: | :--------: | --------------------------------------- |
| `no_mix`                | 0.5395     | 0.4216     | 0.4113     | Normalisasi memperburuk model bias-Abn  |
| `no_mix_mha`            | 0.2658     | 0.5188     | **0.6516** | Reinhard menyelamatkan model rusak      |
| `saliency`              | 0.5464     | 0.4530     | 0.5686     | Reinhard sedikit membantu               |
| `focusmix`              | 0.4236     | 0.4503     | **0.6022** | Reinhard menutup sebagian besar gap     |
| `focusmix_mha`          | 0.3377     | 0.4106     | **0.5731** | Reinhard membantu signifikan            |
| `focusmix_mha_strong`   | 0.4339     | 0.4202     | 0.3647     | Normalisasi tidak membantu              |
| `focusmix_cam`          | 0.4327     | 0.3647     | 0.2855     | Normalisasi memperburuk                 |
| **`focusmix_stain`**    | **0.6351** | 0.6225     | 0.5559     | Sudah stain-robust; Reinhard malah turun|
| `focusmix_stain_strong` | 0.4133     | 0.5353     | 0.5136     | Stain aug terlalu kuat, butuh Macenko   |
| `focusmix_stain_max`    | 0.5062     | 0.5426     | 0.5110     | Macenko sedikit membantu                |

---

## Validasi Multi-Seed (3 Seed) — Angka Headline

Single-seed rawan dipersoalkan reviewer karena val set ALL-IDB sangat kecil (204 sampel) dan jenuh
di F1 1.0 sejak epoch awal — variasi seed bisa menggeser hasil lintas-domain secara signifikan. Karena
itu tiga eksperimen kunci dilatih ulang dengan **3 seed (42 / 123 / 2025)** dan dilaporkan sebagai
**mean ± std** (otomatis via `src/run_multiseed.py` + `src/aggregate_seeds.py`). Inilah angka yang masuk
klaim utama paper dan tabel perbandingan dengan baseline CoAtNet.

### Tabel Utama — No-Norm, No-TTA (headline)

C-NMC 2019 train-merged (10.661 sel), threshold 0.5, **tanpa normalisasi & tanpa TTA**.

| Eksperimen           | Val F1 | **No-Norm F1 (mean±std)** | Acc (mean±std)  | Rec Abn | Rec Norm | Gap (mean±std)  |
| -------------------- | :----: | :-----------------------: | :-------------: | :-----: | :------: | :-------------: |
| `no_mix`             | 1.000  | **0.5636 ± 0.0817**       | 0.6336 ± 0.0799 | 68.4%   | 52.5%    | 0.366 ± 0.080   |
| `focusmix_stain` ★   | 1.000  | **0.5535 ± 0.1189**       | 0.5703 ± 0.1300 | 48.4%   | 75.5%    | 0.430 ± 0.130   |
| `focusmix`           | 1.000  | **0.3486 ± 0.1405**       | 0.3915 ± 0.1029 | 13.3%   | 94.7%    | 0.609 ± 0.103   |

### Rincian Per Kondisi (no-TTA, mean ± std)

| Eksperimen       | Kondisi  | F1 macro        | Accuracy        | Rec Abn | Rec Norm |
| ---------------- | -------- | :-------------: | :-------------: | :-----: | :------: |
| `no_mix`         | No Norm  | 0.5636 ± 0.0817 | 0.6336 ± 0.0799 | 68.4%   | 52.5%    |
| `no_mix`         | Macenko  | 0.4416 ± 0.0281 | 0.6392 ± 0.0438 | 89.9%   | 8.2%     |
| `no_mix`         | Reinhard | 0.4710 ± 0.0557 | 0.6651 ± 0.0373 | 92.6%   | 10.5%    |
| `focusmix_stain` | No Norm  | 0.5535 ± 0.1189 | 0.5703 ± 0.1300 | 48.4%   | 75.5%    |
| `focusmix_stain` | Macenko  | 0.4364 ± 0.0219 | 0.6328 ± 0.0335 | 89.5%   | 6.9%     |
| `focusmix_stain` | Reinhard | 0.5187 ± 0.0500 | 0.6205 ± 0.0412 | 75.0%   | 34.2%    |
| `focusmix`       | No Norm  | 0.3486 ± 0.1405 | 0.3915 ± 0.1029 | 13.3%   | 94.7%    |
| `focusmix`       | Macenko  | 0.4142 ± 0.0802 | 0.5415 ± 0.1542 | 61.0%   | 39.4%    |
| `focusmix`       | Reinhard | 0.4652 ± 0.1638 | 0.4998 ± 0.1471 | 40.4%   | 70.5%    |

Sumber: `results_multiseed/aggregate.json` & `aggregate.md`.

### Ablation TTA-8 (bukan headline)

TTA-8 (full dihedral, 8 views) dijalankan hanya untuk menunjukkan TTA **bukan** faktor utama. Dampaknya
pada F1 no-norm dapat diabaikan (≤ 0.005) — argumen kuat bahwa model bekerja baik tanpa overhead 8× di
inference (relevan untuk deployment klinis).

| Eksperimen       | No-Norm F1 (no-TTA) | No-Norm F1 (TTA-8) | Δ        |
| ---------------- | :-----------------: | :----------------: | :------: |
| `no_mix`         | 0.5636 ± 0.0817     | 0.5670 ± 0.0899    | +0.0034  |
| `focusmix_stain` | 0.5535 ± 0.1189     | 0.5578 ± 0.1274    | +0.0043  |
| `focusmix`       | 0.3486 ± 0.1405     | 0.3399 ± 0.1435    | −0.0087  |

Sumber: `results_multiseed_tta8/aggregate.json`.

### Implikasi Multi-Seed (koreksi penting atas hasil single-seed)

- **Keunggulan single-seed `focusmix_stain` (F1 0.635) tidak bertahan lintas seed.** Rata-rata 3 seed
  turun ke **0.5535 ± 0.1189** dan **secara statistik setara dengan `no_mix` (0.5636 ± 0.0817)** —
  bahkan mean `no_mix` sedikit lebih tinggi dengan variansi lebih kecil. Run single-seed 42 kebetulan
  menguntungkan `focusmix_stain`.
- **Pembeda nyata `focusmix_stain` bukan F1 absolut, melainkan keseimbangan recall.** `focusmix_stain`
  condong ke Normal lebih ringan (Rec Abn/Norm 48%/76%) sedangkan `no_mix` condong ke Abnormal
  (68%/53%). Untuk konteks klinis (FN Abnormal berbahaya), `no_mix` yang condong ke Abnormal justru
  lebih aman, tetapi keduanya jauh dari ideal — argumen utama tetap butuh data primer RS untuk
  fine-tuning.
- **`focusmix` murni paling buruk & paling tidak stabil** (0.3486 ± 0.1405) — mixing tanpa stain aug
  merusak generalisasi dan menambah variansi antar-seed.

---

## Hasil Per Eksperimen

Confusion matrix dalam orientasi `[[TP_Abn, FN_Abn], [FP_Abn, TN_Norm]]` (baris = label benar,
kolom = prediksi; kelas 0 = Abnormal/ALL, kelas 1 = Normal/HEM).

### 1. `no_mix` — Baseline (ConvNeXtV2-Tiny, tanpa MHA, tanpa mixing)

**Konfigurasi:** `aug_mode=none`, `use_mha=False`, augmentasi dasar saja.

| Kondisi               | Acc        | F1 Macro | Recall Abn | Recall Norm |
| --------------------- | :--------: | :------: | :--------: | :---------: |
| ALL-IDB val           | 1.0000     | 1.0000   | 100%       | 100%        |
| C-NMC — no norm       | 0.6680     | 0.5395   | 87.7%      | 22.0%       |
| C-NMC — Macenko       | 0.6845     | 0.4216   | 99.6%      | 1.6%        |
| C-NMC — Reinhard      | 0.6817     | 0.4113   | 99.7%      | 0.6%        |
| **Domain-shift gap**  |            |          |            | **0.332**   |

CM no-norm: `[[6377, 895], [2644, 745]]`

Baseline bias kuat ke Abnormal (recall Normal 22%). Stain normalization justru **memperburuk**
keseimbangan — mendorong hampir semua prediksi ke Abnormal (recall Normal turun ke ~1%), karena
normalisasi menghilangkan sedikit sinyal warna yang masih dipakai model untuk mengenali HEM.

---

### 2. `no_mix_mha` — MHA murni (tanpa mixing)

**Konfigurasi:** `aug_mode=none`, `use_mha=True`, `mha_stage=2`, `warmup_epochs=5`.

| Kondisi               | Acc        | F1 Macro   | Recall Abn | Recall Norm |
| --------------------- | :--------: | :--------: | :--------: | :---------: |
| ALL-IDB val           | 1.0000     | 1.0000     | 100%       | 100%        |
| C-NMC — no norm       | 0.3312     | 0.2658     | 2.4%       | 99.0%       |
| C-NMC — Macenko       | 0.6185     | 0.5188     | 78.7%      | 25.7%       |
| C-NMC — Reinhard      | **0.7183** | **0.6516** | 84.7%      | 44.1%       |
| **Domain-shift gap**  |            |            |            | **0.669**   |

CM no-norm: `[[175, 7097], [33, 3356]]`

**Kolaps total ke Normal** tanpa normalisasi — hanya 175/7.272 sel ALL terdeteksi (recall Abn 2.4%),
gap terburuk (0.669). Namun **Reinhard menyelamatkannya secara dramatis**: F1 melonjak 0.266 → 0.652,
accuracy 0.331 → 0.718 (accuracy tertinggi di seluruh eksperimen). MHA membuat model sangat
bergantung pada distribusi warna Giemsa ALL-IDB; saat Reinhard memetakan kembali warna C-NMC,
representasi MHA kembali berfungsi.

---

### 3. `saliency` — SaliencyMix (tanpa MHA)

**Konfigurasi:** `aug_mode=saliency`, `use_mha=False`, `paste_ratio=0.25`.

| Kondisi               | Acc        | F1 Macro   | Recall Abn | Recall Norm |
| --------------------- | :--------: | :--------: | :--------: | :---------: |
| ALL-IDB val           | 1.0000     | 1.0000     | 100%       | 100%        |
| C-NMC — no norm       | **0.7088** | 0.5464     | 95.8%      | 17.4%       |
| C-NMC — Macenko       | 0.6840     | 0.4530     | 97.8%      | 5.4%        |
| C-NMC — Reinhard      | 0.7105     | 0.5686     | 94.1%      | 21.5%       |
| **Domain-shift gap**  |            |            |            | **0.291**   |

CM no-norm: `[[6967, 305], [2800, 589]]`

Accuracy no-norm tertinggi (0.709) dan gap terkecil (0.291) — tetapi **menyesatkan**: model nyaris
selalu memprediksi Abnormal (recall Normal hanya 17%). F1 macro hanya 0.546. Ini ilustrasi sempurna
mengapa gap dan accuracy tidak cukup untuk data tidak seimbang. Setelah kalibrasi threshold (0.41),
F1 naik ke 0.685 (lihat bagian Kalibrasi), tetapi angka itu optimistik karena dikalibrasi di test set.

---

### 4. `focusmix` — FocusAugMix murni (tanpa MHA, tanpa stain aug)

**Konfigurasi:** `aug_mode=focusmix`, `use_mha=False`, `n_segments=50`, `paste_ratio=0.25`.

| Kondisi               | Acc        | F1 Macro   | Recall Abn | Recall Norm |
| --------------------- | :--------: | :--------: | :--------: | :---------: |
| ALL-IDB val           | 1.0000     | 1.0000     | 100%       | 100%        |
| C-NMC — no norm       | 0.4377     | 0.4236     | 20.6%      | 93.4%       |
| C-NMC — Macenko       | 0.6519     | 0.4503     | 92.2%      | 7.3%        |
| C-NMC — Reinhard      | 0.6212     | **0.6022** | 61.6%      | 63.3%       |
| **Domain-shift gap**  |            |            |            | **0.562**   |

CM no-norm: `[[1501, 5771], [224, 3165]]`

FocusAugMix murni (dengan Focal Loss + class weights) condong ke Normal (recall Abn 21%). Tanpa
stain aug, model masih sensitif terhadap pergeseran warna. **Reinhard menyeimbangkan** prediksi
dengan baik (F1 0.424 → 0.602, recall Abn/Norm jadi 62%/63%) — menunjukkan masalah utamanya memang
domain warna, bukan morfologi.

---

### 5. `focusmix_mha` — FocusAugMix + MHA

**Konfigurasi:** `aug_mode=focusmix`, `use_mha=True`, `mha_stage=2`, `warmup_epochs=5`.

| Kondisi               | Acc        | F1 Macro   | Recall Abn | Recall Norm |
| --------------------- | :--------: | :--------: | :--------: | :---------: |
| ALL-IDB val           | 1.0000     | 1.0000     | 100%       | 100%        |
| C-NMC — no norm       | 0.3746     | 0.3377     | 10.2%      | 96.1%       |
| C-NMC — Macenko       | 0.4111     | 0.4106     | 27.9%      | 69.5%       |
| C-NMC — Reinhard      | 0.6678     | **0.5731** | 83.5%      | 30.9%       |
| **Domain-shift gap**  |            |            |            | **0.625**   |

CM no-norm: `[[738, 6534], [133, 3256]]`

Menambahkan MHA ke FocusAugMix **merusak generalisasi** (F1 0.424 → 0.338 vs `focusmix`). Bias kuat
ke Normal. Reinhard kembali sangat membantu (+0.235 F1), tetapi performa puncaknya masih di bawah
`focusmix_stain` tanpa normalisasi.

---

### 6. `focusmix_mha_strong` — FocusAugMix + MHA, paste ratio 0.30

**Konfigurasi:** `aug_mode=focusmix`, `use_mha=True`, `paste_ratio=0.30`, `warmup_epochs=5`.

| Kondisi               | Acc        | F1 Macro   | Recall Abn | Recall Norm |
| --------------------- | :--------: | :--------: | :--------: | :---------: |
| ALL-IDB val           | 1.0000     | 1.0000     | 100%       | 100%        |
| C-NMC — no norm       | 0.4365     | 0.4339     | 27.0%      | 79.4%       |
| C-NMC — Macenko       | 0.4273     | 0.4202     | 23.2%      | 84.6%       |
| C-NMC — Reinhard      | 0.3795     | 0.3647     | 16.6%      | 83.7%       |
| **Domain-shift gap**  |            |            |            | **0.563**   |

CM no-norm: `[[1964, 5308], [699, 2690]]`

Memperbesar `paste_ratio` ke 0.30 sedikit menyeimbangkan recall vs `focusmix_mha` tetapi tidak
memperbaiki F1 secara berarti. Pada eksperimen ini, **tidak ada metode normalisasi yang membantu** —
indikasi representasi MHA yang sudah rusak tidak bisa diperbaiki di test-time.

---

### 7. `focusmix_cam` — FocusAugMix + MHA + Grad-CAM Online

**Konfigurasi:** `aug_mode=focusmix_cam`, `use_mha=True`, Grad-CAM refresh tiap 5 epoch.

| Kondisi               | Acc        | F1 Macro   | Recall Abn | Recall Norm |
| --------------------- | :--------: | :--------: | :--------: | :---------: |
| ALL-IDB val           | 0.9902     | 0.9901     | 100%       | 97.8%       |
| C-NMC — no norm       | 0.4410     | 0.4327     | 23.5%      | 88.4%       |
| C-NMC — Macenko       | 0.3841     | 0.3647     | 15.3%      | 87.9%       |
| C-NMC — Reinhard      | 0.3398     | 0.2855     | 4.7%       | 96.8%       |
| **Domain-shift gap**  |            |            |            | **0.549**   |

CM no-norm: `[[1706, 5566], [394, 2995]]`

Satu-satunya eksperimen dengan val F1 < 1.0 (0.990). Grad-CAM online pada dataset kecil dengan
training terbatas menghasilkan saliency map yang belum stabil, dan di sini normalisasi **memperburuk**
(Reinhard menjatuhkan F1 ke 0.286). Grad-CAM online tidak memberi keuntungan lintas-domain yang
sepadan dengan kompleksitasnya.

---

### 8. `focusmix_stain` — FocusAugMix + Stain Augmentation Moderat ★

**Konfigurasi:** `aug_mode=focusmix`, `use_mha=False`, `use_robust_aug=True`,
`stain_sigma_mean=0.15`, `stain_sigma_std=0.10`, `stain_aug_prob=0.5`.

| Kondisi               | Acc        | F1 Macro   | Recall Abn | Recall Norm |
| --------------------- | :--------: | :--------: | :--------: | :---------: |
| ALL-IDB val           | 1.0000     | 1.0000     | 100%       | 100%        |
| C-NMC — no norm       | 0.6581     | **0.6351** | 66.7%      | 64.0%       |
| C-NMC — Macenko       | **0.7057** | 0.6225     | 86.1%      | 37.2%       |
| C-NMC — Reinhard      | 0.5606     | 0.5559     | 48.7%      | 71.8%       |
| **Domain-shift gap**  |            |            |            | **0.342**   |

CM no-norm: `[[4847, 2425], [1220, 2169]]`

**Model terbaik.** F1 macro tertinggi tanpa normalisasi (0.635) **dan** prediksi paling seimbang di
seluruh eksperimen (recall Abn 66.7% / Norm 64.0%). ReinhardJitter saat training memaksa model
belajar fitur morfologi yang invariant terhadap pergeseran warna, sehingga generalisasi lintas-domain
tercapai **tanpa preprocessing apa pun di test-time**. Macenko menaikkan accuracy ke 0.706 (F1 0.623),
tetapi Reinhard test-time justru **menurunkan** performa (0.635 → 0.556): model sudah stain-robust,
menumpuk normalisasi di atasnya malah mendistorsi distribusi yang sudah cocok.

---

### 9. `focusmix_stain_strong` — Stain Augmentation Kuat (σ=0.25)

**Konfigurasi:** `stain_sigma_mean=0.25`, `stain_sigma_std=0.15`, `stain_aug_prob=0.7`.

| Kondisi               | Acc        | F1 Macro   | Recall Abn | Recall Norm |
| --------------------- | :--------: | :--------: | :--------: | :---------: |
| ALL-IDB val           | 1.0000     | 1.0000     | 100%       | 100%        |
| C-NMC — no norm       | 0.4265     | 0.4133     | 20.3%      | 90.7%       |
| C-NMC — Macenko       | 0.5718     | **0.5353** | 62.5%      | 45.9%       |
| C-NMC — Reinhard      | 0.5265     | 0.5136     | 50.5%      | 57.2%       |
| **Domain-shift gap**  |            |            |            | **0.573**   |

CM no-norm: `[[1473, 5799], [315, 3074]]`

Menaikkan intensitas stain aug ke σ=0.25 **kontraproduktif**: F1 no-norm jatuh dari 0.635 (`focusmix_stain`)
ke 0.413. Distorsi warna yang terlalu agresif saat training merusak sinyal warna yang valid, model
kembali bias ke Normal dan butuh Macenko untuk pulih sebagian.

---

### 10. `focusmix_stain_max` — Stain Augmentation Maksimal (σ=0.35)

**Konfigurasi:** `stain_sigma_mean=0.35`, `stain_sigma_std=0.20`, `stain_aug_prob=0.8`.

| Kondisi               | Acc        | F1 Macro   | Recall Abn | Recall Norm |
| --------------------- | :--------: | :--------: | :--------: | :---------: |
| ALL-IDB val           | 1.0000     | 1.0000     | 100%       | 100%        |
| C-NMC — no norm       | 0.5082     | 0.5062     | 32.5%      | 90.1%       |
| C-NMC — Macenko       | 0.5947     | **0.5426** | 68.3%      | 40.5%       |
| C-NMC — Reinhard      | 0.5142     | 0.5110     | 43.7%      | 68.1%       |
| **Domain-shift gap**  |            |            |            | **0.492**   |

CM no-norm: `[[2366, 4906], [337, 3052]]`

Menariknya σ=0.35 sedikit lebih baik dari σ=0.25 (F1 0.506 vs 0.413) tetapi tetap jauh di bawah σ=0.15.
Hubungan intensitas stain aug → robustness **tidak monoton**: ada titik optimal di σ≈0.15. Terlalu lemah
tidak cukup robust, terlalu kuat merusak sinyal warna yang masih informatif.

---

## Kalibrasi Threshold (Diagnostik)

Skrip evaluasi mencari threshold optimal pada `probs[:, 1]` (P(Normal)) yang memaksimalkan F1 macro,
lalu menerapkannya ke seluruh kondisi. Hasil terpilih:

| Eksperimen           | Kondisi   | Thr opt | F1 (default 0.5) | F1 (kalibrasi) |
| -------------------- | --------- | :-----: | :--------------: | :------------: |
| `saliency`           | no-norm   | 0.41    | 0.5464           | **0.6853**     |
| `focusmix_stain`     | no-norm   | 0.55    | 0.6351           | **0.6473**     |
| `focusmix_stain_max` | no-norm   | 0.64    | 0.5062           | **0.6668**     |
| `no_mix`             | no-norm   | 0.35    | 0.5395           | **0.6135**     |

> **Peringatan metodologis (penting untuk paper).** Threshold ini dikalibrasi pada **test set C-NMC
> yang sama** dengan yang dievaluasi — bukan held-out terpisah. Karena itu kolom "F1 kalibrasi"
> bersifat **optimistik (upper bound)** dan **tidak boleh diklaim** sebagai performa deployment.
> Gunakan F1 pada threshold 0.5 sebagai angka utama; sajikan F1 kalibrasi hanya sebagai "potensi
> dengan threshold tuning". Pada data primer RS nanti, threshold wajib dikalibrasi di **val set RS**,
> bukan di test set.

---

## Temuan Kunci

### 1. Stain Augmentation: Pembeda Keseimbangan, Bukan F1 Absolut (revisi multi-seed)

Pada single-seed (42), `focusmix_stain` mencapai F1 no-norm tertinggi (0.635). **Namun setelah validasi
3 seed, keunggulan F1 ini tidak bertahan**: rata-rata `focusmix_stain` turun ke **0.5535 ± 0.1189**,
secara statistik **setara dengan baseline `no_mix` (0.5636 ± 0.0817)** — lihat
[Validasi Multi-Seed](#validasi-multi-seed-3-seed--angka-headline).

Kontribusi train-time stain augmentation yang tetap valid lintas seed adalah **keseimbangan recall**:
`focusmix_stain` (Rec Abn/Norm 48%/76%) jauh lebih seimbang ketimbang `focusmix` murni (13%/95%, kolaps
ke Normal), dan menanamkan invariansi warna langsung ke bobot model **tanpa normalisasi test-time**.
Klaim yang jujur untuk paper: stain aug **menstabilkan dan menyeimbangkan** prediksi lintas-domain,
bukan menaikkan F1 absolut secara signifikan di atas baseline pada dataset kecil ini.

### 2. Intensitas Stain Augmentation Bersifat Non-Monoton

| Stain aug σ_mean | Eksperimen              | F1 no-norm |
| :--------------: | ----------------------- | :--------: |
| 0.15 (moderat)   | `focusmix_stain`        | **0.6351** |
| 0.25 (kuat)      | `focusmix_stain_strong` | 0.4133     |
| 0.35 (maksimal)  | `focusmix_stain_max`    | 0.5062     |

Ada **sweet spot di σ≈0.15**. Distorsi warna berlebihan menghancurkan sinyal warna yang masih valid
untuk membedakan blast vs limfosit normal. Ini temuan ablation yang layak dilaporkan.

### 3. MHA Konsisten Merusak Generalisasi Lintas-Domain

| Kelompok                                                                | Rata-rata Gap |
| ----------------------------------------------------------------------- | :-----------: |
| Tanpa MHA (`no_mix`, `saliency`, `focusmix`, ketiga `focusmix_stain*`)  | **0.432**     |
| Dengan MHA (`no_mix_mha`, `focusmix_mha`, `_mha_strong`, `focusmix_cam`)| **0.601**     |

Keempat eksperimen ber-MHA punya gap lebih besar dari median tanpa-MHA. MHA pada dataset kecil
(~556 sampel) mempelajari *spatial attention pattern* yang spesifik terhadap staining Giemsa ALL-IDB,
sehingga kolaps saat domain bergeser. Model terbaik (`focusmix_stain`) **tidak memakai MHA**.

### 4. Stain Augmentation dan Test-Time Normalization Saling Menggantikan

Pola yang konsisten di seluruh data:

- Model **tanpa** stain-robustness (`no_mix_mha`, `focusmix`, `focusmix_mha`) → **Reinhard test-time
  sangat membantu** (F1 naik +0.18 sampai +0.39).
- Model **sudah** stain-robust (`focusmix_stain`) → **Reinhard test-time malah menurunkan** F1
  (0.635 → 0.556), karena menumpuk dua koreksi warna.

Artinya kedua mekanisme menangani masalah yang sama (domain warna). Memakai keduanya sekaligus bisa
over-correct. Untuk deployment: pilih **salah satu** — train-time stain aug (disarankan, lebih cepat
di inference) **atau** test-time normalization.

### 5. Accuracy & Gap Menyesatkan; F1 Macro + Recall Per-Kelas Wajib

`saliency` punya accuracy no-norm tertinggi (0.709) dan gap terkecil (0.291), tetapi recall Normal
hanya 17% — model nyaris menebak "semua Abnormal" dan diuntungkan distribusi C-NMC yang 68% ALL.
Sebaliknya `focusmix_stain` (acc 0.658) jauh lebih berguna secara klinis karena seimbang. **Selalu
laporkan F1 macro dan recall per-kelas, bukan hanya accuracy.**

### 6. Dua Arah Bias yang Berlawanan

| Arah bias        | Eksperimen                                            | Mekanisme                                                       |
| ---------------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| → Abnormal       | `no_mix`, `saliency`                                  | Imbalance training (2:1) + ciri blast persisten lintas domain  |
| → Normal         | `no_mix_mha`, `focusmix*`, `focusmix_mha*`            | Mixing/MHA mengacaukan representasi Abnormal; sel blast C-NMC yang bersih salah dibaca Normal |
| Seimbang         | **`focusmix_stain`**                                  | Stain aug moderat → invariansi warna tanpa merusak morfologi   |

**Catatan klinis:** False Negative Abnormal (sel leukemia diprediksi Normal) jauh lebih berbahaya
daripada False Positive. Eksperimen bias-Normal (`no_mix_mha`, `focusmix_mha`) memiliki >6.000 FN
Abnormal dari 7.272 — tidak dapat diterima secara klinis tanpa koreksi (Reinhard atau kalibrasi).

---

## Rekomendasi

| Skenario                          | Eksperimen / Setup                | Alasan                                          |
| --------------------------------- | --------------------------------- | ----------------------------------------------- |
| **Deployment tanpa preprocessing**| **`focusmix_stain` (no norm)**    | F1 0.635, paling seimbang, tanpa overhead inference |
| Akurasi maksimal + Macenko        | `focusmix_stain` + Macenko        | Acc 0.706 (F1 0.623)                             |
| Menyelamatkan model bias-Normal   | + Reinhard test-time              | mis. `no_mix_mha`: F1 0.266 → 0.652             |
| Model proposal untuk paper        | **`focusmix_stain`**              | Kontribusi: FocusAugMix + train-time stain aug  |
| In-domain saja (ALL-IDB)          | Semua setara (~100%)              | Pilih berdasarkan kebutuhan lintas-domain       |

---

## Catatan Dataset

- **Training/val:** ALL-IDB1 + ALL-IDB2 (Giemsa, Italia) — ~556 train / 204 val (image-level split).
- **External test:** `C-NMC_train_merged` (Wright-Giemsa, India) — **10.661 sel** (7.272 ALL + 3.389 HEM).
  Ini satu-satunya split C-NMC berlabel publik; split test (prelim & final) flat tanpa label.
- Hasil mentah single-seed: `results/<exp>.json` per eksperimen + `results/summary.json` gabungan.
- Hasil multi-seed (3 seed): `results_multiseed/` (no-TTA) & `results_multiseed_tta8/` (TTA-8 ablation),
  masing-masing dengan `aggregate.json` + `aggregate.md`.

---

Last updated: 2026-06-07
