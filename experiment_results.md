# Hasil & Analisis Eksperimen

Model dilatih pada **ALL-IDB** (Giemsa stain, Italia) dan dievaluasi secara eksternal pada
**C-NMC 2019 train-merged** (Wright-Giemsa, India) — 10.661 sel berlabel — untuk mengukur
generalisasi lintas-domain.

---

## Analisis: Val Accuracy 100%

Semua 7 eksperimen mencapai val accuracy **1.0000** (204 sampel, binary task). Kondisi ini sempat
menimbulkan kecurigaan data leakage, tetapi investigasi membuktikan sebaliknya.

### Tidak Ada Data Leakage

Split dilakukan **per gambar mikroskopi original** (bukan per crop):

```text
Train: Im001, Im002, Im003, Im006, Im007, ... (86 gambar original ALL-IDB1)
Val  : Im004, Im005, Im012, Im014, Im015, ... (22 gambar original ALL-IDB1)

Semua sel crop dari Im004 masuk ke val saja — tidak ada yang masuk ke train.
```

Tidak ada overlap di antara keduanya. Kode yang melakukan split ada di `src/segment_dataset.py`,
dengan strategi *image-level split* sebelum crop.

### Mengapa 100% Bisa Genuine?

| Faktor                  | Penjelasan                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------ |
| Pretrained sangat kuat  | ConvNeXtV2 fine-tuned di ImageNet-22k sudah punya fitur visual sangat kaya           |
| Val set kecil           | 204 gambar, binary task — lebih mudah mencapai perfect fit                          |
| Kelas visually distinct | Blast cell (inti besar, kromatin kasar) vs normal WBC berbeda jelas secara morfologi |
| Dataset terkontrol      | ALL-IDB: satu lab Italia, satu mesin mikroskop, satu protokol staining               |

### Catatan Penting

Val accuracy 100% **tidak mengindikasikan** model akan generalisasi ke dataset lain.
Model kemungkinan belajar *staining artefacts* khas Giemsa ALL-IDB Italia, bukan morfologi sel
yang benar-benar general. Domain-shift gap di bawah adalah ukuran yang lebih jujur.

---

## Ringkasan Lintas-Domain (Semua Eksperimen)

Evaluasi menggunakan `C-NMC_train_merged` (10.661 sel: 7.272 ALL + 3.389 HEM).

| Eksperimen                |     Val Acc     |  C-NMC No Norm  | C-NMC Macenko |  C-NMC Reinhard  |       Gap       |
| ------------------------- | :--------------: | :--------------: | :-----------: | :--------------: | :-------------: |
| `baseline`              |      1.0000      |      0.6904      |    0.6537    | **0.7035** |      0.310      |
| `mha_only`              |      1.0000      |      0.6755      |    0.5648    |      0.6810      |      0.324      |
| `saliency_mix`          |      1.0000      |      0.6831      |    0.6827    |      0.6827      |      0.317      |
| **`focusmix_v2`** | **1.0000** | **0.7260** |    0.5677    |      0.6888      | **0.274** |
| `focusmix_mha`          |      1.0000      |      0.4702      |    0.3403    |      0.5893      |      0.530      |
| `focusmix_aggressive`   |      1.0000      |      0.4459      |    0.4018    |      0.5179      |      0.554      |
| `focusmix_full`         |      1.0000      |      0.3507      |    0.3648    |      0.3752      |      0.649      |

**Gap** = `val_acc − cnmc_no_norm_acc`. Semakin kecil, semakin robust lintas-domain.

---

## Hasil Per Eksperimen

### 1. `baseline` — ConvNeXtV2-Tiny, tanpa MHA, tanpa mixing aug

**Konfigurasi:** `use_mha=False`, `aug_mode=none`, `aug_prob=0.5`, `paste_ratio=0.25`

| Kondisi                    |       Acc       | F1 Macro | F1 Weighted | Precision |     Recall     |
| -------------------------- | :--------------: | :------: | :---------: | :-------: | :-------------: |
| ALL-IDB val (in-domain)    |      1.0000      |  1.0000  |   1.0000   |  1.0000  |     1.0000     |
| C-NMC — no norm           |      0.6904      |  0.5289  |   0.6293   |  0.6244  |     0.5501     |
| C-NMC — Macenko           |      0.6537      |  0.4864  |   0.5931   |  0.5307  |     0.5140     |
| C-NMC — Reinhard          | **0.7035** |  0.5101  |   0.6222   |  0.7012  |     0.5473     |
| **Domain-shift gap** |                  |          |            |          | **0.310** |

Confusion matrix C-NMC no-norm `[Abnormal, Normal]`:

```
[[6801,  471],   ← 6801/7272 ALL benar, 471 salah ke Normal
 [2830,  559]]   ← 2830/3389 HEM salah ke Abnormal, 559 benar
```

**Catatan:** Reinhard memberikan hasil terbaik (+0.014 acc vs no-norm). Model cenderung
over-predict Abnormal pada C-NMC karena bias kelas training (ALL-IDB lebih banyak sel Abnormal).

---

### 2. `mha_only` — ConvNeXtV2-Tiny + MHA, tanpa mixing aug

**Konfigurasi:** `use_mha=True`, `aug_mode=none`, `aug_prob=0.5`

| Kondisi                    |  Acc  | F1 Macro | F1 Weighted | Precision |     Recall     |
| -------------------------- | :----: | :------: | :---------: | :-------: | :-------------: |
| ALL-IDB val (in-domain)    | 1.0000 |  1.0000  |   1.0000   |  1.0000  |     1.0000     |
| C-NMC — no norm           | 0.6755 |  0.4454  |   0.5755   |  0.5496  |     0.5083     |
| C-NMC — Macenko           | 0.5648 |  0.5426  |   0.5793   |  0.5511  |     0.5586     |
| C-NMC — Reinhard          | 0.6810 |  0.4277  |   0.5664   |  0.5757  |     0.5058     |
| **Domain-shift gap** |        |          |            |          | **0.324** |

Confusion matrix C-NMC no-norm:

```
[[7035,  237],
 [3222,  167]]
```

**Catatan:** MHA tanpa augmentasi memperburuk F1 macro vs baseline (0.445 vs 0.529). Model
sangat bias ke kelas Abnormal — hanya 167/3389 Normal yang terklasifikasi benar (recall Normal
sangat rendah). Macenko justru membantu menyeimbangkan prediksi (F1 0.543 vs 0.445).

---

### 3. `saliency_mix` — ConvNeXtV2-Tiny + SaliencyMix, tanpa MHA

**Konfigurasi:** `use_mha=False`, `aug_mode=saliency`, `aug_prob=0.5`, `paste_ratio=0.25`

| Kondisi                    |       Acc       | F1 Macro | F1 Weighted | Precision |     Recall     |
| -------------------------- | :--------------: | :------: | :---------: | :-------: | :-------------: |
| ALL-IDB val (in-domain)    |      1.0000      |  1.0000  |   1.0000   |  1.0000  |     1.0000     |
| C-NMC — no norm           |      0.6831      |  0.4189  |   0.5616   |  0.6218  |     0.5044     |
| C-NMC — Macenko           | **0.6827** |  0.5067  |   0.6140   |  0.6013  |     0.5363     |
| C-NMC — Reinhard          |      0.6827      |  0.4085  |   0.5552   |  0.6985  |     0.5012     |
| **Domain-shift gap** |                  |          |            |          | **0.317** |

Confusion matrix C-NMC no-norm:

```
[[7235,   37],
 [3342,   47]]
```

**Catatan:** Bias Abnormal paling ekstrem di kelompok ini — hanya 47/3389 Normal yang benar.
Accuracy terlihat tinggi karena C-NMC train-merged dominated oleh kelas ALL (68% dari total).
F1 macro lebih representatif: hanya 0.419. Macenko cukup membantu menyeimbangkan (+0.088 F1).

---

### 4. `focusmix_v2` — ConvNeXtV2-Tiny + FocusAugMix, tanpa MHA

**Konfigurasi:** `use_mha=False`, `aug_mode=focusmix`, `aug_prob=0.5`, `paste_ratio=0.25`

| Kondisi                    |       Acc       |     F1 Macro     |   F1 Weighted   | Precision |     Recall     |
| -------------------------- | :--------------: | :--------------: | :--------------: | :-------: | :-------------: |
| ALL-IDB val (in-domain)    |      1.0000      |      1.0000      |      1.0000      |  1.0000  |     1.0000     |
| C-NMC — no norm           | **0.7260** | **0.6711** | **0.7200** |  0.6806  |     0.6655     |
| C-NMC — Macenko           |      0.5677      |      0.5336      |      0.5795      |  0.5368  |     0.5411     |
| C-NMC — Reinhard          |      0.6888      |      0.4953      |      0.6091      |  0.6254  |     0.5341     |
| **Domain-shift gap** |                  |                  |                  |          | **0.274** |

Confusion matrix C-NMC no-norm:

```
[[6048, 1224],
 [1697, 1692]]
```

**Catatan:** **Terbaik** di antara semua eksperimen — accuracy 72.6% dan F1 macro 0.671 tanpa
normalisasi, serta domain-shift gap paling kecil (0.274). Model lebih seimbang dalam prediksi
Normal (recall Normal: 0.499, terbaik di kelompok tanpa normalisasi). FocusAugMix tanpa MHA
ternyata lebih robust dari kombinasi FocusAugMix+MHA.

---

### 5. `focusmix_mha` — ConvNeXtV2-Tiny + FocusAugMix + MHA

**Konfigurasi:** `use_mha=True`, `aug_mode=focusmix`, `aug_prob=0.5`, `paste_ratio=0.25`

| Kondisi                    |  Acc  |     F1 Macro     | F1 Weighted | Precision |     Recall     |
| -------------------------- | :----: | :--------------: | :---------: | :-------: | :-------------: |
| ALL-IDB val (in-domain)    | 1.0000 |      1.0000      |   1.0000   |  1.0000  |     1.0000     |
| C-NMC — no norm           | 0.4702 |      0.4666      |   0.4506   |  0.5923  |     0.5767     |
| C-NMC — Macenko           | 0.3403 |      0.3167      |   0.2705   |  0.4439  |     0.4703     |
| C-NMC — Reinhard          | 0.5893 | **0.5858** |   0.5996   |  0.6263  |     0.6409     |
| **Domain-shift gap** |        |                  |            |          | **0.530** |

Confusion matrix C-NMC no-norm:

```
[[2068, 5204],
 [ 444, 2945]]
```

**Catatan:** Kontras tajam dengan `focusmix_v2` — menambahkan MHA ke FocusAugMix justru
merusak generalisasi (0.470 vs 0.726). Model cenderung misprediksi Abnormal sebagai Normal
(5204 false negatives). Reinhard sangat membantu eksperimen ini (+0.119 acc vs no-norm),
menghasilkan F1 macro tertinggi dengan normalisasi (0.586).

---

### 6. `focusmix_aggressive` — FocusAugMix + MHA, parameter agresif

**Konfigurasi:** `use_mha=True`, `aug_mode=focusmix`, `aug_prob=0.7`, `paste_ratio=0.35`

| Kondisi                    |  Acc  | F1 Macro | F1 Weighted | Precision |     Recall     |
| -------------------------- | :----: | :------: | :---------: | :-------: | :-------------: |
| ALL-IDB val (in-domain)    | 1.0000 |  1.0000  |   1.0000   |  1.0000  |     1.0000     |
| C-NMC — no norm           | 0.4459 |  0.4382  |   0.4141   |  0.5858  |     0.5634     |
| C-NMC — Macenko           | 0.4018 |  0.3931  |   0.3666   |  0.5186  |     0.5136     |
| C-NMC — Reinhard          | 0.5179 |  0.5164  |   0.5261   |  0.5686  |     0.5740     |
| **Domain-shift gap** |        |          |            |          | **0.554** |

Confusion matrix C-NMC no-norm:

```
[[1751, 5521],
 [ 386, 3003]]
```

**Catatan:** Augmentasi agresif (`aug_prob=0.7`, `paste_ratio=0.35`) pada dataset kecil ini
memperbesar domain-shift gap secara signifikan vs baseline (0.554 vs 0.310). Model belajar
mencampur artefak staining ALL-IDB secara berlebihan, bukan morfologi sel yang generalisable.
Untuk dataset skala ini, parameter default (`aug_prob=0.5`, `paste_ratio=0.25`) lebih tepat.

---

### 7. `focusmix_full` — FocusAugMix + MHA + Grad-CAM Online

**Konfigurasi:** `use_mha=True`, `aug_mode=focusmix_cam`, `aug_prob=0.5`, `paste_ratio=0.25`

| Kondisi                    |  Acc  | F1 Macro | F1 Weighted | Precision |     Recall     |
| -------------------------- | :----: | :------: | :---------: | :-------: | :-------------: |
| ALL-IDB val (in-domain)    | 1.0000 |  1.0000  |   1.0000   |  1.0000  |     1.0000     |
| C-NMC — no norm           | 0.3507 |  0.3055  |   0.2409   |  0.5426  |     0.5115     |
| C-NMC — Macenko           | 0.3648 |  0.3380  |   0.2895   |  0.5107  |     0.5051     |
| C-NMC — Reinhard          | 0.3752 |  0.3590  |   0.3219   |  0.4989  |     0.4993     |
| **Domain-shift gap** |        |          |            |          | **0.649** |

Confusion matrix C-NMC no-norm:

```
[[ 509, 6763],
 [ 159, 3230]]
```

**Catatan:** **Terburuk** di antara semua eksperimen. Grad-CAM online (diupdate setiap 5 epoch)
pada dataset training yang sangat kecil dan few-epoch menghasilkan saliency map yang tidak stabil,
mendistorsi proses augmentasi. Domain-shift gap 0.649 jauh melampaui eksperimen lain. Model
sangat bermasalah — mayoritas ALL diprediksi sebagai Normal (6763 false negatives dari 7272).
Stain normalization tidak cukup membantu pada eksperimen ini.

---

## Temuan Kunci

### 1. FocusAugMix Murni (tanpa MHA) Paling Robust

`focusmix_v2` menjadi satu-satunya eksperimen yang melampaui baseline secara signifikan dalam
lintas-domain (+3.6% acc, gap 0.274 vs 0.310). Ini menunjukkan FocusAugMix membantu model belajar
fitur morfologi sel yang lebih general, **asalkan tidak dikombinasi dengan MHA**.

### 2. MHA Merusak Generalisasi Lintas-Domain

Semua eksperimen yang menggunakan MHA (`mha_only`, `focusmix_mha`, `focusmix_aggressive`,
`focusmix_full`) memiliki domain-shift gap lebih besar dari baseline. MHA kemungkinan belajar
*spatial attention patterns* yang spesifik terhadap staining ALL-IDB Italia, bukan fitur
morfologi yang invariant terhadap pewarnaan.

| Kelompok                                                                                |  Rata-rata Gap  |
| --------------------------------------------------------------------------------------- | :-------------: |
| Tanpa MHA (`baseline`, `saliency_mix`, `focusmix_v2`)                             | **0.300** |
| Dengan MHA (`mha_only`, `focusmix_mha`, `focusmix_aggressive`, `focusmix_full`) | **0.514** |

### 3. Grad-CAM Online Kontraproduktif

`focusmix_full` (satu-satunya eksperimen dengan Grad-CAM online) menghasilkan performa lintas-
domain terburuk. Pada dataset kecil dengan training pendek (≤7 epoch), Grad-CAM belum stabil —
menggunakan saliency yang berubah-ubah sebagai panduan augmentasi justru menambah noise.

### 4. Stain Normalization: Efek Beragam

Tidak ada metode normalisasi yang konsisten unggul di semua eksperimen:

- **Reinhard** paling konsisten membantu (memberikan acc tertinggi pada 5 dari 7 eksperimen)
- **Macenko** bervariasi — membantu pada beberapa kasus tetapi merusak pada yang lain
  (terutama `focusmix_mha`: −0.130 acc vs no-norm)
- Untuk eksperimen dengan bias Abnormal ekstrem (`mha_only`, `saliency_mix`),
  normalisasi membantu menyeimbangkan prediksi (F1 macro naik signifikan)

### 5. Bias Kelas Dominan pada Model Berbasis MHA

Eksperimen dengan MHA cenderung kolaps ke satu kelas, baik over-predict Abnormal (`mha_only`,
`saliency_mix`) maupun over-predict Normal (`focusmix_mha`, `focusmix_full`). Ini konsisten
dengan hipotesis bahwa MHA overfits ke distribusi kelas training ALL-IDB.

### 6. Gap Per-Kelas: Recall Abnormal vs Normal pada Data Eksternal

Salah satu temuan paling krusial adalah **asimetri recall per kelas** antara Abnormal (ALL/blast)
dan Normal (HEM) pada C-NMC. Gap ini jauh lebih informatif dari sekedar gap keseluruhan.

#### Recall Per-Kelas dari Confusion Matrix (C-NMC No Norm)

| Eksperimen            | Recall Abnormal | Recall Normal | Gap (Abn − Norm) | Arah Bias     |
| --------------------- | :-------------: | :-----------: | :--------------: | ------------- |
| `baseline`          |     93.5%       |     16.5%     |    **+77.0%**  | → Abnormal    |
| `mha_only`          |     96.7%       |      4.9%     |    **+91.8%**  | → Abnormal    |
| `saliency_mix`      |     99.5%       |      1.4%     |    **+98.1%**  | → Abnormal    |
| `focusmix_v2`       |     83.2%       |     49.9%     |    **+33.3%**  | → Abnormal    |
| `focusmix_mha`      |     28.4%       |     86.9%     |    **−58.5%**  | → Normal      |
| `focusmix_aggressive` |   24.1%       |     88.6%     |    **−64.5%**  | → Normal      |
| `focusmix_full`     |      7.0%       |     95.3%     |    **−88.3%**  | → Normal      |

> Recall per kelas dihitung langsung dari diagonal confusion matrix:
> Recall Abnormal = TP_abn / (TP_abn + FN_abn), Recall Normal = TP_norm / (TP_norm + FN_norm)

#### Mengapa Ada Gap dan Kenapa Arahnya Berbeda?

Ada dua pola berlawanan yang masing-masing punya penjelasan tersendiri.

**Pola 1 — Bias ke Abnormal (baseline, mha_only, saliency_mix):**

1. **Imbalance kelas training.** Training ALL-IDB memiliki ~372 Abnormal vs ~184 Normal (rasio ≈2:1).
   Model yang belajar dari distribusi ini secara natural cenderung memprediksi Abnormal karena
   itu pilihan yang lebih "aman" secara statistik.

2. **Ciri morfologi blast cell lebih persisten lintas domain.** Sel Abnormal (blast) memiliki
   ciri khas yang kuat — inti besar, kromatin kasar, rasio nukleus-sitoplasma tinggi — yang
   relatif tetap terlihat walaupun protokol pewarnaan berbeda (Giemsa Italia vs Wright-Giemsa India).
   Sebaliknya, sel Normal (HEM/limfosit) lebih sensitif terhadap perubahan staining: sitoplasma
   dan membran sel berubah warna secara berbeda di Wright-Giemsa, membuat fitur warna yang dipelajari
   dari ALL-IDB tidak lagi cocok.

3. **Distibusi C-NMC menguntungkan bias Abnormal.** C-NMC train-merged berisi 68% ALL dan 32% HEM.
   Model yang memprediksi "semua Abnormal" secara naif pun mendapat accuracy ~68%, sehingga
   akurasi keseluruhan terlihat wajar padahal recall Normal nyaris nol.

**Pola 2 — Bias ke Normal (focusmix_mha, focusmix_aggressive, focusmix_full):**

1. **FocusAugMix mengacaukan representasi Abnormal.** FocusAugMix mem-paste potongan superpixel
   dari satu gambar ke gambar lain. Ketika diterapkan ke sel Abnormal di dataset kecil, patch
   Normal yang di-paste ke wilayah sel blast menciptakan pola "tambal sulam" yang tidak
   konsisten. Model belajar bahwa penampilan yang tidak seragam itu adalah ciri Abnormal —
   tetapi di C-NMC, sel blast terlihat uniform tanpa tambalan, sehingga model salah
   mengklasifikasikannya sebagai Normal.

2. **MHA memperkuat fitur lokal yang domain-spesifik.** MHA mempelajari *spatial attention
   patterns* pada token 14×14 dari stage backbone. Pada ALL-IDB yang homogen (satu lab, satu
   protokol), pola atensi ini sangat spesifik terhadap warna dan tekstur Giemsa Italia. Ketika
   ditambah FocusAugMix yang agresif, model mengkonsolidasikan perhatiannya ke pola yang tidak
   ada di C-NMC domain, mengakibatkan prediksi Normal secara masif.

3. **Representasi Abnormal yang terdegradasi.** Kombinasi FocusAugMix+MHA pada dataset kecil
   (~556 training samples) dalam hanya ≤7 epoch membuat representasi kelas Abnormal "terpecah"
   dan tidak kompak di feature space. Di domain sumber pun representasi ini cukup untuk
   mengklasifikasikan dengan benar, tetapi ketika domain bergeser, representasi yang lemah ini
   tidak bisa mempertahankan discoverability kelas Abnormal.

#### Implikasi Praktis

Gap per-kelas ini sangat kritis dalam konteks medis:

- **False Negative Abnormal (FN Abn) lebih berbahaya** — sel leukemia yang diprediksi Normal
  berarti pasien tidak terdeteksi. Eksperimen `focusmix_mha`, `focusmix_aggressive`, dan
  `focusmix_full` memiliki FN Abn sangat tinggi (>5000 dari 7272 ALL).
- **Accuracy overall dapat menyesatkan.** Accuracy 70%+ bisa terjadi dengan recall Normal
  yang mendekati nol, hanya karena distribusi kelas C-NMC yang tidak seimbang.
- **F1 macro adalah metrik yang lebih jujur** untuk skenario lintas-domain dengan class imbalance.

---

## Rekomendasi

| Skenario                    | Eksperimen yang Direkomendasikan  | Alasan                               |
| --------------------------- | --------------------------------- | ------------------------------------ |
| Deployment tanpa stain norm | `focusmix_v2`                   | Acc tertinggi (72.6%), gap terkecil  |
| Deployment dengan Reinhard  | `focusmix_v2` atau `baseline` | Keduanya kompetitif (68.9% vs 70.4%) |
| Deployment dengan Macenko   | `saliency_mix`                  | F1 macro 0.507, paling stabil        |
| In-domain saja (ALL-IDB)    | Semua setara (100%)               | Pilih berdasarkan kebutuhan lain     |

---

## Catatan Dataset

- **Training/val model:** ALL-IDB1 + ALL-IDB2 (Giemsa stain, Italia) — 204 val samples
- **External test:** `C-NMC_train_merged` (Wright-Giemsa, India) — **10.661 sel**
  (7.272 ALL + 3.389 HEM). Ini adalah split training C-NMC yang memiliki label kelas.
  Split test C-NMC (prelim & final) tidak memiliki label publik sehingga tidak dapat
  digunakan untuk evaluasi kuantitatif.
- Hasil tersimpan di `results/` (JSON per eksperimen + `summary.json`)

---

Last updated: 2026-06-05
