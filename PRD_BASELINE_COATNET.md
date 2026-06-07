# PRD — Baseline CoAtNet-0 untuk Perbandingan dengan ConvNeXtV2-Tiny + FocusAugMix

> **Status:** Draft 1 · **Owner:** tim baseline · **Tanggal:** 2026-06-07
> **Dokumen pasangan:** [`PERBANDINGAN_BASELINE.md`](PERBANDINGAN_BASELINE.md) (protokol fair-comparison),
> [`experiment_results.md`](experiment_results.md) (hasil & analisis ConvNeXtV2), [`README.md`](README.md) (pipeline & tooling).

PRD ini mendefinisikan **apa yang harus dibangun, diuji, dan diserahkan** agar baseline **CoAtNet-0**
dapat masuk ke **satu tabel perbandingan yang sama** dengan model proposal **ConvNeXtV2-Tiny +
FocusAugMix (`focusmix_stain`)** secara *fair*. PRD ini bersifat **kontrak implementasi** — ia memetakan
protokol di `PERBANDINGAN_BASELINE.md` ke titik-integrasi konkret di kode (`src/`) dan ke skema output
yang sudah dikonsumsi oleh `src/aggregate_seeds.py`.

---

## 1. Latar Belakang & Masalah

Model proposal (ConvNeXtV2-Tiny + FocusAugMix + ReinhardJitter, alias `focusmix_stain`) sudah dievaluasi
penuh: 3 seed (42/123/2025), no-TTA headline, plus ablation TTA-8. Untuk publikasi conference, klaim
"model kami robust lintas-domain" **tidak kredibel tanpa baseline arsitektur pembanding yang dilatih pada
protokol identik**. CoAtNet-0 dipilih sebagai baseline karena (a) kapasitas params setara (~25M vs ~28.5M),
(b) arsitektur hybrid conv+attention yang berbeda paradigma dari ConvNeXtV2 murni-conv, sehingga
perbandingan informatif, (c) tersedia pretrained ImageNet di `timm`.

**Masalah yang diselesaikan PRD ini:** memastikan hasil CoAtNet-0 (1) diperoleh pada protokol shared yang
*identik*, dan (2) keluar dalam format JSON yang **langsung teragregasi** oleh tooling multi-seed yang ada,
tanpa perlu menulis ulang `aggregate_seeds.py`.

### Non-Goals (di luar cakupan PRD ini)

- Mengubah arsitektur/augmentasi ConvNeXtV2 (`focusmix_stain`) — itu model proposal, sudah final.
- Tuning hyperparameter CoAtNet untuk "menang". Baseline harus **jujur**: protokol sama, bukan dioptimasi.
- Mengganti dataset, split, atau loss. Semua komponen "Harus Sama" di §3 bersifat **terkunci**.

---

## 2. Target yang Harus Dilampaui (angka "Ours", FINAL)

Sumber: `results_multiseed/aggregate.json` — `focusmix_stain`, **3 seed (42/123/2025), no-TTA**.
Metrik utama = **F1 macro**, dilaporkan `mean ± std`. Recall pada kondisi headline **no_norm**.

| Kondisi   | Acc (mean±std)  | **F1 Macro (mean±std)** | Recall Abn | Recall Norm |
| --------- | :-------------: | :---------------------: | :--------: | :---------: |
| No Norm   | 0.5703 ± 0.1300 | **0.5535 ± 0.1189**     | 48.4%      | 75.5%       |
| Macenko   | 0.6328 ± 0.0335 | 0.4364 ± 0.0219         | 89.5%      | 6.9%        |
| Reinhard  | 0.6205 ± 0.0412 | 0.5187 ± 0.0500         | 75.0%      | 34.2%       |

Pembanding baseline ConvNeXtV2 tanpa stain-aug = `no_mix`: no-norm F1 **0.5636 ± 0.0817** (Rec Abn/Norm
68.4%/52.5%). CoAtNet-0 baseline diharapkan berperilaku mirip `no_mix` (tanpa stain-aug → cenderung
**butuh** Reinhard test-time untuk menyeimbangkan prediksi). **Itu hipotesis, bukan keharusan** — laporkan
apa adanya.

> **Catatan penting bagi implementer:** tujuan bukan membuat CoAtNet kalah. Tujuan adalah angka yang
> *fair*. Jika CoAtNet ternyata lebih baik, itu temuan valid — laporkan. Yang dilarang adalah
> ketidaksetaraan protokol (mis. epoch beda, TTA beda, seed beda).

---

## 3. Requirements — Komponen Terkunci (HARUS SAMA dengan ConvNeXtV2)

Diturunkan dari `PERBANDINGAN_BASELINE.md` §2 dan dari `src/main.py` (`ExperimentConfig` default).
Tabel ini adalah **acceptance gate**: jika salah satu berbeda, hasil CoAtNet **tidak fair** dan ditolak.

| # | Komponen | Nilai terkunci | Sumber-of-truth |
| - | -------- | -------------- | --------------- |
| R1 | Dataset train | ALL-IDB1+IDB2, split image-level 80/20 (~556 train / 204 val) | `src/segment_dataset.py`, folder `dataset/` |
| R2 | Dataset eval eksternal | C-NMC 2019 `C-NMC_train_merged` (10.661 sel) | `PKG_C_NMC 2019/` |
| R3 | Image size | 224 × 224 | `ExperimentConfig` |
| R4 | Augmentasi dasar | Resize→HFlip→VFlip→Rotation(20°)→ColorJitter(0.25/0.25/0.25/0.08)→ToTensor→Normalize(ImageNet) | `PERBANDINGAN_BASELINE.md` §2.2 |
| R5 | Loss | Weighted Focal Loss, γ=2.0, class_weights = inverse-freq dari train | `src/lightning_model.py`, `data_module.get_class_weights()` |
| R6 | Optimizer | AdamW | `ExperimentConfig` |
| R7 | LR / weight decay | 1e-4 / 0.05 | `ExperimentConfig` |
| R8 | Scheduler | Cosine Annealing + linear warmup 3 epoch | `lightning_model.py` |
| R9 | Max epochs / early stop | 30 epoch, EarlyStopping(`val_loss`, patience=10) | `src/main.py` |
| R10 | Gradient clip | 1.0 | `src/main.py` Trainer |
| R11 | Precision | `bf16-mixed` (fallback fp32) | `src/main.py` Trainer |
| R12 | Checkpoint monitor | `val_f1` (max), save_top_k=1 | `src/main.py` |
| R13 | Normalisasi Test | Lapor **no_norm**, **macenko**, **reinhard** (pakai `src/stain_normalize.py`, ref fit dari ALL-IDB train) | `src/evaluate_external.py` |
| R14 | TTA | Headline **n_tta=1 (no-TTA)**; TTA-8 hanya ablation opsional | `evaluate_external.py --tta-n` |
| R15 | Multi-seed | **42 / 123 / 2025**, lapor `mean ± std` | `src/run_multiseed.py` (`SEEDS`) |
| R16 | Class index mapping | 0=Abnormal, 1=Normal (urutan alfabet ImageFolder) | `data_module.py` |

> **Re-use, jangan tulis ulang.** R4, R5, R8, R13, R14 sudah terimplementasi di `src/data_module.py`,
> `src/lightning_model.py`, dan `src/evaluate_external.py`. Implementer CoAtNet **wajib** memanggil
> komponen yang sama, bukan membuat versi paralel yang berpotensi menyimpang.

---

## 4. Requirements — Komponen Pembeda (BOLEH BEDA, kontribusi masing-masing)

| Komponen | CoAtNet-0 (baseline) | ConvNeXtV2 (ours) |
| -------- | -------------------- | ----------------- |
| Backbone | **CoAtNet-0** (`timm`, pretrained IN1k, ~25M) | ConvNeXtV2-Tiny (~28.5M) |
| Augmentasi mixing | **CutMix + Mixup** (per-batch, p=0.5) | FocusAugMix (SLIC+saliency+Grad-CAM) |
| Stain aug train-time | **Tidak ada** | ReinhardJitter σ_mean=0.15, p=0.5 |
| MHA | Tidak ada | Tidak ada (di `focusmix_stain`) |

### 4.1 Backbone CoAtNet-0

```
Model  : timm.create_model('coatnet_0_rw_224', pretrained=True, num_classes=2)
         # cek timm.list_models('coatnet*') jika nama tidak tersedia di versi timm terpasang
Input  : 224 × 224 × 3
Head   : ganti classifier ke Linear(feat_dim, 2); Dropout(0.3) sebelum head (samakan dgn ConvNeXtV2)
```

### 4.2 Augmentasi Mixing: CutMix + Mixup

Per **batch**, pilih salah satu secara acak (prob 0.5/0.5); penerapan mixing per batch dengan prob
`aug_prob=0.5` (samakan dengan `aug_prob` ConvNeXtV2):

```python
if random() < 0.5:
    images, y_a, y_b, lam = cutmix(images, labels, alpha=1.0)
else:
    images, y_a, y_b, lam = mixup(images, labels, alpha=0.2)
# Loss kompatibel dengan Weighted Focal Loss (R5):
loss = lam * focal(logits, y_a) + (1 - lam) * focal(logits, y_b)
```

Parameter: CutMix α=1.0, Mixup α=0.2, prob mixing 0.5. **Tidak ada** ReinhardJitter, **tidak ada** MHA.

---

## 5. Kontrak Output (KRITIS — agar teragregasi otomatis)

`src/aggregate_seeds.py` mengumpulkan **semua** file `*_seed*.json` di folder hasil dan mengagregasinya.
Agar JSON CoAtNet ikut terbaca, ia **harus** memenuhi semua poin berikut.

### 5.1 Penamaan & lokasi file

Artefak CoAtNet disimpan **di folder terpisah** dari ConvNeXtV2 agar tidak tercampur:

```
checkpoints_coatnet/coatnet_0_seed{42,123,2025}/   <- bobot
logs_coatnet/coatnet_0_seed{42,123,2025}/          <- CSV log
results_coatnet/coatnet_0_seed{42,123,2025}.json   <- hasil eval no-TTA
```

- Pola nama **wajib** `coatnet_0_seed<seed>.json` (regex `*_seed*.json`, exp diturunkan dari prefix).
- Ablation TTA-8 → folder terpisah lagi `results_coatnet_tta8/coatnet_0_seed<seed>.json` (jangan menimpa no-TTA).
- Folder ConvNeXtV2 tetap `results_multiseed/` (tidak diubah).

### 5.2 Skema per-seed (key yang dibaca `aggregate_seeds.py`)

`aggregate_seeds.py` membaca, untuk tiap seed: `in_domain_val.f1_macro`, lalu untuk tiap kondisi
`cnmc_no_norm` / `cnmc_macenko` / `cnmc_reinhard` membaca `f1_macro`, `accuracy`, `confusion_matrix`
(recall per-kelas **diturunkan** dari CM — tak perlu dihitung manual), dan `domain_shift_gap`, `n_tta`,
`seed`, `exp`. Skema minimal:

```json
{
  "exp": "coatnet_0",
  "seed": 42,
  "n_tta": 1,
  "checkpoint": "checkpoints_multiseed/coatnet_0_seed42/best.ckpt",
  "in_domain_val":  { "accuracy": 0.0, "f1_macro": 0.0 },
  "cnmc_no_norm":   { "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0,
                      "precision_macro": 0.0, "recall_macro": 0.0,
                      "confusion_matrix": [[0,0],[0,0]] },
  "cnmc_macenko":   { "accuracy": 0.0, "f1_macro": 0.0, "confusion_matrix": [[0,0],[0,0]] },
  "cnmc_reinhard":  { "accuracy": 0.0, "f1_macro": 0.0, "confusion_matrix": [[0,0],[0,0]] },
  "domain_shift_gap": 0.0
}
```

> **`confusion_matrix` = `[[TP_Abn, FN_Abn], [FP_Abn, TN_Norm]]`** (baris=label asli, kolom=prediksi;
> index 0=Abnormal, 1=Normal). Ini konvensi yang dipakai `per_class_recall()` di `aggregate_seeds.py` —
> salah orientasi → recall Abn/Norm tertukar. `domain_shift_gap = val_acc − cnmc_no_norm_acc`.

### 5.3 Cara termudah memenuhi kontrak (jalur rekomendasi)

**Re-use `src/evaluate_external.py` apa adanya.** Ia sudah menghasilkan persis skema §5.2 (lihat
`results['cnmc_no_norm']`, `results['domain_shift_gap']`, `--tta-n`, `--output-json`). Syaratnya hanya:
checkpoint CoAtNet dapat di-load & dijalankan oleh script itu. Dua opsi integrasi:

- **Opsi A (paling kecil risikonya):** buat `CoAtNetLightningModel` yang **API-compatible** dengan
  `LeukemiaLightningModel` (sama `forward`, sama cara `load_from_checkpoint`, sama Focal Loss & optimizer
  config), sehingga `evaluate_external.py` dan `run_multiseed.py` bisa memanggilnya tanpa perubahan besar.
  Daftarkan sebagai entri `EXPERIMENTS['coatnet_0']` di `src/main.py` (dengan CutMix/Mixup di datamodule
  CoAtNet) dan tambahkan `'coatnet_0'` ke `KEY_EXPERIMENTS`/argumen `--exps` di `run_multiseed.py`.
- **Opsi B:** script evaluasi CoAtNet terpisah yang meng-**import** `compute_metrics`, normalizer dari
  `stain_normalize.py`, dan loader C-NMC dari `data_module.py`, lalu menulis JSON skema §5.2 sendiri.
  Lebih fleksibel, tapi wajib mereproduksi key & orientasi CM persis.

Opsi A lebih disukai karena memaksimalkan kode bersama (mengurangi sumber divergensi protokol).

> **Status: Opsi A SUDAH DIIMPLEMENTASI (2026-06-07).** Tidak ada `CoAtNetLightningModel` baru — sebagai
> gantinya `LeukemiaLightningModel` dibuat *backbone-switchable* (`backbone='convnextv2'|'coatnet'`),
> sehingga `evaluate_external.py` & `run_multiseed.py` **tidak diubah** dan tetap me-load CoAtNet via
> `load_from_checkpoint` (hyperparameter `backbone` tersimpan otomatis). Detail:
> - `src/lightning_model.py`: kelas `CoAtNetClassifier` (timm `coatnet_0_rw_224.sw_in1k`, head Dropout(0.3)+Linear),
>   fungsi `cutmix_mixup_batch()` (CutMix/Mixup level-batch), dan cabang backbone di `LeukemiaLightningModel`.
>   Mixing diterapkan di `training_step` saat `mixing='cutmix_mixup'`; optimizer CoAtNet = AdamW uniform-LR.
> - `src/main.py`: `ExperimentConfig` dapat field `backbone`/`coatnet_model_name`/`mixing`/`cutmix_alpha`/
>   `mixup_alpha`/`mix_prob`; entri baru `EXPERIMENTS['coatnet_0']` (`aug_mode='none'`, tanpa stain-aug/MHA).
> - Pretrained default = **`coatnet_0_rw_224.sw_in1k`** (~26.7M params, terverifikasi end-to-end: bf16 +
>   CutMix/Mixup + Focal Loss jalan, checkpoint round-trip reconstruct CoAtNet OK).
>
> **Cara menjalankan (hanya CoAtNet-0, 3 seed, no-TTA headline) — semua artefak di folder terpisah:**
> ```bash
> # 1 seed cepat (uji): python src/main.py --exp coatnet_0 --seed 42
> python src/run_multiseed.py --exps coatnet_0 \
>     --ckpt-root checkpoints_coatnet --log-root logs_coatnet --results-root results_coatnet
> # gabungkan dgn ConvNeXtV2 untuk tabel perbandingan:
> python src/aggregate_seeds.py --results-dir results_multiseed results_coatnet --out-dir results_comparison
> # (opsional) ablation TTA-8 tanpa latih ulang:
> python src/run_multiseed.py --exps coatnet_0 --no-train --tta-n 8 \
>     --ckpt-root checkpoints_coatnet --results-root results_coatnet_tta8
> ```

### 5.4 Agregasi final

Karena folder dipisah, gabungkan kedua folder dalam satu tabel dengan memberi `aggregate_seeds.py`
beberapa `--results-dir` (CoAtNet otomatis muncul bersama `focusmix_stain` & `no_mix`):

```bash
# CoAtNet sendiri:
python src/aggregate_seeds.py --results-dir results_coatnet
# Gabungan untuk tabel perbandingan akhir (output ke --out-dir):
python src/aggregate_seeds.py --results-dir results_multiseed results_coatnet --out-dir results_comparison
```

---

## 6. Deliverables

| ID | Artefak | Lokasi |
| -- | ------- | ------ |
| D1 | Implementasi backbone + CutMix/Mixup CoAtNet-0 | `src/` (model + datamodule/collate, sesuai Opsi A/B §5.3) |
| D2 | 3 checkpoint terbaik (per `val_f1`) | `checkpoints_coatnet/coatnet_0_seed{42,123,2025}/` |
| D3 | 3 JSON hasil eval no-TTA (skema §5.2) | `results_coatnet/coatnet_0_seed{42,123,2025}.json` |
| D4 | (Opsional) 3 JSON ablation TTA-8 | `results_coatnet_tta8/coatnet_0_seed*.json` |
| D5 | Baris CoAtNet di tabel gabungan | `results_comparison/aggregate.{md,json}` via `aggregate_seeds.py` |
| D6 | Update tabel perbandingan akhir | `PERBANDINGAN_BASELINE.md` §6 (isi baris _TBD_ CoAtNet) |
| D7 | Catatan singkat: versi `timm`, nama model persis, anomali training | bagian "Catatan" di JSON atau README |

---

## 7. Acceptance Criteria

Hasil CoAtNet diterima masuk paper **hanya jika** semua poin terpenuhi:

- [ ] **AC1 — Protokol identik.** Semua R1–R16 (§3) sesuai. Bukti: config tercetak di log training
      cocok dengan `ExperimentConfig` default ConvNeXtV2 (epoch 30, lr 1e-4, wd 0.05, focal γ=2.0,
      warmup 3, clip 1.0, bf16, monitor `val_f1`).
- [ ] **AC2 — Pembeda hanya yang diizinkan (§4).** Backbone CoAtNet-0, mixing CutMix+Mixup, tanpa
      stain-aug, tanpa MHA. Tidak ada perubahan lain.
- [ ] **AC3 — Multi-seed lengkap.** Tiga JSON `coatnet_0_seed{42,123,2025}.json` ada & valid skema §5.2.
- [ ] **AC4 — Headline no-TTA.** Angka utama `n_tta=1`. TTA-8 (jika ada) di folder terpisah, ditandai ablation.
- [ ] **AC5 — Kedua kondisi normalisasi dilaporkan.** `no_norm` **dan** `reinhard` (Macenko opsional tapi
      disarankan, karena `evaluate_external.py` sudah menghitungnya gratis).
- [ ] **AC6 — Teragregasi otomatis.** `python src/aggregate_seeds.py` menghasilkan baris `coatnet_0`
      tanpa error dan tanpa edit manual JSON.
- [ ] **AC7 — Sanity in-domain.** `val_f1` per seed tinggi (≈ 1.0 seperti ConvNeXtV2; val set 204 sel,
      task biner). Jika jauh di bawah, ada bug training — selidiki sebelum lapor.
- [ ] **AC8 — Confusion matrix waras.** Orientasi `[[TP_Abn,FN_Abn],[FP_Abn,TN_Norm]]` benar
      (cek: total baris 0 = 7.272 Abn, total baris 1 = 3.389 Norm pada C-NMC).

---

## 8. Rencana Kerja (Milestones)

| M | Tujuan | Output | Definition of Done |
| - | ------ | ------ | ------------------ |
| M1 | Backbone + head CoAtNet-0 jalan 1 epoch | training loop hidup | loss turun, checkpoint `val_f1` tersimpan |
| M2 | CutMix+Mixup + Focal Loss terintegrasi | datamodule/collate | loss mixing benar (`lam*FL_a+(1-lam)*FL_b`) |
| M3 | Eval 1 seed end-to-end | `coatnet_0_seed42.json` | skema §5.2, `aggregate_seeds.py` membacanya |
| M4 | Multi-seed 42/123/2025 | 3 JSON + checkpoint | AC3 terpenuhi |
| M5 | Agregasi + isi tabel | `aggregate.md` + `PERBANDINGAN_BASELINE.md` §6 | AC6, D6 |
| M6 | (Opsional) Ablation TTA-8 | `results_multiseed_tta8/` | D4 |

Estimasi waktu didominasi training: 30 epoch × 3 seed. Jalankan via pola `run_multiseed.py` agar
train+eval+penyisipan metadata `seed`/`exp` otomatis.

---

## 9. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
| ------ | ------ | -------- |
| Nama model `coatnet_0_rw_224` tak ada di versi `timm` terpasang | training gagal start | `timm.list_models('coatnet*')`; catat nama persis di D7 |
| Orientasi confusion matrix terbalik | recall Abn/Norm tertukar, salah klaim | pakai `compute_metrics()` dari `evaluate_external.py` (sudah benar); cek AC8 |
| Diam-diam ikut menyalakan stain-aug/MHA | comparison tidak fair | kunci config; review AC2 |
| File JSON tak terbaca aggregate (nama/skema salah) | tabel kosong | ikuti §5.1–5.2 persis; jalankan AC6 lebih awal di M3 |
| CoAtNet butuh warmup/LR beda agar konvergen | tergoda mengubah R7/R8 | **jangan** ubah hyperparameter terkunci; jika benar-benar perlu, dokumentasikan sebagai deviasi eksplisit dan diskusikan — bukan diam-diam |
| Mixing kuat + dataset kecil → underfit | val_f1 rendah (AC7 gagal) | verifikasi `aug_prob` & implementasi label-mixing; bandingkan dengan run tanpa mixing sebagai sanity |

---

## 10. Tabel Perbandingan Akhir (target pengisian)

Setelah CoAtNet selesai, baris ini di `PERBANDINGAN_BASELINE.md` §6 terisi. Semua angka **no-TTA, mean ±
std atas 3 seed**, F1 macro = metrik utama. Pilih kondisi normalisasi terbaik per model sebagai headline,
tapi tampilkan no_norm **dan** reinhard untuk transparansi.

| Model | Val F1 | No-Norm F1 (mean±std) | Reinhard F1 (mean±std) | Recall Abn | Recall Norm | Gap |
| ----- | :----: | :-------------------: | :--------------------: | :--------: | :---------: | :-: |
| CoAtNet-0 + CutMix+Mixup (Baseline)  | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| ConvNeXtV2-Tiny + FocusAugMix (Ours) | 1.000 | **0.5535 ± 0.1189** | 0.5187 ± 0.0500 | 48.4% | 75.5% | 0.430 ± 0.130 |

### Tabel Ablation TTA (opsional, hanya jika M6 dikerjakan)

| Config | No-Norm F1 | Catatan |
| ------ | :--------: | ------- |
| no-TTA | _TBD_ | angka headline |
| TTA-8  | _TBD_ | full dihedral 8-view (`--tta-n 8`) |

---

Last updated: 2026-06-07
