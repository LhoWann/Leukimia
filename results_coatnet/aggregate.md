# Hasil Multi-Seed (mean +/- std)

Agregasi otomatis dari `results_multiseed/` oleh `aggregate_seeds.py`.
Metrik utama = **F1 macro**. Recall per-kelas = rata-rata lintas seed.

## Ringkasan — kondisi No-Norm (headline)

| Eksperimen | n_tta | Val F1 | No-Norm F1 (mean±std) | Acc (mean±std) | Rec Abn | Rec Norm | Gap |
| ---------- | :---: | :----: | :-------------------: | :------------: | :-----: | :------: | :-: |
| `coatnet_0` | 1 | 1.000 | 0.4185 ± 0.0223 | 0.6587 ± 0.0382 | 95.1% | 3.2% | 0.341±0.038 |

## `coatnet_0`  (3 run, seeds=[42, 123, 2025], n_tta=1)

Val F1: 1.0000 ± 0.0000

| Kondisi | F1 macro | Accuracy | Recall Abn | Recall Norm |
| ------- | :------: | :------: | :--------: | :---------: |
| No Norm | 0.4185 ± 0.0223 | 0.6587 ± 0.0382 | 95.1% | 3.2% |
| Macenko | 0.3762 ± 0.0684 | 0.4658 ± 0.1511 | 39.6% | 61.5% |
| Reinhard | 0.4063 ± 0.0004 | 0.6819 ± 0.0006 | 99.9% | 0.1% |

Domain-shift gap: 0.3413 ± 0.0382
