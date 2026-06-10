# Hasil Multi-Seed (mean +/- std)

Agregasi otomatis dari `results_multiseed/` oleh `aggregate_seeds.py`.
Metrik utama = **F1 macro**. Recall per-kelas = rata-rata lintas seed.

## Ringkasan — kondisi No-Norm (headline)

| Eksperimen | n_tta | Val F1 | No-Norm F1 (mean±std) | Acc (mean±std) | Rec Abn | Rec Norm | Gap |
| ---------- | :---: | :----: | :-------------------: | :------------: | :-----: | :------: | :-: |
| `coatnet_0` | 8 | 1.000 | 0.4147 ± 0.0157 | 0.6599 ± 0.0376 | 95.6% | 2.5% | 0.340±0.038 |

## `coatnet_0`  (3 run, seeds=[42, 123, 2025], n_tta=8)

Val F1: 1.0000 ± 0.0000

| Kondisi | F1 macro | Accuracy | Recall Abn | Recall Norm |
| ------- | :------: | :------: | :--------: | :---------: |
| No Norm | 0.4147 ± 0.0157 | 0.6599 ± 0.0376 | 95.6% | 2.5% |
| Macenko | 0.3727 ± 0.0674 | 0.4663 ± 0.1573 | 39.6% | 61.8% |
| Reinhard | 0.4059 ± 0.0005 | 0.6821 ± 0.0003 | 100.0% | 0.0% |

Domain-shift gap: 0.3401 ± 0.0376
