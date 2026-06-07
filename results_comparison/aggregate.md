# Hasil Multi-Seed (mean +/- std)

Agregasi otomatis dari `results_multiseed/` oleh `aggregate_seeds.py`.
Metrik utama = **F1 macro**. Recall per-kelas = rata-rata lintas seed.

## Ringkasan — kondisi No-Norm (headline)

| Eksperimen | n_tta | Val F1 | No-Norm F1 (mean±std) | Acc (mean±std) | Rec Abn | Rec Norm | Gap |
| ---------- | :---: | :----: | :-------------------: | :------------: | :-----: | :------: | :-: |
| `coatnet_0` | 1 | 1.000 | 0.4185 ± 0.0223 | 0.6587 ± 0.0382 | 95.1% | 3.2% | 0.341±0.038 |
| `focusmix` | 1 | 1.000 | 0.3486 ± 0.1405 | 0.3915 ± 0.1029 | 13.3% | 94.7% | 0.609±0.103 |
| `focusmix_stain` | 1 | 1.000 | 0.5535 ± 0.1189 | 0.5703 ± 0.1300 | 48.4% | 75.5% | 0.430±0.130 |
| `no_mix` | 1 | 1.000 | 0.5636 ± 0.0817 | 0.6336 ± 0.0799 | 68.4% | 52.5% | 0.366±0.080 |

## `coatnet_0`  (3 run, seeds=[42, 123, 2025], n_tta=1)

Val F1: 1.0000 ± 0.0000

| Kondisi | F1 macro | Accuracy | Recall Abn | Recall Norm |
| ------- | :------: | :------: | :--------: | :---------: |
| No Norm | 0.4185 ± 0.0223 | 0.6587 ± 0.0382 | 95.1% | 3.2% |
| Macenko | 0.3762 ± 0.0684 | 0.4658 ± 0.1511 | 39.6% | 61.5% |
| Reinhard | 0.4063 ± 0.0004 | 0.6819 ± 0.0006 | 99.9% | 0.1% |

Domain-shift gap: 0.3413 ± 0.0382

## `focusmix`  (3 run, seeds=[42, 123, 2025], n_tta=1)

Val F1: 1.0000 ± 0.0000

| Kondisi | F1 macro | Accuracy | Recall Abn | Recall Norm |
| ------- | :------: | :------: | :--------: | :---------: |
| No Norm | 0.3486 ± 0.1405 | 0.3915 ± 0.1029 | 13.3% | 94.7% |
| Macenko | 0.4142 ± 0.0802 | 0.5415 ± 0.1542 | 61.0% | 39.4% |
| Reinhard | 0.4652 ± 0.1638 | 0.4998 ± 0.1471 | 40.4% | 70.5% |

Domain-shift gap: 0.6085 ± 0.1029

## `focusmix_stain`  (3 run, seeds=[42, 123, 2025], n_tta=1)

Val F1: 1.0000 ± 0.0000

| Kondisi | F1 macro | Accuracy | Recall Abn | Recall Norm |
| ------- | :------: | :------: | :--------: | :---------: |
| No Norm | 0.5535 ± 0.1189 | 0.5703 ± 0.1300 | 48.4% | 75.5% |
| Macenko | 0.4364 ± 0.0219 | 0.6328 ± 0.0335 | 89.5% | 6.9% |
| Reinhard | 0.5187 ± 0.0500 | 0.6205 ± 0.0412 | 75.0% | 34.2% |

Domain-shift gap: 0.4297 ± 0.1300

## `no_mix`  (3 run, seeds=[42, 123, 2025], n_tta=1)

Val F1: 1.0000 ± 0.0000

| Kondisi | F1 macro | Accuracy | Recall Abn | Recall Norm |
| ------- | :------: | :------: | :--------: | :---------: |
| No Norm | 0.5636 ± 0.0817 | 0.6336 ± 0.0799 | 68.4% | 52.5% |
| Macenko | 0.4416 ± 0.0281 | 0.6392 ± 0.0438 | 89.9% | 8.2% |
| Reinhard | 0.4710 ± 0.0557 | 0.6651 ± 0.0373 | 92.6% | 10.5% |

Domain-shift gap: 0.3664 ± 0.0799
