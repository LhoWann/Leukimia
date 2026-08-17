# Hasil Multi-Seed (mean +/- std)

Agregasi otomatis dari `results_multiseed/` oleh `aggregate_seeds.py`.
Metrik utama = **F1 macro**. Recall per-kelas = rata-rata lintas seed.

## Ringkasan — kondisi No-Norm (headline)

| Eksperimen | n_tta | Val F1 | No-Norm F1 (mean±std) | Acc (mean±std) | Rec Abn | Rec Norm | Gap |
| ---------- | :---: | :----: | :-------------------: | :------------: | :-----: | :------: | :-: |
| `focusmix` | 8 | 1.000 | 0.3399 ± 0.1435 | 0.3861 ± 0.1042 | 12.2% | 95.4% | 0.614±0.104 |
| `focusmix_stain` | 8 | 1.000 | 0.5578 ± 0.1274 | 0.5750 ± 0.1369 | 48.6% | 76.6% | 0.425±0.137 |
| `no_mix` | 8 | 1.000 | 0.5670 ± 0.0899 | 0.6414 ± 0.0819 | 69.6% | 52.4% | 0.359±0.082 |

## `focusmix`  (3 run, seeds=[42, 123, 2025], n_tta=8)

Val F1: 1.0000 ± 0.0000

| Kondisi | F1 macro | Accuracy | Recall Abn | Recall Norm |
| ------- | :------: | :------: | :--------: | :---------: |
| No Norm | 0.3399 ± 0.1435 | 0.3861 ± 0.1042 | 12.2% | 95.4% |
| Macenko | 0.4089 ± 0.0868 | 0.5435 ± 0.1616 | 61.4% | 39.2% |
| Reinhard | 0.4631 ± 0.1728 | 0.5019 ± 0.1536 | 40.5% | 70.9% |

Domain-shift gap: 0.6139 ± 0.1042

## `focusmix_stain`  (3 run, seeds=[42, 123, 2025], n_tta=8)

Val F1: 1.0000 ± 0.0000

| Kondisi | F1 macro | Accuracy | Recall Abn | Recall Norm |
| ------- | :------: | :------: | :--------: | :---------: |
| No Norm | 0.5578 ± 0.1274 | 0.5750 ± 0.1369 | 48.6% | 76.6% |
| Macenko | 0.4328 ± 0.0223 | 0.6353 ± 0.0349 | 90.2% | 6.2% |
| Reinhard | 0.5185 ± 0.0520 | 0.6260 ± 0.0418 | 76.0% | 33.7% |

Domain-shift gap: 0.4250 ± 0.1369

## `no_mix`  (3 run, seeds=[42, 123, 2025], n_tta=8)

Val F1: 1.0000 ± 0.0000

| Kondisi | F1 macro | Accuracy | Recall Abn | Recall Norm |
| ------- | :------: | :------: | :--------: | :---------: |
| No Norm | 0.5670 ± 0.0899 | 0.6414 ± 0.0819 | 69.6% | 52.4% |
| Macenko | 0.4395 ± 0.0272 | 0.6428 ± 0.0410 | 90.8% | 7.5% |
| Reinhard | 0.4639 ± 0.0512 | 0.6674 ± 0.0371 | 93.7% | 9.0% |

Domain-shift gap: 0.3586 ± 0.0819
