# Uji Signifikansi 3-Seed — F1 macro (cnmc_no_norm)

Paired test atas F1 macro per-seed (seed 42/123/2025) C-NMC 2019. **n=3 → daya uji rendah**; baca sebagai arah + effect size, bukan bukti definitif.

## F1 per-seed

| Model | seed42 | seed123 | seed2025 | mean ± std |
| ----- | :----: | :-----: | :------: | :--------: |
| ConvNeXtV2+FocusAugMix+stain (Ours) | 0.4327 | 0.6703 | 0.5576 | 0.5535 ± 0.1189 |
| ConvNeXtV2 no_mix | 0.4954 | 0.5411 | 0.6542 | 0.5636 ± 0.0817 |
| ConvNeXtV2 focusmix | 0.5086 | 0.2452 | 0.2920 | 0.3486 ± 0.1405 |
| CoAtNet-0 baseline | 0.4051 | 0.4443 | 0.4061 | 0.4185 ± 0.0223 |

## Perbandingan berpasangan (a − b)

| a vs b | Δmean (a−b) | Cohen d | paired t p | Wilcoxon p |
| ------ | :---------: | :-----: | :--------: | :--------: |
| ConvNeXtV2+FocusAugMix+stain (Ours) vs CoAtNet-0 baseline | +0.1350 | 1.35 | 0.145 | 0.250 |
| ConvNeXtV2 no_mix vs CoAtNet-0 baseline | +0.1451 | 1.63 | 0.106 | 0.250 |
| ConvNeXtV2+FocusAugMix+stain (Ours) vs ConvNeXtV2 no_mix | -0.0101 | -0.08 | 0.899 | 1.000 |
| ConvNeXtV2+FocusAugMix+stain (Ours) vs ConvNeXtV2 focusmix | +0.2049 | 0.80 | 0.300 | 0.500 |

> **Interpretasi.** Wilcoxon p=0.25 adalah p **terkecil** yang mungkin untuk n=3 (jadi tak pernah <0.05) — gunakan paired t-test + Cohen d sebagai indikator utama. Cohen d > 0.8 = efek besar.