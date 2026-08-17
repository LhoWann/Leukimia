import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from scipy import stats

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

MODELS = [
    ('focusmix_stain', 'results_multiseed'),
    ('no_mix', 'results_multiseed'),
    ('focusmix', 'results_multiseed'),
    ('coatnet_0', 'results_coatnet'),
    ('coatnet_0_stain', 'results_coatnet'),
]
PAIRS = [
    ('focusmix_stain', 'coatnet_0'),
    ('no_mix', 'coatnet_0'),
    ('focusmix_stain', 'no_mix'),
    ('focusmix_stain', 'focusmix'),
    ('coatnet_0_stain', 'coatnet_0'),
    ('focusmix_stain', 'coatnet_0_stain')
]
LABEL = {
    'focusmix_stain': 'ConvNeXtV2+FocusAugMix+stain (Ours)',
    'no_mix': 'ConvNeXtV2 no_mix',
    'focusmix': 'ConvNeXtV2 focusmix',
    'coatnet_0': 'CoAtNet-0 baseline',
    'coatnet_0_stain': 'CoAtNet-0+stain',
}


def load_per_seed_f1(exp, folder, condition):
    out = {}
    for p in sorted(Path(folder).glob(f'{exp}_seed*.json')):
        d = json.load(open(p))
        seed = d.get('seed')
        if seed is None:
            seed = int(p.stem.rsplit('_seed', 1)[1])
        c = d.get(condition)
        if c:
            out[seed] = c['f1_macro']
    return out


def paired_stats(a_map, b_map):
    seeds = sorted(set(a_map) & set(b_map))
    a = [a_map[s] for s in seeds]
    b = [b_map[s] for s in seeds]
    diff = [x - y for x, y in zip(a, b)]
    mean_d = statistics.mean(diff)
    sd_d = statistics.stdev(diff) if len(diff) > 1 else 0.0
    cohen_d = mean_d / sd_d if sd_d > 1e-12 else float('inf')

    t_p = w_p = float('nan')
    try:
        t_p = float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        pass
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        pass
    return seeds, a, b, mean_d, sd_d, cohen_d, t_p, w_p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--condition', default='cnmc_no_norm',
                    choices=['cnmc_no_norm', 'cnmc_macenko', 'cnmc_reinhard'])
    ap.add_argument('--out', default='results_comparison/significance.md')
    args = ap.parse_args()

    f1 = {exp: load_per_seed_f1(exp, folder, args.condition) for exp, folder in MODELS}

    L = [f'# Uji Signifikansi 3-Seed — F1 macro ({args.condition})', '',
         'Paired test atas F1 macro per-seed (seed 42/123/2025) C-NMC 2019. '
         '**n=3 → daya uji rendah**; baca sebagai arah + effect size, bukan bukti definitif.',
         '', '## F1 per-seed', '',
         '| Model | seed42 | seed123 | seed2025 | mean ± std |',
         '| ----- | :----: | :-----: | :------: | :--------: |']
    for exp, _ in MODELS:
        m = f1[exp]
        if not m:
            continue
        vals = [m.get(s, float('nan')) for s in (42, 123, 2025)]
        present = [v for v in vals if v == v]
        ms = f'{statistics.mean(present):.4f} ± {statistics.stdev(present):.4f}' if len(present) > 1 else '—'
        L.append(f'| {LABEL[exp]} | ' + ' | '.join(f'{v:.4f}' for v in vals) + f' | {ms} |')

    L += ['', '## Perbandingan berpasangan (a − b)', '',
          '| a vs b | Δmean (a−b) | Cohen d | paired t p | Wilcoxon p |',
          '| ------ | :---------: | :-----: | :--------: | :--------: |']
    print(f'Kondisi: {args.condition}\n')
    for a_exp, b_exp in PAIRS:
        if not f1[a_exp] or not f1[b_exp]:
            continue
        seeds, a, b, md, sd, cd, tp, wp = paired_stats(f1[a_exp], f1[b_exp])
        cd_s = '∞' if cd == float('inf') else f'{cd:.2f}'
        L.append(f'| {LABEL[a_exp]} vs {LABEL[b_exp]} | {md:+.4f} | {cd_s} '
                 f'| {tp:.3f} | {wp:.3f} |')
        print(f'{a_exp:16} vs {b_exp:12}  dmean={md:+.4f}  d={cd_s:>5}  t_p={tp:.3f}  w_p={wp:.3f}')

    L += ['', '> **Interpretasi.** Wilcoxon p=0.25 adalah p **terkecil** yang mungkin untuk n=3 '
          '(jadi tak pernah <0.05) — gunakan paired t-test + Cohen d sebagai indikator utama. '
          'Cohen d > 0.8 = efek besar.']

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L), encoding='utf-8')
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
