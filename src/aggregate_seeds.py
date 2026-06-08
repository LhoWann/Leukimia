import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CONDITIONS = ['cnmc_no_norm', 'cnmc_macenko', 'cnmc_reinhard']
COND_LABEL = {
    'cnmc_no_norm': 'No Norm',
    'cnmc_macenko': 'Macenko',
    'cnmc_reinhard': 'Reinhard',
}


def per_class_recall(cm):
    abn_total = cm[0][0] + cm[0][1]
    norm_total = cm[1][0] + cm[1][1]
    rec_abn = cm[0][0] / abn_total if abn_total else 0.0
    rec_norm = cm[1][1] / norm_total if norm_total else 0.0
    return rec_abn, rec_norm


def mean_std(xs):
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def collect(results_dir: Path):
    runs = defaultdict(list)
    for p in sorted(results_dir.glob('*_seed*.json')):
        with open(p) as f:
            data = json.load(f)
        exp = data.get('exp') or p.stem.rsplit('_seed', 1)[0]
        seed = data.get('seed')
        if seed is None:
            m = p.stem.rsplit('_seed', 1)
            seed = int(m[1]) if len(m) == 2 and m[1].isdigit() else None
        runs[exp].append((seed, data))
    return runs


def aggregate(runs):
    out = {}
    for exp, entries in sorted(runs.items()):
        seeds = sorted(s for s, _ in entries if s is not None)
        n_tta = entries[0][1].get('n_tta', 1)
        exp_out = {'n_tta': n_tta, 'seeds': seeds, 'n_runs': len(entries)}

        val_f1s = [d['in_domain_val']['f1_macro'] for _, d in entries
                   if d.get('in_domain_val')]
        if val_f1s:
            m, s = mean_std(val_f1s)
            exp_out['val_f1_mean'] = round(m, 4)
            exp_out['val_f1_std'] = round(s, 4)

        for cond in CONDITIONS:
            f1s, accs, rabn, rnorm = [], [], [], []
            for _, d in entries:
                c = d.get(cond)
                if not c:
                    continue
                f1s.append(c['f1_macro'])
                accs.append(c['accuracy'])
                ra, rn = per_class_recall(c['confusion_matrix'])
                rabn.append(ra)
                rnorm.append(rn)
            if not f1s:
                continue
            f1m, f1sd = mean_std(f1s)
            accm, accsd = mean_std(accs)
            exp_out[cond] = {
                'f1_macro_mean': round(f1m, 4), 'f1_macro_std': round(f1sd, 4),
                'accuracy_mean': round(accm, 4), 'accuracy_std': round(accsd, 4),
                'recall_abnormal_mean': round(statistics.mean(rabn), 4),
                'recall_normal_mean': round(statistics.mean(rnorm), 4),
            }

        gaps = [d['domain_shift_gap'] for _, d in entries
                if d.get('domain_shift_gap') is not None]
        if gaps:
            gm, gsd = mean_std(gaps)
            exp_out['domain_shift_gap_mean'] = round(gm, 4)
            exp_out['domain_shift_gap_std'] = round(gsd, 4)

        out[exp] = exp_out
    return out


def to_markdown(agg):
    L = ['# Hasil Multi-Seed (mean +/- std)', '',
         'Agregasi otomatis dari `results_multiseed/` oleh `aggregate_seeds.py`.',
         'Metrik utama = **F1 macro**. Recall per-kelas = rata-rata lintas seed.', '']

    L += ['## Ringkasan — kondisi No-Norm (headline)', '',
          '| Eksperimen | n_tta | Val F1 | No-Norm F1 (mean±std) | Acc (mean±std) | Rec Abn | Rec Norm | Gap |',
          '| ---------- | :---: | :----: | :-------------------: | :------------: | :-----: | :------: | :-: |']
    for exp, d in agg.items():
        nn = d.get('cnmc_no_norm')
        if not nn:
            continue
        val = f"{d['val_f1_mean']:.3f}" if 'val_f1_mean' in d else '—'
        gap = (f"{d['domain_shift_gap_mean']:.3f}±{d['domain_shift_gap_std']:.3f}"
               if 'domain_shift_gap_mean' in d else '—')
        L.append(
            f"| `{exp}` | {d['n_tta']} | {val} "
            f"| {nn['f1_macro_mean']:.4f} ± {nn['f1_macro_std']:.4f} "
            f"| {nn['accuracy_mean']:.4f} ± {nn['accuracy_std']:.4f} "
            f"| {nn['recall_abnormal_mean'] * 100:.1f}% | {nn['recall_normal_mean'] * 100:.1f}% | {gap} |"
        )
    L.append('')

    for exp, d in agg.items():
        L += [f"## `{exp}`  ({d['n_runs']} run, seeds={d['seeds']}, n_tta={d['n_tta']})", '']
        if 'val_f1_mean' in d:
            L.append(f"Val F1: {d['val_f1_mean']:.4f} ± {d['val_f1_std']:.4f}\n")
        L += ['| Kondisi | F1 macro | Accuracy | Recall Abn | Recall Norm |',
              '| ------- | :------: | :------: | :--------: | :---------: |']
        for cond in CONDITIONS:
            c = d.get(cond)
            if not c:
                continue
            L.append(
                f"| {COND_LABEL[cond]} "
                f"| {c['f1_macro_mean']:.4f} ± {c['f1_macro_std']:.4f} "
                f"| {c['accuracy_mean']:.4f} ± {c['accuracy_std']:.4f} "
                f"| {c['recall_abnormal_mean'] * 100:.1f}% | {c['recall_normal_mean'] * 100:.1f}% |"
            )
        if 'domain_shift_gap_mean' in d:
            L.append(f"\nDomain-shift gap: {d['domain_shift_gap_mean']:.4f} "
                     f"± {d['domain_shift_gap_std']:.4f}")
        L.append('')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', nargs='+', default=['results_multiseed'],
                    help='Satu atau beberapa folder hasil per-seed untuk diagregasi bersama.')
    ap.add_argument('--out-dir', default=None,
                    help='Folder output aggregate.{json,md} (default: folder --results-dir pertama).')
    args = ap.parse_args()

    results_dirs = [Path(d) for d in args.results_dir]
    runs = defaultdict(list)
    for d in results_dirs:
        for exp, entries in collect(d).items():
            runs[exp].extend(entries)
    if not runs:
        searched = ', '.join(str(d.resolve()) for d in results_dirs)
        raise SystemExit(f'Tidak ada file *_seed*.json di {searched}. '
                         f'Jalankan run_multiseed.py dulu.')

    agg = aggregate(runs)

    out_dir = Path(args.out_dir) if args.out_dir else results_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / 'aggregate.json'
    out_md = out_dir / 'aggregate.md'
    with open(out_json, 'w') as f:
        json.dump(agg, f, indent=2)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(to_markdown(agg))

    print(f'Eksperimen teragregasi: {list(agg)}')
    for exp, d in agg.items():
        nn = d.get('cnmc_no_norm', {})
        if nn:
            print(f"  {exp:<22} no-norm F1 = {nn['f1_macro_mean']:.4f} "
                  f"± {nn['f1_macro_std']:.4f}  ({d['n_runs']} run)")
    print(f'\nSaved -> {out_json}')
    print(f'Saved -> {out_md}')


if __name__ == '__main__':
    main()
