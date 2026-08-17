import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def bootstrap_metric_ci(probs, labels, metric_fn, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(labels[idx], probs[idx].argmax(axis=1)))
    vals = np.array(vals)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(vals, [alpha * 100, (1 - alpha) * 100])
    return {'mean': float(vals.mean()), 'ci_low': float(lo), 'ci_high': float(hi)}


def main():
    ap = argparse.ArgumentParser(
        description='Bootstrap confidence interval (default 95%) untuk F1-macro & accuracy, '
                     'dihitung dari probs+labels mentah (.npz) hasil '
                     '`evaluate_external.py --save-probs`.'
    )
    ap.add_argument('--npz', nargs='+', required=True,
                    help='Satu atau lebih file .npz berisi array probs, labels.')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--ci', type=float, default=0.95)
    args = ap.parse_args()

    for npz_path in args.npz:
        d = np.load(npz_path)
        probs, labels = d['probs'], d['labels']

        f1_ci = bootstrap_metric_ci(
            probs, labels,
            lambda y, p: f1_score(y, p, average='macro', zero_division=0),
            n_boot=args.n_boot, ci=args.ci,
        )
        acc_ci = bootstrap_metric_ci(probs, labels, accuracy_score,
                                     n_boot=args.n_boot, ci=args.ci)

        name = Path(npz_path).stem
        print(f'{name}  (n={len(labels)}, n_boot={args.n_boot}, ci={args.ci:.0%})')
        print(f'  F1 macro : {f1_ci["mean"]:.4f}  [{f1_ci["ci_low"]:.4f}, {f1_ci["ci_high"]:.4f}]')
        print(f'  Accuracy : {acc_ci["mean"]:.4f}  [{acc_ci["ci_low"]:.4f}, {acc_ci["ci_high"]:.4f}]')


if __name__ == '__main__':
    main()
