import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

FIG_DIR = Path('figures')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def load_roc_pr(exp: str, folder: str) -> list[dict]:
    """Kumpulkan `results['roc_pr']` dari semua file hasil per-seed {exp}_seed*.json di `folder`.
    roc_pr dihitung di evaluate_external.py selalu atas kondisi cnmc_no_norm."""
    curves = []
    for p in sorted(Path(folder).glob(f'{exp}_seed*.json')):
        d = json.load(open(p))
        rp = d.get('roc_pr')
        if rp:
            curves.append(rp)
    return curves


def mean_roc(curves: list[dict], n_points: int = 200):
    # Vertical averaging: interpolasi TPR pada grid FPR tetap, lalu rata-rata lintas seed.
    fpr_grid = np.linspace(0, 1, n_points)
    tprs, aucs = [], []
    for c in curves:
        fpr, tpr = np.array(c['roc_curve']['fpr']), np.array(c['roc_curve']['tpr'])
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        aucs.append(c['roc_auc'])
    tprs = np.stack(tprs)
    return fpr_grid, tprs.mean(axis=0), tprs.std(axis=0), float(np.mean(aucs)), float(np.std(aucs))


def mean_pr(curves: list[dict], n_points: int = 200):
    # Interpolasi precision pada grid recall tetap (recall dari sklearn tidak monoton naik -> sort dulu).
    rec_grid = np.linspace(0, 1, n_points)
    precs, aps = [], []
    for c in curves:
        rec, prec = np.array(c['pr_curve']['recall']), np.array(c['pr_curve']['precision'])
        order = np.argsort(rec)
        precs.append(np.interp(rec_grid, rec[order], prec[order]))
        aps.append(c['pr_auc'])
    precs = np.stack(precs)
    return rec_grid, precs.mean(axis=0), precs.std(axis=0), float(np.mean(aps)), float(np.std(aps))


def plot_roc_pr(models: list[tuple[str, str, str]], out_stem: str) -> None:
    """models: list of (label, exp_name, results_folder)"""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)

    for label, exp, folder in models:
        curves = load_roc_pr(exp, folder)
        if not curves:
            print(f'  [skip] {exp} ({folder}): tidak ada roc_pr di hasil JSON '
                  f'(jalankan ulang evaluate_external.py dengan kode terbaru)')
            continue

        fpr, tpr_mean, tpr_std, auc_mean, auc_std = mean_roc(curves)
        line, = ax_roc.plot(fpr, tpr_mean, label=f'{label} (AUC={auc_mean:.3f}±{auc_std:.3f})')
        ax_roc.fill_between(fpr, tpr_mean - tpr_std, tpr_mean + tpr_std,
                            color=line.get_color(), alpha=0.15)

        rec, prec_mean, prec_std, ap_mean, ap_std = mean_pr(curves)
        line, = ax_pr.plot(rec, prec_mean, label=f'{label} (AP={ap_mean:.3f}±{ap_std:.3f})')
        ax_pr.fill_between(rec, prec_mean - prec_std, prec_mean + prec_std,
                           color=line.get_color(), alpha=0.15)

    ax_roc.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve (kelas Abnormal, C-NMC no-norm)')
    ax_roc.legend(fontsize=6.5, loc='lower right')

    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision–Recall Curve (kelas Abnormal, C-NMC no-norm)')
    ax_pr.legend(fontsize=6.5, loc='lower left')

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        out = FIG_DIR / f'{out_stem}.{ext}'
        fig.savefig(out, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


def main():
    plot_roc_pr([
        ('CoAtNet-0 (baseline)', 'coatnet_0', 'results_coatnet'),
        ('CoAtNet-0 + stain aug', 'coatnet_0_stain', 'results_coatnet'),
        ('ConvNeXt V2, no_mix (baseline)', 'no_mix', 'results_multiseed'),
        ('ConvNeXt V2 + FocusAugMix (Ours)', 'focusmix_stain', 'results_multiseed'),
    ], 'roc_pr_no_norm_models')


if __name__ == '__main__':
    main()
