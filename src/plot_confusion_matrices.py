"""
Gambar confusion matrix C-NMC 2019 untuk paper.

Membaca JSON per-seed (skema aggregate_seeds.py), menjumlahkan confusion matrix
lintas 3 seed (C-NMC = 10.661 sel yang sama tiap seed → jumlah = total prediksi),
lalu memplot heatmap row-normalized (recall) dengan anotasi jumlah + persen.

Menghasilkan:
    figures/cm_no_norm_3models.png   <- 3 model utama, kondisi no-norm (headline)
    figures/cm_focusmix_stain_conditions.png  <- Ours pada no_norm/macenko/reinhard

Orientasi CM: [[TP_Abn, FN_Abn], [FP_Abn, TN_Norm]] (baris=label, kolom=prediksi;
0=Abnormal/ALL, 1=Normal/HEM).

Jalankan dari root proyek:
    python src/plot_confusion_matrices.py
"""
import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CLASSES = ['Abnormal\n(ALL)', 'Normal\n(HEM)']
FIG_DIR = Path('figures')


def sum_cm(exp, folder, condition):
    cms = []
    for p in sorted(Path(folder).glob(f'{exp}_seed*.json')):
        d = json.load(open(p))
        c = d.get(condition)
        if c and c.get('confusion_matrix'):
            cms.append(np.array(c['confusion_matrix'], dtype=float))
    if not cms:
        return None
    return np.sum(cms, axis=0)


def draw(ax, cm, title):
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel('Prediksi', fontsize=9)
    ax.set_ylabel('Label sebenarnya', fontsize=9)
    for i in range(2):
        for j in range(2):
            pct = norm[i, j] * 100
            cnt = int(cm[i, j])
            ax.text(j, i, f'{pct:.1f}%\n({cnt})', ha='center', va='center',
                    fontsize=9, color='white' if norm[i, j] > 0.5 else 'black')
    # recall per kelas di sumbu y
    rec = [norm[0, 0], norm[1, 1]]
    return im, rec


def main():
    FIG_DIR.mkdir(exist_ok=True)

    # --- Fig 1: 3 model utama, no-norm ---
    models = [
        ('CoAtNet-0\n(baseline)', 'coatnet_0', 'results_coatnet'),
        ('ConvNeXtV2 no_mix\n(baseline)', 'no_mix', 'results_multiseed'),
        ('ConvNeXtV2+FocusAugMix\n(Ours)', 'focusmix_stain', 'results_multiseed'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (title, exp, folder) in zip(axes, models):
        cm = sum_cm(exp, folder, 'cnmc_no_norm')
        if cm is None:
            ax.set_visible(False)
            continue
        im, rec = draw(ax, cm, title)
        ax.set_xlabel(f'Prediksi\nRec Abn={rec[0]*100:.1f}%  Rec Norm={rec[1]*100:.1f}%', fontsize=8)
    fig.suptitle('Confusion Matrix C-NMC 2019 (no-norm, agregat 3 seed, row-normalized = recall)',
                 fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label='Recall')
    out1 = FIG_DIR / 'cm_no_norm_3models.png'
    fig.savefig(out1, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out1}')

    # --- Fig 2: Ours pada 3 kondisi normalisasi ---
    conds = [('No Norm', 'cnmc_no_norm'), ('Macenko', 'cnmc_macenko'), ('Reinhard', 'cnmc_reinhard')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (clabel, ckey) in zip(axes, conds):
        cm = sum_cm('focusmix_stain', 'results_multiseed', ckey)
        if cm is None:
            ax.set_visible(False)
            continue
        im, rec = draw(ax, cm, clabel)
        ax.set_xlabel(f'Prediksi\nRec Abn={rec[0]*100:.1f}%  Rec Norm={rec[1]*100:.1f}%', fontsize=8)
    fig.suptitle('Ours (focusmix_stain) — efek normalisasi test-time (agregat 3 seed)',
                 fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label='Recall')
    out2 = FIG_DIR / 'cm_focusmix_stain_conditions.png'
    fig.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out2}')


if __name__ == '__main__':
    main()
