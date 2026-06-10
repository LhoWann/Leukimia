import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

FIG_DIR = Path('figures')
TICK = ['Abnormal\n(ALL)', 'Normal\n(HEM)']

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


def mean_cm(exp, folder, condition):
    cms = []
    for p in sorted(Path(folder).glob(f'{exp}_seed*.json')):
        d = json.load(open(p))
        c = d.get(condition)
        if c and c.get('confusion_matrix'):
            cms.append(np.array(c['confusion_matrix'], dtype=float))
    if not cms:
        return None
    return np.mean(cms, axis=0)


def draw_panel(ax, cm, title, show_y):
    row = cm.sum(axis=1, keepdims=True)
    rec = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    im = ax.imshow(rec, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(TICK)
    ax.set_xlabel('Predicted label')
    if show_y:
        ax.set_yticklabels(TICK)
        ax.set_ylabel('True label')
    else:
        ax.set_yticklabels([])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{rec[i, j] * 100:.1f}%\n(n={int(round(cm[i, j])):,})',
                    ha='center', va='center', fontsize=9,
                    color='white' if rec[i, j] > 0.5 else '#1a1a1a')

    ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    for d in (0, 1):
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                               edgecolor='#d62728', linewidth=1.6))

    ax.set_title(f'{title}\nRecall: ALL {rec[0, 0] * 100:.0f}%, HEM {rec[1, 1] * 100:.0f}%')
    return im


def make_figure(panels, out_stem):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.95 * n + 0.6, 3.5),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]
    im = None
    for k, (ax, (title, cm)) in enumerate(zip(axes, panels)):
        if cm is None:
            ax.set_visible(False); continue
        im = draw_panel(ax, cm, title, show_y=(k == 0))
    cbar = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.02)
    cbar.set_label('Recall (row-normalized)')
    for ext in ('pdf', 'png'):
        out = FIG_DIR / f'{out_stem}.{ext}'
        fig.savefig(out, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)

    make_figure([
        ('CoAtNet-0\n(baseline)', mean_cm('coatnet_0', 'results_coatnet', 'cnmc_no_norm')),
        ('ConvNeXt V2, no_mix\n(baseline)', mean_cm('no_mix', 'results_multiseed', 'cnmc_no_norm')),
        ('ConvNeXt V2 + FocusAugMix\n(Ours)', mean_cm('focusmix_stain', 'results_multiseed', 'cnmc_no_norm')),
    ], 'cm_no_norm_3models')

    make_figure([
        ('No normalization', mean_cm('focusmix_stain', 'results_multiseed', 'cnmc_no_norm')),
        ('Macenko', mean_cm('focusmix_stain', 'results_multiseed', 'cnmc_macenko')),
        ('Reinhard', mean_cm('focusmix_stain', 'results_multiseed', 'cnmc_reinhard')),
    ], 'cm_focusmix_stain_conditions')


if __name__ == '__main__':
    main()
