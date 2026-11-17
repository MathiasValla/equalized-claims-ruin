#!/usr/bin/env python3
"""Draw the worked surplus path and the saved paired simulation comparison."""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'equalized-revision-mpl'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INK = '#243039'
BLUE = '#315c73'
RUST = '#9b5035'
GRAY = '#687783'
GRID = '#dfe5e8'


def save_figure(fig, name):
    folder = ROOT / 'figures'
    folder.mkdir(exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(folder / f'{name}.{extension}', dpi=600, facecolor='white')
    plt.close(fig)
    path = folder / f'{name}.png'
    with Image.open(path) as image:
        rgb = image.convert('RGB')
    rgb.save(path, dpi=(600, 600))


def plot_early_claim():
    # Original clock. Starting at zero at -u/c gives t + 1 in premiums.
    start, end = -1., 2.
    arrivals = np.array([-.5, 1.])
    amounts = np.array([.25, 1.25])
    time = np.r_[start, np.repeat(arrivals, 2), end]
    premiums = time - start
    classical = premiums - np.repeat(np.r_[0., amounts.cumsum()], 2)
    equalized = premiums - np.repeat(np.r_[0., np.cumsum([amounts.mean()] * 2)], 2)

    fig, ax = plt.subplots(figsize=(5.9, 2.78), layout='constrained')
    ax.axvspan(start, 0., facecolor='#eaf1f5', linewidth=0)
    ax.axhline(0., color=GRAY, linewidth=.8)
    ax.axvline(0., color=GRAY, linewidth=.9, linestyle=(0, (2, 3)))
    ax.plot(time[:2], classical[:2], color=GRAY, linewidth=1.5)
    ax.plot(time[4:], classical[4:], color=GRAY, linewidth=1.5)
    ax.plot(time[1:5], classical[1:5], color=BLUE, linewidth=2.0, label='Classical claims')
    ax.plot(time[1:5], equalized[1:5], color=RUST, linewidth=1.8,
            linestyle=(0, (4, 2)), label='Equalized claims')
    ax.plot(arrivals[0], classical[2], marker='o', ms=5, color=BLUE)
    ax.plot(arrivals[0], equalized[2], marker='D', ms=4, color=RUST)
    ax.text(-.5, 1.98, r'Extra interval $u/c$', ha='center', va='top', color=BLUE)
    ax.annotate('Ruin after equalization', xy=(arrivals[0], equalized[2]), xytext=(-.07, -.34),
                ha='left', va='center', color=RUST,
                arrowprops=dict(arrowstyle='->', color=RUST, lw=.95))
    ax.set(xlim=(-1.03, 2.035), ylim=(-.5, 2.1),
           xticks=[-1., -.5, 0., 1., 2.], yticks=[-.5, 0., .5, 1., 1.5, 2.],
           xlabel='Time relative to the original start',
           ylabel='Surplus (monetary units)')
    ax.set_xticklabels(['−1', '−0.5', '0', '1', '2'])
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    save_figure(fig, 'early_claim_paths')


def plot_estimator_intervals():
    benchmark = json.loads((ROOT / 'data' / 'paired_benchmark.json').read_text())
    fig, ax = plt.subplots(figsize=(5.9, 2.03), layout='constrained')
    methods = [
        ('ordinary_paired', 1, GRAY, 'o'),
        ('conditional_importance', 0, BLUE, 'D'),
    ]
    for key, y, color, marker in methods:
        run = benchmark[key]
        estimate = run['mean'] * 1e3
        low, high = (value * 1e3 for value in run['normal_95_interval'])
        ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]],
                    fmt=marker, color=color, markerfacecolor='white' if y else color,
                    markeredgewidth=1.5, markersize=6.2, capsize=4, elinewidth=1.7,
                    zorder=3)
    ax.set(xlim=(1.132, 1.279), ylim=(-.55, 1.55), yticks=[1, 0],
           xticks=[1.14, 1.17, 1.20, 1.23, 1.26],
           xlabel=r'Ruin-probability gap $\Delta$ (units of $10^{-3}$)')
    ordinary_n = benchmark['ordinary_paired']['samples']
    weighted_n = benchmark['conditional_importance']['samples']
    ax.set_yticklabels([f'Ordinary paired simulation\n{ordinary_n:,} paths',
                        f'Conditional importance sampling\n{weighted_n:,} weighted draws'])
    ax.tick_params(axis='y', length=0, pad=9)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    save_figure(fig, 'paired_estimator_intervals')


if __name__ == '__main__':
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'mathtext.fontset': 'stix', 'font.size': 10,
        'axes.edgecolor': GRAY, 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
        'ytick.color': INK, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })
    plot_early_claim()
    plot_estimator_intervals()
