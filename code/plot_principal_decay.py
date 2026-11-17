#!/usr/bin/env python3
"""Visualize how the corrected Lomax gap falls with the initial reserve."""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'equalized-revision-mpl'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator
import numpy as np
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    records = json.loads((root / 'data' / 'checks.json').read_text())
    runs = sorted(records['importance_sampling'], key=lambda row: row['parameters']['u'])
    u = np.array([row['parameters']['u'] for row in runs])
    gaps = np.array([row['mean'] for row in runs])
    intervals = np.array([[row['normal_95_lower'], row['normal_95_upper']] for row in runs])
    limits = records['main_example_zero_reserve']
    coefficient = np.mean([limits['quadratic_coefficient_lower'],
                           limits['quadratic_coefficient_upper']])
    line_u = np.geomspace(8e-5, 1.25e-2, 120)
    ink, blue, gray = '#243039', '#315c73', '#687783'
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'mathtext.fontset': 'stix',
        'font.size': 10, 'axes.edgecolor': gray, 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': ink, 'axes.labelcolor': ink, 'xtick.color': ink,
        'ytick.color': ink, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(5.9, 2.82), layout='constrained')
    ax.set(xscale='log', yscale='log', xlim=(1.4e-2, 7e-5), ylim=(2e-15, 9e-11),
           xlabel=r'Initial reserve $u$ (decreasing to the right)',
           ylabel=r'Ruin-probability gap $\Delta(u,T)$')
    ax.plot(line_u, coefficient * line_u**2, linestyle=(0, (4, 2)),
            color=gray, lw=1.35, label=r'Quadratic prediction $Ku^2$')
    ax.errorbar(u, gaps, yerr=(gaps - intervals[:, 0], intervals[:, 1] - gaps),
                fmt='o', ms=5.7, color=blue, capsize=2.3,
                elinewidth=1.2, label='Conditional estimates')
    ax.legend(loc='upper right', frameon=False, fontsize=9, handletextpad=.6)
    ax.yaxis.set_major_locator(FixedLocator(10.**np.arange(-15, -9)))
    ax.xaxis.set_major_locator(FixedLocator(u))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(LogFormatterMathtext())
        axis.set_minor_locator(NullLocator())
    ax.grid(axis='y', color='#dfe5e8', lw=.55)
    ax.set_axisbelow(True)
    fig.savefig(root / 'figures' / 'principal_decay.pdf')
    fig.savefig(root / 'figures' / 'principal_decay.svg')
    path = root / 'figures' / 'principal_decay.png'
    fig.savefig(path, dpi=600, facecolor='white')
    plt.close(fig)
    with Image.open(path) as image:
        rgb = image.convert('RGB')
    rgb.save(path, dpi=(600, 600))


if __name__ == '__main__':
    main()
