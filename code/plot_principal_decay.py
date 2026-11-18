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
    prediction = coefficient * u**2
    relative_gaps = 100 * (gaps / prediction - 1)
    relative_intervals = 100 * (intervals / prediction[:, None] - 1)
    line_u = np.geomspace(8e-5, 1.25e-2, 120)
    ink, blue, gray = '#243039', '#315c73', '#687783'
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'mathtext.fontset': 'stix',
        'font.size': 10, 'axes.edgecolor': gray, 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': ink, 'axes.labelcolor': ink, 'xtick.color': ink,
        'ytick.color': ink, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })
    fig, (ax, detail) = plt.subplots(
        2, 1, sharex=True, height_ratios=(1.6, 1), figsize=(5.9, 3.75),
        layout='constrained',
    )
    ax.set(xscale='log', yscale='log', xlim=(1.4e-2, 7e-5), ylim=(2e-15, 9e-11),
           ylabel=r'Ruin-probability gap $\Delta(u,T)$')
    ax.plot(line_u, coefficient * line_u**2, linestyle=(0, (4, 2)),
            color=gray, lw=1.35, label=r'Quadratic prediction $Ku^2$')
    ax.plot(u, gaps, 'o', ms=5.3, color=blue, label='Conditional estimates')
    ax.legend(loc='upper right', frameon=False, fontsize=9, handletextpad=.6)
    ax.yaxis.set_major_locator(FixedLocator(10.**np.arange(-15, -9)))
    detail.set(xlabel=r'Initial reserve $u$ (decreasing to the right)',
               ylabel='Difference from\n' r'$Ku^2$ (%)')
    detail.axhline(0, linestyle=(0, (4, 2)), color=gray, lw=1.35)
    detail.errorbar(
        u, relative_gaps,
        yerr=(relative_gaps - relative_intervals[:, 0],
              relative_intervals[:, 1] - relative_gaps),
        fmt='o', ms=5.3, color=blue, capsize=3, elinewidth=1.3, capthick=1.2,
    )
    detail.yaxis.set_major_locator(FixedLocator([-.5, -.25, 0, .25, .5]))
    detail.xaxis.set_major_locator(FixedLocator(u))
    for axis in (detail.xaxis, ax.yaxis):
        axis.set_major_formatter(LogFormatterMathtext())
        axis.set_minor_locator(NullLocator())
    for panel in (ax, detail):
        panel.grid(axis='y', color='#dfe5e8', lw=.55)
        panel.set_axisbelow(True)
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
