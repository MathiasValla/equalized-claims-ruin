#!/usr/bin/env python3
"""Render manuscript figures from the retained records; no simulation."""
from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'equalized-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'equalized-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
records = json.loads((ROOT / 'data' / 'checks.json').read_text())
BLUE, INK, GRAY = '#315c73', '#20252a', '#87949c'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'axes.edgecolor': INK, 'axes.linewidth': .65,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': .65, 'ytick.major.width': .65,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'legend.fontsize': 8, 'savefig.facecolor': 'white',
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


def save(fig, name):
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(FIG / f'{name}.{extension}', dpi=600, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)


def clean_log_x(ax):
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='minor', length=2)


fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), layout='constrained')
for rho, color, marker in [(.5, BLUE, 'o'), (1, INK, 's'), (2, GRAY, '^')]:
    rows = sorted((r for r in records['two_claim_lower_tail_orders']
                   if float(r['rho']) == rho), key=lambda r: float(r['u']))
    axes[0].semilogx([float(r['u']) for r in rows],
                     [float(r['ratio']) for r in rows], marker=marker,
                     color=color, ms=4, label=fr'$\rho={rho:g}$')
axes[0].axhline(1, ls='--', lw=.8, color=INK, zorder=0)
axes[0].set(xlabel='Initial reserve $u$', ylabel='Gap / leading term',
            ylim=(.80, 1.035), title='(a) Continuous lower tails')
axes[0].legend(loc='lower right')
atom = sorted(records['zero_atom_poisson'], key=lambda r: float(r['u']))
u = np.array([float(r['u']) for r in atom])
lead = float(atom[0]['linear_coefficient'])
curve = np.geomspace(u[0]*.8, u[-1]*1.2, 100)
axes[1].loglog(curve, lead*curve, color=INK, lw=1, label='Linear prediction')
axes[1].loglog(u, [float(r['gap']) for r in atom], 'o', color=BLUE, ms=4,
               label='Direct Poisson sum')
axes[1].set(xlabel='Initial reserve $u$', ylabel='Equalization gap',
            title='(b) A mass at zero')
axes[1].legend(loc='upper right')
for ax in axes:
    ax.invert_xaxis()
    clean_log_x(ax)
save(fig, 'lower_tail_orders')
print('Rendered the local-asymptotic figure as PDF, SVG and 600 dpi PNG.')
