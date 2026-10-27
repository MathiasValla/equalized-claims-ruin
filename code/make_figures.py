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
axes[0].legend(loc='lower left')
atom = sorted(records['zero_atom_poisson'], key=lambda r: float(r['u']))
u = np.array([float(r['u']) for r in atom])
lead = float(atom[0]['linear_coefficient'])
curve = np.geomspace(u[0]*.8, u[-1]*1.2, 100)
axes[1].loglog(curve, lead*curve, color=INK, lw=1, label='Linear prediction')
axes[1].loglog(u, [float(r['gap']) for r in atom], 'o', color=BLUE, ms=4,
               label='Direct Poisson sum')
axes[1].set(xlabel='Initial reserve $u$', ylabel='Equalization gap',
            title='(b) A mass at zero')
axes[1].legend(loc='upper left')
for ax in axes:
    clean_log_x(ax)
save(fig, 'lower_tail_orders')

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), layout='constrained')
published = json.loads((ROOT/'data/published_main.json').read_text())
corrected = json.loads((ROOT/'data/six_examples_importance.json').read_text())[0]
bound = json.loads((ROOT/'data/certified_bounds.json').read_text())[0]
axes[0].axvline(0,color=GRAY,lw=.8,ls='--',zorder=0)
axes[0].errorbar(published['estimate']/1e-8,1,xerr=published['reported_half_width']/1e-8,
                 fmt='o',color=INK,capsize=4,ms=4)
axes[0].errorbar(corrected['mean']/1e-8,0,xerr=1.96*corrected['standard_error']/1e-8,
                 fmt='o',color=BLUE,capsize=4,ms=4)
axes[0].set(yticks=[0,1],yticklabels=['Corrected','Published'],ylim=(-.5,1.5),
            xlim=(-3.1,.45),xlabel=r'Gap $\Delta$ ($10^{-8}$ units)',title='(a) The reported sign')
axes[0].tick_params(axis='y',length=0)
lo,hi = float(bound['delta_lower'])/1e-13,float(bound['delta_upper'])/1e-13
axes[1].axvline(0,color=GRAY,lw=.8,ls='--',zorder=0)
axes[1].hlines(0,lo,hi,color=GRAY,lw=2.5)
axes[1].vlines([lo,hi],-.08,.08,color=GRAY,lw=.9)
axes[1].errorbar(corrected['mean']/1e-13,1,xerr=1.96*corrected['standard_error']/1e-13,
                 fmt='o',color=BLUE,capsize=4,ms=4)
axes[1].set(yticks=[0,1],yticklabels=['Analytic bound','Sampling'],ylim=(-.5,1.5),
            xlim=(-.2,5),xlabel=r'Gap $\Delta$ ($10^{-13}$ units)',title='(b) The positive gap')
axes[1].tick_params(axis='y',length=0)
save(fig,'counterexample_correction')
print('Rendered two figures as PDF, SVG and 600 dpi PNG.')
