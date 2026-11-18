#!/usr/bin/env python3
"""Generate manuscript values directly from retained numerical records."""
from pathlib import Path
from decimal import Decimal as D, ROUND_FLOOR, ROUND_CEILING
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    six=json.loads((ROOT/'data/six_examples_importance.json').read_text())
    bounds=json.loads((ROOT/'data/certified_bounds.json').read_text())
    validation=json.loads((ROOT/'data/main_example_validation.json').read_text())
    timing=json.loads((ROOT/'data/timing.json').read_text())
    labels=['Main','A1','A2','A3','A4','Positive loading']
    lines=[r'\begin{table}[!htbp]',r'\centering',
        r'\caption{Corrected gaps for the six parameter sets of \citet{loisel2023}, with \(\lambda=1\). Each estimate uses \(500{,}000\) weighted observations. The multiplier in the last column applies to both the estimate and its normal 95\% sampling interval.}',
        r'\label{tab:parameters}',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrl}',r'\toprule',
        r'Example & \(u\) & \(T\) & \(c\) & \(\alpha\) & \(\beta\) & Estimate [95\% interval]\\',r'\midrule']
    for label,r in zip(labels,six):
        p=r['parameters'];exponent=math.floor(math.log10(r['mean']));scale=10.**exponent
        nums=' & '.join((f'{p[k]:,.0f}'.replace(',', '{,}') if p[k] >= 10000 and float(p[k]).is_integer() else f'{p[k]:g}') for k in ['u','T','c','alpha','beta'])
        values=rf"\({r['mean']/scale:.5f}\ [{r['normal_95_lower']/scale:.5f},{r['normal_95_upper']/scale:.5f}]\times10^{{{exponent}}}\)"
        lines.append(f'{label} & {nums} & {values}'+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'case_table.tex').write_text('\n'.join(lines)+'\n')
    exponent=math.floor(math.log10(validation['mean']));scale=10.**exponent
    (out/'main_run.tex').write_text(r'\begin{equation}'+'\n'+rf"\widehat\Delta={validation['mean']/scale:.5f}\times10^{{{exponent}}},\qquad\text{{normal 95\% interval }}[{validation['normal_95_lower']/scale:.5f},{validation['normal_95_upper']/scale:.5f}]\times10^{{{exponent}}}."+'\n'+r'\label{eq:main-run}'+'\n'+r'\end{equation}'+'\n')
    def scientific(x,mode):
        d=D(x);e=d.adjusted();m=d.scaleb(-e).quantize(D('.0001'),rounding=mode)
        return rf'\({m}\times10^{{{e}}}\)'
    lines=[r'\begin{table}[!htbp]',r'\centering',
        r'\caption{Analytic sign enclosures for Table~\ref{tab:parameters}. Rounding error in evaluating \eqref{eq:lower-six}--\eqref{eq:upper-six} and the omitted count tail \eqref{eq:count-tail} are included. Displayed endpoints are rounded outwards.}',
        r'\label{tab:enclosures}',r'\begin{tabular}{lll}',r'\toprule',r'Example & Lower endpoint & Upper endpoint\\',r'\midrule']
    for label,b in zip(labels,bounds):
        lines.append(label+' & '+scientific(b['delta_lower'],ROUND_FLOOR)+' & '+scientific(b['delta_upper'],ROUND_CEILING)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'enclosure_table.tex').write_text('\n'.join(lines)+'\n')
    main=next(r for r in timing['runs'] if r['label']=='Main example (Table 1)')
    independent=next(r for r in timing['runs'] if r['label']=='independent main run')
    total_six=sum(r['seconds'] for r in timing['runs'] if r['samples']==500000)
    text=rf'''In a recorded run on an ARM64 macOS machine, using one Python process
with numerical-library thread limits set to one, the \(500{{,}}000\)-observation
principal example took {main['seconds']:.2f} seconds and the independent two-million-observation
run took {independent['seconds']:.2f} seconds. All six \(500{{,}}000\)-observation cases took
{total_six:.2f} seconds in total. The six outward-rounded analytic enclosures took
{timing['certify_six_seconds']:.2f} seconds. Conditional volumes used 80-digit decimal
arithmetic. The environment, seeds and individual timings are recorded
alongside the results.
'''
    (out/'timing_summary.tex').write_text(text)
    benchmark = json.loads((ROOT / 'data/paired_benchmark.json').read_text())
    lines = [
        r'\begin{table}[!htbp]',
        r'\centering',
        r'\caption{Agreement of the two estimators in the larger-gap benchmark. The gap and both interval endpoints are in units of \(10^{-3}\). Intervals are normal 95\% sampling intervals.}',
        r'\label{tab:paired-benchmark}',
        r'\small',
        r'\begin{tabular}{lrrl}',
        r'\toprule',
        r'Estimator & Observations & Gap & 95\% interval\\',
        r'\midrule',
    ]
    for key, label in [('ordinary_paired', 'Ordinary paired paths'),
                       ('conditional_importance', 'Conditional importance sampling')]:
        run = benchmark[key]
        lower, upper = run['normal_95_interval']
        lines.append(
            f"{label} & {run['samples']:,} & {1e3*run['mean']:.5f}"
            f" & [{1e3*lower:.5f}, {1e3*upper:.5f}]" + r'\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out / 'paired_table.tex').write_text('\n'.join(lines) + '\n')
    print('Generated all three tables, the independent main result, and timing paragraph.')


if __name__=='__main__':
    main()
