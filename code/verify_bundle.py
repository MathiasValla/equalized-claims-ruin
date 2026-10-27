#!/usr/bin/env python3
"""Recompute the retained results without modifying the data files."""
from pathlib import Path
import json
import math
import mpmath as mp
import numpy as np

from reproduce_bounds import CASES, calculate, self_check
from verify_local_gap import (
    check_subset_recursion, power_two_claim_checks, atom_poisson_checks,
    lomax_zero_reserve, importance_lomax, fixed_vector_gap,
    independent_determinant_survival,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def check_record(new, old):
    for name in ('mean', 'standard_error', 'normal_95_lower', 'normal_95_upper',
                 'relative_standard_error'):
        if not (math.isclose(new[name], old[name], rel_tol=1e-10, abs_tol=0)):
            raise AssertionError((name, new, old))
    if not (new['count_histogram'] == {int(k): v for (k, v) in old['count_histogram'].items()}):
        raise AssertionError('Numerical verification failed')
    if not (new['negative_numerical_gaps'] == old['negative_numerical_gaps'] == 0):
        raise AssertionError('Numerical verification failed')


def reproduce(old):
    p = old['parameters']
    new = importance_lomax(old['samples'], old['seed'], p['u'], p['lam'],
                           p['T'], p['c'], p['alpha'], p['beta'])
    check_record(new, old)


def main():
    self_check()
    check_subset_recursion()
    mp.mp.dps = 70
    for pair in [('.005', '.4'), ('.4', '.005'), ('.005', '.01'), ('.4', '.5'),
                 ('.005', '2'), ('0', '.05')]:
        xs = list(map(mp.mpf, pair))
        u, capital = mp.mpf('.03'), mp.mpf(1)
        eq = independent_determinant_survival([sum(xs)/2]*2, u, capital)
        cl = (independent_determinant_survival(xs, u, capital)
              + independent_determinant_survival(xs[::-1], u, capital))/2
        observed = fixed_vector_gap([list(map(float, xs))], float(u), float(capital))[0]
        if not (abs(mp.mpf(str(observed)) - (eq - cl)) < mp.mpf('1e-15')):
            raise AssertionError('Numerical verification failed')
    power_two_claim_checks()
    atom_poisson_checks()
    checks = json.loads((DATA / 'checks.json').read_text())
    z = lomax_zero_reserve(.95, 10, 1000, .1)
    for name in ('upper_survival', 'lower_survival'):
        if not (math.isclose(z[name], checks['main_example_zero_reserve'][name], abs_tol=1e-13)):
            raise AssertionError('Numerical verification failed')
    for old in checks['importance_sampling']:
        reproduce(old)
    six = json.loads((DATA / 'six_examples_importance.json').read_text())
    for case, old in zip(CASES, six):
        reproduce(old)
        bound = calculate(case, 80, 80)
        if not (float(bound['delta_lower']) < old['normal_95_lower']):
            raise AssertionError('Numerical verification failed')
        if not (old['normal_95_upper'] < float(bound['delta_upper'])):
            raise AssertionError('Numerical verification failed')
    reproduce(json.loads((DATA / 'main_example_validation.json').read_text()))
    print('All deterministic checks and ten retained Monte Carlo runs reproduced.')
    print('All six normal intervals lie inside the separate analytic enclosures.')
    print('Conditional kernels: Decimal, normally 80 digits; variates and summaries: binary64.')


if __name__ == '__main__':
    main()
