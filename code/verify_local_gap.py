#!/usr/bin/env python3
"""Checks for the proposed local equalization-gap theorem and estimator.

Requires numpy and mpmath. No GPU is needed. This is research code, not a
formal interval-arithmetic implementation. See the manuscript for statements,
proofs and attribution, and README.md for the limits of the novelty search.
"""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

from risk_core import (permutation_survival, equalized_survival, fixed_vector_gap,
                       lomax_zero_reserve, importance_lomax)








def independent_determinant_survival(xs, u, capital):
    """Independent high-precision order-statistic determinant for QA."""
    n = len(xs)
    if n == 0:
        return mp.mpf(1)
    running = mp.mpf(0)
    boundaries = []
    for x in xs:
        running += x
        boundaries.append(max(mp.mpf(0), (running - u) / capital))
    if boundaries[-1] >= 1:
        return mp.mpf(0)
    points = [1 - x for x in boundaries[::-1]]
    matrix = mp.matrix(n)
    for i in range(n):
        for j in range(n):
            power = j - i + 1
            matrix[i, j] = points[i] ** power / mp.factorial(power) if power >= 0 else 0
    return mp.factorial(n) * mp.det(matrix)


def check_subset_recursion():
    mp.mp.dps = 70
    maximum = mp.mpf(0)
    for n in range(2, 6):
        xs = [mp.mpf(1) / 200] + [mp.mpf(k + 1) / (4 * n * n) for k in range(n - 1)]
        u, capital = mp.mpf('.03'), mp.mpf(1)
        reference = mp.fsum(
            independent_determinant_survival(perm, u, capital)
            for perm in itertools.permutations(xs)
        ) / mp.factorial(n)
        observed = permutation_survival(
            [[np.longdouble(str(x)) for x in xs]], np.longdouble(str(u)), np.longdouble(1)
        )[0]
        error = abs(mp.mpf(str(observed)) - reference)
        maximum = max(maximum, error)
        if not (error < mp.mpf('1e-14')):
            raise AssertionError((n, error))
    return {"maximum_absolute_error_vs_independent_determinants": str(maximum)}


def power_two_claim_checks():
    """F(x)=x**rho on [0,1], N=2, cT=1; independent exact-kernel quadrature."""
    mp.mp.dps = 70
    records = []
    for rho in map(mp.mpf, ['.5', '1', '2']):
        for u in map(mp.mpf, ['.01', '.001', '.0001']):
            def moment(k, lo, hi):
                return rho / (rho + k) * (hi ** (rho + k) - lo ** (rho + k))

            def integrand(t):
                x = u * t ** (1 / rho)
                h, split = 1 + u - x, 2 * u - x
                first = (
                    (h + u) * moment(1, u, split)
                    - h * u * moment(0, u, split)
                    - moment(2, u, split)
                )
                second = (u - x) * (h * moment(0, split, 1) - moment(1, split, 1))
                return first + second

            exact = 2 * u ** rho * mp.quad(integrand, [0, 1])
            predicted = 2 * u ** (rho + 1) / (rho + 1) ** 2
            # The non-asymptotic sandwich, specialized to deterministic N=2.
            shortfall = u ** (rho + 1) / (rho + 1)
            kappa = 2 / (rho + 1)
            epsilon = 2 * (1 + u) * (2 * u) ** rho
            remainder = 2 * (u / (rho + 1) + (1 + u) * 2 * u)
            lower = shortfall * (kappa - epsilon)
            upper = shortfall * (kappa + remainder)
            if not (lower <= exact <= upper):
                raise AssertionError('Numerical verification failed')
            if rho == 1:
                polynomial = u ** 2 / 2 - 2 * u ** 3 / 3
                if not (abs(exact - polynomial) < mp.mpf('1e-60')):
                    raise AssertionError('Numerical verification failed')
            records.append({
                "rho": str(rho), "u": str(u), "gap": str(exact),
                "leading_term": str(predicted), "ratio": str(exact / predicted),
                "analytic_sandwich_lower": str(lower), "analytic_sandwich_upper": str(upper),
            })
    return records


def atom_poisson_checks():
    """Full Poisson model, P(X=0)=P(X=1)=1/2, c=2,T=lambda=1.

    Classical survival uses thinning and the positive-claim support gap.
    Equalized survival explicitly sums over the original N and Bernoulli marks.
    The uncomputed count tail is less than 1e-84.
    """
    mp.mp.dps = 80
    p, m, capital = mp.mpf('.5'), mp.mpf(1), mp.mpf(2)
    mu = m * (1 - p)
    zero_survival = mp.exp(-mu) * (1 + mu / 2)
    derivative = p / capital * (zero_survival - mp.exp(-mu))
    records = []
    for u in map(mp.mpf, ['.01', '.001', '.0001']):
        length = 1 + u / capital
        classical = mp.fsum(
            mp.exp(-mu) * mu ** k / mp.factorial(k)
            * (length - k / capital) * length ** (k - 1)
            for k in range(3)
        )
        equalized = mp.exp(-m)
        for n in range(1, 61):
            conditional = p ** n
            for k in range(1, min(n, 2) + 1):
                if not (mp.mpf(k) / n > u):
                    raise AssertionError('Numerical verification failed')
                marks = mp.binomial(n, k) * (1 - p) ** k * p ** (n - k)
                conditional += marks * (length - k / capital) * length ** (n - 1)
            equalized += mp.exp(-m) * m ** n / mp.factorial(n) * conditional
        gap = equalized - classical
        records.append({
            "u": str(u), "gap": str(gap), "linear_coefficient": str(derivative),
            "gap_divided_by_linear_prediction": str(gap / (derivative * u)),
        })
    return records






def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=200000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    mp.mp.dps = 70
    results = {
        "environment": {"python": sys.version, "numpy": np.__version__, "mpmath": mp.__version__},
        "subset_recursion_check": check_subset_recursion(),
        "two_claim_lower_tail_orders": power_two_claim_checks(),
        "zero_atom_poisson": atom_poisson_checks(),
    }
    zero = lomax_zero_reserve(.95, 10, 1000, .1)
    zero['quadratic_coefficient_lower'] = .095 / 20000 * (zero['lower_survival'] - math.exp(-.1))
    zero['quadratic_coefficient_upper'] = .095 / 20000 * (zero['upper_survival'] - math.exp(-.1))
    results['main_example_zero_reserve'] = zero
    results['importance_sampling'] = [
        importance_lomax(args.samples, 20261027 + i, u, 1, .1, 10000, .95, 10)
        for i, u in enumerate([.01, .001, .0001])
    ]
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
