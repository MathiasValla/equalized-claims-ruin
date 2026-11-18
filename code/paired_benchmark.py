#!/usr/bin/env python3
"""Compare ordinary paired paths with conditional importance sampling."""
from argparse import ArgumentParser
import json
from pathlib import Path
import math
import platform
from time import perf_counter

import numpy as np

from risk_core import importance_lomax, lomax_inverse


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = dict(u=2., lam=1., horizon=1., premium=10., alpha=.95, beta=10.)
PAIRED_SAMPLES = 4_000_000
WEIGHTED_SAMPLES = 150_000
PAIRED_SEED = 2026111701
WEIGHTED_SEED = 2026111702


def paired_ruin(times, claims, u, premium):
    """Return classical and equalized ruin flags for sorted arrival times."""
    available = u + premium * times
    totals = claims.cumsum(axis=1)
    count = claims.shape[1]
    classical = np.any(totals > available, axis=1)
    equalized = totals[:, [-1]] / count > available / np.arange(1, count + 1)
    return classical, np.any(equalized, axis=1)


def ordinary_paired(samples, seed, u, lam, horizon, premium, alpha, beta):
    rng = np.random.default_rng(seed)
    cap = u + premium * horizon + 1
    class_only = equal_only = 0
    for start in range(0, samples, 100_000):
        counts = rng.poisson(lam * horizon, min(100_000, samples - start))
        for count, size in enumerate(np.bincount(counts)):
            # Both indicators coincide on every path with zero or one claim.
            if count < 2 or size == 0:
                continue
            shape = (size, count)
            times = np.sort(rng.random(shape), axis=1) * horizon
            claims = lomax_inverse(rng.random(shape), alpha, beta, cap)
            classical, equalized = paired_ruin(times, claims, u, premium)
            class_only += int(np.count_nonzero(classical & ~equalized))
            equal_only += int(np.count_nonzero(equalized & ~classical))
    net = class_only - equal_only
    mean = net / samples
    variance = (class_only + equal_only - net * mean) / (samples - 1)
    return dict(samples=samples, seed=seed, mean=mean,
                standard_error=math.sqrt(variance / samples),
                classical_only=class_only, equalized_only=equal_only)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'paired_benchmark.json')
    args = parser.parse_args()
    start = perf_counter()
    paired = ordinary_paired(PAIRED_SAMPLES, PAIRED_SEED, **PARAMETERS)
    paired['elapsed_seconds'] = perf_counter() - start
    start = perf_counter()
    weighted = importance_lomax(WEIGHTED_SAMPLES, WEIGHTED_SEED, **PARAMETERS)
    weighted['elapsed_seconds'] = perf_counter() - start
    for run in (paired, weighted):
        run['normal_95_interval'] = [
            run['mean'] + sign * 1.96 * run['standard_error'] for sign in (-1, 1)
        ]
    se_paired, se_weighted = [r['standard_error'] for r in (paired, weighted)]
    var_paired = se_paired**2 * paired['samples']
    var_weighted = se_weighted**2 * weighted['samples']
    variance_ratio = var_paired / var_weighted
    time_ratio = (se_paired**2 * paired['elapsed_seconds']
                  / (se_weighted**2 * weighted['elapsed_seconds']))
    output = {
        'parameters': PARAMETERS,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'machine': platform.machine(), 'platform': platform.platform()},
        'ordinary_paired': paired,
        'conditional_importance': weighted,
        'se_of_difference': math.hypot(se_paired, se_weighted),
        'estimated_variance_ratio_per_observation': variance_ratio,
        'estimated_time_ratio_at_equal_precision': time_ratio,
    }
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
