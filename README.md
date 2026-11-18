# Finite-time ruin with equalized claims

Reproducible calculations by Mathias Valla for *Revisiting reported counterexamples to the De Vylder–Goovaerts conjecture*.

The quantity is Delta = P(classical ruin) - P(equalized ruin), for a fixed horizon, premium rate and initial reserve. The six parameter sets come from [Loisel and Minier (2023)](https://doi.org/10.3390/math11061501). The general comparison is proved by [Kim, Kim and Kim (2026)](https://doi.org/10.1016/j.insmatheco.2025.103205).

This repository contains analytic sign enclosures for all six examples, conditional importance sampling, the small-reserve asymptotic checks, and a comparison with ordinary paired Monte Carlo. It includes saved records and scripts for all three manuscript figures and tables. These are independent calculations for the model inputs in the paper.

## Reproduce

The recorded environment uses Python 3.10.18, NumPy 1.26.4, mpmath 1.3.0 and Matplotlib 3.10.0. The pinned dependencies also support Python 3.12. No CUDA device is required.

```sh
python -m pip install -r requirements.txt
python code/test_numerics.py
python -O code/test_numerics.py
python code/verify_bundle.py
```

The first two test commands include independent determinant comparisons, randomized claim vectors, exact two-claim cases, boundary and zero-count cases, extreme tails, very small reserves, proposal and weight checks, and failure modes. The verification driver checks the ten retained Monte Carlo runs and both runs in the paired benchmark against the archived records.

To regenerate the manuscript tables and figures from the saved records:

```sh
python code/write_manuscript_tables.py
python code/plot_principal_decay.py
python code/plot_revision_figures.py
```

The plot scripts write PDF, SVG and 600 dpi PNG under `figures/`. Figure 1 draws a worked path with two claims. Figure 2 reads three independent weighted runs and the small-reserve coefficient from `data/checks.json`. Its lower panel shows the estimates and 95% sampling intervals as percentage differences from the quadratic prediction. Figure 3 reads the two gap estimates and their normal 95% intervals from `data/paired_benchmark.json`.

To rerun the paired benchmark and write a separate record, including fresh timings:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python code/paired_benchmark.py --output /tmp/paired_benchmark_reproduced.json
```

To regenerate the six example records, asymptotic checks and auxiliary lower-tail plot:

```sh
python code/regenerate_results.py
python code/make_figures.py
```

To evaluate the six outward-rounded analytic enclosures alone:

```sh
python code/certify_bounds.py --precision 80 --cutoff 60
```

## What is checked

- `risk_core.py`: validated inputs, finite terminal-ruin cap, overflow-safe Lomax transforms, conditional volumes normally at 80 Decimal digits, and explicit resource guards. Larger precision is used automatically for very small reserve-to-premium ratios. Exact rational arithmetic resolves affordable ambiguous gaps.
- `certify_bounds.py`: directed arithmetic intervals for the explicit analytic bounds, including the count tail. Correctly rounded Decimal logarithms and exponentials are expanded to adjacent values outwards. Parameters enter as exact decimal strings.
- `test_numerics.py`: independent mathematical and failure-mode tests using standard `unittest`.
- `verify_local_gap.py`: determinant, quadrature, atom and convolution benchmarks.
- `paired_benchmark.py`: two independent estimators using the same model parameters. An ordinary run pairs both models on each simulated path. The weighted run conditions on claim sizes and samples small claims directly.
- `data/`: fixed seeds, sample counts, estimates, sampling errors, bound endpoints, arithmetic precision and serial timings.
- `figures/`: the three manuscript figures as vector SVGs, plus the auxiliary `lower_tail_orders.svg` plot. The original calculation snapshot is [commit `be9185e`](https://github.com/MathiasValla/equalized-claims-ruin/tree/be9185e6dbc9ee31acaa52cff9431f653c1e7f75).

## Scope and limits

An outward-rounded analytic enclosure and a normal Monte Carlo interval are different objects. The former includes arithmetic rounding and the explicit count-tail remainder, subject to Python Decimal's documented arithmetic contract and the underlying analytic inequalities. The latter reports sampling error only. The auxiliary FFT coefficient check has no formal roundoff enclosure.

The theoretical importance sampler is unbiased in exact arithmetic. Finite random-number generators and inverse transforms are numerical approximations; the entire software stack is not formally verified. Invalid or numerically unsupported inputs cause errors. The count resource limit aborts a run instead of truncating its law: retain the same seed if raising a limit, rather than discarding unsuccessful runs. The subset recursion is intended for small and moderate claim counts.

The former float-volume draft and this version agree at the displayed precision: the largest change in the ten retained estimates was less than 1.7e-7 of a Monte Carlo standard error. See `AUDIT.md` for the detailed check record.
