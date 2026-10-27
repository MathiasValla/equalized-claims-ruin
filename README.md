# Finite-time ruin with equalized claims

Reproducible calculations by Mathias Valla for *Revisiting reported counterexamples to the De Vylder–Goovaerts conjecture* (manuscript in preparation).

The quantity is Delta = P(classical ruin) - P(equalized ruin), for a fixed horizon, premium rate and initial reserve. The six parameter sets come from [Loisel and Minier (2023)](https://doi.org/10.3390/math11061501). The general comparison is proved by [Kim, Kim and Kim (2026)](https://doi.org/10.1016/j.insmatheco.2025.103205).

This repository contains independent analytic sign enclosures, conditional importance sampling, the local asymptotic checks, saved records, and figure-generation code. It does not reconstruct the historical CUDA runs. The manuscript and editorial correspondence are not included here.

## Reproduce

The recorded environment uses Python 3.10.18, NumPy 1.26.4, mpmath 1.3.0 and Matplotlib 3.10.0. The pinned dependencies also support Python 3.12. No CUDA device is required.

```sh
python -m pip install -r requirements.txt
python code/test_numerics.py
python -O code/test_numerics.py
python code/verify_bundle.py
```

The first two test commands include independent determinant comparisons, randomized claim vectors, exact two-claim cases, boundary and zero-count cases, extreme tails, very small reserves, proposal and weight checks, and failure modes. The full verification driver regenerates the ten retained Monte Carlo runs without overwriting them.

To regenerate the calculation records and plots:

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
- `data/`: fixed seeds, sample counts, estimates, sampling errors, bound endpoints, arithmetic precision and descriptive serial timings.
- `figures/`: two vector SVG figures; the script also exports PDF and 600 dpi PNG.

## Scope and limits

An outward-rounded analytic enclosure and a normal Monte Carlo interval are different objects. The former includes arithmetic rounding and the explicit count-tail remainder, subject to Python Decimal's documented arithmetic contract and the underlying analytic inequalities. The latter reports sampling error only. The auxiliary FFT coefficient check has no formal roundoff enclosure.

The theoretical importance sampler is unbiased in exact arithmetic. Finite random-number generators and inverse transforms are numerical approximations; the entire software stack is not formally verified. Invalid or numerically unsupported inputs cause errors. The count resource limit aborts a run instead of truncating its law: retain the same seed if raising a limit, rather than discarding unsuccessful runs. The subset recursion is intended for small and moderate claim counts.

The former float-volume draft and this version agree at the displayed precision: the largest change in the ten retained estimates was less than 1.7e-7 of a Monte Carlo standard error. See `AUDIT.md` for the detailed check record.
