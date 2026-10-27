# Numerical audit record

Audit date: 27 October 2026. This record describes the implemented checks, not a guarantee of error-free software.

## Changes made after critical review

The first working draft used binary floating-point conditional volumes. Its reported values were reproducible, but its public helper functions lacked complete parameter and boundary checks and subtracting nearly equal volumes could lose precision. The revised implementation:

- validates nonnegative finite claims, scalar parameters, sample counts, precision and resource bounds;
- handles zero and one claim, zero reserve, equal marks, terminal ruin, positive-minimum zero gaps and empty batches explicitly;
- evaluates conditional volumes with Decimal arithmetic, normally at 80 digits, with extra precision based on the reserve-to-premium ratio;
- uses the positive two-claim kernel and a symmetric implementation for either order of a claim pair;
- resolves affordable ambiguous near-zero differences with exact rational arithmetic, otherwise raising an error;
- treats large sampled amounts by a mathematically invariant terminal-ruin cap and uses overflow-safe inverse transforms;
- checks the forced small claim, finite importance weights, nonnegative conditional gaps and existence of a usable variance estimate;
- bounds exponential computational work, aborting rather than dropping high-count samples;
- avoids an arbitrarily slow rejection sampler for very small Poisson means;
- keeps verification checks active under Python optimization and avoids simulations or file writes on importing verification drivers.

The largest change in the ten retained estimates after replacing binary volume subtraction was below 1.7e-7 of a Monte Carlo standard error. The manuscript's displayed numerical results are unchanged at their stated precision. Details are in data/revision_comparison.json.

## Independent checks

The standard unittest suite contains 17 test methods with many parameterized subcases. It checks all ordered pairs from an eight-point claim grid at three reserves against high-precision order-statistic determinants; randomized vectors for claim counts 3–5 against the determinant averaged over every permutation; zero-reserve ballot identities through count 14; exact zero regions; permutation and monetary-scale invariance; reserves down to 1e-200 for a conditional kernel; and agreement at 80 and 120 digits for randomized vectors.

The importance weight is checked by complete discrete enumeration, separately from Monte Carlo. Proposal counts, sample accounting, computational chunk invariance, capped transforms, extreme tails, arithmetic interval primitives and invalid inputs are tested. Exact binary fractions define equality-boundary tests, so decimal-looking inputs are not incorrectly assumed to sum exactly in binary.

Both normal execution and python -O retain the checks. The full verification driver rechecks the analytic benchmarks and all ten saved Monte Carlo runs. No run or seed was discarded.

## Outward-rounded sign enclosures

The six analytic bounds isolate the two-claim contribution and bound every remaining Poisson count. Basic arithmetic uses directed rounding; Decimal.exp and Decimal.ln, which Python documents as correctly rounded, are widened outwards by one adjacent representable value. Noninteger powers use enclosed exp/ln composition. The omitted count tail is included explicitly. Exact decimal parameter strings are used.

At 80 digits the widths of the evaluated bound expressions are below 2e-80. All six lower endpoints are positive, also at 40 digits. This conclusion uses the analytic inequalities and Decimal's documented arithmetic contract. It does not use the general equalization comparison theorem.

## Limits

The exact-arithmetic importance-sampling theorem is separate from its implementation. A finite pseudorandom generator and floating-point inverse transforms do not implement exact continuous random variables. Normal Monte Carlo intervals measure sampling error, not a complete arithmetic error budget. The auxiliary FFT calculation of the local coefficient does not enclose FFT roundoff. Python, NumPy, Decimal and the operating system are not formally verified by these tests.

The sample-count guard must not be turned into a truncated-law estimator or a reason to select seeds. If a guard is reached, keep the same seed when increasing an admissible computational limit. This is a small/moderate-count implementation, not a uniformly fast solver for every parameter set.
