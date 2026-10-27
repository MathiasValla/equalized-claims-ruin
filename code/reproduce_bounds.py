#!/usr/bin/env python3
"""Independent bounds for every numerical example in Loisel & Minier (2023).

Uses only Python's standard library. No Monte Carlo, CUDA, or assumption that
the general De Vylder--Goovaerts conjecture is true. See manuscript Appendix A.

    python3 reproduce_bounds.py
    python3 reproduce_bounds.py --precision 80 --self-check

The inequalities are analytic; Decimal evaluates their closed-form endpoints.
This is high-precision evaluation, not a formal interval-arithmetic certificate.
"""

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, localcontext


@dataclass(frozen=True)
class Case:
    label: str
    u: str
    lam: str
    horizon: str
    premium: str
    alpha: str
    beta: str


CASES = (
    Case("Main example (Table 1)", ".001", "1", ".1", "10000", ".95", "10"),
    Case("Alternative 1", ".001", "1", "1", "100000", ".01", ".01"),
    Case("Alternative 2", ".01", "1", "1", "100000", ".01", ".1"),
    Case("Alternative 3", ".001", "1", "1", "10000", "1", "10"),
    Case("Alternative 4", ".1", "1", "1", "10000", ".1", "10"),
    Case("Positive safety loading", ".001", "1", "1", "10000", "1.0011", "10"),
)


def calculate(case, precision=70, cutoff=60):
    if isinstance(precision,bool) or not isinstance(precision,int) or precision < 35:
        raise ValueError('precision must be an integer >=35')
    if isinstance(cutoff,bool) or not isinstance(cutoff,int) or cutoff < 2:
        raise ValueError('cutoff must be an integer >=2')
    with localcontext() as ctx:
        ctx.prec = precision
        D = Decimal
        one, two = D(1), D(2)
        u, lam, horizon, premium, alpha, beta = map(
            D, (case.u, case.lam, case.horizon, case.premium, case.alpha, case.beta)
        )
        if any(not x.is_finite() or x<=0 for x in (u,lam,horizon,premium,alpha,beta)):
            raise ValueError('Model parameters must be finite and positive')
        capital = premium * horizon
        mean_count = lam * horizon
        v = u / capital
        length = one + v
        if two*u >= capital:
            raise ValueError('The two-claim bound requires 2u<cT')
        if mean_count*length >= D(cutoff+2):
            raise ValueError('The cutoff does not support the geometric tail bound')

        def survival(x):
            return (one + x / beta) ** (-alpha)

        def cdf(x):
            return one - survival(x)

        def moment1(lo, hi):
            """Integral_lo^hi x*f(x) dx, including the alpha=1 limit."""
            if alpha == one:
                integral_survival = beta * ((hi + beta) / (lo + beta)).ln()
            else:
                power = one - alpha
                integral_survival = (
                    beta ** alpha
                    * ((hi + beta) ** power - (lo + beta) ** power)
                    / power
                )
            return lo * survival(lo) - hi * survival(hi) + integral_survival

        # Exact analytic lower/upper bounds on the N=2 contribution.
        shortfall = u * cdf(u) - moment1(D(0), u)
        if shortfall <= 0:
            raise ArithmeticError('Insufficient precision for the claim shortfall')
        b_lower = cdf(capital) - cdf(two * u) - moment1(two * u, capital) / capital
        b_upper = (
            length * (cdf(capital + u) - cdf(u))
            - moment1(u, capital + u) / capital
        )
        p0 = (-mean_count).exp()
        p2 = p0 * mean_count ** 2 / two
        n2_lower = p2 * two / capital * shortfall * b_lower
        n2_upper = p2 * two / capital * shortfall * b_upper

        # For n>=3: -(L^n-1)F(nu)^n <= Delta_n <= (L^n-1)F(u).
        negative_tail_bound, positive_tail_bound = D(0), D(0)
        probability = p2
        for n in range(3, cutoff + 1):
            probability *= mean_count / D(n)
            time_factor = length ** n - one
            negative_tail_bound += probability * time_factor * cdf(D(n) * u) ** n
            positive_tail_bound += probability * time_factor * cdf(u)

        # Bound sum_{n>K} P(N=n)L^n by a geometric series. Ratios decrease.
        first_omitted = probability * mean_count / D(cutoff + 1) * length ** (cutoff + 1)
        remainder = first_omitted / (one - mean_count * length / D(cutoff + 2))
        lower = n2_lower - negative_tail_bound - remainder
        upper = n2_upper + positive_tail_bound + remainder
        return {
            "label": case.label,
            "parameters": {
                "u": case.u, "lambda": case.lam, "T": case.horizon,
                "c": case.premium, "alpha": case.alpha, "beta": case.beta,
            },
            "precision": precision,
            "cutoff": cutoff,
            "F_u": str(cdf(u)),
            "weighted_n2_lower": str(n2_lower),
            "weighted_n2_upper": str(n2_upper),
            "negative_tail_bound": str(negative_tail_bound),
            "positive_tail_bound": str(positive_tail_bound),
            "omitted_tail_bound": str(remainder),
            "delta_lower": str(lower),
            "delta_upper": str(upper),
            "positive": lower > 0,
        }


def self_check():
    for case in CASES:
        a, b = calculate(case, 50, 60), calculate(case, 80, 80)
        for key in ("delta_lower", "delta_upper"):
            x, y = Decimal(a[key]), Decimal(b[key])
            if not (abs(x - y) < abs(y) * Decimal('1e-25')):
                raise AssertionError((case.label, key))
        if not (a['positive'] and b['positive']):
            raise AssertionError('Numerical verification failed')
    print("Self-check passed: all six signs positive; endpoints stable at 50/80 digits and cutoffs 60/80.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=70)
    parser.add_argument("--cutoff", type=int, default=60)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print full endpoint evaluations as JSON")
    args = parser.parse_args()
    if args.precision < 35 or args.cutoff < 30:
        parser.error("Use precision >=35 and cutoff >=30.")
    if args.self_check:
        self_check()
    results = [calculate(case, args.precision, args.cutoff) for case in CASES]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("Delta = P(classical ruin) - P(equalized ruin)")
        for r in results:
            print(f"{r['label']}: [{Decimal(r['delta_lower']):.12E}, {Decimal(r['delta_upper']):.12E}]")
        print("All lower endpoints are strictly positive:", all(r["positive"] for r in results))


if __name__ == "__main__":
    main()
