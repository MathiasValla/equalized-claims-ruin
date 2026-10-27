#!/usr/bin/env python3
"""Outward-rounded evaluation of the manuscript's six analytic enclosures.

Basic Decimal arithmetic is rounded down/up. Decimal exp and ln are
correctly rounded to nearest according to Python's documented contract;
one adjacent representable number is added outwards at each endpoint.
Noninteger powers are composed from these enclosed exp/ln operations.
This establishes numerical enclosures subject to that arithmetic contract,
not an independent formal verification of Python or of the analytic proof.
"""
import argparse
from dataclasses import dataclass
from decimal import Decimal as D, localcontext, ROUND_FLOOR, ROUND_CEILING
import json
import math
from pathlib import Path

from reproduce_bounds import CASES


def rounded(operation, mode):
    with localcontext() as ctx:
        ctx.rounding = mode
        result = operation()
    if not result.is_finite():
        raise ArithmeticError('Nonfinite interval endpoint')
    return result


@dataclass(frozen=True)
class Interval:
    lo: D
    hi: D

    def __post_init__(self):
        if not self.lo.is_finite() or not self.hi.is_finite() or self.lo > self.hi:
            raise ValueError('Invalid finite interval')

    @staticmethod
    def exact(x):
        x = D(x)
        return Interval(x,x)

    def __add__(self, other):
        b = box(other)
        return Interval(rounded(lambda:self.lo+b.lo,ROUND_FLOOR),
                        rounded(lambda:self.hi+b.hi,ROUND_CEILING))
    __radd__ = __add__

    def __neg__(self):
        return Interval(self.hi.copy_negate(),self.lo.copy_negate())

    def __sub__(self, other):
        return self+-box(other)

    def __rsub__(self, other):
        return box(other)+-self

    def __mul__(self, other):
        b = box(other)
        lows = [rounded(lambda:x*y,ROUND_FLOOR) for x in (self.lo,self.hi) for y in (b.lo,b.hi)]
        highs = [rounded(lambda:x*y,ROUND_CEILING) for x in (self.lo,self.hi) for y in (b.lo,b.hi)]
        return Interval(min(lows),max(highs))
    __rmul__ = __mul__

    def __truediv__(self, other):
        b = box(other)
        if b.lo <= 0 <= b.hi:
            raise ArithmeticError('Interval division contains zero')
        lows = [rounded(lambda:x/y,ROUND_FLOOR) for x in (self.lo,self.hi) for y in (b.lo,b.hi)]
        highs = [rounded(lambda:x/y,ROUND_CEILING) for x in (self.lo,self.hi) for y in (b.lo,b.hi)]
        return Interval(min(lows),max(highs))

    def __rtruediv__(self, other):
        return box(other)/self

    def __pow__(self, n):
        if isinstance(n,bool) or not isinstance(n,int) or n < 0:
            raise ValueError('Interval power requires a nonnegative integer')
        result,base = box(1),self
        while n:
            if n & 1:
                result = result*base
            n >>= 1
            if n:
                base = base*base
        return result

    def exp(self):
        lo,hi = self.lo.exp(),self.hi.exp()
        return Interval(lo.next_minus(),hi.next_plus())

    def ln(self):
        if self.lo <= 0:
            raise ValueError('Logarithm requires a positive interval')
        return Interval(self.lo.ln().next_minus(),self.hi.ln().next_plus())


def box(x):
    return x if isinstance(x,Interval) else Interval.exact(x)


def certify(case, precision=80, cutoff=60):
    if isinstance(precision,bool) or not isinstance(precision,int) or not 40<=precision<=1000:
        raise ValueError('precision must be an integer between 40 and 1000')
    if isinstance(cutoff,bool) or not isinstance(cutoff,int) or not 2<=cutoff<=10000:
        raise ValueError('cutoff must be an integer between 2 and 10000')
    with localcontext() as ctx:
        ctx.prec = precision
        raw = [D(x) for x in (case.u,case.lam,case.horizon,case.premium,case.alpha,case.beta)]
        if any(not x.is_finite() or x<=0 for x in raw):
            raise ValueError('Parameters must be finite and positive')
        u,lam,T,c,alpha,beta = map(box,raw)
        C,m = c*T,lam*T
        if (2*u).hi >= C.lo:
            raise ValueError('The two-claim enclosure requires 2u<C')
        L = 1+u/C
        if (m*L/(cutoff+2)).hi >= 1:
            raise ValueError('Count cutoff is too small for the geometric bound')
        def sf(x):
            return (-alpha*(1+box(x)/beta).ln()).exp()
        def F(x):
            return 1-sf(x)
        def moment(a,b):
            a,b = box(a),box(b)
            if raw[4] == 1:
                integral = beta*((b+beta)/(a+beta)).ln()
            else:
                power = 1-alpha
                integral = (alpha*beta.ln()).exp()*((power*(b+beta).ln()).exp()
                            -(power*(a+beta).ln()).exp())/power
            return a*sf(a)-b*sf(b)+integral
        A = u*F(u)-moment(0,u)
        bm = F(C)-F(2*u)-moment(2*u,C)/C
        bp = L*(F(C+u)-F(u))-moment(u,C+u)/C
        p0 = (-m).exp()
        p2 = p0*m**2/2
        low,high = p2*2*A*bm/C,p2*2*A*bp/C
        probability = p2
        for n in range(3,cutoff+1):
            probability = probability*m/n
            factor = L**n-1
            low = low-probability*factor*F(n*u)**n
            high = high+probability*factor*F(u)
        remainder = p0*(m*L)**(cutoff+1)/math.factorial(cutoff+1)/(1-m*L/(cutoff+2))
        low,high = low-remainder,high+remainder
        if low.lo <= 0:
            raise ArithmeticError('This evaluation does not certify a positive gap')
        return dict(label=case.label,precision=precision,cutoff=cutoff,
                    delta_lower=str(low.lo),delta_upper=str(high.hi),
                    lower_expression_width=str(rounded(lambda:low.hi-low.lo,ROUND_CEILING)),
                    upper_expression_width=str(rounded(lambda:high.hi-high.lo,ROUND_CEILING)),
                    omitted_tail_upper=str(remainder.hi),positive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision',type=int,default=80)
    parser.add_argument('--cutoff',type=int,default=60)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    records = [certify(case,args.precision,args.cutoff) for case in CASES]
    content = json.dumps(records,indent=2)+'\n'
    if args.output:
        args.output.write_text(content)
    print(content)


if __name__ == '__main__':
    main()
