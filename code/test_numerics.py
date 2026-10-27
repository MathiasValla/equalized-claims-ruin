"""Independent mathematical and failure-mode tests (standard unittest)."""
import itertools
import math
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
import mpmath as mp
import numpy as np

from risk_core import (fixed_vector_gap, permutation_survival, equalized_survival,
                       importance_lomax, lomax_cdf, lomax_inverse, lomax_zero_reserve)
from verify_local_gap import independent_determinant_survival
from reproduce_bounds import CASES, calculate
from certify_bounds import Interval, certify


def determinant_gap(xs,u,C):
    xs,u,C = [mp.mpf(float(x)) for x in xs],mp.mpf(float(u)),mp.mpf(float(C))
    n = len(xs)
    if n<2:
        return mp.mpf(0)
    eq = independent_determinant_survival([sum(xs)/n]*n,u,C)
    cl = mp.fsum(independent_determinant_survival(list(p),u,C)
                 for p in itertools.permutations(xs))/math.factorial(n)
    return eq-cl


class MathematicalChecks(unittest.TestCase):
    def setUp(self):
        mp.mp.dps = 120

    def test_empty_and_single_claim_cases(self):
        np.testing.assert_array_equal(fixed_vector_gap([[],[]],0,1),[0,0])
        np.testing.assert_array_equal(permutation_survival([[],[]],0,1),[1,1])
        np.testing.assert_array_equal(fixed_vector_gap([[0],[.2],[2]],.1,1),[0,0,0])
        np.testing.assert_allclose(permutation_survival([[0],[.2],[2]],.1,1),[1,.9,0],atol=1e-15)
        np.testing.assert_array_equal(equalized_survival([0,0],0,0,1),[1,1])
        self.assertEqual(fixed_vector_gap(np.zeros((0,3)),.1,1).shape,(0,))

    def test_exact_zero_regions_and_terminal_boundary(self):
        rows = [[0,0,0],[.2,.2,.2],[.25,.3,.4],[.03125,.5,.59375],[.5,.5,.5]]
        np.testing.assert_array_equal(fixed_vector_gap(rows,.125,1),np.zeros(5))
        np.testing.assert_array_equal(fixed_vector_gap([[0,.1,.2],[.01,.02,.03]],1,1),[0,0])
        self.assertEqual(fixed_vector_gap([[0,.5,.625]],.125,1)[0],0)

    def test_all_two_claim_orders_against_determinant(self):
        for u in [.125,.75,2.]:
            for pair in itertools.product([0,.01,.125,.2,.4,.8,1.2,2.1],repeat=2):
                with self.subTest(u=u,pair=pair):
                    g = fixed_vector_gap([pair],u,1)[0]
                    reference = determinant_gap(pair,u,1)
                    self.assertLess(abs(mp.mpf(float(g))-reference),mp.mpf('5e-16'))

    def test_random_vectors_against_permutation_determinants(self):
        rng = np.random.default_rng(2026102701)
        for n in range(3,6):
            for j in range(16):
                u = [.001,.03,.25,.75][j%4]
                row = rng.dirichlet(np.ones(n))*(1+u)*rng.uniform(.2,.95)
                row[0] = u*rng.uniform(.05,.95)
                with self.subTest(n=n,j=j):
                    g = fixed_vector_gap([row],u,1)[0]
                    reference = determinant_gap(row,u,1)
                    self.assertLess(abs(mp.mpf(float(g))-reference),mp.mpf('5e-16'))
                    g120 = fixed_vector_gap([row],u,1,precision=120)[0]
                    self.assertEqual(g,g120)

    def test_permutation_and_scale_invariance(self):
        row = [.002,.04,.13,.22,.35]
        g = fixed_vector_gap([row],.03,1)[0]
        rng = np.random.default_rng(1749)
        for _ in range(20):
            self.assertEqual(fixed_vector_gap([rng.permutation(row)],.03,1)[0],g)
        for scale in [2**-20,2**-3,8,2**20]:
            self.assertEqual(fixed_vector_gap([np.array(row)*scale],.03*scale,scale)[0],g)

    def test_higher_counts_against_zero_reserve_ballot_identity(self):
        for n in [6,8,10,12,14]:
            row=np.arange(1,n+1,dtype=float)/(n*(n+1))
            reference=1-mp.fsum(mp.mpf(float(x)) for x in row)
            observed=permutation_survival([row],0,1)[0]
            self.assertLess(abs(mp.mpf(float(observed))-reference),mp.mpf('1e-15'))
            self.assertAlmostEqual(equalized_survival([.75],n,0,1)[0],.25,places=15)

    def test_very_small_reserves(self):
        mp.mp.dps = 350
        for u in [1e-8,1e-20,1e-60,1e-200]:
            row = [u/2,.2,.3]
            g = fixed_vector_gap([row],u,1)[0]
            r = determinant_gap(row,u,1)
            self.assertGreater(g,0)
            self.assertLess(abs(mp.mpf(float(g))/r-1),mp.mpf('1e-14'))

    def test_importance_weight_by_exact_discrete_enumeration(self):
        marks = [0.,.125,.5]
        probabilities = [.2,.3,.5]
        u,fu = .2,.5
        for n in range(2,5):
            original = forced = 0.
            for indices in itertools.product(range(3),repeat=n):
                xs = [marks[i] for i in indices]
                p = math.prod(probabilities[i] for i in indices)
                g = fixed_vector_gap([xs],u,1)[0]
                original += p*g
                if xs[0] <= u:
                    J = sum(x<=u for x in xs)
                    forced += p/fu*g/J
            self.assertAlmostEqual(n*fu*forced,original,places=16)

    def test_sampling_chunk_invariance(self):
        a = importance_lomax(600,987,.01,1,.1,10,1.1,.2,chunk_size=17)
        b = importance_lomax(600,987,.01,1,.1,10,1.1,.2,chunk_size=8192)
        for field in ['mean','standard_error','count_histogram']:
            self.assertEqual(a[field],b[field])

    def test_zero_truncated_count_law(self):
        for m,seed in [(.01,91),(1.,92)]:
            r = importance_lomax(12000,seed,.02,1,m,1/m,1,.2)
            expected = m*math.exp(-m)/(1-math.exp(-m))
            actual = r['count_histogram'].get(2,0)/12000
            sd = math.sqrt(expected*(1-expected)/12000)
            self.assertLess(abs(actual-expected),6*sd)
            self.assertEqual(sum(r['count_histogram'].values()),12000)

    def test_extreme_inverse_transform(self):
        np.testing.assert_array_equal(lomax_inverse([0.,.5],1,1e300,1e-300),[0.,1e-300])
        for alpha,beta,cap in [(.01,.01,100001.),(.001,1e-300,1e200),(1,1e300,1e308)]:
            z = np.array([0.,.01,.5,.6321205588285577,np.nextafter(1.,0.)])
            out = lomax_inverse(z,alpha,beta,cap)
            for a,b in zip(z,out):
                true = min(mp.mpf(cap),mp.mpf(beta)*(mp.expm1(-mp.log1p(-mp.mpf(float(a)))/mp.mpf(alpha))))
                if true:
                    self.assertLess(abs(mp.mpf(float(b))/true-1),mp.mpf('5e-12'))
                else:
                    self.assertEqual(b,0)
        self.assertTrue(np.isfinite(lomax_cdf(1e200,1e-4,1e-300)))
        reference = -mp.expm1(-mp.mpf(1e-4)*mp.log1p(mp.mpf(1e200)/mp.mpf(1e-300)))
        self.assertLess(abs(mp.mpf(float(lomax_cdf(1e200,1e-4,1e-300)))-reference),mp.mpf('1e-15'))

    def test_directed_interval_primitives_and_transcendentals(self):
        with localcontext() as ctx:
            ctx.prec = 18
            a,b = Interval.exact('1.23456789'),Interval.exact('.000000123456789')
            operations = [(a+b,Fraction(Decimal('1.23456789'))+Fraction(Decimal('.000000123456789'))),
                          (a/b,Fraction(10_000_000)),
                          (a.exp(),mp.exp(mp.mpf('1.23456789'))),
                          (b.ln(),mp.log(mp.mpf('.000000123456789')))]
            for interval,true in operations:
                if isinstance(true,Fraction):
                    self.assertLessEqual(Fraction(interval.lo),true)
                    self.assertGreaterEqual(Fraction(interval.hi),true)
                else:
                    self.assertLessEqual(mp.mpf(str(interval.lo)),true)
                    self.assertGreaterEqual(mp.mpf(str(interval.hi)),true)

    def test_certified_bounds_at_two_precisions(self):
        for case in CASES:
            a,b = certify(case,40,60),certify(case,80,60)
            self.assertGreater(Decimal(a['delta_lower']),0)
            self.assertLessEqual(Decimal(a['delta_lower']),Decimal(b['delta_lower']))
            self.assertGreaterEqual(Decimal(a['delta_upper']),Decimal(b['delta_upper']))
            approximate = calculate(case,80,60)
            self.assertLess(abs(Decimal(approximate['delta_lower'])-Decimal(b['delta_lower'])),Decimal('1e-65'))


class FailureModeChecks(unittest.TestCase):
    def test_invalid_volume_arguments(self):
        for row in [[[math.nan]],[[math.inf]],[[-.1]],[[1+2j]],np.zeros((2,2,2))]:
            with self.subTest(row=repr(row)),self.assertRaises(ValueError):
                fixed_vector_gap(row,.1,1)
        for u,C in [(-1,1),(math.nan,1),(.1,0),(.1,math.inf)]:
            with self.assertRaises(ValueError):
                fixed_vector_gap([[.2,.3]],u,C)
        with self.assertRaises(ValueError):
            equalized_survival([1],0,0,1)
        with self.assertRaises(ValueError):
            fixed_vector_gap([[.01]*15],.1,1)

    def test_invalid_sampler_arguments_and_resource_limit(self):
        base = dict(samples=10,seed=1,u=.01,lam=1,horizon=1,premium=1,alpha=1,beta=.2)
        for key,value in [('samples',1),('samples',0),('samples',2.5),('seed',-1),('u',0),
                          ('u',math.nan),('lam',0),('horizon',-1),('premium',math.inf),
                          ('alpha',0),('beta',-1),('precision',20),('chunk_size',0)]:
            args = dict(base);args[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                importance_lomax(**args)
        with self.assertRaises(ValueError):
            importance_lomax(1000,1,.01,1,1,1,1,.2,max_count=2)

    def test_invalid_inverse_and_fft_parameters(self):
        for z in [[-1.],[1.],[math.nan],[math.inf],[1+1j]]:
            with self.assertRaises(ValueError):
                lomax_inverse(z,1,1,2)
        for args in [dict(grid=1),dict(grid=3.5),dict(cutoff=-1),dict(cutoff=0,mean_count=3)]:
            params=dict(alpha=1,beta=1,capital=1,mean_count=1);params.update(args)
            with self.assertRaises(ValueError):
                lomax_zero_reserve(**params)
        z = lomax_zero_reserve(1,1,1,0,grid=16)
        self.assertEqual(z['upper_survival'],1)
        self.assertEqual(z['lower_survival'],1)

    def test_invalid_analytic_bound_parameters(self):
        with self.assertRaises(ValueError):
            calculate(CASES[0],precision=2)
        with self.assertRaises(ValueError):
            calculate(CASES[4],cutoff=1)
        with self.assertRaises(ArithmeticError):
            _ = Interval.exact(1)/Interval(Decimal(-1),Decimal(1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
