"""Guarded numerical kernels for ruin with equalized claims.

Inputs are binary64 numbers. Conditional volumes use Decimal arithmetic,
normally at 80 digits. The theoretical estimator is unbiased in exact
arithmetic; this implementation does not claim exact continuous randomness.
Resource limits abort a run rather than truncate its count distribution.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import numbers
import numpy as np

MAX_COUNT = 14
MAX_SAMPLES = 10_000_000


def positive(value, name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value)!=0:
        raise ValueError(f'{name} must be a finite real number')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite real number') from exc
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
    return value


def integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside its supported range')
    return value


def _matrix(xs, max_count=MAX_COUNT):
    if np.iscomplexobj(xs):
        raise ValueError('Claims must be real and nonnegative')
    try:
        xs = np.asarray(xs, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Claims must form a finite nonnegative numeric matrix') from exc
    if xs.ndim == 1:
        xs = xs.reshape(1, -1)
    if xs.ndim != 2 or not np.isfinite(xs).all() or (xs < 0).any():
        raise ValueError('Claims must form a finite nonnegative matrix')
    if xs.shape[1] > max_count:
        raise ValueError('Claim count exceeds the explicit computational limit; no counts were discarded')
    return xs


def _parameters(u, capital, precision):
    u = positive(u, 'u', allow_zero=True)
    capital = positive(capital, 'capital')
    precision = integer(precision, 'precision', 40, 1000)
    # Preserve at least 30 working digits after the small-reserve subtraction.
    loss = max(0, math.ceil((math.log(capital)-math.log(u))/math.log(10))) if u else 0
    return u, capital, max(precision, loss + 40)


def _subset_survival(xs, u, capital):
    zero, one = u-u, u-u+1
    n = len(xs)
    sums, values = [zero]*(1 << n), [one]
    for mask in range(1, 1 << n):
        bit = mask & -mask
        sums[mask] = sums[mask ^ bit] + xs[bit.bit_length()-1]
        b = max((sums[mask]-u)/capital, zero)
        size = mask.bit_count()
        value = -(b**size)
        sub = (mask-1) & mask
        while sub:
            value -= values[sub]*b**(size-sub.bit_count())
            sub = (sub-1) & mask
        values.append(value)
    return sum(values, zero)


def _equal_survival(total, n, u, capital):
    zero, one = u-u, u-u+1
    values = [one]
    for k in range(1, n+1):
        b = max((total*k/n-u)/capital, zero)
        values.append(-sum((math.comb(k,j)*values[j]*b**(k-j)
                            for j in range(k)), zero))
    return sum((math.comb(n,j)*values[j] for j in range(n+1)), zero)


def _gap(xs, u, capital):
    zero = u-u
    n, total = len(xs), sum(xs, zero)
    if n < 2 or u == 0 or total <= u or total >= capital+u:
        return zero
    lo, hi = min(xs), max(xs)
    if lo >= u or lo == hi or total-lo <= u:
        return zero
    if n == 2:
        return (1-(total-u)/capital)*min(max(hi-u,zero), u-lo)/capital
    return _equal_survival(total,n,u,capital)-_subset_survival(xs,u,capital)


def fixed_vector_gap(xs, u, capital, *, precision=80, max_count=MAX_COUNT):
    """Return symmetric gaps, including zero/one-claim and terminal cases.

    Recursion uses adaptive Decimal precision. Ambiguous near-zero values
    for n<=8 are resolved with exact rational arithmetic; larger ambiguous
    cases raise instead of clipping or silently returning a negative gap.
    """
    max_count = integer(max_count, 'max_count', 0, MAX_COUNT)
    xs = _matrix(xs, max_count)
    u, capital, precision = _parameters(u, capital, precision)
    out = np.zeros(xs.shape[0], dtype=float)
    with localcontext() as ctx:
        ctx.prec = precision
        du, dc = Decimal.from_float(u), Decimal.from_float(capital)
        tiny = Decimal(10)**(-precision+15)
        for i, row in enumerate(xs):
            dx = [Decimal.from_float(float(x)) for x in row]
            g = _gap(dx, du, dc)
            # Tiny values after subtraction may have lost their sign. Resolve
            # these cases exactly when the rational calculation is affordable.
            total = sum(dx,Decimal(0))
            structural_zero = (len(dx)<2 or du==0 or total<=du or total>=du+dc
                               or min(dx)>=du or min(dx)==max(dx) or total-min(dx)<=du)
            if not structural_zero and (g <= 0 or abs(g) < tiny):
                if len(dx) > 8:
                    raise ArithmeticError('Ambiguous conditional gap; increase precision')
                g = _gap([Fraction.from_float(float(x)) for x in row],
                         Fraction.from_float(u), Fraction.from_float(capital))
            bound = ((1+du/dc)**max(0,len(dx)-1)/dc
                     *sum((max(du-x,Decimal(0)) for x in dx),Decimal(0)))
            value = float(g)
            if not math.isfinite(value) or value < 0 or value > min(1.,float(bound))*(1+1e-12)+1e-300:
                raise ArithmeticError('Conditional gap violates its mathematical bounds')
            if g > 0 and value == 0:
                raise ArithmeticError('Conditional gap underflows binary64 output')
            out[i] = value
    return out


def permutation_survival(xs, u, capital, *, precision=80):
    xs = _matrix(xs)
    u, capital, precision = _parameters(u, capital, precision)
    out = np.empty(xs.shape[0])
    with localcontext() as ctx:
        ctx.prec = precision
        du, dc = Decimal.from_float(u), Decimal.from_float(capital)
        for i,row in enumerate(xs):
            dx = [Decimal.from_float(float(x)) for x in row]
            total = sum(dx,Decimal(0))
            q = Decimal(1) if total <= du else Decimal(0) if total >= du+dc else _subset_survival(dx,du,dc)
            if not 0 <= q <= 1:
                raise ArithmeticError('Survival recursion left [0,1]')
            out[i] = float(q)
    return out


def equalized_survival(totals, n, u, capital, *, precision=80):
    n = integer(n, 'n', 0, MAX_COUNT)
    if np.iscomplexobj(totals):
        raise ValueError('Totals must be real')
    totals = np.atleast_1d(np.asarray(totals, dtype=float))
    if totals.ndim != 1 or not np.isfinite(totals).all() or (totals < 0).any():
        raise ValueError('Totals must be finite and nonnegative')
    if n == 0 and (totals != 0).any():
        raise ValueError('Zero claims must have zero total')
    u, capital, precision = _parameters(u, capital, precision)
    out = np.empty(len(totals))
    with localcontext() as ctx:
        ctx.prec = precision
        du, dc = Decimal.from_float(u), Decimal.from_float(capital)
        for i,s in enumerate(totals):
            ds = Decimal.from_float(float(s))
            q = Decimal(1) if ds <= du else Decimal(0) if ds >= du+dc else _equal_survival(ds,n,du,dc)
            if not 0 <= q <= 1:
                raise ArithmeticError('Equalized survival left [0,1]')
            out[i] = float(q)
    return out


def _log1p_ratio(x, beta):
    x = np.asarray(x,dtype=float)
    with np.errstate(over='ignore'):
        ratio = x/beta
    ordinary = np.isfinite(ratio)
    result = np.empty_like(x)
    result[ordinary] = np.log1p(ratio[ordinary])
    result[~ordinary] = np.log(x[~ordinary])-math.log(beta)+np.log1p(beta/x[~ordinary])
    return result


def lomax_cdf(x, alpha, beta):
    alpha, beta = positive(alpha,'alpha'), positive(beta,'beta')
    x = np.asarray(x,dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('CDF arguments must be finite and nonnegative')
    with np.errstate(over='ignore'):
        return -np.expm1(-alpha*_log1p_ratio(x,beta))


def lomax_inverse(uniforms, alpha, beta, cap):
    """Inverse CDF with an explicit terminal-ruin cap; handles extreme tails."""
    alpha,beta,cap = positive(alpha,'alpha'),positive(beta,'beta'),positive(cap,'cap')
    if np.iscomplexobj(uniforms):
        raise ValueError('Uniform variates must be real')
    z = np.asarray(uniforms,dtype=float)
    if not np.isfinite(z).all() or (z < 0).any() or (z >= 1).any():
        raise ValueError('Uniform variates must lie in [0,1)')
    with np.errstate(over='ignore'):
        exponent = -np.log1p(-z)/alpha
    threshold = float(_log1p_ratio(cap,beta))
    capped = exponent >= threshold
    result = np.full(z.shape,cap)
    active = ~capped
    a = exponent[active]
    ordinary = a < 700
    values = np.zeros(a.shape)
    with np.errstate(over='ignore'):
        values[ordinary] = beta*np.expm1(a[ordinary])
    special = ~ordinary | ~np.isfinite(values)
    if special.any():
        b = a[special]
        with np.errstate(divide='ignore'):
            values[special] = np.exp(math.log(beta)+b+np.log(-np.expm1(-b)))
    result[active] = values
    result[z==0] = 0.
    # This is the explicit modeled cap, not a clipping of conditional gaps.
    result = np.minimum(result,cap)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ArithmeticError('Inverse CDF left its finite range')
    return result


def lomax_zero_reserve(alpha, beta, capital, mean_count, grid=262144, cutoff=16):
    alpha,beta,capital = positive(alpha,'alpha'),positive(beta,'beta'),positive(capital,'capital')
    m = positive(mean_count,'mean_count',allow_zero=True)
    grid,cutoff = integer(grid,'grid',2,2**22),integer(cutoff,'cutoff',0,10000)
    if m >= cutoff+2 or math.exp(-m) == 0:
        raise ValueError('Count cutoff does not support the geometric tail bound')
    x = np.linspace(0,capital,grid+1)
    cdf = lomax_cdf(x,alpha,beta)
    down = np.diff(cdf)
    up = np.zeros(grid)
    up[1:] = np.diff(cdf[:grid])
    if (down < 0).any() or (up < 0).any():
        raise ArithmeticError('Rounded claim probabilities became negative')
    weight = 1-np.arange(grid)/grid
    output = {}
    for label,probabilities in [('upper_survival',down),('lower_survival',up)]:
        transform = np.fft.rfft(probabilities,n=2*grid)
        convolution = np.zeros(grid)
        convolution[0] = 1
        p_n = math.exp(-m)
        survival = p_n
        for n in range(1,cutoff+1):
            convolution = np.fft.irfft(np.fft.rfft(convolution,n=2*grid)*transform,n=2*grid)[:grid]
            p_n *= m/n
            survival += p_n*float(np.dot(weight,convolution))
        output[label] = survival
    remainder = p_n*m/(cutoff+1)/(1-m/(cutoff+2))
    output.update(upper_survival=output['upper_survival']+remainder,
                  poisson_tail_bound=remainder,discretization_width_bound=m/grid,grid=grid)
    return output


def importance_lomax(samples, seed, u, lam, horizon, premium, alpha, beta,
                     *, precision=80, max_count=MAX_COUNT, chunk_size=8192):
    samples = integer(samples,'samples',2,MAX_SAMPLES)
    seed = integer(seed,'seed',0)
    max_count = integer(max_count,'max_count',2,MAX_COUNT)
    chunk_size = integer(chunk_size,'chunk_size',1,MAX_SAMPLES)
    u,lam,horizon,premium,alpha,beta = [positive(x,name) for x,name in
        [(u,'u'),(lam,'lam'),(horizon,'horizon'),(premium,'premium'),(alpha,'alpha'),(beta,'beta')]]
    m,capital = lam*horizon,premium*horizon
    positive(m,'lam*horizon'); positive(capital,'premium*horizon')
    terminal = u+capital
    if not math.isfinite(terminal):
        raise ValueError('u+capital must remain representable')
    cap = max(terminal+1,math.nextafter(terminal,math.inf))
    if not math.isfinite(cap):
        raise ValueError('Terminal-ruin cap is not representable')
    fu = float(lomax_cdf(u,alpha,beta))
    mu2 = m*(-math.expm1(-m))
    if fu <= 0 or mu2 <= 0 or not math.isfinite(mu2):
        raise ArithmeticError('Importance weight is outside the representable range')
    if math.log(mu2)+math.log(fu)+math.log(u)-math.log(capital) < math.log(np.nextafter(0.,1.))+20:
        raise ArithmeticError('Weighted signal is too small for binary64 output')
    rng = np.random.default_rng(seed)
    if m < .05:
        # Inversion avoids an arbitrarily slow zero-truncated rejection loop.
        uniforms = rng.random(samples)
        k = np.ones(samples,dtype=np.int64)
        probability = m/math.expm1(m)
        cumulative = probability
        remaining = uniforms > cumulative
        while remaining.any():
            next_k = int(k[remaining][0])+1
            if next_k >= max_count:
                raise ValueError('Sampled count exceeds resource limit; retain seed and increase the limit')
            probability *= m/next_k
            cumulative = min(1.,cumulative+probability)
            k[remaining] = next_k
            remaining = uniforms > cumulative
    else:
        k = np.zeros(samples,dtype=np.int64)
        pending = np.arange(samples)
        attempts = 0
        while pending.size:
            attempts += 1
            if attempts > 10000:
                raise ArithmeticError('Zero-truncated count rejection exceeded its iteration guard')
            draws = rng.poisson(m,pending.size)
            good = draws > 0
            k[pending[good]] = draws[good]
            pending = pending[~good]
    counts = 1+k
    if counts.max() > max_count:
        raise ValueError('Sampled count exceeds resource limit; retain seed and increase the limit')
    estimates = np.zeros(samples)
    histogram = {}
    minimum_gap = math.inf
    for n in np.unique(counts):
        positions = np.flatnonzero(counts==n)
        histogram[int(n)] = int(positions.size)
        for start in range(0,positions.size,chunk_size):
            ids = positions[start:start+chunk_size]
            uniforms = rng.random((ids.size,int(n)))
            xs = lomax_inverse(uniforms,alpha,beta,cap)
            xs[:,0] = lomax_inverse(uniforms[:,0]*fu,alpha,beta,cap)
            bad = xs[:,0] > u
            if bad.any():
                # Resolve rare endpoint rounding without silently clipping it.
                with localcontext() as ctx:
                    ctx.prec = 100
                    a,b,du = Decimal.from_float(alpha),Decimal.from_float(beta),Decimal.from_float(u)
                    exact_fu = 1-(-a*(1+du/b).ln()).exp()
                    for j in np.flatnonzero(bad):
                        z = Decimal.from_float(float(uniforms[j,0]))*exact_fu
                        xs[j,0] = float(b*((-(1-z).ln()/a).exp()-1))
            small_count = (xs <= u).sum(axis=1)
            if (small_count < 1).any() or (xs[:,0] > u).any():
                raise ArithmeticError('Conditional mark sampler violated X1<=u')
            gaps = fixed_vector_gap(xs,u,capital,precision=precision,max_count=max_count)
            minimum_gap = min(minimum_gap,float(gaps.min()))
            estimates[ids] = mu2*fu*gaps/small_count
    if not np.isfinite(estimates).all():
        raise ArithmeticError('Nonfinite weighted observation')
    value = float(estimates.mean())
    if value <= 0 or not np.any(estimates != estimates[0]):
        raise ArithmeticError('No usable variance estimate; do not report a zero-width normal interval')
    se = float(estimates.std(ddof=1)/math.sqrt(samples))
    return dict(parameters=dict(u=u,lam=lam,T=horizon,c=premium,alpha=alpha,beta=beta),
                samples=samples,seed=seed,mean=value,standard_error=se,
                normal_95_lower=value-1.96*se,normal_95_upper=value+1.96*se,
                relative_standard_error=se/value,count_histogram=histogram,
                negative_numerical_gaps=0,minimum_numerical_gap=minimum_gap,
                arithmetic='Decimal conditional volumes',precision=precision)
