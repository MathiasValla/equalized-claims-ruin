#!/usr/bin/env python3
"""Regenerate every reported run and record reproducible timing information."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import json
import math
import platform
import subprocess
import sys
import time
import mpmath as mp
import numpy as np
from verify_local_gap import check_subset_recursion,power_two_claim_checks,atom_poisson_checks
from risk_core import importance_lomax,lomax_zero_reserve
from certify_bounds import certify
from reproduce_bounds import CASES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'


def save(name,value):
    target = DATA/name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(target)


def main():
    start = time.perf_counter()
    processor = platform.processor()
    if platform.system()=='Darwin':
        r = subprocess.run(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True)
        if r.returncode==0:
            processor = r.stdout.strip()
    timing = dict(environment=dict(python=platform.python_version(),numpy=np.__version__,mpmath=mp.__version__,
                                   system=platform.system(),machine=platform.machine(),processor=processor,
                                   processes=1,thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),runs=[])
    def run(samples,seed,parameters,label):
        begin = time.perf_counter()
        r = importance_lomax(samples,seed,*parameters)
        elapsed = time.perf_counter()-begin
        timing['runs'].append(dict(label=label,samples=samples,seed=seed,seconds=elapsed))
        print(label,format(r['mean'],'.12e'),f'{elapsed:.3f} s',flush=True)
        return r
    checks = dict(environment=dict(python=sys.version,numpy=np.__version__,mpmath=mp.__version__),
                  subset_recursion_check=check_subset_recursion(),
                  two_claim_lower_tail_orders=power_two_claim_checks(),zero_atom_poisson=atom_poisson_checks())
    zero = lomax_zero_reserve(.95,10,1000,.1)
    for label,side in [('lower','lower_survival'),('upper','upper_survival')]:
        zero['quadratic_coefficient_'+label] = .095/20000*(zero[side]-math.exp(-.1))
    checks['main_example_zero_reserve'] = zero
    checks['importance_sampling'] = [run(200000,20261027+i,(u,1,.1,10000,.95,10),f'reserve {u}')
                                     for i,u in enumerate([.01,.001,.0001])]
    save('checks.json',checks)
    parameters = [(float(c.u),float(c.lam),float(c.horizon),float(c.premium),float(c.alpha),float(c.beta)) for c in CASES]
    six = [run(500000,20261101+i,p,c.label) for i,(p,c) in enumerate(zip(parameters,CASES))]
    save('six_examples_importance.json',six)
    validation = run(2000000,20261110,parameters[0],'independent main run')
    save('main_example_validation.json',validation)
    begin = time.perf_counter()
    certified = [certify(c) for c in CASES]
    timing['certify_six_seconds'] = time.perf_counter()-begin
    save('certified_bounds.json',certified)
    timing['total_regeneration_seconds'] = time.perf_counter()-start
    save('timing.json',timing)
    old = ROOT/'tmp/revision_v1/data'
    if old.exists():
        comparisons=[]
        for name,current in [('checks.json',checks['importance_sampling']),('six_examples_importance.json',six),('main_example_validation.json',[validation])]:
            previous=json.loads((old/name).read_text())
            if name=='checks.json': previous=previous['importance_sampling']
            if isinstance(previous,dict): previous=[previous]
            for a,b in zip(previous,current):
                comparisons.append(dict(seed=b['seed'],mean_old=a['mean'],mean_new=b['mean'],
                                        change_in_standard_errors=abs(a['mean']-b['mean'])/b['standard_error']))
        save('revision_comparison.json',comparisons)
    print('All ten runs and six outward-rounded enclosures saved.',flush=True)


if __name__=='__main__':
    main()
