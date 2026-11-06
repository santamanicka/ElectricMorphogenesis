"""Is the Stigmergic operating point special in Potentia?  Endogenous criterion: perpetual novelty
(never settles, never repeats), measured by the novelty score (mV) and checked by Lyapunov exponents.

Reads the fine scans F1 (bias), F2 (R_s) and F3 (2-D grid) for the original (lattice) and Potentia.
 - 1-D: location of the novelty maximum (mean over 3 jitter replicates), the interval where novelty >= 50% of its
        maximum, and where the Stigmergic value falls relative to both
 - 2-D: rank of the Stigmergic cell among all cells, its novelty relative to the global maximum, and the ridge
        (best bias at each R_s)
usage: python analysis_optimum.py
"""
import os, pickle
import numpy as np
from sweeps import STIG

B0 = STIG['bias']


def load(sweep, tag):
    fn = f'results/atlas_{sweep}_11_{tag}.pkl'
    return pickle.load(open(fn, 'rb')) if os.path.exists(fn) else None


def vals(R, key='novelty'):
    return np.array([r['desc'][key] for r in R['results']]).reshape(R['shape'])


def oned(sweep, xs_fn, ref):
    print(f'\n=== {sweep}: novelty along the axis, mean +/- sd over 3 jitter replicates')
    for tag in ('lattice', 'potentia2', 'potentia4'):
        R = load(sweep, tag)
        if R is None: continue
        x = xs_fn(np.array(R['axes'][list(R['axes'])[0]], float))
        Y = vals(R); mu, sd = Y.mean(axis=1), Y.std(axis=1, ddof=1)
        i = int(np.argmax(mu))
        half = x[mu >= 0.5 * mu[i]]
        j = int(np.argmin(np.abs(x - ref)))
        # is the Stigmergic value's novelty distinguishable from the maximum? (difference vs combined sd)
        z = (mu[i] - mu[j]) / np.sqrt(sd[i] ** 2 + sd[j] ** 2 + 1e-12)
        print(f'  {tag:10s} max novelty {mu[i]:5.1f} +/- {sd[i]:3.1f} mV at x = {x[i]:.3g}   | >=50% of max for x in [{half.min():.3g}, {half.max():.3g}]'
              f'   | at Stigmergic x={x[j]:.3g}: {mu[j]:5.1f} +/- {sd[j]:3.1f} mV ({100 * mu[j] / mu[i]:.0f}% of max, {z:.1f} sd below max)')


def twod():
    print('\n=== F3: 2-D novelty grid (R_s x bias/bias*), one run per cell')
    for tag in ('lattice', 'potentia2'):
        R = load('F3', tag)
        if R is None: continue
        rs = np.array(R['axes']['fieldScreenSize'], float); bs = np.array(R['axes']['bias'], float) / B0
        N = vals(R)
        i4, j1 = int(np.argmin(np.abs(rs - STIG['fieldScreenSize']))), int(np.argmin(np.abs(bs - 1.0)))
        order = np.argsort(-N.ravel())
        rank = int(np.where(order == np.ravel_multi_index((i4, j1), N.shape))[0][0]) + 1
        gi, gj = np.unravel_index(int(np.argmax(N)), N.shape)
        print(f'  {tag:10s} global max {N.max():5.1f} mV at R_s={rs[gi]:g}, bias={bs[gj]:.2f}x | Stigmergic cell {N[i4, j1]:5.1f} mV = {100 * N[i4, j1] / N.max():.0f}% of max, rank {rank}/{N.size}')
        print('             ridge (best bias/bias* at each R_s):  ' + '  '.join(f'R_s={r:g}: {bs[int(np.argmax(N[k]))]:.2f}x ({N[k].max():.0f})' for k, r in enumerate(rs)))
        frac = (N >= 0.5 * N.max()).mean()
        print(f'             fraction of the grid with novelty >= 50% of the max: {frac:.2f}  (cells >= 3 mV: {(N >= 3).mean():.2f})')


if __name__ == '__main__':
    oned('F1', lambda v: v / B0, 1.0)
    oned('F2', lambda v: v, STIG['fieldScreenSize'])
    twod()
