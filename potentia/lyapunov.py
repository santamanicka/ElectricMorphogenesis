"""Largest Lyapunov exponent at the Stigmergic point by renormalised twin trajectories (Benettin).

Reference + twin whose (V,g) difference is rescaled back to rms dV = d0 every `tr` seconds; the exponent is
the mean log-growth per second.  Reported over blocks to give a standard error, after discarding the first
`burn` seconds (transient front phase).  Homogeneous start, finite 11x11 tissue, Stigmergic parameters.
 asym: random perturbation of all pixels;   sym: the same perturbation averaged over the 8 D4 images
 (stays in the symmetric invariant subspace the homogeneous start lives in).
usage: python lyapunov.py <m> <T> [asym|sym]
"""
import sys, numpy as np
from sweeps import STIG
from potentia import Potentia

def d4(X):
    return (X + X[::-1] + X[:, ::-1] + X[::-1, ::-1] + X.T + X.T[::-1] + X.T[:, ::-1] + X.T[::-1, ::-1]) / 8

def lyap(m, T, mode, d0=1e-9, tr=2.0, burn=100.0, nblocks=6, seed=0, N=11, params=None):
    P = Potentia(size_cells=(N, N), m=m, **{**STIG, **(params or {})})
    rng = np.random.default_rng(seed)
    V, g = P.initial()
    dV = d0 * rng.standard_normal(P.Np)
    if mode == 'sym': dV = d4(dV)
    dV *= d0 / np.sqrt((dV ** 2).mean())
    V2, g2 = V + dV, g.copy()
    ntr = int(round(tr / P.dt)); nrenorm = int(round(T / tr))
    logs = []
    for k in range(nrenorm):
        for _ in range(ntr):
            V, g = P.step(V, g); V2, g2 = P.step(V2, g2)
        dVv, dgv = V2 - V, g2 - g
        d = np.sqrt((dVv ** 2).mean())
        logs.append(np.log(d / d0) / tr)
        s = d0 / d
        V2 = V + dVv * s; g2 = g + dgv * s
    logs = np.array(logs); t = (np.arange(nrenorm) + 1) * tr
    use = logs[t > burn]
    blocks = np.array_split(use, nblocks)
    bm = np.array([b.mean() for b in blocks])
    return use.mean(), bm.std(ddof=1) / np.sqrt(nblocks), P.dt

if __name__ == '__main__':
    m = int(sys.argv[1]); T = float(sys.argv[2]); mode = sys.argv[3] if len(sys.argv) > 3 else 'asym'
    lam, se, dt = lyap(m, T, mode)
    print(f'm={m} dt={dt:g} T={T:g}s {mode}: largest Lyapunov exponent = {lam:+.4f} +/- {se:.4f} /s   (e-folding {1/lam if lam>0 else float("inf"):.0f} s)', flush=True)
