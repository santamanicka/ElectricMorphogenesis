"""Run the parameter sweeps (finite tissue, homogeneous start, no clamp) for the lattice original and Potentia.
usage: python atlas.py <sweep> <size> <model:m> [T]
   sweeps: S1 (R_s x bias)  S2 (rate kappa x bias @R_s=4)  S3 (GJ x R_s)
   model:  lattice | potentia:2 | potentia:4"""
import sys, os, pickle, time
import numpy as np
from sweeps import STIG, run_grid, grid_cfgs

B0 = STIG['bias']
SWEEPS = {
    'S1': dict(axes={'fieldScreenSize': [1, 2, 3, 4, 5, 6, 8, 10],
                     'bias': [B0 * 2 ** (k / 2) for k in range(-6, 5)]}, params={}),
    'S2': dict(axes={'weight': [1000 * 2.0 ** k for k in range(-4, 5)],
                     'bias': [B0 * 2 ** (k / 2) for k in range(-6, 5)]}, params={}),
    'S3': dict(axes={'GJStrength': [0.0, 0.01, 0.05, 0.2, 1.0],
                     'fieldScreenSize': [1, 2, 3, 4, 5, 6, 8, 10]}, params={}),
}

def run(sweep, n, model, T=300.0, procs=4, keep=(3, 10, 30, 100)):
    name, _, m = model.partition(':')
    m = int(m) if m else 1
    S = SWEEPS[sweep]
    base = dict(size=(n, n), m=m, T=T, rec_dt=0.5, model=name, keep=list(keep), params=S['params'])
    cfgs, shape = grid_cfgs(S['axes'], base)
    t0 = time.time()
    res = run_grid(cfgs, procs=procs, label=f'{sweep} {n}x{n} {model}')
    out = dict(sweep=sweep, size=n, model=model, T=T, axes=S['axes'], shape=shape, results=res, wall=time.time() - t0)
    os.makedirs('results', exist_ok=True)
    fn = f'results/atlas_{sweep}_{n}_{name}{m if name == "potentia" else ""}.pkl'
    pickle.dump(out, open(fn, 'wb'))
    print(f'saved {fn}  wall {out["wall"]:.0f}s', flush=True)
    return out

if __name__ == '__main__':
    sweep, n, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    T = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
    run(sweep, n, model, T)
