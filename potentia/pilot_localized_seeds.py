"""Pilot: does Potentia support localized, self-maintaining objects (Lenia-style 'creatures') seeded in a quiescent tissue?

30x30-cell finite tissue (m=2 -> 60x60 px), hyperpolarised background (g = 2, Vmem at its equilibrium), a depolarised seed (G_pol = 0 disc) in the
centre, Stigmergic parameters except bias and field range, T = 60 s.  A finite tissue's edges can switch on by themselves, so every parameter set also
runs a CONTROL without the seed; the seed's effect is the difference map |V_seed - V_control| > 10 mV, and the 'clean window' is the time until the control
first shows activity (Vmem > -30 mV) in the outer 2 cells.
Seed outcomes: dies (the effect is gone by the end), spreads (effect reaches the outer 5 cells, or covers > 30%), persistent localized (effect covers 0.2-25% for
the last 30 s, confined to the interior), other.   usage: python pilot_localized_seeds.py
"""
import time, json
import numpy as np
from multiprocessing import Pool
from scipy import ndimage
from sweeps import STIG
from potentia import Potentia

N, M, T, REC = 30, 2, 60.0, 1.0
EDGE = 2 * M            # outer 2 cells: edge self-activation
MARGIN = 5 * M          # outer 5 cells: where a spreading effect counts as 'reached the edge'

def trajectory(ratio, Rs, seed_r):
    P = Potentia(size_cells=(N, N), m=M, **{**STIG, 'bias': STIG['bias'] * ratio, 'fieldScreenSize': Rs})
    g = np.full(P.Np, 2.0)
    if seed_r:
        yy, xx = np.mgrid[:P.Np[0], :P.Np[1]]; c = (P.Np[0] - 1) / 2
        g[(yy - c) ** 2 + (xx - c) ** 2 <= seed_r ** 2] = 0.0
    V, g = P.initial_from_g(g, 'dep')
    nrec, frames = int(round(REC / P.dt)), []
    for it in range(int(round(T / P.dt)) + 1):
        if it % nrec == 0: frames.append((V * 1e3).astype(np.float32))
        if it < int(round(T / P.dt)): V, g = P.step(V, g)
    return np.array(frames)

def edge_active(F):
    a = F > -30
    return a[:, :EDGE].any(axis=(1, 2)) | a[:, -EDGE:].any(axis=(1, 2)) | a[:, :, :EDGE].any(axis=(1, 2)) | a[:, :, -EDGE:].any(axis=(1, 2))

def job(args):
    ratio, Rs = args
    ctrl = trajectory(ratio, Rs, 0)
    ea = edge_active(ctrl); t_edge = float(np.argmax(ea) * REC) if ea.any() else np.inf
    out = []
    for sr in (3, 6, 10):
        F = trajectory(ratio, Rs, sr); D = np.abs(F - ctrl) > 10.0
        area = D.mean(axis=(1, 2)); core = D[:, MARGIN:-MARGIN, MARGIN:-MARGIN].sum(axis=(1, 2)); allp = D.sum(axis=(1, 2))
        reached = (allp > core).any(); late = slice(int(len(area) - 30 / REC), None)
        if area[-1] < 0.001: cls = 'dies'
        elif reached or area[-1] > 0.30: cls = 'spreads'
        elif 0.002 <= area[late].min() and area[late].max() <= 0.25: cls = 'persistent localized'
        else: cls = 'other'
        out.append(dict(ratio=ratio, Rs=Rs, seed_r=sr, cls=cls, t_edge_control=t_edge, effect_end=float(area[-1]), effect_peak=float(area.max())))
    return out

if __name__ == '__main__':
    grid = [(r, Rs) for r in (0.7, 1.0, 1.4, 2.0, 3.0, 4.0) for Rs in (2.0, 3.0, 4.0, 6.0)]
    t0 = time.time()
    with Pool(4) as pool: res = [x for chunk in pool.map(job, grid, chunksize=1) for x in chunk]
    print(f'{len(res)} seeded runs + {len(grid)} controls, {time.time() - t0:.0f} s wall')
    from collections import Counter
    print('seed outcomes:', dict(Counter(r['cls'] for r in res)))
    print(f'\n{"bias/bias*":>10} {"R_s":>4} {"control: edge switches on at":>29}   seed radius 3 px -> 6 px -> 10 px')
    for r, Rs in grid:
        rr = [x for x in res if x['ratio'] == r and x['Rs'] == Rs]
        te = rr[0]['t_edge_control']
        print(f'{r:10.1f} {Rs:4.0f} {("never in 60 s" if np.isinf(te) else f"{te:.0f} s"):>29}   ' + ' -> '.join(f'{x["cls"]} ({100 * x["effect_peak"]:.0f}% peak)' for x in rr))
    json.dump(res, open('results/pilot_localized_seeds.json', 'w'), default=lambda o: None if np.isinf(o) else o)
