"""Run Potentia inside an animal outline from a homogeneous start and save G_pol (and Vmem) snapshots.

Free parameters (the only ones varied in the anatomy search): tissue size (rows, in cells), proportions
(box length / width, and for the tadpole the tail depth p and the head / tail ratio p2) and the field range
(as a percentage of the shorter side of the box).  Everything else is the Stigmergic set, including the bias.

    python anatomy_run.py planarian 16 4.6 25          # shape rows aspect range% [p p2] -> anatomy/runs/<tag>.npz
"""
import os, sys, time
import numpy as np
from potentia import Potentia
from shapes import shape_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anatomy', 'runs')
M = 2                       # pixels per cell pitch
T = 300.0                   # simulated seconds
SNAP = 5.0                  # seconds between snapshots


def tag(shape, rows, aspect, rng, p=None, p2=None, m=M):
    s = f'{shape}_r{rows:g}_a{aspect:g}_f{rng:g}'
    if p is not None:
        s += f'_p{p:g}_q{p2:g}'
    if m != M:
        s += f'_m{m}'
    return s


def run(shape, rows, aspect, rng, p=None, p2=None, m=M, T=T, snap=SNAP, noise_g=0.0, seed=0, save=True, suffix=''):
    cols = int(round(rows * aspect))
    Np = (int(round(rows * m)), int(round(cols * m)))
    mask = shape_mask(shape, Np, p, p2)
    R_um = rng / 100.0 * rows * 10.0                       # field range in um (cell pitch 10 um)
    R_s = R_um * 1e-6 / (Potentia.a * np.sqrt(2)) - 0.001
    P = Potentia(size_cells=(rows, cols), m=m, fieldScreenSize=R_s, mask=mask)
    V, g = P.initial(noise_g=noise_g, seed=seed)
    nsnap = int(round(T / snap)); per = int(round(snap / P.dt))
    G = np.zeros((nsnap + 1,) + Np, np.float32); Vs = np.zeros_like(G)
    G[0], Vs[0] = g, V
    t0 = time.time()
    for k in range(1, nsnap + 1):
        for _ in range(per):
            V, g = P.step(V, g)
        G[k], Vs[k] = g, V
    out = dict(G=G, V=Vs, mask=mask, t=np.arange(nsnap + 1) * snap, rows=rows, cols=cols, aspect=aspect, range_pct=rng, range_um=R_um,
               R_s=R_s, m=m, p=np.nan if p is None else p, p2=np.nan if p2 is None else p2, shape=shape, wall=time.time() - t0)
    if save:
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(os.path.join(OUT, tag(shape, rows, aspect, rng, p, p2, m) + suffix + '.npz'), **out)
    return out


if __name__ == '__main__':
    a = sys.argv[1:]
    o = run(a[0], float(a[1]), float(a[2]), float(a[3]), *(float(x) for x in a[4:6]))
    print(tag(a[0], float(a[1]), float(a[2]), float(a[3]), *(float(x) for x in a[4:6])), 'wall %.1f s' % o['wall'])
