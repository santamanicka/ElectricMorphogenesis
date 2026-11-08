"""Is the flat-top hexagon's pattern four-fold in its middle?   python fourfold_check.py  (writes results/fourfold_hexagon.txt)

A homogeneous start in a disc or ring is exactly four-fold on a square grid; a flat-top hexagon only has the two mirror symmetries. This measures the four-fold error
(rms(V - rot90 V) / spread, over the central 30 x 30 pixels, away from the boundary) of the hexagon against a rotated hexagon, a rectangle and the disc, and the mirror error.
"""
import os
import numpy as np
from potentia import Potentia
from shapes import shape_mask
from sweeps import STIG

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anisotropy_study.py')).read().split("# ------------------------------------------------------------------ (A) operators")[0]
ns = {'__file__': os.path.abspath('anisotropy_study.py')}
exec(compile(src, 'anisotropy_study_head', 'exec'), ns)            # imports, PotentiaIsoGate (the experimental isotropic-gate variant) and hexagon_rot
Iso = ns['PotentiaIsoGate']; hexrot = ns['hexagon_rot']
lines = []


def central(V, half=15):
    R, C = V.shape
    return V[R // 2 - half:R // 2 + half, C // 2 - half:C // 2 + half]


def c4_err(V):
    c = central(V)
    return np.sqrt(np.mean((c - np.rot90(c)) ** 2)) / np.sqrt(np.mean((c - c.mean()) ** 2))


def d2_err(V):
    c = central(V)
    return max(np.sqrt(np.mean((c - c[::-1]) ** 2)), np.sqrt(np.mean((c - c[:, ::-1]) ** 2))) / np.sqrt(np.mean((c - c.mean()) ** 2))


cases = [('hexagon (flat top), standard gate', Potentia, shape_mask('hexagon', (60, 60)), (30, 30)), ('hexagon (flat top), isotropic gate', Iso, shape_mask('hexagon', (60, 60)), (30, 30)),
         ('hexagon rotated 15 deg, standard gate', Potentia, hexrot((60, 60), 15.0), (30, 30)), ('rectangle 30 x 50 cells, standard gate', Potentia, None, (30, 50)),
         ('disc (D4), standard gate', Potentia, shape_mask('disc', (60, 60)), (30, 30))]
lines.append('Four-fold error of the central 30 x 30 pixels (rms(V - rot90 V) / spread; 0 = exactly four-fold, about 1 = unrelated), homogeneous start, Stigmergic, m = 2')
lines.append(f'    {"case":<42}' + ''.join(f'{t:>9d} s' for t in (10, 30, 60, 100, 140)) + '   mirror (D2) error at 140 s')
for name, cls, mk, size in cases:
    P = cls(size_cells=size, m=2, mask=mk, **STIG)
    V, g = P.initial_from_g(1.0, 'dep'); rec = {}
    for it in range(14001):
        if it % 1000 == 0 and it // 100 in (10, 30, 60, 100, 140):
            rec[it // 100] = c4_err(V)
        if it < 14000:
            V, g = P.step(V, g)
    lines.append(f'    {name:<42}' + ''.join(f'{rec[t]:>11.3f}' for t in (10, 30, 60, 100, 140)) + f'   {d2_err(V):.1e}')
print('\n'.join(lines))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'fourfold_hexagon.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
