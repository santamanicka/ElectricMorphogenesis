"""Tests of tissue shapes (masks) in Potentia: python test_masks.py

A mask must make the tissue behave like a finite tissue of that shape: nothing outside it, no gap junctions across its boundary, charge only inside.
"""
import sys
import numpy as np
from potentia import Potentia
from shapes import shape_mask, SHAPES
from sweeps import STIG

ok = True


def check(name, cond, extra=''):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'   {extra}' if extra else ''))


# 1. A rectangular island inside a larger grid is the smaller finite tissue (translation invariance of the model + a correct mask).
for m, vector, offs in [(2, True, (3, 5)), (3, True, (2, 4)), (2, False, (4, 2))]:
    small = (5, 6)
    Ps = Potentia(size_cells=small, m=m, **{**STIG, 'fieldVector': vector})
    big = (small[0] * m + 2 * offs[0], small[1] * m + 2 * offs[1])
    mask = np.zeros(big, bool); mask[offs[0]:offs[0] + Ps.Np[0], offs[1]:offs[1] + Ps.Np[1]] = True
    Pb = Potentia(size_cells=(big[0] / m, big[1] / m), m=m, mask=mask, **{**STIG, 'fieldVector': vector})
    rng = np.random.default_rng(m)
    V0 = rng.uniform(-55e-3, -5e-3, Ps.Np); g0 = rng.uniform(0.2, 1.8, Ps.Np)
    Vb = np.zeros(big); gb = np.zeros(big); Vb[mask] = V0.ravel(); gb[mask] = g0.ravel()
    Vs, gs = V0.copy(), g0.copy()
    for _ in range(300):
        Vs, gs = Ps.step(Vs, gs); Vb, gb = Pb.step(Vb, gb)
    sl = (slice(offs[0], offs[0] + Ps.Np[0]), slice(offs[1], offs[1] + Ps.Np[1]))
    d = max(np.abs(Vb[sl] - Vs).max(), np.abs(gb[sl] - gs).max() * 1e-3)
    out = max(np.abs(Vb[~mask]).max(), np.abs(gb[~mask]).max())
    check(f'rectangular island in a larger grid == the smaller finite tissue (m={m}, {"vector" if vector else "scalar"} field), 300 steps', d < 1e-12 and out == 0.0, f'max|dV|={d:.1e}, outside={out}')

# 2. Outside pixels stay at zero; the gap-junction term conserves charge inside the tissue and creates no flux across the boundary
Pd = Potentia(size_cells=(10, 10), m=2, mask=shape_mask('annulus', (20, 20)), **STIG)
rng = np.random.default_rng(1)
V = rng.uniform(-55e-3, -5e-3, Pd.Np) * Pd.maskf
check('gap-junction term sums to zero over the tissue (no flux through the boundary or the hole)', abs(Pd.gj(V).sum()) < 1e-15 * np.abs(V).sum() * Pd.gj_coef, f'sum={Pd.gj(V).sum():.1e}')
check('gap-junction term is exactly zero outside the tissue', np.all(Pd.gj(V)[~Pd.mask] == 0.0))
Pt = Potentia(size_cells=(8, 18), m=2, mask=shape_mask('dumbbell', (16, 36), 0.0), **STIG)   # two separate discs, 4 px (20 um) apart: inside the field range R = 28 um
lab_left = np.zeros(Pt.Np, bool); lab_left[:, :18] = True
Vt = rng.uniform(-55e-3, -5e-3, Pt.Np) * Pt.maskf
gl = Pt.gj(Vt)
check('two separate islands exchange no gap-junction current (each conserves its own charge)', abs(gl[lab_left].sum()) < 1e-12 and abs(gl[~lab_left].sum()) < 1e-12 and Pt.mask[:, 16:20].sum() == 0, f'left {gl[lab_left].sum():.1e}, right {gl[~lab_left].sum():.1e}')
Vg = rng.uniform(-55e-3, -5e-3, Pd.Np)
check('the field ignores whatever is stored outside the tissue (charge only inside)', np.allclose(Pd.field(Vg), Pd.field(Vg * Pd.maskf), rtol=0, atol=0))
# ... but the field of one island does reach the other (the interaction is through the field only)
Va = np.zeros(Pt.Np); Va[Pt.mask & lab_left] = -9e-3; Vb2 = Va.copy(); Vb2[Pt.mask & ~lab_left] = -50e-3
Ea, Eb = Pt.field(Va), Pt.field(Vb2)
chg = np.abs(Eb - Ea)[:Pt.Np[0], :18].max() / np.abs(Ea).max()
check('islands closer than the field range interact through the field: changing the right island changes the field at the left one', chg > 1e-4, f'relative change {chg:.1e}')

# 3. Symmetry: a homogeneous start in a symmetric shape on a square grid keeps the symmetry of the grid (a misplaced mask or a one-sided pair mask would break it)
for name, syms in [('disc', 'rot'), ('annulus', 'rot'), ('cross', 'rot'), ('hexagon', 'flip'), ('triangle', 'mirror'), ('stadium', 'rot')]:
    Ps = Potentia(size_cells=(12, 12), m=2, mask=shape_mask(name, (24, 24)), **STIG)
    V, g = Ps.initial_from_g(1.0, 'dep')
    for _ in range(3000): V, g = Ps.step(V, g)
    if syms == 'rot':
        d = max(np.abs(V - np.rot90(V)).max(), np.abs(V - V[::-1, :]).max(), np.abs(V - V.T).max())
    elif syms == 'flip':
        d = max(np.abs(V - V[::-1, :]).max(), np.abs(V - V[:, ::-1]).max())
    else:
        d = np.abs(V - V[:, ::-1]).max()
    spread = np.ptp(V[Ps.mask])
    check(f'{name}: the homogeneous start keeps the symmetry of the shape for 30 s', d < 1e-9 and spread > 1e-3, f'max asymmetry {d:.1e} V, Vmem spread {1e3 * spread:.1f} mV')

# 4. The shapes themselves
for name in SHAPES:
    good = True
    for Np in [(24, 24), (30, 50), (50, 30), (21, 33), (64, 64)]:
        mk = shape_mask(name, Np)
        good &= mk.shape == Np and mk.any()
    check(f'shape {name}: non-empty on 5 grid sizes', good)
sq = shape_mask('disc', (40, 40))
check('disc: symmetric under flips about the grid centre and covers pi/4 of the square', np.array_equal(sq, sq[::-1, :]) and np.array_equal(sq, sq[:, ::-1]) and np.array_equal(sq, sq.T) and abs(sq.mean() - np.pi / 4) < 0.01, f'{sq.mean():.3f}')
an = shape_mask('annulus', (40, 40), 0.5)
check('annulus: hole of half the radius, hole pixels excluded', not an[20, 20] and an[20, 3] and abs(an.mean() - np.pi / 4 * 0.75) < 0.01, f'{an.mean():.3f}')
two = shape_mask('dumbbell', (20, 60), 0.0)
check('dumbbell with neck 0: two separate discs', two[:, 25:35].sum() == 0 and two[:, :20].any() and two[:, 40:].any())
tp = shape_mask('tadpole', (20, 60))
check('tadpole: round head at the left end (20 pixels tall), narrow tail (6 pixels) reaching the right end', tp[:, 10].sum() == 20 and tp[:, 50].sum() == 6 and tp[10, 59], f'{tp[:, 10].sum()} / {tp[:, 50].sum()}')

pl = shape_mask('planarian', (40, 200))
wid = pl.sum(axis=0)                                                 # width (pixels) at each column, head at the left
c_aur, c_neck, c_mid, c_tail = int(0.13 * 200), int(0.20 * 200), int(0.52 * 200), int(0.97 * 200)
from scipy import ndimage as _ndi
check('planarian: arrowhead head with auricles wider than the neck behind them, a full-width body, a narrow tail, one piece',
      wid[c_aur] >= wid[c_neck] + 4 and wid[c_mid] == 40 and 0 < wid[c_tail] <= 6 and wid[2] < wid[c_aur] and _ndi.label(pl)[1] == 1, f'widths: nose {wid[2]}, auricles {wid[c_aur]}, neck {wid[c_neck]}, body {wid[c_mid]}, tail {wid[c_tail]}')
check('planarian: head at the left end of a wide box and at the top of a tall one (mirror images of each other under transposition)', np.array_equal(shape_mask('planarian', (200, 40)), shape_mask('planarian', (40, 200)).T))
check('planarian: the parameter widens the head (auricle width 0.85 against 0.45)', shape_mask('planarian', (40, 200), 0.85)[:, 26].sum() > shape_mask('planarian', (40, 200), 0.45)[:, 26].sum() + 10)

# 5. Errors
for label, fn in [('wrong mask shape', lambda: Potentia(size_cells=(4, 4), m=2, mask=np.ones((3, 3), bool))),
                  ('periodic with a shape', lambda: Potentia(size_cells=(4, 4), m=2, periodic=True, mask=shape_mask('disc', (8, 8)))),
                  ('empty mask', lambda: Potentia(size_cells=(4, 4), m=2, mask=np.zeros((8, 8), bool))),
                  ('unknown shape', lambda: shape_mask('torus', (8, 8)))]:
    try:
        fn(); check(f'{label} is rejected', False)
    except ValueError as e:
        check(f'{label} is rejected', True, str(e)[:70])
check('an all-True mask is the plain rectangle (no mask stored)', Potentia(size_cells=(4, 4), m=2, mask=np.ones((8, 8), bool)).mask is None)

print('\nALL PASS' if ok else '\nFAILURES')
sys.exit(0 if ok else 1)
