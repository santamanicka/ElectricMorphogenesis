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
tp = shape_mask('tadpole', (40, 150))                                # snout at the left, dorsal side up
def colinfo(mk, j):
    r = np.where(mk[:, j])[0]
    return (len(r), r[0], r[-1]) if len(r) else (0, -1, -1)
from scipy import ndimage as _ndi0
n_body, top_body, bot_body = colinfo(tp, int(0.25 * 150)); n_head, top_head, _ = colinfo(tp, int(0.15 * 150)); n_fin, top_fin, bot_fin = colinfo(tp, int(0.55 * 150)); n_tip, _, _ = colinfo(tp, 146); n_snout, _, _ = colinfo(tp, 3)
check('tadpole: oval body bulging below the tail, tail and fins rising above the head (dorsal fin), a narrow snout and a narrow tail tip, one piece',
      n_body >= 0.7 * 40 and bot_body > bot_fin and top_fin < top_head and n_snout < n_body / 2 and 0 < n_tip <= 5 and _ndi0.label(tp)[1] == 1, f'depth: body {n_body}, tail {n_fin}, snout {n_snout}, tip {n_tip}; belly rows {bot_body} vs {bot_fin}, back rows {top_head} vs {top_fin}')
edge_ok = True
for Rr, Cc in [(40, 150), (22, 82), (11, 41), (44, 164), (23, 85), (40, 104)]:
    mk = shape_mask('tadpole', (Rr, Cc)); edge_ok &= not (mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any())
check('tadpole: a clear margin all round (6 grids, odd and even)', edge_ok)
check('tadpole: head at the left end of a wide box and at the top of a tall one (transposition), dorsal side to the top or the left', np.array_equal(shape_mask('tadpole', (150, 40)), tp.T))
check('tadpole: the body keeps its proportions in a shorter box (the first 20 columns are the same in boxes 2.6 and 3.75 times longer than high)', np.array_equal(shape_mask('tadpole', (40, 104))[:, :20], tp[:, :20]))
d9, d5 = shape_mask('tadpole', (40, 150), 0.9)[:, 82].sum(), shape_mask('tadpole', (40, 150), 0.5)[:, 82].sum()
check('tadpole: the parameter is the depth of the tail (0.5 gives a tail depth about 5/9 of the 0.9 one at the deepest point)', abs(d5 / d9 - 0.5 / 0.9) < 0.08, f'{d5} / {d9}')

pl = shape_mask('planarian', (40, 200))
wid = pl.sum(axis=0)                                                 # width (pixels) at each column, head at the left
plat = wid.max(); cols = np.where(wid > 0)[0]; c0, c1 = cols[0], cols[-1]
c_full = int(np.argmax(wid == plat)); c_half = int(1.6 + 0.5 * (200 - 3.2))
from scipy import ndimage as _ndi
check('planarian: pointed head already as wide as the body shortly after the nose (auricle corners at 13% of the length), no neck (full width from there to the middle), one piece',
      wid[c0] < 12 and c_full - c0 <= int(0.134 * 197) + 2 and (wid[c_full:c_half] == plat).all() and np.all(np.diff(wid[:c_full + 1]) >= 0) and _ndi.label(pl)[1] == 1, f'widths: nose {wid[c0]}, body {plat}, full width from column {c_full} (nose at {c0}) to {c_half}')
check('planarian: the tail tapers monotonically to a point over the second half', np.all(np.diff(wid[c_half:]) <= 0) and 0 < wid[c1] <= 3 and wid[150] < plat and wid[170] < wid[150], f'widths: s=0.75 {wid[150]}, s=0.85 {wid[170]}, tip {wid[c1]}')
edge_ok = True
for Rr, Cc in [(40, 200), (22, 102), (11, 51), (44, 204), (23, 105)]:
    mk = shape_mask('planarian', (Rr, Cc))
    edge_ok &= not (mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any())
check('planarian: a clear margin all round, so no part of the outline sits on the edge of the box and looks cut off (5 grids, odd and even)', edge_ok and not pl[:1].any() and pl[:, c0].sum() == wid[c0])
check('planarian: head at the left end of a wide box and at the top of a tall one (mirror images of each other under transposition)', np.array_equal(shape_mask('planarian', (200, 40)), shape_mask('planarian', (40, 200)).T))
check('planarian: has no parameter (the head is as wide as the body by construction)', SHAPES['planarian'][1] is None and np.array_equal(shape_mask('planarian', (40, 200), 0.3), pl))

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
