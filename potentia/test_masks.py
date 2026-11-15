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
from scipy import ndimage as _ndi0
flip_long = lambda mk: mk[::-1] if mk.shape[1] >= mk.shape[0] else mk[:, ::-1]               # mirror image about the long axis (horizontal in a wide box, vertical in a tall one)


def target_grid(R, aspect):
    """(R, C) whose inset box (inside the margin) has the given aspect: the grid on which an outline has the proportions of its target."""
    mg = max(1.0, 0.04 * R); return (R, int(round((R - 2 * mg) * aspect + 2 * mg)))


def column_depths(mk):
    return mk.sum(axis=0)


# tadpole (dorsal view; target 4.82 times longer than high): a large oval body, an S-shaped neck, a long tail tapering to a thin whip, mirror-symmetric
G = target_grid(120, 4.82); tp = shape_mask('tadpole', G); dep = column_depths(tp); cols = np.where(dep > 0)[0]; c0, c1 = cols[0], cols[-1]; Hin = G[0] - 2 * max(1.0, 0.04 * G[0]); Lin = c1 - c0 + 1
at = lambda s: dep[int(c0 + s * (Lin - 1))]
check('tadpole: a large oval body (depth at 20% of the length at least 90% of the height), a narrow tail (about 12% of the height at 75%) ending in a point, a small snout spike, one piece',
      at(0.2) >= 0.9 * Hin and 0.06 * Hin <= at(0.75) <= 0.2 * Hin and dep[c1] <= 3 and dep[c0] <= 0.12 * Hin and _ndi0.label(tp)[1] == 1, f'depths: snout {dep[c0]}, 20% {at(0.2)}, 75% {at(0.75)}, tip {dep[c1]} of {Hin:.0f}')
i_body = int(np.argmax(dep[: c0 + int(0.4 * Lin)])); tail_part = dep[i_body:c1 + 1]
check('tadpole: from the body to the tip the depth never increases by more than 1 pixel (a smooth S-shaped neck into the tail, no waist and no second bulge)', np.all(np.diff(tail_part) <= 1), f'largest increase {int(np.diff(tail_part).max())} px')
check('tadpole: shape of the target: the neck narrows the outline from the body (>= 0.9 of the height at 25% of the length) to under half the height at 45%', at(0.25) >= 0.9 * Hin and at(0.45) <= 0.5 * Hin, f'{at(0.25)} / {at(0.45)} of {Hin:.0f}')
check('tadpole: bilaterally symmetric: the mask equals its mirror image about the long axis (7 grids, odd and even, 3 tail depths)',
      all(np.array_equal(mk, flip_long(mk)) for mk in (shape_mask('tadpole', g, pp) for g in [(40, 150), (22, 82), (11, 51), (23, 85), (44, 164), (150, 40), (41, 120)] for pp in (None, 0.2, 0.6))))
edge_ok = True
for Rr, Cc in [(40, 150), (22, 100), (11, 51), (44, 164), (23, 85), (40, 104)]:
    mk = shape_mask('tadpole', (Rr, Cc)); edge_ok &= not (mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any())
check('tadpole: a clear margin all round (6 grids, odd and even)', edge_ok)
check('tadpole: head at the left end of a wide box and at the top of a tall one (transposition)', np.array_equal(shape_mask('tadpole', (G[1], G[0])), tp.T))
# the first parameter: depth of the tail at its root; the second: head size / tail size
d_def, _ = column_depths(shape_mask('tadpole', G)), None
d_thin = column_depths(shape_mask('tadpole', G, 0.2)); d_small = column_depths(shape_mask('tadpole', G, None, 1.9)); d_big = column_depths(shape_mask('tadpole', G, None, 2.36))
i75 = int(c0 + 0.75 * (Lin - 1)); i20 = int(c0 + 0.2 * (Lin - 1))
check('tadpole: the first parameter is the depth of the tail (0.2 gives a tail about 0.2 / 0.41 as deep at 75% of the length)', abs(d_thin[i75] / d_def[i75] - 0.2 / 0.41) < 0.08, f'{d_thin[i75]} / {d_def[i75]}')
check('tadpole: head size / tail size scales the head and leaves the tail alone (body depth in proportion to the ratio; the tail from half the length on identical for every ratio)',
      abs(d_small[i20] / d_def[i20] - 1.9 / 2.354) < 0.1 and (d_small[int(c0 + 0.5 * Lin):] == d_def[int(c0 + 0.5 * Lin):]).all() and (d_big[int(c0 + 0.5 * Lin):] == d_def[int(c0 + 0.5 * Lin):]).all(), f'body depth {d_small[i20]} / {d_def[i20]} for ratios 1.9 / 2.354')
huge = shape_mask('tadpole', G, 0.41, 3.0)
check('tadpole: a head larger than the box is limited to it (ratio 3 with the default tail depth), still with a margin and symmetric', column_depths(huge).max() <= Hin + 1 and not (huge[0].any() or huge[-1].any() or huge[:, 0].any() or huge[:, -1].any()) and np.array_equal(huge, huge[::-1]), f'depth {column_depths(huge).max()} of {Hin:.0f}')
check('tadpole: margin all round and symmetric for every combination of the two parameters tried (3 grids x 3 tail depths x 3 ratios)',
      all(not (mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any()) and np.array_equal(mk, mk[::-1]) for mk in (shape_mask('tadpole', gg, pp, qq) for gg in [(40, 150), (23, 85), (11, 51)] for pp in (None, 0.2, 0.7) for qq in (1.5, 2.354, 5.0))))

# planarian (dorsal view; target 4.70 times longer than high): a pointed head with two small auricle spikes, a notch behind them, a spindle-shaped body, a tail tapering to a point
G2 = target_grid(120, 4.70); pl = shape_mask('planarian', G2); dp = column_depths(pl); cp = np.where(dp > 0)[0]; p0, p1 = cp[0], cp[-1]; Hin2 = G2[0] - 2 * max(1.0, 0.04 * G2[0]); Lp = p1 - p0 + 1
atp = lambda s: dp[int(p0 + s * (Lp - 1))]
i_spike = int(np.argmax(dp[: p0 + int(0.3 * Lp)]))
check('planarian: pointed head, auricle spikes (the tallest points, at 12-16% of the length) with a notch behind them (at least 12% of the height narrower at 20%), widest body at 40-58%, one piece',
      dp[p0] <= 0.1 * Hin2 and 0.12 <= (i_spike - p0) / Lp <= 0.16 and dp[i_spike] - atp(0.2) >= 0.12 * Hin2 and atp(0.5) >= 0.9 * dp[i_spike] and _ndi0.label(pl)[1] == 1,
      f'spike at {(i_spike - p0) / Lp:.3f}: {dp[i_spike]} px, at 20%: {atp(0.2)}, body {atp(0.5)}, nose {dp[p0]} of {Hin2:.0f}')
check('planarian: the body widens smoothly from the notch to its widest part and tapers to a point (the depth never drops by more than 1 pixel up to 40% or rises after 58%)',
      np.all(np.diff(dp[int(p0 + 0.2 * Lp):int(p0 + 0.4 * Lp)]) >= -1) and np.all(np.diff(dp[int(p0 + 0.58 * Lp):p1 + 1]) <= 1) and dp[p1] <= 3, f'tip {dp[p1]} px')
edge_ok = True
for Rr, Cc in [(40, 200), (22, 96), (11, 48), (44, 204), (23, 105)]:
    mk = shape_mask('planarian', (Rr, Cc)); edge_ok &= not (mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any()) and np.array_equal(mk, flip_long(mk))
check('planarian: a clear margin all round and symmetric about the long axis (5 grids, odd and even)', edge_ok)
check('planarian: head at the left end of a wide box and at the top of a tall one (transposition)', np.array_equal(shape_mask('planarian', (G2[1], G2[0])), pl.T))
check('planarian: has no parameter', SHAPES['planarian'][1] is None and np.array_equal(shape_mask('planarian', (40, 200), 0.3), shape_mask('planarian', (40, 200))))

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
