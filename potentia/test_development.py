"""Tests of the development option (a prescribed morph from an egg to a tadpole): python test_development.py"""
import sys
import numpy as np
from scipy import ndimage
from potentia import Potentia
from development import Development, STAGES, stage_mask, signed_distance, grow_state
from shapes import shape_mask
from sweeps import STIG

ok = True


def check(name, cond, extra=''):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'   {extra}' if extra else ''))


# 1. signed distance: exact against brute force (the grid is padded by one outside pixel)
rng = np.random.default_rng(0)
worst = 0.0
for R, C in [(9, 12), (14, 7), (16, 16)]:
    m = rng.uniform(size=(R, C)) > 0.5
    m[0, :] = True; m[:, 0] = True                                 # touching the grid edge: the padding ring is the nearest outside
    P = np.pad(m, 1, constant_values=False)
    ii, jj = np.mgrid[:R + 2, :C + 2]
    ins = np.argwhere(P); outs = np.argwhere(~P)
    sdf = signed_distance(m)
    for i in range(R):
        for j in range(C):
            tgt = outs if m[i, j] else ins
            d = np.sqrt(((tgt - np.array([i + 1, j + 1])) ** 2).sum(1).min())
            worst = max(worst, abs(abs(sdf[i, j]) - d))
check('signed distance equals a brute-force search (3 random masks, grid edge touching)', worst == 0.0, f'max difference {worst}')

# 2. keyframes and the morph
R, C = 100, 260                                                      # 50 x 130 cells at m = 2
dev = Development((R, C), duration=300.0)
exact = all(np.array_equal(dev.mask_at(t), dev.masks[k]) for k, t in enumerate(dev.taus))
check('at every keyframe time the mask is exactly the keyframe shape', exact)
tad = stage_mask(STAGES[-1], R, C)
check('the last keyframe is the tadpole preset on the full grid', np.array_equal(tad, shape_mask('tadpole', (R, C))))
taus = np.linspace(0, 1, 201)
areas, comps, holes, inside_grid = [], [], [], True
for t in taus:
    m = dev.mask_at(t)
    areas.append(int(m.sum()))
    comps.append(ndimage.label(m)[1])
    holes.append(int((ndimage.binary_fill_holes(m) & ~m).sum()))
areas = np.array(areas)
check('egg is a small disc (about 17 cells across) and the tadpole fills about 60% of the grid', abs(areas[0] - np.pi * (R / 6) ** 2) / areas[0] < 0.05 and 0.4 < areas[-1] / (R * C) < 0.8, f'egg {areas[0]} px, tadpole {areas[-1]} px of {R * C}')
check('the tissue is one connected piece with no holes at every one of 201 times', max(comps) == 1 and max(holes) == 0, f'components max {max(comps)}, hole pixels max {max(holes)}')
nest = [int((dev.masks[k] & ~dev.masks[k + 1]).sum()) for k in range(len(dev.masks) - 1)]
sizes = [int(m.sum()) for m in dev.masks]
check('every keyframe shape lies inside the next one, except that the neurula\'s back may protrude by a row or two behind the tailbud\'s head (pixels not nested; at most 2% of the neurula)',
      all(n == 0 for k, n in enumerate(nest) if k != 3) and nest[3] <= 0.02 * sizes[3], f'{nest} of {sizes}')
dec = int((np.diff(areas) < 0).sum())
check('the area never decreases along the sequence (pure growth)', dec == 0, f'{dec} decreasing steps; area {areas[0]} -> {areas[-1]}')
st = [dev.stage_at(t)['name'] for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
check('stage labels along the sequence', st == ['Egg', 'Blastula', 'Gastrula', 'Neurula', 'Tailbud', 'Tadpole'], ','.join(st))
# other grid sizes (the sequence must work on any grid at least ~2.7 heights wide, and degrade sensibly on narrower ones)
good = True
for Np in [(40, 108), (60, 160), (120, 330), (50, 50), (30, 200)]:
    d2 = Development(Np)
    for t in (0.0, 0.3, 0.6, 0.9, 1.0):
        m = d2.mask_at(t)
        good &= bool(m.any()) and ndimage.label(m)[1] == 1
check('the sequence builds on five other grid sizes (connected, non-empty), including a square and a very wide grid', good)

# 3. new tissue inherits; removed tissue is zero
V = np.zeros((5, 5)); g = np.zeros((5, 5)); old = np.zeros((5, 5), bool)
old[2, 1] = old[2, 3] = True; V[2, 1], V[2, 3] = 0.0, 10.0; g[2, 1], g[2, 3] = 1.0, 2.0
new = old.copy(); new[2, 2] = True
Vo, go = grow_state(V, g, old, new)
check('a new pixel between two tissue pixels takes the mean of its neighbours', Vo[2, 2] == 5.0 and go[2, 2] == 1.5 and Vo[2, 1] == 0.0 and Vo[2, 3] == 10.0)
old = np.zeros((7, 7), bool); old[3, 3] = True; V = np.zeros((7, 7)); g = np.zeros((7, 7)); V[3, 3], g[3, 3] = 7.0, 1.5
new = np.zeros((7, 7), bool); new[2:5, 2:5] = True
Vo, go = grow_state(V, g, old, new)
check('growth spreads the state layer by layer (3x3 block grown from one pixel is uniform)', np.all(Vo[2:5, 2:5] == 7.0) and np.all(go[2:5, 2:5] == 1.5) and Vo.sum() == 63.0)
rngs = np.random.default_rng(3)
old = rngs.uniform(size=(9, 11)) > 0.7; new = old | (ndimage.binary_dilation(old, iterations=2)); V = rngs.uniform(size=(9, 11)) * old; g = rngs.uniform(size=(9, 11)) * old
A, B = grow_state(V, g, old, new)
Af, Bf = grow_state(V[::-1, ::-1], g[::-1, ::-1], old[::-1, ::-1], new[::-1, ::-1])
check('the result does not depend on pixel order (the flipped problem gives the flipped result to round-off)', np.allclose(A, Af[::-1, ::-1], rtol=1e-14, atol=1e-15) and np.allclose(B, Bf[::-1, ::-1], rtol=1e-14, atol=1e-15), f'max difference {max(np.abs(A - Af[::-1, ::-1]).max(), np.abs(B - Bf[::-1, ::-1]).max()):.1e}')
shrunk = old & ~ndimage.binary_erosion(old)
Vs, gs = grow_state(V, g, old, old & ~shrunk)
check('removed pixels are exactly zero and persisting pixels are unchanged', np.all(Vs[shrunk] == 0) and np.all(Vs[old & ~shrunk] == V[old & ~shrunk]))
old = np.zeros((6, 6), bool); old[0, 0] = True; new = np.zeros((6, 6), bool); new[4:6, 4:6] = True
Vo, go = grow_state(np.ones((6, 6)), np.ones((6, 6)), old, new, fill=(-30.0, 0.8))
check('a new island that touches no tissue gets the background state', np.all(Vo[4:6, 4:6] == -30.0) and np.all(go[4:6, 4:6] == 0.8) and Vo[0, 0] == 0.0)

# 4. the model with a changing mask
Np = (20, 54)
Pd = Potentia(size_cells=(10, 27), m=2, mask=Development(Np, 30.0).mask_at(0.0), **STIG)
V, g = Pd.initial_from_g(1.0, 'dep')
d3 = Development(Np, 30.0); bad = 0; maxarea = 0; fin = True
for n in range(1, 3001):
    V, g = Pd.step(V, g)
    if n % 50 == 0:
        mnew = d3.mask_at(d3.tau(n * Pd.dt))
        if not np.array_equal(mnew, Pd.mask if Pd.mask is not None else np.ones(Np, bool)):
            before_inside = V.copy()
            V, g = Pd.set_mask(mnew, V, g, (-9.24e-3, 1.0))
            keep = (before_inside != 0) & mnew
            bad += int(not np.array_equal(V[keep], before_inside[keep]))
    fin &= bool(np.isfinite(V).all() and np.isfinite(g).all())
    bad += int(np.abs(V[~(Pd.mask if Pd.mask is not None else np.ones(Np, bool))]).sum() != 0)
check('a 30 s development run on 10x27 cells: finite, nothing outside the tissue at any step, persisting pixels unchanged at every mask change', fin and bad == 0, f'violations {bad}; tissue {int(Pd.mask.sum())} px at the end')
check('... and the tissue has grown to the final shape', np.array_equal(Pd.mask, d3.masks[-1]))

# 5. set_range equals building the model with that range
Pa = Potentia(size_cells=(8, 10), m=2, **STIG); Pa.set_range(9.0)
Pb = Potentia(size_cells=(8, 10), m=2, **{**STIG, 'fieldScreenSize': 9.0})
rng = np.random.default_rng(4); V0 = rng.uniform(-55e-3, -5e-3, Pa.Np); g0 = rng.uniform(0.2, 1.8, Pa.Np)
Va, ga, Vb, gb = V0.copy(), g0.copy(), V0.copy(), g0.copy()
for _ in range(200):
    Va, ga = Pa.step(Va, ga); Vb, gb = Pb.step(Vb, gb)
check('set_range(R_s) gives the same dynamics as a model built with that range (200 steps)', np.abs(Va - Vb).max() < 1e-15 and abs(Pa.R - Pb.R) == 0, f'max|dV| {np.abs(Va - Vb).max():.1e}')

# 6. field-range schedule
h_um = 5.0
mask = dev.mask_at(0.0)
r_const = dev.range_um('const', 0.0, mask, h_um, const_um=28.3)
r_egg = dev.range_um('prop', 0.0, dev.masks[0], h_um, pct=25.0); r_tad = dev.range_um('prop', 1.0, dev.masks[-1], h_um, pct=25.0)
ext_egg = min(np.ptp(np.where(dev.masks[0].any(1))[0]) + 1, np.ptp(np.where(dev.masks[0].any(0))[0]) + 1) * h_um
check('range schedule: constant stays; proportional follows the shorter extent of the tissue (egg < tadpole)', r_const == 28.3 and abs(r_egg - 0.25 * ext_egg) < 1e-9 and r_tad > 2 * r_egg, f'egg {r_egg:.1f} um, tadpole {r_tad:.1f} um')
su = [10, 20, 30, 40, 50, 60]
check('per-stage range: equals the stage value at each keyframe, in between by smoothstep, and never below 3 pixels', all(abs(dev.range_um('stage', t, mask, h_um, stage_um=su) - max(v, 15.0)) < 1e-12 for t, v in zip(dev.taus, su))
      and su[0] < dev.range_um('stage', 0.05, mask, h_um, stage_um=su) < su[1] and dev.range_um('const', 0.0, mask, h_um, const_um=2.0) == 15.0)

print('\nALL PASS' if ok else '\nFAILURES')
sys.exit(0 if ok else 1)
