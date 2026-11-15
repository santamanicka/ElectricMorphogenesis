"""Tests of the position-dependent field range: python test_localrange.py

(1) the local thickness of the tissue: what it should be on simple shapes; (2) a range map in the model: a uniform map is the single range, a cell whose range is exactly one of the levels senses
that level's field, the interpolation is linear between levels, narrower regions get smaller ranges and the field they sense is that of a smaller range; (3) a run stays finite and confined to the tissue.
"""
import sys
import numpy as np
from potentia import Potentia
from shapes import shape_mask
from localrange import local_thickness, local_width, local_gaussian, gaussian_blur, range_map, _d2_to_zero
from scipy import ndimage
from sweeps import STIG

ok = True


def check(name, cond, extra=''):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + (f'   {extra}' if extra else ''))


# ---- 1. local thickness
T = local_thickness(np.ones((22, 60), bool))
mid = T[:, 20:40]
check('rectangle 22 x 60: thickness is the height (22 px) everywhere except the corners', (mid == 22).all() and (T == 22).mean() > 0.9 and T.min() >= 6, f'{(T == 22).mean():.2f} of the pixels at 22, corners down to {T.min()}')
Td = local_thickness(shape_mask('disc', (40, 40)))
mk = shape_mask('disc', (40, 40))
check('disc of 40 px: thickness is the diameter (38 px, pixel centres) over the whole disc, rim included', (Td[mk] == Td.max()).all() and 37 <= Td.max() <= 40, f'{Td[mk].min()}..{Td.max()}')
Ts = local_thickness(shape_mask('stadium', (24, 72)))
check('stadium (capsule, 24 px high): thickness uniform within one pixel of the height', Ts[shape_mask('stadium', (24, 72))].min() >= 22 and Ts.max() <= 24, f'{Ts[shape_mask("stadium", (24, 72))].min()}..{Ts.max()}')
mkd = shape_mask('dumbbell', (24, 72)); Tb = local_thickness(mkd)
check('dumbbell: the discs are thick (22 px), the neck is thin (about 0.3 of the disc: 8 px), and the thin part is in the middle', Tb[12, 10] == 22 and Tb[12, 62] == 22 and Tb[12, 36] <= 9 and Tb[12, 36] >= 6, f'disc {Tb[12, 10]}, neck {Tb[12, 36]}')
mka = shape_mask('annulus', (40, 40)); Ta = local_thickness(mka)
check('annulus: thickness is the ring width (half the radius, 10 px) all along the ring', abs(float(np.median(Ta[mka])) - 10) <= 1 and (Ta[mka] >= 8).mean() > 0.95, f'median {np.median(Ta[mka]):.0f}')
mkt = shape_mask('tadpole', (22, 100)); Tt = local_thickness(mkt)
body, neck, tail = Tt[11, 20], Tt[11, 50], Tt[11, 85]
check('tadpole: thick body (20 px), thinner neck, thin tail (2-3 px), decreasing along the length', body == 20 and neck < body and tail <= 3 and Tt[11, 30] >= Tt[11, 45] >= Tt[11, 60] >= Tt[11, 80], f'body {body}, neck {neck}, tail {tail}')
mkp = shape_mask('planarian', (22, 96)); Tp = local_thickness(mkp)
check('planarian: the middle of the body is the thickest part and the tail end is thin', Tp[11, 48] == Tp.max() and Tp[11, 92] <= 8 and Tp[11, 48] >= 18, f'middle {Tp[11, 48]}, tail end {Tp[11, 92]}')
check('thickness is 0 outside the tissue and positive inside', (Tt[~mkt] == 0).all() and (Tt[mkt] > 0).all())
check('thickness is mirror-symmetric for a symmetric shape (tadpole about its long axis)', np.array_equal(Tt, Tt[::-1]))
check('the range map is pct % of the local width, at least 3 pixels', np.allclose(range_map(mkt, 25.7, 5e-6)[11, 20], 0.257 * 20 * 5e-6) and range_map(mkt, 25.7, 5e-6)[11, 85] == 3 * 5e-6)

# ---- 1b. the smooth width: falls to every boundary, is the thickness on the middle line of a broad region, has no steps
def neighbour_jump(F, sel):
    worst = 0.0
    for ax in (0, 1):
        b = np.roll(F, -1, axis=ax); ok = sel & np.roll(sel, -1, axis=ax); sl = [slice(None)] * 2; sl[ax] = slice(0, -1)
        rel = (np.abs(F - b) / np.maximum(np.maximum(F, b), 1.0))[tuple(sl)][ok[tuple(sl)]]
        worst = max(worst, float(rel.max()) if rel.size else 0.0)
    return worst
Wr = local_width(np.ones((22, 60), bool))
check('smooth width: on the middle line of a 22-pixel strip it is the thickness (22 px), and it falls to the edge: below 45% of it at the edge row, 60% one row in, rising all the way', abs(Wr[11, 30] - 22.0) < 0.5 and Wr[0, 30] < 0.45 * Wr[11, 30] and Wr[1, 30] < 0.6 * Wr[11, 30] and np.all(np.diff(Wr[:12, 30]) >= 0) and Wr[0, 30] < Wr[5, 30] < Wr[10, 30] + 1e-9, f'centre {Wr[11, 30]:.1f}, edge row {Wr[0, 30]:.1f}, one row in {Wr[1, 30]:.1f}')
mkd2 = shape_mask('disc', (40, 40)); Wd = local_width(mkd2)
row = Wd[20, 20:]; row = row[row > 0]
check('smooth width: along a radius of a disc it decreases monotonically from the centre (about the diameter) to the rim', np.all(np.diff(row) <= 1e-12) and abs(row[0] - 38.8) < 1.0 and row[-1] < 0.45 * row[0], f'{row[0]:.1f} -> {row[-1]:.1f}')
worst_ratio = 0.0; ratios = {}
for name, Np in [('disc', (40, 40)), ('rectangle', (22, 60)), ('stadium', (24, 72)), ('ellipse', (24, 40)), ('tadpole', (22, 100)), ('planarian', (22, 96)), ('dumbbell', (24, 72)), ('annulus', (40, 40))]:
    mk_ = shape_mask(name, Np); W_ = local_width(mk_); edge = mk_ & ~ndimage.binary_erosion(mk_, structure=np.ones((3, 3)), border_value=0); inner = mk_ & ~edge
    ratios[name] = float(np.median(W_[edge]) / np.median(W_[inner])); worst_ratio = max(worst_ratio, ratios[name])
check('smooth width: in every shape the cells at the edge have a median width below 70% of that of the interior cells (the thickness alone gives 100% for a disc or a capsule; the ratio is higher where the whole width is a few pixels, as in a ring)', worst_ratio < 0.7, ', '.join(f'{k} {v:.2f}' for k, v in ratios.items()))
mkt2 = shape_mask('tadpole', (22, 100)); Wt = local_width(mkt2)
check('smooth width: still local thickness: the middle of the tadpole body is about its thickness (20 px), the middle of the tail 2 px, the neck in between, and an edge cell of the body is smaller than the middle of the body', abs(Wt[11, 20] - 20.0) < 0.5 and Wt[11, 85] <= 3 and 3 < Wt[11, 50] < 20 and Wt[1, 20] < 0.5 * Wt[11, 20], f'body {Wt[11, 20]:.1f}, neck {Wt[11, 50]:.1f}, tail {Wt[11, 85]:.1f}, body edge {Wt[1, 20]:.1f}')
smoother = []
for name, Np in [('tadpole', (22, 100)), ('planarian', (22, 96)), ('dumbbell', (24, 72))]:
    mk_ = shape_mask(name, Np); d_ = np.sqrt(_d2_to_zero(np.pad(mk_, 1, constant_values=False))[1:-1, 1:-1]); deep = mk_ & (d_ >= 4)
    smoother.append((neighbour_jump(local_width(mk_), deep), neighbour_jump(local_thickness(mk_).astype(float), deep)))
check('smooth width has no steps: between neighbouring cells more than 4 px from an edge it never changes by more than 12%, where the thickness changes by up to 48% at the neck of the dumbbell', all(s <= 0.12 for s, _ in smoother) and all(s < t for s, t in smoother), ', '.join(f'{s:.2f} vs {t:.2f}' for s, t in smoother))
Rs_ = range_map(mkt2, 25.7, 5e-6, 'smooth')
check('range map in smooth mode: pct % of the smooth width at least 3 pixels (15 um to 25.7 um for the tadpole body)', np.allclose(Rs_[11, 20], 0.257 * Wt[11, 20] * 5e-6) and abs(Rs_[mkt2].min() - 15e-6) < 1e-12 and abs(Rs_[mkt2].max() * 1e6 - 25.7) < 0.01, f'{Rs_[mkt2].min() * 1e6:.1f} to {Rs_[mkt2].max() * 1e6:.2f} um')
try:
    range_map(mkt2, 25.7, 5e-6, 'sideways'); check('an unknown mode is rejected', False)
except ValueError:
    check('an unknown mode is rejected', True)

# ---- 1c. Gaussian-smoothed thickness: the stepped thickness without its steps
Tt = local_thickness(mkt2).astype(float); Gt = local_gaussian(mkt2)
check('Gaussian thickness: a constant is unchanged (normalised over the tissue pixels, so the edge does not pull it down), and the result lies between the smallest and the largest thickness and is 0 outside',
      np.allclose(gaussian_blur(np.full(mkt2.shape, 7.0), mkt2, 2.5)[mkt2], 7.0, rtol=1e-12) and Tt[mkt2].min() - 1e-9 <= Gt[mkt2].min() and Gt[mkt2].max() <= Tt[mkt2].max() + 1e-9 and np.all(Gt[~mkt2] == 0), f"G in [{Gt[mkt2].min():.2f}, {Gt[mkt2].max():.2f}], T in [{Tt[mkt2].min():.0f}, {Tt[mkt2].max():.0f}]")
mkr = shape_mask('stadium', (24, 72)); Gr = local_gaussian(mkr)
check('Gaussian thickness keeps the plateau of a broad region to its rim: in a capsule (24 px high) the edge pixels away from the ends have the same width as the middle to 1%, the broad body of the tadpole keeps 20 px up to its edge row',
      np.all(np.abs(Gr[0:3, 30:42] / Gr[12, 36] - 1.0)[mkr[0:3, 30:42]] < 0.01) and abs(Gt[1, 20] - 20.0) < 1.0 and abs(Gt[11, 20] - 20.0) < 1e-9 and Gt[mkt2][:].max() > 19.99, f"capsule edge/middle {Gr[1, 36] / Gr[12, 36]:.4f}; tadpole body edge row {Gt[1, 20]:.2f}, middle {Gt[11, 20]:.2f}")
gj = []
for name, Np in [('tadpole', (22, 100)), ('tadpole', (50, 130)), ('planarian', (22, 96)), ('dumbbell', (22, 66)), ('triangle', (40, 40))]:
    mk_ = shape_mask(name, Np); f_ = lambda X: np.maximum(0.257 * X, 3.0)              # the range in pixels (25.7%, at least 3 px)
    gj.append((name, neighbour_jump(f_(local_gaussian(mk_)), mk_), neighbour_jump(f_(local_thickness(mk_).astype(float)), mk_)))
check('Gaussian thickness has no steps: the range of neighbouring cells never differs by more than 17% (the stepped thickness: up to 61%)', all(g <= 0.17 for _, g, _ in gj) and max(s for _, _, s in gj) > 0.4, ', '.join(f'{n} {g:.2f} vs {s:.2f}' for n, g, s in gj))
# a thin part next to a broad one is lifted a little towards it, a wider Gaussian lifts it further, a tiny one is the thickness
Gt_w = local_gaussian(mkt2, 0.25); Gt_n = local_gaussian(mkt2, 0.005)
check('Gaussian thickness: the width of the Gaussian is the smoothing length: the thin tail next to the body is lifted more by 25% than by 10%, and a tiny length (0.5%, one pixel) stays within 1.3 px of the thickness',
      Gt_w[11, 72] > Gt[11, 72] > Tt[11, 72] and np.abs(Gt_n - Tt)[mkt2].max() < 1.3, f"tail (column 72): T {Tt[11, 72]:.0f}, 10%: {Gt[11, 72]:.2f}, 25%: {Gt_w[11, 72]:.2f}; max|G(1 px) - T| {np.abs(Gt_n - Tt)[mkt2].max():.2f}")
Rg_ = range_map(mkt2, 25.7, 5e-6, 'gaussian')
check('range map in Gaussian mode: pct % of the smoothed thickness at least 3 pixels (15 um to 25.7 um for the tadpole body; the default smoothing length 10% is the argument sigma_frac = 0.10)',
      np.allclose(Rg_[11, 20], 0.257 * Gt[11, 20] * 5e-6) and abs(Rg_[mkt2].min() - 15e-6) < 1e-12 and abs(Rg_[mkt2].max() * 1e6 - 25.7) < 0.01 and np.allclose(range_map(mkt2, 25.7, 5e-6, 'gaussian', 0.25)[11, 40], 0.257 * Gt_w[11, 40] * 5e-6), f"{Rg_[mkt2].min() * 1e6:.1f} to {Rg_[mkt2].max() * 1e6:.2f} um")
try:
    local_gaussian(mkt2, 0.0); check('a smoothing length of zero is rejected', False)
except ValueError:
    check('a smoothing length of zero is rejected', True)

# ---- 2. the model
rng = np.random.default_rng(0)
P0 = Potentia(size_cells=(11, 50), m=2, mask=mkt, **STIG)
V = rng.uniform(-55e-3, -5e-3, P0.Np) * P0.maskf
e0 = P0.perceived_field(V)
Pu = Potentia(size_cells=(11, 50), m=2, mask=mkt, range_map=np.full(P0.Np, P0.R), **STIG)
check('a uniform range map is the single range (same perceived field to round-off, one level)', len(Pu._lv['R']) == 1 and np.abs(Pu.perceived_field(V) - e0).max() <= 1e-14 * np.abs(e0).max())
Rm = range_map(mkt, 60.0, P0.h)
Pl = Potentia(size_cells=(11, 50), m=2, mask=mkt, range_map=Rm, **STIG)
lv = Pl._lv; el = Pl.perceived_field(V)
check('the range map has several levels, evenly spaced from the smallest to the largest range (3 to 12 pixels)', len(lv['R']) >= 5 and abs(lv['lo'] / P0.h - 3.0) < 1e-9 and abs(lv['hi'] / P0.h - 12.0) < 1e-9 and np.allclose(np.diff(lv['R']), np.diff(lv['R'])[0]), f'{len(lv["R"])} levels: {[round(r / P0.h, 2) for r in lv["R"]]} px')
worst = 0.0
for j, Rj in enumerate(lv['R']):
    Pj = Potentia(size_cells=(11, 50), m=2, mask=mkt, **STIG); Pj.R = Rj; Pj._build_kernel()
    sel = (Rm == Rj) & mkt
    if sel.any():
        ej = Pj.perceived_field(V); worst = max(worst, float(np.abs(el[sel] - ej[sel]).max() / np.abs(ej[sel]).max()))
check('a cell whose range is exactly one of the levels senses that level\'s field', worst < 1e-12, f'worst relative difference {worst:.1e}')
# a block of pixels whose range is half way between two levels senses the average of the two levels' fields (linear interpolation in the range)
floor = np.sqrt(Potentia.epsilon)                                           # the perceived field never falls below the softening sqrt(epsilon) = 3.2e-8
R_lo, R_hi = lv['R'][2], lv['R'][3]; Rmid = 0.5 * (R_lo + R_hi)
Rm4 = Rm.copy(); block = np.zeros(P0.Np, bool); block[:, 12:40] = True; block &= mkt; Rm4[block] = Rmid
P4 = Potentia(size_cells=(11, 50), m=2, mask=mkt, range_map=Rm4, **STIG)
Pa, Pb = Potentia(size_cells=(11, 50), m=2, mask=mkt, **STIG), Potentia(size_cells=(11, 50), m=2, mask=mkt, **STIG)
Pa.R = R_lo; Pa._build_kernel(); Pb.R = R_hi; Pb._build_kernel()
expect = 0.5 * Pa.perceived_field(V) + 0.5 * Pb.perceived_field(V)
check('between two levels the sensed field is their linear interpolation in the range (a block of pixels half way between two levels)', block.sum() > 200 and len(P4._lv['R']) == len(lv['R']) and np.abs(P4.perceived_field(V)[block] - expect[block]).max() < 1e-12 * np.abs(expect[block]).max(), f'{int(block.sum())} pixels')
# the sensing range of a cell is its own range: a single charge, cells at known distances (the perceived field is 3.2e-8 where nothing is sensed)
def sensed(Pm_, charge_col, cells):
    q = np.zeros(P0.Np); q[11, charge_col] = -50e-3; return Pm_.perceived_field(q * P0.maskf)[11, cells]
fa = sensed(Pl, 11, slice(20, 24))                                            # body cells 9-13 px from a charge: the body's range is 12 px
fb = sensed(P0, 11, slice(20, 24))                                            # ... the single range is 5.7 px
check('a thick region senses farther than the single range: body cells 9-13 px from a charge sense it with the local range (12 px), not with the single range (5.7 px)', fa.min() > 100 * floor and fb.max() < 1.01 * floor, f'local {fa.min():.2e}, single {fb.max():.2e}, floor {floor:.1e}')
fc = sensed(Pl, 82, slice(88, 92))                                            # tail cells 6-10 px from a charge: the tail's range is 3 px
fd = sensed(P0, 82, slice(88, 92))
check('a thin region senses less far than the single range: tail cells 6-10 px from a charge do not sense it with the local range (3 px), but do with the single range (5.7 px at its near end)', fc.max() < 1.01 * floor and fd.max() > 100 * floor, f'local {fc.max():.2e}, single {fd.max():.2e}, floor {floor:.1e}')
far = sensed(Pl, 55, slice(78, 84))                                           # 23-29 px away: beyond every range
check('far from a charge (beyond the largest range) nothing is sensed', far.max() < 1.01 * floor)

# smooth mode in the model: a cell at the edge of the body senses a shorter range than a cell in the middle of the body (the thickness mode gives them the same range)
Rms = range_map(mkt, 60.0, P0.h, 'smooth'); Rmt = range_map(mkt, 60.0, P0.h, 'thickness')
Ps = Potentia(size_cells=(11, 50), m=2, mask=mkt, range_map=Rms, **STIG); Pt = Potentia(size_cells=(11, 50), m=2, mask=mkt, range_map=Rmt, **STIG)
qE = np.zeros(P0.Np); qE[6, 12] = -50e-3; qE = qE * P0.maskf                          # a charge in the body, 8.6 px from an edge cell (row 1) and from a middle cell (row 11) at column 19
ea, ma, eta = Ps.perceived_field(qE)[1, 19], Ps.perceived_field(qE)[11, 19], Pt.perceived_field(qE)[1, 19]
check('in smooth mode a body cell at the edge senses a charge 8.6 px away less than a middle cell at the same distance does (ranges 5.2 and 12 px); in thickness mode the edge cell senses it as well', bool(mkt[6, 12]) and ea < 1.01 * floor and ma > 100 * floor and eta > 100 * floor and Rms[1, 19] < 0.5 * Rms[11, 19] and Rmt[1, 19] == Rmt[11, 19], f'smooth: edge {ea:.1e}, middle {ma:.1e}; thickness: edge {eta:.1e}; ranges {Rms[1, 19] / P0.h:.1f} / {Rms[11, 19] / P0.h:.1f} px')

# ---- 3. a run
V1, g1 = Pl.initial_from_g(1.0, 'dep')
for _ in range(1000):
    V1, g1 = Pl.step(V1, g1)
check('a 10 s run with the local range is finite, confined to the tissue and has structure', np.isfinite(V1).all() and (V1[~mkt] == 0).all() and np.ptp(V1[mkt]) > 1e-3, f'Vmem spread {1e3 * np.ptp(V1[mkt]):.1f} mV')
for label, fn in [('periodic with a range map', lambda: Potentia(size_cells=(4, 4), m=2, periodic=True, range_map=np.full((8, 8), 3e-5), **STIG)),
                  ('range map of the wrong shape', lambda: Potentia(size_cells=(4, 4), m=2, range_map=np.ones((3, 3)), **STIG))]:
    try:
        fn(); check(f'{label} is rejected', False)
    except ValueError as e:
        check(f'{label} is rejected', True, str(e)[:60])
try:
    Pl.set_range(3.0); check('changing the single range while a map is set is rejected', False)
except ValueError:
    check('changing the single range while a map is set is rejected', True)
Pl.set_range_map(None); check('set_range_map(None) restores the single range', Pl._lv is None and np.allclose(Pl.perceived_field(V), e0))
print('\nALL PASS' if ok else '\nFAILURES')
sys.exit(0 if ok else 1)
