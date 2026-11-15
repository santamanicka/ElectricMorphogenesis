"""Tests of the position-dependent field range: python test_localrange.py

(1) the local thickness of the tissue: what it should be on simple shapes; (2) a range map in the model: a uniform map is the single range, a cell whose range is exactly one of the levels senses
that level's field, the interpolation is linear between levels, narrower regions get smaller ranges and the field they sense is that of a smaller range; (3) a run stays finite and confined to the tissue.
"""
import sys
import numpy as np
from potentia import Potentia
from shapes import shape_mask
from localrange import local_thickness, range_map
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
