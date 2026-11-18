"""Tests of the pattern metrics (complexity.py): positional information and persistent features of G_pol.   python test_complexity.py"""
import sys
import numpy as np
from scipy import ndimage
from shapes import shape_mask
from complexity import positional_information, features, pattern_metrics, cell_means, block_pixels, sample_offsets

fails = 0


def check(name, cond, extra=''):
    global fails
    if not cond:
        fails += 1
    print(('PASS ' if cond else 'FAIL ') + name + (('   ' + extra) if extra else ''))


def stripes(mk, m, P):                                  # transverse square-wave bands of period P (um), levels 0 and 2
    R, C = mk.shape; x = (np.arange(C) + 0.5) * (10.0 / m); return np.tile(1 + np.sign(np.sin(2 * np.pi * x / P))[None, :], (R, 1))


def domains(mk, m, S, rng, nbins=8):                    # random domains of mean size S (um), a random level in each
    R, C = mk.shape; px = 10.0 / m; n = max(int(R * C * px * px / (S * S)), 1); pts = np.stack([rng.uniform(0, R, n), rng.uniform(0, C, n)], 1); lev = rng.choice(np.linspace(0.125, 1.875, nbins), n)
    I, J = np.indices((R, C)); d = (I[..., None] - pts[:, 0]) ** 2 + (J[..., None] - pts[:, 1]) ** 2; return lev[np.argmin(d, axis=2)]


rng = np.random.default_rng(0)
mk = shape_mask('tadpole', (100, 260)); R, C = mk.shape
check('block: 20 um = 2 cells = 2m pixels (4, 6, 8 for m = 2, 3, 4), at least 4 pixels (40 um at m = 1); 4 x 4 sample lattice at the centres of the quarters', [block_pixels(m) for m in (1, 2, 3, 4)] == [4, 4, 6, 8] and list(sample_offsets(4)) == [0, 1, 2, 3] and list(sample_offsets(8)) == [1, 3, 5, 7])
r0 = pattern_metrics(np.where(mk, 1.0, 0.0), mk, 2)
check('a uniform tissue has no positional information and no features', r0['bits'] == 0.0 and r0['features'] == 0 and r0['cells'] > 2000, str(r0))
raw = [positional_information(rng.uniform(0, 2, mk.shape), mk, 2, raw=True) for _ in range(20)]
check('random noise has no positional information: the leave-one-out estimate is below zero in all of 20 draws (mean %.2f), reported as 0' % np.mean(raw), max(raw) < -0.1 and positional_information(rng.uniform(0, 2, mk.shape), mk, 2) == 0.0)
gr = np.tile(np.linspace(0, 2, C)[None, :], (R, 1)); rg = pattern_metrics(np.where(mk, gr, 0), mk, 2)
check('a smooth gradient from head to tail: at least 2 bits of positional information (3 bits is the largest possible) and exactly 2 features (one peak, one basin)', rg['bits'] >= 2.0 and rg['bits'] <= 3.0 and rg['features'] == 2 and rg['peaks'] == 1, str(rg))
rect = np.ones((40, 200), bool)
res = {P: pattern_metrics(stripes(rect, 2, P), None, 2) for P in (40, 60, 80, 120, 20)}
check('transverse bands (the bands fall on the blocks: the phase of the pattern matters, see below): at most the 1 bit of the two levels, at least 0.9 bit when whole bands fill the blocks (40, 80 and 120 um), 0.5 bit for 60 um, and 0 for a period of 20 um (finer than the blocks)',
      all(0.9 <= res[P]['bits'] <= 1.0 for P in (40, 80, 120)) and 0.5 <= res[60]['bits'] <= 1.0 and res[20]['bits'] == 0.0, ', '.join(f'{P} um: {res[P]["bits"]:.2f}' for P in (20, 40, 60, 80, 120)))
shifts = [positional_information(stripes(rect, 2, 40)[:, np.arange(200) % 200] if s == 0 else np.roll(stripes(rect, 2, 40), s, axis=1), None, 2) for s in range(4)]
rd = domains(mk, 2, 80, np.random.default_rng(1)); sh = [positional_information(np.roll(np.roll(rd, dx, axis=1), dy, axis=0), np.roll(np.roll(mk, dx, axis=1), dy, axis=0), 2) for dx in range(4) for dy in range(4)]
check('phase of the blocks: perfectly regular 40 um bands, which are as fine as the blocks can resolve, read 0.9 bit or 0 depending on where they fall (shifts of 0-3 pixels: %s), but an irregular pattern (random domains of 80 um) changes by less than 0.03 bit over 16 shifts (sd %.3f)' % (' '.join(f'{x:.2f}' for x in shifts), np.std(sh)),
      max(shifts) > 0.9 and min(shifts) < 0.1 and np.std(sh) < 0.03 and max(sh) - min(sh) < 0.1)
check('transverse bands: two features per period (a peak and a basin; one more or fewer at the ends) on a 40 x 200-pixel rectangle', all(abs(res[P]['features'] - 2 * (200 * 5.0 / P)) <= 2 for P in (40, 60, 80, 120)), ', '.join(f'{P} um: {res[P]["features"]} (expected {2 * 200 * 5.0 / P:.0f})' for P in (40, 60, 80, 120)))
dm = pattern_metrics(domains(mk, 2, 80, rng), mk, 2)
check('random domains of 80 um with 8 levels (3 bits): at least 2 bits (the estimator is a lower bound), a few features', 2.0 <= dm['bits'] <= 3.0 and 3 <= dm['features'] <= 40, str(dm))
rin = pattern_metrics(np.clip(1 + 0.3 * rng.standard_normal(mk.shape), 0, 2), mk, 2); rno = pattern_metrics(rng.uniform(0, 2, mk.shape), mk, 2)
check('a random initial condition (G_pol = 1 + 0.3 N(0,1)) has no positional information and fewer than 1 feature per 100 cells; full-range white noise has a feature in 5-10% of the cells and no positional information (many features without organisation)',
      rin['bits'] == 0.0 and rin['per100'] < 1.0 and rno['bits'] == 0.0 and 5.0 <= rno['per100'] <= 10.0, f"random IC {rin['features']} features ({rin['per100']:.2f} per 100 cells); white noise {rno['features']} ({rno['per100']:.1f} per 100 cells)")
# comparability: the same physical pattern on other tissues and resolutions
tis = [('tadpole', 50, 130, 2), ('tadpole', 22, 100, 2), ('planarian', 24, 106, 2), ('ellipse', 30, 60, 2), ('disc', 40, 40, 2), ('tadpole', 30, 80, 3), ('tadpole', 25, 65, 4)]
bb, fb, dd = [], [], []
for shp, nx, ny, m in tis:
    mk_ = shape_mask(shp, (nx * m, ny * m)); bb.append(pattern_metrics(np.where(mk_, stripes(mk_, m, 60), 0), mk_, m)['bits'])
    rs = [pattern_metrics(np.where(mk_, domains(mk_, m, 60, np.random.default_rng(s)), 0), mk_, m) for s in range(6)]; dd.append((np.mean([r['bits'] for r in rs]), np.mean([r['per100'] for r in rs]), rs[0]['cells']))
check('comparable across tissues: the same 60 um bands on 7 tissues (900 to 2600 cells; tadpole, planarian, ellipse, disc; m = 2, 3, 4) give positional information within 0.08 bit; random 60 um domains give 2.2 to 2.4 bits (within 0.2) and features per 100 cells within 25% of each other (for tissues of 1400 cells and more)',
      max(bb) - min(bb) <= 0.08 and max(d[0] for d in dd) - min(d[0] for d in dd) <= 0.2 and max(d[1] for d in dd if d[2] >= 1400) <= 1.25 * min(d[1] for d in dd if d[2] >= 1400),
      'bands: ' + ' '.join(f'{x:.2f}' for x in bb) + ';  domains bits ' + ' '.join(f'{d[0]:.2f}' for d in dd) + ', per 100 cells ' + ' '.join(f'{d[1]:.2f}' for d in dd))
mk_s = shape_mask('tadpole', (22, 100)); rs_s = [pattern_metrics(np.where(mk_s, domains(mk_s, 2, 60, np.random.default_rng(s)), 0), mk_s, 2) for s in range(6)]
check('small tissues read lower in bits and higher in features per 100 cells (a tissue of 220 cells: 1.7-2.0 bits against 2.2-2.4 and about 2 features per 100 cells against 1): compare tissues of similar size',
      1.5 <= np.mean([r['bits'] for r in rs_s]) <= 2.0 and np.mean([r['per100'] for r in rs_s]) > 1.4, f"{np.mean([r['bits'] for r in rs_s]):.2f} bits, {np.mean([r['per100'] for r in rs_s]):.2f} per 100 cells, {rs_s[0]['cells']} cells")
# the same field as all-True mask and as None; mirror image
g_ = domains(rect, 2, 60, np.random.default_rng(4)); a = pattern_metrics(g_, None, 2); a2 = pattern_metrics(g_, np.ones(g_.shape, bool), 2)
check('no mask and an all-true mask give the same numbers', a == a2)
gm = domains(mk, 2, 60, np.random.default_rng(6)); am = pattern_metrics(np.where(mk, gm, 0), mk, 2); bm = pattern_metrics(np.where(mk[::-1], gm[::-1], 0), mk[::-1], 2)
check('mirror image (flipped about the long axis, mask included): the same number of features and the same positional information within 0.05 bit (the block grid is anchored at the corner)', am['features'] == bm['features'] and abs(am['bits'] - bm['bits']) < 0.05, f"{am['features']} / {bm['features']} features, {am['bits']:.3f} / {bm['bits']:.3f} bits")
# persistence: the contrast needed for a feature
def row(vs):
    g = np.array([vs], float); return features(g, None, 1)
check('persistence of at least 0.5 (a quarter of the range 0 to 2) is needed: a second peak 0.1 above its saddle is not a feature (1 peak), one 0.7 above it is (2 peaks); a range under 0.5 has no features, a bump of exactly 0.5 does (1 peak, 2 basins)',
      row([0.0, 1.0, 0.8, 0.9, 0.0])[:3] == (3, 1, 2) and row([0.0, 1.0, 0.2, 0.9, 0.0])[:3] == (5, 2, 3) and row([1.0, 1.2, 1.4])[0] == 0 and row([1.0, 1.5, 1.0])[:3] == (3, 1, 2), f"{row([0.0, 1.0, 0.8, 0.9, 0.0])}, {row([0.0, 1.0, 0.2, 0.9, 0.0])}, {row([1.0, 1.2, 1.4])}, {row([1.0, 1.5, 1.0])}")
# two separate islands each have their own extremes
isl = shape_mask('dumbbell', (24, 72), 0.0); gi = np.where(isl, np.tile(np.linspace(0, 2, 72)[None, :], (24, 1)) % 1.0 * 2, 0.0); ri = pattern_metrics(gi, isl, 2)
check('separate islands (dumbbell with neck 0) each count their own peak and basin', ri['peaks'] >= 2 and ri['basins'] >= 2, str(ri))
# g outside [0, 2] is clipped into the end bins; a tissue too small for the lattice reads 0
check('G_pol outside 0 to 2 goes into the end bins (no error), and a tissue smaller than one block has 0 bits', np.isfinite(positional_information(np.where(mk, 3.0 * np.sin(np.arange(260) / 7.0)[None, :], 0), mk, 2)) and positional_information(np.ones((6, 6)), np.ones((6, 6), bool), 2) == 0.0)
cmv, vl = cell_means(np.arange(16.0).reshape(4, 4), np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], bool), 2)
check('cell means: the mean of the tissue pixels of each m x m cell, a cell is tissue if at least half of its pixels are', cmv[0, 0] == 2.5 and vl[0, 0] and not vl[0, 1] and vl[1, 0] and cmv[1, 0] == 10.0 and not vl[1, 1], f'{cmv.tolist()} {vl.tolist()}')
print('\nALL PASS' if fails == 0 else f'\n{fails} FAILURES')
sys.exit(1 if fails else 0)
