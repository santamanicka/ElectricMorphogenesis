"""Feature filters for the anatomy search: coarse-grained features of a pattern, compared with the same features of the target.

A pixel-by-pixel fit (anatomy_search.py) gives no credit to a structure that is present but a little out of register (segments at
a slightly different spacing or phase, a cord a pixel to the side), and so prefers plain patterns.  Here both the G_pol pattern and
the schematic target are first straightened into body coordinates (s along the axis, u across the body as a fraction of the local
half-width; NSR x NUR samples) and passed through the same bank of filters; what is compared is the filters' output, pooled over
position, not the pixels:

    coarse    the mean level in three lateral zones (midline, flank, edge) x 8 bins along the axis (signed): the body-scale layout
    cords     energy of stripes that run along the body: Gabor filters across the body at 1, 2, 3 and 4 cycles per body width,
              in 4 bins along the axis (head, front trunk, rear trunk, tail)  (lateral cords, a midline, a gut trunk)
    segments  energy of repeats along the body: Gabor filters along the axis at six wavelengths (2% to 15% of the body length) in
              the three lateral zones x the 4 bins  (commissures, gut lobes, myomeres)
    spots     energy of blob-sized structure (difference of Gaussians) in the three zones x the 4 bins  (eye cups, ganglion, brain)

The pooling is deliberately coarse.  A first version pooled in 24 bins along the axis; it then mattered exactly where along the
body each stripe or repeat fell, and richly structured frames scored no better than noise (0.06 to 0.13 against 0.08).

Energies are phase-free and sign-free, so a structure counts whether it is marked by high or by low G_pol.  The filter outputs are
then reduced to a short description (`summary`): how strong each kind of structure is, at what spacing, and roughly where; `match`
compares two descriptions.  Two earlier versions correlated the filter maps themselves (in 24 and then 4 axial bins); a frame with a
clearly segmented midline then scored 0 on `segments` against a target whose segments lie on the flanks at another spacing, so
visibly structured frames scored no better than noise.
Baselines to read a score against (printed by `python anatomy_features.py`): white noise, a plain midline stripe, a uniform
rim-to-centre gradient, and the other animal's target.
"""
import numpy as np
from scipy import ndimage as ndi
from anatomy_targets import layers, planarian_profile, tadpole_profile, TAIL_BASE, NS, NU

NSR, NUR = 240, 41
UMAX = 0.92
NBIN = 4                                                   # pooling bins along the axis for the energy channels: head, front trunk, rear trunk, tail
NBIN_COARSE = 8
WAVES = (0.02, 0.03, 0.045, 0.07, 0.10, 0.15)             # segment wavelengths, as fractions of the body length
CYCLES = (1, 2, 3, 4)                                      # stripes across the body, cycles per body width
ZONES = ((0.0, 0.25), (0.25, 0.65), (0.65, 1.01))          # |u|: midline, flank, edge
CHANNELS = ('coarse', 'cords', 'segments', 'spots')
_S = (np.arange(NSR) + 0.5) / NSR
_U = np.linspace(-UMAX, UMAX, NUR)


def rectifier(shape, Np, tail=TAIL_BASE, mask=None, pinch=False):
    """Index arrays that straighten a pixel grid into (s, u): rows, cols (NSR, NUR) and which samples fall on tissue."""
    R, C = Np
    b, a = R / 2.0, C / 2.0
    mg = max(1.0, 0.04 * R)
    prof = planarian_profile(_S, pinch) if shape == 'planarian' else tadpole_profile(_S, tail)
    hw = (b - mg) * prof
    x = mg + _S * (2 * a - 2 * mg)                                         # pixels from the left edge
    ci = np.clip(np.floor(x).astype(int), 0, C - 1)[:, None] + np.zeros((NSR, NUR), int)
    ri = np.clip(np.floor(R / 2.0 + _U[None, :] * hw[:, None]).astype(int), 0, R - 1)
    valid = (hw[:, None] >= 1.0) + np.zeros((NSR, NUR), bool)
    if mask is not None:
        valid &= mask[ri, ci]
    return ri, ci, valid


def rectify(img, rect):
    ri, ci, valid = rect
    I = img[ri, ci].astype(float)
    m, sd = I[valid].mean(), I[valid].std()
    I = (I - m) / (sd if sd > 1e-9 else 1.0)
    I[~valid] = 0.0
    return I


def target_image(shape):
    """The schematic target in the same straightened frame: 1 on any structure (the union of the layers), 0 elsewhere, standardised."""
    L = layers(shape)
    T = np.max(np.stack(list(L.values())), 0)                               # (NS, NU) over s in (0,1), u in [-1,1]
    si = np.clip((_S * NS).astype(int), 0, NS - 1)
    ui = np.clip(np.rint((_U + 1) / 2 * (NU - 1)).astype(int), 0, NU - 1)
    I = ndi.zoom(T, 1, order=0)[si][:, ui]
    # area-average along s (the template is 2.5 times finer than the frame)
    k = NS // NSR
    if k > 1:
        I = T[:NSR * k].reshape(NSR, k, NU).mean(1)[:, ui]
    return (I - I.mean()) / I.std()


def _gabor(lam_px):
    sg = 0.8 * lam_px
    n = int(np.ceil(3 * sg))
    x = np.arange(-n, n + 1)
    g = np.exp(-0.5 * (x / sg) ** 2) * np.exp(2j * np.pi * x / lam_px)
    g -= g.mean()                                                            # no response to a constant
    return g / np.abs(g).sum()


_ZI = [(np.abs(_U) >= lo) & (np.abs(_U) < hi) for lo, hi in ZONES]
_BIN = (np.arange(NSR) * NBIN // NSR)
_GS = [_gabor(w * NSR) for w in WAVES]
_GU = [_gabor((NUR / UMAX) / c) for c in CYCLES]                             # NUR samples span 2*UMAX of the full width 2


def _pool(a, nb=NBIN):
    """Mean over nb equal bins along the axis (axis 0)."""
    idx = np.arange(NSR) * nb // NSR
    return np.stack([a[idx == b].mean(0) for b in range(nb)], 0)


def features(I):
    F = {}
    F['coarse'] = np.stack([_pool(I[:, z].mean(1), NBIN_COARSE) for z in _ZI])            # mean level in 3 lateral zones x 8 bins along the axis
    cords = []
    for g in _GU:
        e = np.abs(ndi.convolve1d(I, g.real, axis=1, mode='nearest') + 1j * ndi.convolve1d(I, g.imag, axis=1, mode='nearest'))
        cords.append(_pool(e.mean(1)))
    F['cords'] = np.stack(cords)
    seg = []
    for g in _GS:
        e = np.abs(ndi.convolve1d(I, g.real, axis=0, mode='nearest') + 1j * ndi.convolve1d(I, g.imag, axis=0, mode='nearest'))
        seg.append(np.stack([_pool(e[:, z].mean(1)) for z in _ZI]))
    F['segments'] = np.stack(seg)
    dog = np.abs(ndi.gaussian_filter(I, (4.0, 2.0), mode='nearest') - ndi.gaussian_filter(I, (9.0, 4.5), mode='nearest'))
    F['spots'] = np.stack([_pool(dog[:, z].mean(1)) for z in _ZI])
    return F


def _overlap(p, q):
    """Overlap of two non-negative arrays as distributions: 1 = the same shares, 0 = disjoint."""
    p, q = p.ravel() / max(p.sum(), 1e-12), q.ravel() / max(q.sum(), 1e-12)
    return float(1.0 - 0.5 * np.abs(p - q).sum())


def _ratio(a, b):
    return float(min(a, b) / max(a, b, 1e-12))


def summary(F):
    """Coarse-grained description of a pattern: how strong each kind of structure is, at what spacing, and roughly where."""
    seg, cords, spots = F['segments'], F['cords'], F['spots']
    w = seg.sum((1, 2)); lw = np.log(np.array(WAVES))
    return dict(seg_amp=float(seg.mean()), seg_wave=float(np.exp((w * lw).sum() / max(w.sum(), 1e-12))), seg_zone=seg.sum((0, 2)), seg_axis=seg.sum((0, 1)),
                cord_amp=float(cords.mean()), cord_cycles=cords.sum(1), cord_axis=cords.sum(0), spot_amp=float(spots.mean()), spot_where=spots)


def match(Fp, Ft):
    """Similarity of two patterns' coarse-grained features, each part in [0, 1]:
    layout    |r| between the mean levels in 3 lateral zones x 8 axial bins (the sign of G_pol is free)
    segments  mean of: strength ratio; spacing (1 at the same wavelength, 0.5 at a factor of two); overlap of where they are across the body; along the body
    cords     mean of: strength ratio; overlap of the number of stripes across the body; overlap of where they are along the body
    spots     mean of: strength ratio; overlap of where they are (3 zones x 4 bins)
    score = mean of the four."""
    P, Q = summary(Fp), summary(Ft)
    a, b = Fp['coarse'].ravel(), Ft['coarse'].ravel()
    out = {'coarse': abs(float(np.corrcoef(a, b)[0, 1])) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0}
    out['segments'] = float(np.mean([_ratio(P['seg_amp'], Q['seg_amp']), 0.5 ** abs(np.log2(P['seg_wave'] / Q['seg_wave'])),
                                     _overlap(P['seg_zone'], Q['seg_zone']), _overlap(P['seg_axis'], Q['seg_axis'])]))
    out['cords'] = float(np.mean([_ratio(P['cord_amp'], Q['cord_amp']), _overlap(P['cord_cycles'], Q['cord_cycles']), _overlap(P['cord_axis'], Q['cord_axis'])]))
    out['spots'] = float(np.mean([_ratio(P['spot_amp'], Q['spot_amp']), _overlap(P['spot_where'], Q['spot_where'])]))
    out['score'] = float(np.mean([out[k] for k in CHANNELS]))
    return out


_T = {}


def target_features(shape):
    if shape not in _T:
        _T[shape] = features(target_image(shape))
    return _T[shape]


if __name__ == '__main__':
    u = np.abs(_U)[None, :] + np.zeros((NSR, NUR))
    for sh in ('planarian', 'tadpole'):
        Ft = target_features(sh)
        z = lambda I: (I - I.mean()) / I.std()
        noise = [match(features(np.random.default_rng(i).standard_normal((NSR, NUR))), Ft)['score'] for i in range(50)]
        stripe = np.zeros((NSR, NUR)); stripe[:, NUR // 2 - 2:NUR // 2 + 3] = 1
        print(sh, 'noise %.3f +- %.3f' % (np.mean(noise), np.std(noise)), ' midline stripe %.3f' % match(features(z(stripe)), Ft)['score'],
              ' rim-to-centre gradient %.3f' % match(features(z(1 - u)), Ft)['score'],
              ' other target %.3f' % match(target_features('tadpole' if sh == 'planarian' else 'planarian'), Ft)['score'])
