"""Why do patterns in a disc, hexagon or ring end up aligned with the N-S and E-W axes?   python anisotropy_study.py

Two candidate causes, tested separately:
  (A) the operators are not isotropic: the gap-junction coupling acts along the grid axes with a gate on each axis' voltage difference, and a cell senses the field at the four
      corners of a square;  -> apply each operator to a radially symmetric input and look at the angular variation of the output (a rotation-invariant operator gives a constant)
  (B) a homogeneous start in a domain with mirror symmetry keeps that symmetry exactly (the equations preserve it), so the solution lives in the symmetric subspace;
      -> compare homogeneous with random (symmetry-breaking) starts, and an aligned with a rotated hexagon.
      -> and an experiment that removes the suspected cause: the same model with the gate applied to the local gradient magnitude instead of to each axis' difference
Orientation statistic of the Vmem pattern: gradient direction theta weighted by |grad V|^2 (Gaussian-derivative gradient, interior pixels only, more than 2 sigma + 1 pixels from the boundary);
controls: the same statistic on isotropic test patterns (concentric rings, a smoothed random field) rasterised on the same grid:
  A4 = <cos 4 theta>:  +1 all gradients along the axes (E-W / N-S), -1 all along the diagonals, 0 isotropic;   A2 = <cos 2 theta>: + N-S/E-W-gradient imbalance.
"""
import os
import sys
import numpy as np
from scipy import ndimage

from potentia import Potentia
from shapes import shape_mask
from sweeps import STIG

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
lines = []


def log(s=''):
    print(s, flush=True)
    lines.append(s)


class PotentiaIsoGate(Potentia):
    """Experiment only (not part of the model): the gap-junction gate acts on the local gradient magnitude, evaluated at the midpoint of each pair
    (the difference along the pair and the mean of the two central differences across it), instead of on the difference along the pair alone.
    In the linear regime (gate = 1) it is the same 5-point stencil; in the continuum limit it is div( g(|grad V|) grad V ), which is rotation invariant."""

    def gj(self, V):
        m, V0 = self.m, self.V_0
        out = np.zeros_like(V)
        for ax in (0, 1):
            other = 1 - ax
            cross = np.gradient(V, axis=other)                         # central difference across the pair direction (one-sided at the edge)
            sl_a = [slice(None)] * 2; sl_b = [slice(None)] * 2
            sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None)
            d = V[tuple(sl_b)] - V[tuple(sl_a)]
            t = 0.5 * (cross[tuple(sl_a)] + cross[tuple(sl_b)])
            f = self.gj_coef * d * (2.0 / (1.0 + np.cosh(m * np.sqrt(d * d + t * t) / V0)))
            if self.mask is not None:
                f = f * self._gj_pairs[ax]
            out[tuple(sl_a)] += f
            out[tuple(sl_b)] -= f
        return out


# ------------------------------------------------------------------ (A) operators on radially symmetric input
def ring_values(F, centre, rho, n=360):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    yy = centre[0] + rho * np.sin(th); xx = centre[1] + rho * np.cos(th)
    return ndimage.map_coordinates(F, [yy, xx], order=3, mode='nearest'), th


def variation(v):
    return (v.max() - v.min()) / abs(v.mean())


def harmonic(v, th, k):
    return float(np.abs(np.mean(v * np.exp(1j * k * th))) / abs(v.mean()))


log('(A) angular variation of the operators applied to a radially symmetric Gaussian bump (A = 45 mV, sigma = 20 um, tissue 20 x 20 cells)')
log('    variation = (max - min) / mean of the output on a circle of radius = sigma; 4th harmonic = amplitude of cos(4 theta) relative to the mean')
log(f'    {"m":>2} | {"GJ gated: variation":>20} {"4th harm.":>10} | {"GJ linear: variation":>20} {"4th harm.":>10} | {"GJ isotropic gate: var.":>23} {"4th harm.":>10} | {"perceived |E|: variation":>24} {"4th harm.":>10}')
rows = []
for m in (2, 4, 8):
    P = Potentia(size_cells=(20, 20), m=m, **STIG)
    n = P.Np[0]; c = (n - 1) / 2.0
    y, x = np.mgrid[:n, :n]
    sigma_px = 20.0 / (10.0 / m)                                     # 20 um in pixels
    V = -30e-3 + 45e-3 * np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma_px ** 2))
    out = {}
    g_gate = P.gj(V)
    g_iso = PotentiaIsoGate(size_cells=(20, 20), m=m, **STIG).gj(V)
    P.V_0 = 1e6                                                       # a huge V_0 switches the gate off: plain linear diffusion on the same stencil
    g_lin = P.gj(V)
    P.V_0 = Potentia.V_0
    for tag, F in (('gj', g_gate), ('lin', g_lin), ('iso', g_iso)):
        v, th = ring_values(F, (c, c), sigma_px)
        out[tag] = (variation(v), harmonic(v, th, 4))
    pe = P.perceived_field(V)                                         # indexed by pixel, so the ring is centred on the bump
    vp, thp = ring_values(pe, (c, c), 2.0 * sigma_px)
    out['pe'] = (variation(vp), harmonic(vp, thp, 4))
    rows.append((m, out))
    log(f'    {m:>2} | {out["gj"][0]:>20.4f} {out["gj"][1]:>10.4f} | {out["lin"][0]:>20.4f} {out["lin"][1]:>10.4f} | {out["iso"][0]:>23.4f} {out["iso"][1]:>10.4f} | {out["pe"][0]:>24.4f} {out["pe"][1]:>10.4f}')
log('    (a rotation-invariant operator would give 0 in every column; the perceived field is |E| averaged over the four corner probes at (+-a, +-a), evaluated on a circle of radius 2 sigma)')
log()


# ------------------------------------------------------------------ (B) patterns
def hexagon_rot(Np, deg):
    R, C = Np
    i = np.arange(R)[:, None]; j = np.arange(C)[None, :]
    x = (j + 0.5) - C / 2.0; y = (i + 0.5) - R / 2.0
    th = np.deg2rad(deg)
    xr = x * np.cos(th) + y * np.sin(th); yr = -x * np.sin(th) + y * np.cos(th)
    b = min(R, C) / 2.0; s3 = np.sqrt(3.0)
    return (np.abs(yr) <= s3 * b / 2.0) & (s3 * np.abs(xr) + np.abs(yr) <= s3 * b)


def orientation(V, mask, sigma):
    """(A4, A2, A8) of the gradient direction of Gaussian-smoothed Vmem, weighted by |grad|^2, over interior pixels."""
    Vm = V * 1e3
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    Vf = Vm[tuple(idx)]                                                # outside pixels take the nearest tissue value: no artificial step at the boundary
    gy = ndimage.gaussian_filter(Vf, sigma, order=(1, 0), mode='nearest'); gx = ndimage.gaussian_filter(Vf, sigma, order=(0, 1), mode='nearest')
    margin = int(np.ceil(2 * sigma)) + 1                                # Euclidean distance from the boundary (a city-block erosion would leave only the diagonals of a narrow ring)
    inner = ndimage.distance_transform_edt(mask) > margin
    w = (gx ** 2 + gy ** 2)[inner]; th = np.arctan2(gy, gx)[inner]; s = w.sum()
    return float((w * np.cos(4 * th)).sum() / s), float((w * np.cos(2 * th)).sum() / s), float((w * np.cos(8 * th)).sum() / s)


SIG = 2.0                                                              # pixels (one cell at m = 2)
log('(B0) controls for the orientation statistic: isotropic patterns on the same 60 x 60 disc (concentric rings and a smoothed random field), the same estimator')
mk_c = shape_mask('disc', (60, 60)); yy, xx = np.mgrid[:60, :60] + 0.5 - 30; rr = np.hypot(xx, yy)
ctrl = {'concentric binary rings, wavelength 10 px': np.where(np.sin(2 * np.pi * rr / 10.0) > 0, -9e-3, -50e-3),
        'concentric smooth rings, wavelength 10 px': -30e-3 + 20e-3 * np.sin(2 * np.pi * rr / 10.0),
        'smoothed random binary field': np.where(ndimage.gaussian_filter(np.random.default_rng(0).standard_normal((60, 60)), 2.5) > 0, -9e-3, -50e-3)}
for nm, Vc in ctrl.items():
    a4, a2, a8 = orientation(Vc, mk_c, SIG)
    log(f'    {nm:<46} A4 = {a4:+.3f}   A2 = {a2:+.3f}   A8 = {a8:+.3f}')
log()


def run(P, ic, seed, T=140.0, windows=((20, 50), (100, 140))):
    if ic == 'homogeneous':
        V, g = P.initial_from_g(1.0, 'dep')
    else:
        rng = np.random.default_rng(seed)
        V, g = P.initial_from_g(1.0 + 0.3 * rng.standard_normal(P.Np), 'dep')
    nrec = int(round(1.0 / P.dt)); stats = {w: [] for w in windows}
    mk = P.mask if P.mask is not None else np.ones(P.Np, bool)
    for it in range(int(round(T / P.dt)) + 1):
        t = it * P.dt
        if it % nrec == 0:
            for w in windows:
                if w[0] <= t <= w[1]:
                    stats[w].append(orientation(V, mk, SIG))
        if it < int(round(T / P.dt)):
            V, g = P.step(V, g)
    return {w: np.mean(stats[w], axis=0) for w in windows}, V * 1e3, mk


log(f'(B) orientation of the Vmem pattern (Gaussian gradient, sigma = {SIG} px), Stigmergic parameters, 30 x 30 cells at m = 2, mean over the windows 20-50 s and 100-140 s')
log(f'    {"case":<54} {"A4 20-50 s":>11} {"A4 100-140 s":>13} {"A8 100-140 s":>13}')
box = (30, 30); Np = (60, 60)
masks = {'disc': shape_mask('disc', Np), 'ring': shape_mask('annulus', Np, 0.5), 'hexagon': shape_mask('hexagon', Np), 'hexagon rot 15 deg': hexagon_rot(Np, 15.0)}
plan = [('disc', 'homogeneous', [0]), ('ring', 'homogeneous', [0]), ('hexagon', 'homogeneous', [0]), ('hexagon rot 15 deg', 'homogeneous', [0]),
        ('disc', 'random', [0, 1, 2]), ('ring', 'random', [0, 1]), ('hexagon rot 15 deg', 'random', [0, 1])]
results = {}; summary = {}
for variant, cls in (('standard gate', Potentia), ('isotropic gate', PotentiaIsoGate)):
    for shp, ic, seeds in plan:
        vals = []
        for sd in seeds:
            P = cls(size_cells=box, m=2, mask=masks[shp], **STIG)
            res, snap, mk = run(P, ic, sd)
            ws = list(res.keys()); vals.append((res[ws[0]][0], res[ws[1]][0], res[ws[1]][2]))
            results[(variant, shp, ic, sd)] = (res[ws[1]], snap, mk)
        v = np.array(vals); m_ = v.mean(0); sd_ = v.std(0) if len(v) > 1 else np.zeros(3)
        summary[(variant, shp, ic)] = m_
        tag = f'{variant}: {shp}, {ic} start' + (f' (mean of {len(seeds)})' if len(seeds) > 1 else '')
        log(f'    {tag:<54} {m_[0]:>+11.3f} {m_[1]:>+13.3f} {m_[2]:>13.3f}' + (f'   (sd of A4 over seeds {sd_[1]:.3f})' if len(seeds) > 1 else ''))
log()
log('    A4 > 0: gradients prefer the E-W and N-S axes; A4 < 0: the diagonals; ~0: no preference (compare with the controls above).')

# figure
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = [('disc', 'homogeneous', 0), ('ring', 'homogeneous', 0), ('hexagon', 'homogeneous', 0), ('disc', 'random', 0)]
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.8))
    for r, variant in enumerate(('standard gate', 'isotropic gate')):
        for c, (shp, ic, sd) in enumerate(show):
            (a4, a2, a8), snap, mk = results[(variant, shp, ic, sd)]
            ax = axs[r, c]; ax.imshow(np.where(mk, snap, np.nan), cmap='viridis', vmin=-55, vmax=-5, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{variant}: {shp}, {ic}\nt = 140 s, A4 = {a4:+.2f}', fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_anisotropy_study.png'), dpi=110)
except Exception as e:                                           # the table is the result; the figure is a convenience
    log(f'(figure not written: {e})')
with open(os.path.join(OUT, 'anisotropy_study.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
