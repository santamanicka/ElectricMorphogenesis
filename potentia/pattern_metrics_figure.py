"""The two pattern metrics of the dashboard on reference patterns and on simulated G_pol.   python pattern_metrics_figure.py   (writes results/fig_pattern_metrics.png; about 3 minutes)

Positional information (bits): how well where a pixel is predicts its G_pol / G_ref (8 bins), from a 4 x 4 lattice of samples in blocks of 20 um; features: the persistent peaks plus basins of the cell-level G_pol.
Top: reference patterns on a tadpole of 22 x 100 cells (m = 2): a uniform tissue, a random initial condition (1 + 0.3 N(0,1)), white noise (many features, no positional information), a smooth gradient (much positional
information, 2 features), bands of 80 um (about 1 bit: one bit of state; 2 features per band) and random domains of 80 um with 8 levels (about 2.5 bits).  Bottom: G_pol of the tadpole after 40 s with the position-dependent
range, Gaussian-smoothed thickness 25.7% (the edge shoulder off, the edge shoulder on, the smooth width).  Right: positional information of the three simulations against the block size (20 um is the largest block
that resolves their structure; at m = 2 it is also the smallest with a 4 x 4 lattice).
"""
import os
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from potentia import Potentia
from shapes import shape_mask
from localrange import range_map
from sweeps import STIG
from complexity import pattern_metrics, positional_information

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SIZE, M = (22, 100), 2


def simulate(spec):
    mode, edge = spec
    mask = shape_mask('tadpole', (SIZE[0] * M, SIZE[1] * M))
    P = Potentia(size_cells=SIZE, m=M, mask=mask, **{**STIG, 'fieldScreenSize': 0.4})
    P.set_range_map(range_map(mask, 25.7, P.h, mode, 0.10, edge)); V, g = P.initial_from_g(1.0, 'dep')
    for _ in range(int(round(40.0 / P.dt))): V, g = P.step(V, g)
    return g


if __name__ == '__main__':
    with Pool(3) as pool:
        sims = pool.map(simulate, [('gaussian', 0.0), ('gaussian', 1.0), ('smooth', 0.0)])
    mk = shape_mask('tadpole', (SIZE[0] * M, SIZE[1] * M)); R, C = mk.shape; rng = np.random.default_rng(1)
    x = (np.arange(C) + 0.5) * (10.0 / M)
    I, J = np.indices((R, C)); n = int(R * C * 25.0 / 6400.0); pts = np.stack([rng.uniform(0, R, n), rng.uniform(0, C, n)], 1); lev = rng.choice(np.linspace(0.125, 1.875, 8), n)
    dom = lev[np.argmin((I[..., None] - pts[:, 0]) ** 2 + (J[..., None] - pts[:, 1]) ** 2, axis=2)]
    pats = [('uniform', np.ones((R, C))), ('random initial condition', np.clip(1 + 0.3 * rng.standard_normal((R, C)), 0, 2)), ('white noise', rng.uniform(0, 2, (R, C))), ('smooth gradient', np.tile(np.linspace(0, 2, C)[None, :], (R, 1))),
            ('bands of 80 um', np.tile(1 + np.sign(np.sin(2 * np.pi * x / 80))[None, :], (R, 1))), ('random domains, 80 um', dom),
            ('simulation: edge shoulder off', sims[0]), ('simulation: edge shoulder on', sims[1]), ('simulation: smooth width', sims[2])]
    fig = plt.figure(figsize=(14, 8.5))
    for k, (name, g) in enumerate(pats):
        r, c = divmod(k, 3); ax = fig.add_axes([0.02 + 0.235 * c, 0.70 - 0.31 * r, 0.22, 0.2]); g = np.where(mk, g, np.nan)
        ax.imshow(g, cmap='magma', vmin=0, vmax=2, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
        res = pattern_metrics(np.where(mk, g, 0), mk, M); ax.set_title(f'{name}\n{res["bits"]:.2f} bits, {res["features"]} features ({res["per100"]:.1f} per 100 cells)', fontsize=8.5, loc='left')
    ax = fig.add_axes([0.76, 0.12, 0.22, 0.72]); Ls = [20, 30, 40, 60, 80]
    for (name, g), sty in zip(pats[6:], ['s--', 'o-', '^-']):
        ax.plot(Ls, [positional_information(np.where(mk, g, 0), mk, M, float(L), raw=True) for L in Ls], sty, label=name.replace('simulation: ', ''))
    ax.set_xlabel('block size (um)'); ax.set_ylabel('positional information (bits, not clipped)'); ax.legend(frameon=False, fontsize=8); ax.axvline(20, color='gray', lw=0.8, ls=':')
    fig.savefig(os.path.join(OUT, 'fig_pattern_metrics.png'), dpi=100)
    for name, g in pats: print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in pattern_metrics(np.where(mk, g, 0), mk, M).items()})
