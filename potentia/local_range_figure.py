"""The position-dependent field range on several shapes.   python local_range_figure.py   (writes results/fig_local_range.png)

Each cell senses over a range that is a fixed share (here 25.7%, the Stigmergic range for a 110 um box) of a local width of the tissue, at least 3 pixels (localrange.py).  Colour = the field range in um
(pixel pitch 5 um); the single-range value of the Stigmergic set is 28.3 um everywhere.
Left column: the local thickness (the diameter of the largest circle that fits in the tissue and contains the cell): stepped, the same across a broad region up to its edge.
Middle column: the thickness smoothed by a Gaussian over the tissue pixels (smoothing length 10% of the thickest part; default of the page): the plateaus of the thickness are kept to the rim of a broad
region, the steps become ramps.  Right column: the smooth width (the chord of that circle through the cell, parallel to the nearest boundary): the thickness on the middle line of a broad region,
falling smoothly to every boundary.  Bottom: the range along the tadpole (the middle row, left) and across the tadpole body (a column through its middle, right) in the three modes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapes import shape_mask
from localrange import range_map

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
h = 5e-6
cases = [('tadpole', (22, 100)), ('planarian', (22, 96)), ('dumbbell', (22, 66)), ('ellipse', (22, 66))]
modes = [('thickness', 'local thickness'), ('gaussian', 'Gaussian thickness'), ('smooth', 'smooth width')]
fig = plt.figure(figsize=(15, 10.5))
gs = fig.add_gridspec(len(cases) + 1, 3, height_ratios=[1] * len(cases) + [1.15], hspace=0.4, wspace=0.06)
maps = {}
vmax = 0
for name, Np in cases:
    mk = shape_mask(name, Np)
    for mode, _ in modes:
        Rm = np.where(mk, range_map(mk, 25.7, h, mode) * 1e6, np.nan); maps[name, mode] = Rm; vmax = max(vmax, np.nanmax(Rm))
for r, (name, Np) in enumerate(cases):
    for c, (mode, lab) in enumerate(modes):
        ax = fig.add_subplot(gs[r, c]); Rm = maps[name, mode]
        im = ax.imshow(Rm, cmap='viridis', vmin=0, vmax=vmax, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{name}, {lab}: {np.nanmin(Rm):.1f} to {np.nanmax(Rm):.1f} um', fontsize=8.5, loc='left')
fig.colorbar(im, ax=fig.axes, shrink=0.5, pad=0.02, label='field range (um)')
mk = shape_mask('tadpole', (22, 100)); col = 20; rows = np.where(mk[:, col])[0]; mid = 11
sty = {'thickness': ('s--', 'tab:blue'), 'gaussian': ('o-', 'tab:green'), 'smooth': ('^-', 'tab:orange')}
for k, (ax, xs, sel, xl) in enumerate([(fig.add_subplot(gs[len(cases), 0:2]), np.where(mk[mid, :])[0], lambda M, xs: M[mid, xs], 'column along the middle row of the tadpole (head at the left)'),
                                       (fig.add_subplot(gs[len(cases), 2]), rows, lambda M, xs: M[xs, col], f'row across the body (column x = {col})')]):
    for mode, lab in modes:
        ax.plot(xs, sel(maps['tadpole', mode], xs), sty[mode][0], color=sty[mode][1], label=lab, ms=3, lw=1.2)
    ax.axhline(28.3, color='gray', lw=0.8, ls=':'); ax.set_ylim(0, 32); ax.set_xlabel(xl, fontsize=8); ax.set_ylabel('field range (um)')
    if k == 0: ax.legend(frameon=False, loc='center right', fontsize=8); ax.text(xs[0], 29.1, 'single range of the Stigmergic set (28.3 um)', fontsize=7.5, color='gray', va='bottom')
fig.savefig(os.path.join(OUT, 'fig_local_range.png'), dpi=110, bbox_inches='tight')
print('wrote fig_local_range.png')
for name, Np in cases:
    t, g, s = maps[name, 'thickness'], maps[name, 'gaussian'], maps[name, 'smooth']; print(f'{name:10s} thickness {np.nanmin(t):.1f}-{np.nanmax(t):.1f}   Gaussian {np.nanmin(g):.1f}-{np.nanmax(g):.1f}   smooth {np.nanmin(s):.1f}-{np.nanmax(s):.1f} um')
