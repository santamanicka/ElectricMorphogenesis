"""The position-dependent field range on several shapes.   python local_range_figure.py   (writes results/fig_local_range.png)

Each cell senses over a range that is a fixed share (here 25.7%, the Stigmergic range for a 110 um box) of a local width of the tissue, at least 3 pixels (localrange.py).  Colour = the field range in um
(pixel pitch 5 um); the single-range value of the Stigmergic set is 28.3 um everywhere.
Left column: the local thickness (the diameter of the largest circle that fits in the tissue and contains the cell): stepped, the same across a broad region up to its edge.
Right column: the smooth width (the chord of that circle through the cell, parallel to the nearest boundary; default of the page): the same thickness on the middle line of a broad region,
falling smoothly to every boundary.  Bottom: the range across the tadpole body (a column through its middle) in the two modes.
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
modes = [('thickness', 'local thickness'), ('smooth', 'smooth width')]
fig = plt.figure(figsize=(11, 10.5))
gs = fig.add_gridspec(len(cases) + 1, 2, height_ratios=[1] * len(cases) + [1.15], hspace=0.35, wspace=0.08)
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
        ax.set_title(f'{name}, {lab}: {np.nanmin(Rm):.1f} to {np.nanmax(Rm):.1f} um', fontsize=9, loc='left')
fig.colorbar(im, ax=fig.axes, shrink=0.5, pad=0.02, label='field range (um)')
ax = fig.add_subplot(gs[len(cases), :])
mk = shape_mask('tadpole', (22, 100)); col = 20
rows = np.where(mk[:, col])[0]
for mode, lab in modes:
    ax.plot(rows, maps['tadpole', mode][rows, col], 'o-' if mode == 'smooth' else 's--', label=lab, ms=4)
ax.axhline(28.3, color='gray', lw=0.8, ls=':'); ax.text(rows[0], 29.1, 'single range of the Stigmergic set (28.3 um)', fontsize=8, color='gray', va='bottom')
ax.set_xlabel(f'row (a column through the tadpole body, x = {col})'); ax.set_ylabel('field range (um)'); ax.set_ylim(0, 32); ax.legend(frameon=False, loc='lower center', ncol=2)
fig.savefig(os.path.join(OUT, 'fig_local_range.png'), dpi=110, bbox_inches='tight')
print('wrote fig_local_range.png')
for name, Np in cases:
    t, s = maps[name, 'thickness'], maps[name, 'smooth']; print(f'{name:10s} thickness {np.nanmin(t):.1f}-{np.nanmax(t):.1f}   smooth {np.nanmin(s):.1f}-{np.nanmax(s):.1f} um')
print('tadpole body column, thickness:', np.round(maps['tadpole', 'thickness'][rows, col], 1).tolist())
print('tadpole body column, smooth   :', np.round(maps['tadpole', 'smooth'][rows, col], 1).tolist())
