"""The tapered capsule at several tapering ratios.   python tapered_figure.py   (writes results/fig_tapered.png)

Left: the preset on a 40 x 140 grid (head at the left; tissue pixels dark) for ratios 0 (the plain stadium, to the pixel), 0.3, 0.6 (the default), 0.9 and 0.95 (the largest the slider allows).  The outline is the convex
hull of a head circle of the box height (radius b) and a tail circle of radius (1 - p) b, joined by straight tangent lines.  Right: the depth of the tissue along the length as a fraction of the box height; the
depth at the centre of the tail circle is 1 - p (dots), and it falls linearly between the two circles.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapes import shape_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
R, C = 40, 140
ps = [0.0, 0.3, 0.6, 0.9, 0.95]
fig = plt.figure(figsize=(12, 5.4))
for k, p in enumerate(ps):
    a = fig.add_axes([0.02, 0.84 - 0.165 * k, 0.50, 0.14]); a.imshow(~shape_mask('tapered', (R, C), p), cmap='gray', vmin=-0.3, vmax=1.0, interpolation='nearest'); a.set_xticks([]); a.set_yticks([])
    a.set_title(f'tapering ratio {p:g}' + (' (the plain stadium)' if p == 0 else ' (default)' if p == 0.6 else ''), fontsize=8, loc='left', pad=2)
ax = fig.add_axes([0.60, 0.12, 0.37, 0.78]); x = np.arange(C) / (C - 1)
for p in ps:
    dep = shape_mask('tapered', (R, C), p).sum(axis=0) / R; ax.plot(x, dep, label=f'{p:g}')
    ax.plot((C - (1 - p) * R / 2 - 0.5) / (C - 1), 1 - p, 'o', color='k', ms=3)                # the centre of the tail circle
ax.set_xlabel('position along the length (head at 0)'); ax.set_ylabel('depth / box height'); ax.set_ylim(0, 1.05); ax.legend(title='tapering ratio', frameon=False, loc='lower left')
fig.savefig(os.path.join(OUT, 'fig_tapered.png'), dpi=110)
print('wrote fig_tapered.png; areas', [int(shape_mask('tapered', (R, C), p).sum()) for p in ps])
