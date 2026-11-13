"""The planarian outline against the schematic target it was traced from.   python planarian_figure.py   (writes results/fig_planarian.png)

Left: the preset on three grids (head at the left; tissue pixels dark).  Right: the depth of the preset along the length (its height at each position as a fraction of the height of its tallest point,
the auricle spikes) against the depth of the target's silhouette read at 21 positions (red); s = 0 at the head tip, 1 at the tail tip.  The target is the outline of a schematic planarian (715 x 152 pixels,
4.70 times longer than high, mirror-symmetric): a pointed head, two small auricle spikes at 14% of the length with a notch behind them, a spindle-shaped body widest at 40-58% and a tail tapering to a point;
the internal structure of the schematic (nerve cords, gut, eyes) is ignored.  The preset is a table of 23 half-depth nodes read from the traced silhouette (smoothed with a 3-pixel Gaussian and simplified at a
tolerance of 0.008 of the half-height), joined linearly; its overlap (intersection over union) with the silhouette is 0.990 on a 150-pixel-high grid with the target's proportions.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapes import shape_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
S = np.linspace(0, 1, 21)
TARGET = np.array([0.026, 0.434, 0.658, 0.961, 0.750, 0.796, 0.868, 0.908, 0.954, 0.954, 0.954, 0.954, 0.934, 0.888, 0.842, 0.750, 0.684, 0.579, 0.454, 0.296, 0.026])   # depth / height of the silhouette
# (depth / height of the silhouette: the tallest point, the auricle spike at s = 0.14, is the full height 1.0, so these are in units of the tallest point; s = 0.15 is a sample just behind the spike's peak)

fig = plt.figure(figsize=(12, 5.2))
ax = [fig.add_axes([0.03, 0.70, 0.45, 0.25]), fig.add_axes([0.03, 0.38, 0.30, 0.20]), fig.add_axes([0.03, 0.06, 0.15, 0.20])]
for a, (R, C) in zip(ax, [(40, 190), (24, 114), (11, 48)]):
    a.imshow(~shape_mask('planarian', (R, C)), cmap='gray', vmin=-0.3, vmax=1.0, interpolation='nearest'); a.set_xticks([]); a.set_yticks([])
    a.set_title(f'{R} x {C} pixels', fontsize=9, loc='left')
pr = fig.add_axes([0.55, 0.12, 0.42, 0.78])
R, C = 150, 677; mg = max(1.0, 0.04 * R); M = shape_mask('planarian', (R, C))
cols = np.where(M.any(axis=0))[0]; sm = (cols + 0.5 - mg) / (C - 2 * mg)
dep = np.array([M[:, j].sum() for j in cols], float); dep = dep / dep.max()
pr.plot(sm, dep, 'k-', lw=1.4, label='preset'); pr.plot(S, TARGET, 'o', mfc='none', color='tab:red', label='target (measured)')
pr.set_xlim(0, 1); pr.set_ylim(0, 1.08); pr.set_xlabel('s: head tip (0) to tail tip (1)'); pr.set_ylabel('depth / depth of the tallest point'); pr.legend(frameon=False, loc='lower center')
err = np.abs(np.interp(S, sm, dep) - TARGET)
pr.set_title(f'largest difference from the target: {err[1:-1].max():.3f} (0.05 <= s <= 0.95)', fontsize=9)
fig.savefig(os.path.join(OUT, 'fig_planarian.png'), dpi=110)
print('largest depth difference (0.05 <= s <= 0.95):', float(err[1:-1].max()))
