"""The tadpole outline against the schematic drawing it was fitted to.   python tadpole_figure.py   (writes results/fig_tadpole.png)

Left: the preset on three grids (snout at the left; tissue pixels dark).  Right: the depth of the outline (its height at each position along the length, as a fraction of the box height) against the
depth of the drawing's silhouette read at 21 positions; s = 0 at the snout, 1 at the tail tip.  The drawing is a side view of a Xenopus tadpole with an oval body and a lens-shaped tail with dorsal and
ventral fins.  Fit: the drawing's silhouette (threshold on the green body colour, largest component, holes filled; 512 x 136 pixels, so 3.76 times longer than high) against a body ellipse plus a lens
(half-depth 1 - u^2 towards the head, 1 - t^2.5 towards the tip), 8 parameters optimised for overlap: an intersection over union of 0.963 while the tail's axis was allowed to sit above the body's (the drawing's
body bulges below it).  The preset is now mirror-symmetric about the long axis (body and tail share the axis): its overlap with the drawing is 0.87 centred in the box and 0.90 with the axis put at 0.54 of the
drawing's height; the depth profile is within 0.04 of the drawing's except where the symmetric outline has a shallow waist between body and tail (s = 0.30-0.35: 0.08 and 0.06; depth down to 0.74)
and in the last 5% of the tail (0.08: the drawing's tip is blunter).
What the outline leaves out: the oral apparatus (a small bulge under the snout), the eye and nostril, and the myotome texture.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapes import shape_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
# back and belly edges of the drawing's silhouette (fraction of its height), at s = 0, 0.05, ..., 1
S = np.linspace(0, 1, 21)
TOP = np.array([66, 37, 27, 21, 19, 20, 14, 10, 6, 3, 1, 0, 0, 1, 4, 9, 13, 21, 29, 39, 58]) / 136.0
BOT = (np.array([79, 107, 114, 124, 133, 135, 132, 120, 117, 119, 121, 120, 120, 118, 115, 112, 106, 100, 91, 81, 63]) + 1) / 136.0

fig = plt.figure(figsize=(12, 5.2))
ax = [fig.add_axes([0.03, 0.62, 0.45, 0.34]), fig.add_axes([0.03, 0.32, 0.30, 0.24]), fig.add_axes([0.03, 0.06, 0.15, 0.20])]
for a, (R, C) in zip(ax, [(40, 150), (22, 82), (11, 41)]):
    a.imshow(~shape_mask('tadpole', (R, C)), cmap='gray', vmin=-0.3, vmax=1.0, interpolation='nearest'); a.set_xticks([]); a.set_yticks([])
    a.set_title(f'{R} x {C} pixels', fontsize=9, loc='left')
pr = fig.add_axes([0.55, 0.12, 0.42, 0.78])
R, C = 150, 532; mg = max(1.0, 0.04 * R)
M = shape_mask('tadpole', (R, C)); h = R - 2 * mg; w = C - 2 * mg
cols = np.where(M.any(axis=0))[0]
s_m = (cols + 0.5 - mg) / w
dep_m = np.array([M[:, j].sum() for j in cols]) / h
pr.plot(s_m, dep_m, 'k-', lw=1.4, label='preset'); pr.plot(S, BOT - TOP, 'o', mfc='none', color='tab:red', label='drawing (measured)')
pr.set_ylim(0, 1.05); pr.set_xlim(0, 1); pr.set_xlabel('s: snout (0) to tail tip (1)'); pr.set_ylabel('depth / height'); pr.legend(frameon=False, loc='lower center')
err = abs(np.interp(S, s_m, dep_m) - (BOT - TOP))
pr.set_title(f'depth of the outline against the drawing: largest difference {err[1:-1].max():.3f} of the height (at s = {S[1:-1][err[1:-1].argmax()]:.2f})', fontsize=9)
fig.savefig(os.path.join(OUT, 'fig_tadpole.png'), dpi=110)
print('depth difference (max over 0.05 <= s <= 0.95):', err[1:-1].max())
