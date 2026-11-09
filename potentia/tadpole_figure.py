"""The tadpole outline against the schematic drawing it was fitted to.   python tadpole_figure.py   (writes results/fig_tadpole.png)

Left: the preset on three grids (snout at the left, dorsal side up; tissue pixels dark).  Right: the back and belly edges (as a fraction of the height, 0 = top) against the silhouette of the drawing
(a side view of a Xenopus tadpole with an oval body and a lens-shaped tail with dorsal and ventral fins), read at 21 positions along the length; s = 0 at the snout, 1 at the tail tip.
Fit: the drawing's silhouette (threshold on the green body colour, largest component, holes filled; 512 x 136 pixels, so 3.76 times longer than high) against a body ellipse plus a lens (centre line
at 0.451 of the height, half-depth 1 - u^2 towards the head, 1 - t^2.5 towards the tip), 8 parameters optimised for overlap, gives an intersection over union of 0.963; the shape in shapes.py (parameters
rounded) gives 0.963 on the same silhouette too.  What the outline leaves out: the oral apparatus (a small bulge under the snout), the eye and nostril, and the myotome texture.
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
top_m = np.array([np.where(M[:, j])[0][0] - mg for j in cols]) / h; bot_m = np.array([np.where(M[:, j])[0][-1] + 1 - mg for j in cols]) / h
pr.plot(s_m, top_m, 'k-', lw=1.4, label='preset'); pr.plot(s_m, bot_m, 'k-', lw=1.4)
pr.plot(S, TOP, 'o', mfc='none', color='tab:red', label='drawing (measured)'); pr.plot(S, BOT, 'o', mfc='none', color='tab:red')
pr.set_ylim(1.05, -0.05); pr.set_xlim(0, 1); pr.set_xlabel('s: snout (0) to tail tip (1)'); pr.set_ylabel('position / height (0 = back, 1 = belly)'); pr.legend(frameon=False, loc='lower left')
err = np.array([abs(np.interp(S, s_m, top_m) - TOP).max(), abs(np.interp(S, s_m, bot_m) - BOT).max()])
pr.set_title(f'largest difference of an edge from the drawing: back {err[0]:.3f}, belly {err[1]:.3f} of the height', fontsize=9)
fig.savefig(os.path.join(OUT, 'fig_tadpole.png'), dpi=110)
print('max edge difference (back, belly):', err)
