"""The tadpole outline against the schematic target it was fitted to.   python tadpole_figure.py   (writes results/fig_tadpole.png)

Left: the preset (default settings) and three settings of its two parameters (tail depth at its root / box height, head size / tail size); snout at the left, tissue pixels dark.  Right: the depth of the default
outline (its height at each position along the length, as a fraction of the box height) against the depth of the target's silhouette read at 21 positions; s = 0 at the snout, 1 at the tail tip.  The target is the
outline of a schematic dorsal view of a tadpole (713 x 148 pixels, 4.82 times longer than high, mirror-symmetric): a large oval body with a fuller front and a small snout spike, an S-shaped neck, and a long tail
tapering to a thin whip; its internal structure (brain, cord, myomeres) is ignored.  Model: the half-depth is a smooth maximum of a body, hb = K * 0.62 * sqrt(1 - u^2) * (1 - 0.2 (1 - u^2)) with u the distance from the
centre of the body (0.605 of the height from the snout) in units of its half-length (0.548 in front, 1.394 behind), and a tail, a cubic polynomial in the distance from the tip e = 1 - s, pointed at the tip and
absent at the snout; 14 parameters fitted to the depth profile by least squares (differential evolution then a local polish): rms depth error 0.014 of the height.  With the constants rounded and the tail root
bounded (so that a small head is not swallowed by it), the overlap (intersection over union) with the silhouette is 0.969 on a 150-pixel-high grid with the target's proportions.  The outline is scaled to the
length of the box, as the planarian's is.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapes import shape_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
S = np.linspace(0, 1, 21)
TARGET = np.array([0.027, 0.811, 0.959, 1.0, 1.0, 0.980, 0.865, 0.743, 0.527, 0.412, 0.365, 0.270, 0.230, 0.169, 0.135, 0.115, 0.115, 0.115, 0.095, 0.068, 0.027])   # depth / height of the silhouette
BOT = TARGET; TOP = 0 * TARGET

fig = plt.figure(figsize=(12, 6.2))
ax = [fig.add_axes([0.03, 0.78 - 0.235 * i, 0.46, 0.19]) for i in range(4)]
for a, (R, C, p, p2, tag) in zip(ax, [(40, 192, None, None, 'default (tail depth 0.41, head size / tail size 2.354)'), (40, 192, None, 1.9, 'head size / tail size 1.9'), (40, 192, None, 1.6, 'head size / tail size 1.6'),
                                      (40, 192, 0.2, 4.8, 'tail depth 0.2, head size / tail size 4.8')]):
    a.imshow(~shape_mask('tadpole', (R, C), p, p2), cmap='gray', vmin=-0.3, vmax=1.0, interpolation='nearest'); a.set_xticks([]); a.set_yticks([])
    a.set_title(f'{R} x {C} pixels: {tag}', fontsize=9, loc='left')
pr = fig.add_axes([0.55, 0.12, 0.42, 0.78])
R, C = 150, 677; mg = max(1.0, 0.04 * R)
M = shape_mask('tadpole', (R, C)); h = R - 2 * mg; w = C - 2 * mg
cols = np.where(M.any(axis=0))[0]
s_m = (cols + 0.5 - mg) / w
dep_m = np.array([M[:, j].sum() for j in cols]) / h
pr.plot(s_m, dep_m, 'k-', lw=1.4, label='preset'); pr.plot(S, TARGET, 'o', mfc='none', color='tab:red', label='target (measured)')
pr.set_ylim(0, 1.08); pr.set_xlim(0, 1); pr.set_xlabel('s: snout (0) to tail tip (1)'); pr.set_ylabel('depth / height'); pr.legend(frameon=False, loc='upper right')
err = abs(np.interp(S, s_m, dep_m) - TARGET)
pr.set_title(f'depth of the outline against the target: largest difference {err[1:-1].max():.3f} of the height (at s = {S[1:-1][err[1:-1].argmax()]:.2f})', fontsize=9)
fig.savefig(os.path.join(OUT, 'fig_tadpole.png'), dpi=110)
print('depth difference (max over 0.05 <= s <= 0.95):', err[1:-1].max())
