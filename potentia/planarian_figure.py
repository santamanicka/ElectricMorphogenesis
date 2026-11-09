"""The planarian outline against the schematic drawing it was fitted to.   python planarian_figure.py   (writes results/fig_planarian.png)

Left: the preset on three grids (head at the left; tissue pixels dark).  Right: its half-width profile h(s) (analytic, and the width of the mask on the 40 x 200 grid) against widths measured
on the drawing, normalised to the widest part of the body; s = 0 at the head tip, 1 at the tail tip.  The drawing's own head has auricles about 7% wider than its body and a neck behind them
(width 0.72-0.78 of the auricles, i.e. about 0.77-0.83 of the body); the preset leaves both out on purpose: the head is as wide as the body and there is no bottleneck.
Measured widths of the drawing: threshold on the orange body colour, largest component, holes filled, width of every row, divided by the median width at s = 0.45-0.55 (0.934 of the auricle width).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapes import shape_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
S = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.18, 0.25, 0.35, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98])
W = np.array([0.242, 0.406, 0.55, 0.644, 0.776, 0.975, 0.721, 0.779, 0.882, 0.934, 0.934, 0.904, 0.86, 0.809, 0.735, 0.654, 0.558, 0.436, 0.273, 0.161]) / 0.934


def profile(s, sa=0.134, st=0.5):
    x1 = s / sa
    t = (s - st) / (1.0 - st)
    return np.where(s < sa, np.sqrt(np.maximum(x1, 0)) - x1 + x1 * x1, np.where(s < st, 1.0, 1.0 - t * t * np.sqrt(np.maximum(t, 0))))


fig = plt.figure(figsize=(12, 5.2))
ax = [fig.add_axes([0.03, 0.70, 0.45, 0.25]), fig.add_axes([0.03, 0.38, 0.30, 0.20]), fig.add_axes([0.03, 0.06, 0.15, 0.20])]
for a, (R, C) in zip(ax, [(40, 200), (24, 120), (11, 55)]):
    a.imshow(~shape_mask('planarian', (R, C)), cmap='gray', vmin=-0.3, vmax=1.0, interpolation='nearest'); a.set_xticks([]); a.set_yticks([])
    a.set_title(f'{R} x {C} pixels', fontsize=9, loc='left')
pr = fig.add_axes([0.55, 0.12, 0.42, 0.78])
s = np.linspace(0, 1, 400)
pr.plot(s, profile(s), 'k-', lw=1.6, label='preset (analytic)')
P = shape_mask('planarian', (40, 200)); sx = (np.arange(200) + 0.5) / 200
pr.plot(sx, P.sum(axis=0) / 40.0, '.', ms=3, color='tab:blue', label='mask on 40 x 200 pixels')
pr.plot(S, W, 'o', mfc='none', color='tab:red', label='drawing (measured)')
pr.set_xlabel('s: head tip (0) to tail tip (1)'); pr.set_ylabel('width / body width'); pr.set_ylim(0, 1.15); pr.set_xlim(0, 1); pr.legend(frameon=False, loc='lower left')
rms = lambda sel: float(np.sqrt(np.mean((profile(S) - W)[sel] ** 2)))
pr.set_title(f'rms difference from the drawing, in body widths: head (s <= 0.14) {rms(S <= 0.14):.3f}, tail (s >= 0.5) {rms(S >= 0.5):.3f}\n'
             'between them the drawing narrows to 0.77-0.88 of its body width behind the head; the preset leaves that out', fontsize=8)
fig.savefig(os.path.join(OUT, 'fig_planarian.png'), dpi=110)
print('rms head:', rms(S <= 0.14), ' tail:', rms(S >= 0.5), ' middle (0.14 < s < 0.5):', rms((S > 0.14) & (S < 0.5)))
