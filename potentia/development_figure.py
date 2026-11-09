"""The development sequence: the six keyframes on the default 100 x 260 pixel grid (50 x 130 cells, m = 2).   python development_figure.py   (writes results/fig_development_sequence.png)"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from development import STAGES, stage_mask

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
R, C = 100, 260
fig, axs = plt.subplots(2, 3, figsize=(12, 4.6))
for ax, st in zip(axs.ravel(), STAGES):
    mk = stage_mask(st, R, C)
    ax.imshow(~mk, cmap='gray', vmin=-0.3, vmax=1.0, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{st['name']} ({st['nf']}), time fraction {st['tau']:.2f}, {int(mk.sum())} px", fontsize=9, loc='left')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_development_sequence.png'), dpi=110)
print([int(stage_mask(st, R, C).sum()) for st in STAGES])
