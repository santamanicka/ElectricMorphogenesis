"""The position-dependent field range on several shapes.   python local_range_figure.py   (writes results/fig_local_range.png)

Each cell senses over a range that is a fixed share (here 25.7%, the Stigmergic range for a 110 um box) of the local width of the tissue: the diameter of the largest circle that fits in the tissue and
contains the cell (localrange.py), at least 3 pixels.  Colour = the field range in um (pixel pitch 5 um); the single-range value of the Stigmergic set is 28.3 um everywhere.
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
cases = [('tadpole', (22, 100)), ('planarian', (22, 96)), ('dumbbell', (22, 66)), ('ellipse', (22, 66)), ('triangle', (40, 40)), ('cross', (40, 40))]
fig, axs = plt.subplots(3, 2, figsize=(11, 7.5))
vmax = 0
maps = []
for name, Np in cases:
    mk = shape_mask(name, Np); Rm = range_map(mk, 25.7, h) * 1e6; Rm = np.where(mk, Rm, np.nan); maps.append(Rm); vmax = max(vmax, np.nanmax(Rm))
for ax, (name, Np), Rm in zip(axs.ravel(), cases, maps):
    im = ax.imshow(Rm, cmap='viridis', vmin=0, vmax=vmax, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{name}: {np.nanmin(Rm):.1f} to {np.nanmax(Rm):.1f} um', fontsize=9, loc='left')
fig.colorbar(im, ax=axs, shrink=0.8, label='field range (um)')
fig.savefig(os.path.join(OUT, 'fig_local_range.png'), dpi=110, bbox_inches='tight')
print('wrote fig_local_range.png')
