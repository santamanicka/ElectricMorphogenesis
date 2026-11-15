"""Why the cells at the edge of a broad region go quiet, and what the edge shoulder of the position-dependent range does.   python edge_activity_figure.py   (writes results/fig_edge_activity.png; about 10 minutes)

The tadpole (50 x 130 cells at m = 2, 100 x 260 pixels, the Stigmergic parameters, homogeneous start with g = 1) with the position-dependent range of 30% of the Gaussian-smoothed thickness runs for 70 s, once with
the edge shoulder off (width 0) and once with the default (width 1 range).  Colour = G_pol / G_ref (0 = quiet, 2 = saturated).  Without the shoulder every cell within about one range of the edge of the broad head has
half of its range disc outside the tissue, where there are no charges: the field it senses is lopsided and large, and the cell is suppressed.  With it the range falls towards the edge as the half-chord of a circle of
the cell's own range, so the disc stays (nearly) inside.  Right: the fraction of cells with g < 0.1 in the head (columns up to 35% of the length) against the distance from the edge, mean of snapshots every 5 s from 50 s.
"""
import os
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from potentia import Potentia
from shapes import shape_mask
from localrange import range_map, _d2_to_zero
from sweeps import STIG

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SIZE, M, PCT, T_END = (50, 130), 2, 30.0, 70.0
BANDS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]


def run(edge):
    mask = shape_mask('tadpole', (SIZE[0] * M, SIZE[1] * M))
    P = Potentia(size_cells=SIZE, m=M, mask=mask, **{**STIG, 'fieldScreenSize': 0.4})          # the single range is not used
    P.set_range_map(range_map(mask, PCT, P.h, 'gaussian', 0.10, edge))
    V, g = P.initial_from_g(1.0, 'dep')
    d = np.sqrt(_d2_to_zero(np.pad(mask, 1, constant_values=False))[1:-1, 1:-1]); head = np.zeros_like(mask); head[:, :int(0.35 * mask.shape[1])] = True
    snaps = []
    for it in range(1, int(round(T_END / P.dt)) + 1):
        V, g = P.step(V, g)
        if it * P.dt >= T_END - 20 - 1e-9 and it % int(round(5 / P.dt)) == 0:
            snaps.append([float(np.mean(g[mask & head & (d > a) & (d <= b)] < 0.1)) for a, b in BANDS])
    return edge, g, np.mean(snaps, axis=0)


if __name__ == '__main__':
    with Pool(2) as pool:
        res = dict((e, (g, q)) for e, g, q in pool.map(run, [0.0, 1.0]))
    mask = shape_mask('tadpole', (SIZE[0] * M, SIZE[1] * M))
    fig = plt.figure(figsize=(12, 6.4))
    for k, (e, lab) in enumerate([(0.0, 'edge shoulder off (width 0)'), (1.0, 'edge shoulder, width 1 range (default)')]):
        ax = fig.add_axes([0.03, 0.55 - 0.5 * k, 0.60, 0.40]); ax.imshow(np.where(mask, res[e][0], np.nan), cmap='magma', vmin=0, vmax=2, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{lab}, t = {T_END:.0f} s', fontsize=9, loc='left')
    ax = fig.add_axes([0.70, 0.18, 0.27, 0.70]); x = np.arange(len(BANDS)); labs = ['0-2', '2-4', '4-8', '8-16', '> 16']
    ax.bar(x - 0.2, res[0.0][1], 0.4, label='shoulder off'); ax.bar(x + 0.2, res[1.0][1], 0.4, label='shoulder on')
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlabel('distance from the edge (pixels)'); ax.set_ylabel('fraction of the cells in the head with g < 0.1'); ax.set_ylim(0, 1.2); ax.legend(frameon=False, loc='upper center', ncol=2)
    fig.savefig(os.path.join(OUT, 'fig_edge_activity.png'), dpi=110)
    for e in (0.0, 1.0):
        print(f'edge {e}: quiet fraction by distance band', np.round(res[e][1], 2).tolist())
