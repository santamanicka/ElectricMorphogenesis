"""Figures for the teaser slides (magma G_pol 0-2 on white, like the simulator's own view).   python anatomy_teaser_figures.py  -> anatomy/teaser/*.png
growth.png   the simulator-outline runs of the talk's two frames at four times from the uniform start
outline.png  the same settings in four outlines (own, ellipse, capsule, rectangle) at 120 s, with the head-tail asymmetry of each pattern"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(HERE, 'anatomy', 'feat', 'best'); OUT = os.path.join(HERE, 'anatomy', 'teaser'); os.makedirs(OUT, exist_ok=True)


def asym(g, mask):
    both = mask & mask[:, ::-1]
    return float(1.0 - np.corrcoef(g[both], g[:, ::-1][both])[0, 1])


def frame(ax, g, mask, title=None):
    ax.imshow(np.where(mask, g, np.nan), cmap='magma', vmin=0, vmax=2, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=15, loc='left', color='#333')


def growth():
    runs = [('planarian_r40_a4.4_f20_pinch_sim.npz', 'planarian outline'), ('tadpole_r24_a4.5_f25_w0.4_sim_p0.4_q2.42.npz', 'tadpole outline')]
    times = (10, 40, 90, 160)
    fig, ax = plt.subplots(len(times), 2, figsize=(16.8, 7.6), gridspec_kw=dict(wspace=0.04, hspace=0.28))
    for j, (f, lab) in enumerate(runs):
        d = np.load(os.path.join(B, f)); st = list(d['snap_t'])
        for i, t in enumerate(times):
            frame(ax[i, j], d['snaps'][st.index(t)], d['mask'], f'{lab}, t = {t} s' if True else None)
    fig.savefig(os.path.join(OUT, 'growth.png'), dpi=100, bbox_inches='tight', facecolor='white', pad_inches=0.15)


def outline(t=120):
    sets = [('planarian_r40_a4.4_f20_pinch', 'planarian settings: 40 cells wide, range 20%'), ('tadpole_r24_a4.5_f25_w0.4', 'tadpole settings: 24 cells wide, range 25%')]
    rows = [('', 'own outline'), ('_ellipse', 'ellipse'), ('_stadium', 'capsule'), ('_rectangle', 'rectangle')]
    fig, ax = plt.subplots(len(rows), 2, figsize=(16.8, 7.6), gridspec_kw=dict(wspace=0.04, hspace=0.32))
    info = {}
    for j, (base, head) in enumerate(sets):
        for i, (suf, lab) in enumerate(rows):
            d = np.load(os.path.join(B, base + suf + '.npz')); st = list(d['snap_t']); g = d['snaps'][st.index(t)]; a = asym(g, d['mask'])
            info[(base, lab)] = a
            frame(ax[i, j], g, d['mask'], (head + '\n' if i == 0 else '') + f'{lab}: head-tail asymmetry {a:.2f}')
    fig.savefig(os.path.join(OUT, 'outline.png'), dpi=100, bbox_inches='tight', facecolor='white', pad_inches=0.15)
    return info


if __name__ == '__main__':
    growth()
    for k, v in outline().items():
        print(k, round(v, 2))
