"""Slide figures for the anatomy search (dark theme, 16:9 friendly).   python anatomy_figures.py   -> anatomy/fig_*.png"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from anatomy_targets import targets_on_grid, REF

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'anatomy')
BG, FG, MUTE, AMBER, CYAN, VIOLET, PINK = '#0a111a', '#f1eee6', '#b9c3d6', '#f2b24a', '#4cc9e0', '#b49cf5', '#ff8fa3'
GP = LinearSegmentedColormap.from_list('gpol', ['#0a111a', '#1b3a6b', '#2a5da8', '#4cc9e0', '#f2b24a', '#fff3d6'])
GP.set_bad(BG)
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG, 'text.color': FG, 'axes.labelcolor': FG, 'xtick.color': MUTE,
                     'ytick.color': MUTE, 'axes.edgecolor': '#3a4a66', 'font.family': 'DejaVu Sans', 'font.size': 15})
LAYER_COL = {'planarian': {'nerve': AMBER, 'gut': '#f6e85a', 'eyes': '#ffffff'}, 'tadpole': {'brain': CYAN, 'cord': '#ffffff', 'myomeres': PINK}}
SHAPES = ('planarian', 'tadpole')


def results():
    return [json.loads(l) for l in open(os.path.join(D, 'results.jsonl'))]


def plain(R, sh):
    return [r for r in R if r['shape'] == sh and not r.get('outline') and 'm' not in r and 'seed' not in r and 'T' not in r]


def rgb_target(shape, T, mask):
    import matplotlib.colors as mc
    rgb = np.zeros(mask.shape + (3,)); rgb[:] = mc.to_rgb(BG); rgb[mask] = (0.10, 0.16, 0.25)
    for nm, c in LAYER_COL[shape].items():
        a = np.clip(T[nm], 0, 1)[..., None]
        rgb = rgb * (1 - a) + a * np.array(mc.to_rgb(c))
    return rgb


def reference(shape):
    im = Image.open(os.path.join(REF, 'planarian_drawing.png' if shape == 'planarian' else 'tadpole_photo.png')).convert('RGB')
    if shape == 'planarian':
        im = im.crop((225, 95, 485, 1150)).rotate(90, expand=True)      # the animal only (labels cropped), head at the left
    return im


def show(ax, img, **kw):
    ax.imshow(img, interpolation='nearest', **kw); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_targets():
    fig, ax = plt.subplots(2, 2, figsize=(19.2, 6.6), gridspec_kw=dict(wspace=0.04, hspace=0.22))
    for j, (sh, asp) in enumerate((('planarian', 4.6), ('tadpole', 4.5))):
        Np = (96, int(round(48 * asp)) * 2)
        T, mask = targets_on_grid(sh, Np)
        show(ax[0, j], reference(sh)); ax[0, j].set_title('supplied reference', color=MUTE, fontsize=16, loc='left')
        show(ax[1, j], rgb_target(sh, T, mask))
        ax[1, j].set_title('schematic target:  ' + '   '.join(LAYER_COL[sh]), color=MUTE, fontsize=16, loc='left')
        x = 0.02
        for nm, c in LAYER_COL[sh].items():
            ax[1, j].text(x, -0.1, '■ ' + nm, color=c, transform=ax[1, j].transAxes, fontsize=17, va='top'); x += 0.2
    fig.savefig(os.path.join(D, 'fig_targets_dark.png'), dpi=100, bbox_inches='tight', pad_inches=0.25); plt.close(fig)


def fig_map():
    R = results()
    fig, ax = plt.subplots(1, 2, figsize=(19.2, 6.4), gridspec_kw=dict(wspace=0.22))
    for a, sh in zip(ax, SHAPES):
        rr = [r for r in plain(R, sh) if abs(r['aspect'] - (4.6 if sh == 'planarian' else 4.5)) < 1e-9 and (sh == 'planarian' or abs(r['tail'] - 0.4) < 1e-9)]
        rows = sorted({r['rows'] for r in rr}); rng = sorted({r['range'] for r in rr})
        Z = np.full((len(rows), len(rng)), np.nan)
        for r in rr:
            Z[rows.index(r['rows']), rng.index(r['range'])] = r['score']
        im = a.imshow(Z, cmap=GP, vmin=0, vmax=max(0.3, np.nanmax(Z)), origin='lower', aspect='auto')
        for i in range(len(rows)):
            for k in range(len(rng)):
                if np.isfinite(Z[i, k]):
                    a.text(k, i, f'{Z[i, k]:.2f}', ha='center', va='center', fontsize=10, color=BG if Z[i, k] > 0.55 * im.get_clim()[1] else FG)
        i, k = np.unravel_index(np.nanargmax(Z), Z.shape)
        a.add_patch(plt.Rectangle((k - .5, i - .5), 1, 1, fill=False, ec=FG, lw=3))
        a.set_xticks(range(len(rng))); a.set_xticklabels([f'{x:g}' for x in rng], fontsize=11); a.set_yticks(range(len(rows))); a.set_yticklabels([f'{x:g}' for x in rows])
        a.set_xlabel('field range (% of body width)'); a.set_ylabel('body width (cells)')
        a.set_title(f'{sh}: best-moment score', loc='left', color=FG, fontsize=18)
        cb = fig.colorbar(im, ax=a, fraction=0.035, pad=0.02); cb.outline.set_edgecolor('#3a4a66')
    fig.savefig(os.path.join(D, 'fig_search_map.png'), dpi=100, bbox_inches='tight', pad_inches=0.25); plt.close(fig)


def fig_best(sh, tag=None):
    R = results()
    b = max(plain(R, sh), key=lambda r: r['score']) if tag is None else [r for r in R if r['tag'] == tag][0]
    d = np.load(os.path.join(D, 'best', b['tag'] + '.npz'))
    mask = d['mask']; T = {k[2:]: d[k] for k in d.files if k.startswith('L_')}
    fig = plt.figure(figsize=(19.2, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.28, wspace=0.05)
    a0 = fig.add_subplot(gs[0, 0]); show(a0, reference(sh)); a0.set_title('reference', color=MUTE, fontsize=15, loc='left')
    a1 = fig.add_subplot(gs[0, 1]); show(a1, rgb_target(sh, T, mask)); a1.set_title('schematic target', color=MUTE, fontsize=15, loc='left')
    a2 = fig.add_subplot(gs[0, 2]); show(a2, np.where(mask, d['g'], np.nan), cmap=GP, vmin=0, vmax=2)
    a2.set_title(f"G_pol at the best moment, t = {b['t_best']:g} s   (score {b['score']:.2f})", color=AMBER, fontsize=15, loc='left')
    st = list(d['snap_t']); pick = [t for t in (10, 40, 90, 160, 240) if t in st][:3]
    for k, t in enumerate(pick):
        a = fig.add_subplot(gs[1, k]); show(a, np.where(mask, d['snaps'][st.index(t)], np.nan), cmap=GP, vmin=0, vmax=2)
        a.set_title(f'the same run at t = {t:g} s', color=MUTE, fontsize=15, loc='left')
    at = fig.add_subplot(gs[2, :]); tr = d['trace']; t = np.arange(len(tr)) * 0.5
    ok = tr[:, 3] >= 0.25
    at.plot(t, np.where(ok, tr[:, 0], np.nan), color=AMBER, lw=1.8, label='score: anatomy beyond a rim-to-centre gradient')
    at.plot(t, tr[:, 1], color=CYAN, lw=1.0, alpha=0.8, label='plain R² on the anatomy layers')
    at.axvline(b['t_best'], color=FG, lw=1, ls=':'); at.set_xlim(0, t[-1]); at.set_ylim(0, max(0.5, tr[:, :2].max() * 1.05))
    at.set_xlabel('simulated time (s)'); at.set_ylabel('share of variance'); at.legend(frameon=False, fontsize=13, loc='upper right', ncol=2)
    fig.savefig(os.path.join(D, f'fig_best_{sh}.png'), dpi=100, bbox_inches='tight', pad_inches=0.25); plt.close(fig)
    return b


if __name__ == '__main__':
    fig_targets(); fig_map()
    for sh in SHAPES:
        b = fig_best(sh); print(sh, b['tag'], round(b['score'], 3), b['t_best'])
