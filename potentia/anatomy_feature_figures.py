"""Slide figures for the feature-filter anatomy search (dark theme).   python anatomy_feature_figures.py   -> anatomy/feat/fig_*.png"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anatomy_figures import BG, FG, MUTE, AMBER, CYAN, VIOLET, PINK, GP, LAYER_COL, SHAPES, reference, rgb_target, show
from anatomy_targets import targets_on_grid, TAIL_BASE
import anatomy_features as AF

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'anatomy', 'feat')
REF_ASPECT = {'planarian': 4.6, 'tadpole': 4.5}
CH_LABEL = {'coarse': 'layout', 'segments': 'segments', 'cords': 'cords', 'spots': 'spots'}


def results():
    return [json.loads(l) for l in open(os.path.join(D, 'results.jsonl'))]


def plain(R, sh):
    return [r for r in R if r['shape'] == sh and not r.get('outline') and 'm' not in r]


def baselines(sh):
    Ft = AF.target_features(sh); z = lambda I: (I - I.mean()) / I.std()
    noise = np.mean([AF.match(AF.features(np.random.default_rng(i).standard_normal((AF.NSR, AF.NUR))), Ft)['score'] for i in range(50)])
    stripe = np.zeros((AF.NSR, AF.NUR)); stripe[:, AF.NUR // 2 - 2:AF.NUR // 2 + 3] = 1
    other = AF.match(AF.target_features('tadpole' if sh == 'planarian' else 'planarian'), Ft)['score']
    return dict(noise=float(noise), stripe=AF.match(AF.features(z(stripe)), Ft)['score'], other=other)


def fig_map():
    R = results()
    fig, ax = plt.subplots(1, 2, figsize=(19.2, 6.4), gridspec_kw=dict(wspace=0.2))
    for a, sh in zip(ax, SHAPES):
        rr = [r for r in plain(R, sh) if abs(r['aspect'] - REF_ASPECT[sh]) < 1e-9 and (sh == 'planarian' or abs(r['tail'] - TAIL_BASE) < 1e-9)]
        rows = sorted({r['rows'] for r in rr}); rng = sorted({r['range'] for r in rr})
        Z = np.full((len(rows), len(rng)), np.nan)
        for r in rr:
            Z[rows.index(r['rows']), rng.index(r['range'])] = r['score']
        lo = baselines(sh)['noise']
        im = a.imshow(Z, cmap=GP, vmin=lo, vmax=np.nanmax(Z), origin='lower', aspect='auto')
        for i in range(len(rows)):
            for k in range(len(rng)):
                if np.isfinite(Z[i, k]):
                    a.text(k, i, f'{Z[i, k]:.2f}', ha='center', va='center', fontsize=11, color=BG if Z[i, k] > lo + 0.6 * (np.nanmax(Z) - lo) else FG)
        i, k = np.unravel_index(np.nanargmax(Z), Z.shape)
        a.add_patch(plt.Rectangle((k - .5, i - .5), 1, 1, fill=False, ec=FG, lw=3))
        a.set_xticks(range(len(rng))); a.set_xticklabels([f'{x:g}' for x in rng], fontsize=12); a.set_yticks(range(len(rows))); a.set_yticklabels([f'{x:g}' for x in rows])
        a.set_xlabel('field range (% of body width)'); a.set_ylabel('body width (cells)')
        a.set_title(f'{sh}: feature match at the best moment of each run', loc='left', color=FG, fontsize=16)
        cb = fig.colorbar(im, ax=a, fraction=0.035, pad=0.02); cb.outline.set_edgecolor('#3a4a66')
    fig.savefig(os.path.join(D, 'fig_feat_map.png'), dpi=100, bbox_inches='tight', pad_inches=0.25); plt.close(fig)


def fig_best(sh, tag=None, name=None):
    R = results()
    b = max(plain(R, sh), key=lambda r: r['score']) if tag is None else [r for r in R if r['tag'] == tag][0]
    d = np.load(os.path.join(D, 'best', b['tag'] + '.npz')); mask = d['mask']
    m = int(b.get('m', 2)); Np = mask.shape
    T, tmask = targets_on_grid(sh, Np, b.get('tail', TAIL_BASE))
    rect = AF.rectifier(sh, Np, b.get('tail', TAIL_BASE), mask)
    base = baselines(sh)
    fig = plt.figure(figsize=(19.2, 9.0))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.95], hspace=0.34, wspace=0.12)
    a = fig.add_subplot(gs[0, 0:3]); show(a, reference(sh)); a.set_title('reference', color=MUTE, fontsize=15, loc='left')
    a = fig.add_subplot(gs[0, 3:6]); show(a, np.where(mask, d['g'], np.nan), mask, cmap=GP, vmin=0, vmax=2)
    a.set_title(f"G_pol at the best moment (t = {b['t_best']:g} s): {b['rows']:g} cells wide, {b['cols']} long, field range {b['range']:g}% of the width", color=AMBER, fontsize=15, loc='left')
    a = fig.add_subplot(gs[1, 0:3]); show(a, rgb_target(sh, T, tmask)); a.set_title('schematic target', color=MUTE, fontsize=15, loc='left')
    a = fig.add_subplot(gs[1, 3:6]); show(a, np.where(mask, d['V'] * 1e3, np.nan), mask, cmap='magma', vmin=-55, vmax=-5)
    a.set_title('Vmem at the same moment', color=MUTE, fontsize=15, loc='left')
    ab = fig.add_subplot(gs[2, 0:2])
    ch = list(AF.CHANNELS); vals = [b['channels'][k] for k in ch]
    ab.bar(range(4), vals, color=[CYAN, PINK, AMBER, VIOLET], width=0.62)
    for i, v in enumerate(vals):
        ab.text(i, v + 0.02, f'{v:.2f}', ha='center', color=FG, fontsize=14)
    ab.set_xticks(range(4)); ab.set_xticklabels([CH_LABEL[k] for k in ch], fontsize=14); ab.set_ylim(0, 1.08); ab.set_ylabel('match to the target')
    ab.set_title(f"feature match {b['score']:.2f}", color=AMBER, fontsize=15, loc='left')
    at = fig.add_subplot(gs[2, 2:6]); tr = d['trace']; t = np.arange(len(tr)); ok = tr[:, 6] >= 0.25
    at.plot(t, np.where(ok, tr[:, 0], np.nan), color=AMBER, lw=2, label=f'match to the {sh} target')
    at.plot(t, np.where(ok, tr[:, 5], np.nan), color=VIOLET, lw=1.2, label='the same frames against the other animal’s target')
    at.axhline(base['noise'], color=MUTE, lw=1, ls='--'); at.text(t[-1], base['noise'], ' noise', color=MUTE, va='center', fontsize=12)
    at.axhline(base['stripe'], color=MUTE, lw=1, ls=':'); at.text(t[-1], base['stripe'], ' plain stripe', color=MUTE, va='center', fontsize=12)
    at.axvline(b['t_best'], color=FG, lw=1, ls=':'); at.set_xlim(0, t[-1]); at.set_ylim(0.4, 0.95)
    at.set_xlabel('simulated time (s)'); at.legend(frameon=False, fontsize=13, loc='upper left', ncol=2)
    fig.savefig(os.path.join(D, name or f'fig_feat_best_{sh}.png'), dpi=100, bbox_inches='tight', pad_inches=0.25); plt.close(fig)
    return b, base


def fig_shapes(times=(40, 120), tags=None, name='fig_feat_shapes.png'):
    R = results(); by = {r['tag']: r for r in R}
    outs = ('planarian', 'tadpole', 'stadium', 'ellipse', 'rectangle')
    fig, ax = plt.subplots(len(outs), 4, figsize=(19.2, 9.0), gridspec_kw=dict(wspace=0.04, hspace=0.3))
    info = {}
    for j, sh in enumerate(SHAPES):
        b = max(plain(R, sh), key=lambda r: r['score']) if tags is None else by[tags[sh]]; info[sh] = {}
        for i, o in enumerate(outs):
            r = b if o == sh else by.get(b['tag'] + '_' + o)
            if r is None:
                continue
            info[sh][o] = r['score']
            d = np.load(os.path.join(D, 'best', r['tag'] + '.npz')); st = list(d['snap_t']); mask = d['mask']
            for k, t in enumerate(times):
                a = ax[i, 2 * j + k]
                show(a, np.where(mask, d['snaps'][st.index(t)], np.nan), mask, cmap=GP, vmin=0, vmax=2)
                if k == 0:
                    a.set_title(f"{o} outline" + (' (its own)' if o == sh else '') + f": best match to the {sh} target {r['score']:.2f}", fontsize=12, color=AMBER if o == sh else FG, loc='left')
                else:
                    a.set_title(f't = {t} s' + (f"      {sh} parameters: {b['rows']:g} cells wide, range {b['range']:g}%" if i == 0 else ''), fontsize=12, color=MUTE, loc='left')
    fig.savefig(os.path.join(D, name), dpi=100, bbox_inches='tight', pad_inches=0.25); plt.close(fig)
    return info


def fig_gallery(n=4):
    """The best moments of the n best runs of each animal (different size / range / proportions)."""
    R = results()
    fig, ax = plt.subplots(n, 2, figsize=(19.2, 8.4), gridspec_kw=dict(wspace=0.04, hspace=0.32))
    for j, sh in enumerate(SHAPES):
        for i, r in enumerate(sorted(plain(R, sh), key=lambda r: -r['score'])[:n]):
            d = np.load(os.path.join(D, 'best', r['tag'] + '.npz'))
            show(ax[i, j], np.where(d['mask'], d['g'], np.nan), d['mask'], cmap=GP, vmin=0, vmax=2)
            extra = f", tail {r['tail']:g}" if sh == 'tadpole' else ''
            ax[i, j].set_title(f"{sh}: {r['rows']:g} cells wide, length/width {r['aspect']:g}{extra}, range {r['range']:g}%, t = {r['t_best']:g} s   match {r['score']:.2f}", fontsize=12, color=FG, loc='left')
    fig.savefig(os.path.join(D, 'fig_feat_gallery.png'), dpi=100, bbox_inches='tight', pad_inches=0.25); plt.close(fig)


if __name__ == '__main__':
    fig_map(); fig_gallery()
    for sh in SHAPES:
        b, base = fig_best(sh); print(sh, b['tag'], round(b['score'], 3), b['t_best'], {k: round(v, 2) for k, v in b['channels'].items()}, 'cross', round(b['cross'], 3), base)
    try:
        print(fig_shapes())
    except Exception as e:
        print('shapes figure not made yet:', e)
