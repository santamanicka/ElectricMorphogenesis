"""Figures for the Potentia pattern atlas and the original-vs-Potentia comparison.
usage: python make_figures.py   (uses whatever results/*.pkl exist)
"""
import os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from sweeps import STIG
from patterns import _cell_blocks

os.makedirs('results', exist_ok=True)
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
HUE = {'settled': '#2a78d6', 'recurrent': '#eb6834', 'perpetual novelty': '#1baf7a', 'diverged': '#999999'}   # validated 3-slot palette
GLYPH = {'settled: uniform depolarised': 'D', 'settled: uniform intermediate': 'I', 'settled: uniform hyperpolarised': 'H', 'settled: static pattern': 'S',
         'recurrent: periodic oscillation': 'o', 'recurrent: periodic pattern': 'p',
         'perpetual novelty: oscillation': 'o', 'perpetual novelty: pattern': 'p', 'diverged': 'X'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
AXLAB = {'fieldScreenSize': 'field range $R_s$ (cell-corner-to-centre units)', 'bias': 'threshold  bias / bias$_*$', 'weight': 'transduction weight / weight$_*$',
         'GJStrength': 'GJ strength'}
MODEL_LABEL = {'lattice': 'Original (lattice)', 'potentia2': 'Potentia  m=2', 'potentia4': 'Potentia  m=4'}


def load(sweep, n):
    out = {}
    for tag, fn in (('lattice', f'results/atlas_{sweep}_{n}_lattice.pkl'), ('potentia2', f'results/atlas_{sweep}_{n}_potentia2.pkl'),
                    ('potentia4', f'results/atlas_{sweep}_{n}_potentia4.pkl')):
        if os.path.exists(fn):
            out[tag] = pickle.load(open(fn, 'rb'))
    return out


def ticklabels(name, vals):
    if name == 'bias': return [f'{v / STIG["bias"]:.2f}' for v in vals]
    if name == 'weight': return [f'{v / STIG["weight"]:g}' for v in vals]
    return [f'{v:g}' for v in vals]


def stig_index(name, vals):
    ref = {'fieldScreenSize': STIG['fieldScreenSize'], 'bias': STIG['bias'], 'weight': STIG['weight'], 'GJStrength': STIG['GJStrength']}[name]
    return int(np.argmin(np.abs(np.array(vals, float) - ref)))


def regime_maps(sweep, n):
    D = load(sweep, n)
    if not D: return
    tags = list(D)
    ax_names = list(next(iter(D.values()))['axes'])
    rows_n, cols_n = ax_names
    rv, cv = next(iter(D.values()))['axes'][rows_n], next(iter(D.values()))['axes'][cols_n]
    fig, axs = plt.subplots(2, len(tags), figsize=(3.9 * len(tags), 7.6), squeeze=False)
    nmax = 12.0
    for j, tag in enumerate(tags):
        R = D[tag]['results']; shape = D[tag]['shape']
        a = axs[0, j]
        for k, r in enumerate(R):
            i, c = np.unravel_index(k, shape)
            hue = HUE[r['regime'].split(':')[0]]
            a.add_patch(Rectangle((c - .5, i - .5), 1, 1, facecolor=hue, alpha=0.9, edgecolor=SURF, lw=1.5))
            a.text(c, i, GLYPH[r['regime']], ha='center', va='center', color='white' if hue != HUE['perpetual novelty'] else INK, fontsize=9, fontweight='bold')
        si, sc = stig_index(rows_n, rv), stig_index(cols_n, cv)
        a.add_patch(Rectangle((sc - .5, si - .5), 1, 1, fill=False, edgecolor=INK, lw=2.4))
        a.set_xlim(-.5, shape[1] - .5); a.set_ylim(shape[0] - .5, -.5)
        a.set_title(f'{MODEL_LABEL[tag]}\nregime', fontsize=10)
        b = axs[1, j]
        nv = np.array([r['desc'].get('novelty', np.nan) for r in R]).reshape(shape)
        im = b.imshow(nv, cmap='Greens', vmin=0, vmax=nmax, aspect='auto')
        b.add_patch(Rectangle((sc - .5, si - .5), 1, 1, fill=False, edgecolor=INK, lw=2.4))
        for (i, c), v in np.ndenumerate(nv):
            b.text(c, i, f'{v:.0f}', ha='center', va='center', fontsize=7, color='white' if v > 7 else INK2)
        b.set_title('novelty score (mV)', fontsize=10)
        for x in (a, b):
            x.set_xticks(range(shape[1])); x.set_xticklabels(ticklabels(cols_n, cv), rotation=60, fontsize=7)
            x.set_yticks(range(shape[0])); x.set_yticklabels(ticklabels(rows_n, rv), fontsize=8)
            x.set_xlabel(AXLAB[cols_n], fontsize=8)
            if j == 0: x.set_ylabel(AXLAB[rows_n], fontsize=8)
            for s in x.spines.values(): s.set_visible(False)
    handles = [Patch(color=HUE['settled'], label='settled  (D depolarised, H hyperpolarised, I intermediate, S static pattern)'),
               Patch(color=HUE['recurrent'], label='recurrent  (o oscillation, p pattern)'),
               Patch(color=HUE['perpetual novelty'], label='perpetual novelty: never settles, never repeats  (o, p)'),
               Patch(facecolor='none', edgecolor=INK, lw=2, label='Stigmergic operating point')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f'Endogenous regimes from a homogeneous start: {n}x{n} finite tissue, T={next(iter(D.values()))["T"]:.0f} s, no clamp', fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fn = f'results/fig_regimes_{sweep}_{n}.png'; fig.savefig(fn, dpi=140); plt.close(fig); print('wrote', fn)


def side_by_side(sweep, n, picks, fname, title, tags_wanted=('lattice', 'potentia2', 'potentia4'), tag_n=None):
    D = load(sweep, tag_n or n)
    tags = [t for t in tags_wanted if t in D]
    shape = D[tags[0]]['shape']; T = D[tags[0]]['T']
    times = [3, 10, 30, 100, 'final']
    names = list(D[tags[0]]['axes'])
    nb, nm = len(picks), len(tags)
    heights = ([0.62] + [1.0] * nm) * nb
    fig = plt.figure(figsize=(1.45 * len(times) + 2.6, sum(heights) * 1.45 + 0.7))
    gs = fig.add_gridspec(len(heights), len(times) + 2, height_ratios=heights, width_ratios=[1] * len(times) + [1.25, 0.07],
                          hspace=0.12, wspace=0.07, left=0.06, right=0.97, top=0.955, bottom=0.02)
    for bi, (k, why) in enumerate(picks):
        r0 = bi * (nm + 1)
        p = {nm_: D[tags[0]]['results'][k]['cfg']['params'][nm_] for nm_ in names}
        lab = '   '.join(f'{nm_.replace("fieldScreenSize","R_s").replace("GJStrength","GJ")} = {(p[nm_]/STIG[nm_]) if nm_ in ("bias","weight") else p[nm_]:.3g}{"x*" if nm_ in ("bias","weight") else ""}' for nm_ in names)
        hx = fig.add_subplot(gs[r0, :len(times) + 1]); hx.axis('off')
        hx.text(0, 0.35, f'{why}:   {lab}', fontsize=9, fontweight='bold', color=INK, va='bottom')
        for ti, tag in enumerate(tags):
            r = D[tag]['results'][k]
            for j, t in enumerate(times):
                a = fig.add_subplot(gs[r0 + 1 + ti, j])
                X = r['final'] if t == 'final' else r['keep'][t]
                a.imshow(np.asarray(X, float), cmap='viridis', vmin=-55, vmax=-5, interpolation='nearest')
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values(): sp.set_color(HUE[r['regime'].split(':')[0]]); sp.set_linewidth(2.4 if t == 'final' else 0.6)
                if bi == 0 and ti == 0: a.set_title(f't = {t} s' if t != 'final' else f't = {T:.0f} s', fontsize=9)
                if j == 0: a.set_ylabel(MODEL_LABEL[tag].replace('Original (lattice)', 'Original').replace('Potentia  ', 'Potentia '), fontsize=8)
            tx = fig.add_subplot(gs[r0 + 1 + ti, len(times)]); tx.axis('off')
            tx.text(0.02, 0.5, r['regime'].replace(': ', ':\n') + f'\nnovelty {r["desc"]["novelty"]:.1f} mV', fontsize=7.5, va='center', color=INK2)
    cax = fig.add_subplot(gs[1:3 if nm >= 2 else 2, len(times) + 1])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(-55, -5)), cax=cax); cb.set_label('Vmem (mV)', fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.savefig(fname, dpi=130); plt.close(fig); print('wrote', fname)


def pick_sets(sweep, n):
    D = load(sweep, n)
    L, P = D['lattice']['results'], D['potentia2']['results']
    shape = D['lattice']['shape']
    names = list(D['lattice']['axes'])
    ki = np.ravel_multi_index(tuple(stig_index(nm, D['lattice']['axes'][nm]) for nm in names), shape)
    nl = np.array([r['desc']['novelty'] for r in L]); npot = np.array([r['desc']['novelty'] for r in P])
    coarse = lambda r: r['regime'].split(':')[0]
    agree = np.array([coarse(a) == coarse(b) for a, b in zip(L, P)])
    sets = [(int(ki), 'Stigmergic operating point'), (int(np.argmax(nl)), 'highest novelty in original'), (int(np.argmax(npot)), 'highest novelty in Potentia')]
    def first(cond, why):
        idx = np.where(cond)[0]
        return [(int(idx[len(idx) // 2]), why)] if len(idx) else []
    rg = lambda R, s: np.array([r['regime'] == s for r in R])
    A = first(agree & rg(L, 'settled: uniform hyperpolarised') & rg(P, 'settled: uniform hyperpolarised'), 'both settle hyperpolarised')
    B = first(agree & rg(L, 'settled: uniform depolarised') & rg(P, 'settled: uniform depolarised'), 'both settle depolarised')
    Cc = first(~agree, 'disagreement (coarse regime differs)')
    return sets, A + B + Cc


def atlas_gallery(n, fname):
    D = load('S1', n)
    if not D: return
    tags = [t for t in ('lattice', 'potentia2') if t in D]
    shape = D[tags[0]]['shape']; axes_d = D[tags[0]]['axes']
    fig = plt.figure(figsize=(7.2 * len(tags), 6.2))
    for ti, tag in enumerate(tags):
        sub = fig.add_gridspec(shape[0], shape[1], left=0.04 + ti * 0.5, right=0.46 + ti * 0.5, bottom=0.08, top=0.9, wspace=0.05, hspace=0.05)
        for k, r in enumerate(D[tag]['results']):
            i, c = np.unravel_index(k, shape)
            a = fig.add_subplot(sub[i, c])
            a.imshow(np.asarray(r['final'], float), cmap='viridis', vmin=-55, vmax=-5, interpolation='nearest')
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values(): s.set_color(HUE[r['regime'].split(':')[0]]); s.set_linewidth(2.2)
            if i == 0 and (c % 2 == 0): a.set_title(ticklabels('bias', axes_d['bias'])[c], fontsize=6.5, pad=2)
            if c == 0: a.set_ylabel(f'{axes_d["fieldScreenSize"][i]:g}', fontsize=7, rotation=0, labelpad=8, va='center')
        fig.text(0.25 + ti * 0.5, 0.945, MODEL_LABEL[tag], ha='center', fontsize=11)
        fig.text(0.25 + ti * 0.5, 0.035, 'bias / bias$_*$ (columns)   —   rows: field range $R_s$ (top = 1)', ha='center', fontsize=8)
    fig.suptitle(f'Final Vmem at t={D[tags[0]]["T"]:.0f} s, {n}x{n} tissue; border colour = regime (blue settled, orange recurrent, green perpetual novelty)', fontsize=10)
    fig.savefig(fname, dpi=130); plt.close(fig); print('wrote', fname)


def novelty_vs_bias(sweep, n, fname, at_row_value=4.0):
    D = load(sweep, n)
    if not D: return
    names = list(next(iter(D.values()))['axes'])
    shape = next(iter(D.values()))['shape']
    rows_n, cols_n = names
    rv = np.array(next(iter(D.values()))['axes'][rows_n], float)
    ri = int(np.argmin(np.abs(rv - {'fieldScreenSize': STIG['fieldScreenSize'], 'weight': STIG['weight'], 'GJStrength': STIG['GJStrength']}[rows_n])))
    cv = np.array(next(iter(D.values()))['axes'][cols_n], float)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    cols = {'lattice': INK, 'potentia2': '#1baf7a', 'potentia4': '#2a78d6'}
    for tag, R in D.items():
        for a, key, yl in zip(axs, ('novelty', 'act_last', 'fh'), ('novelty score (mV)', 'late-time activity (mV)', 'hyperpolarised fraction')):
            y = np.array([r['desc'][key] for r in R['results']]).reshape(shape)[ri]
            x = cv / STIG['bias'] if cols_n == 'bias' else cv
            a.plot(x, y, marker='o', ms=4, lw=1.8, color=cols[tag], label=MODEL_LABEL[tag])
            a.set_ylabel(yl); a.set_xscale('log' if cols_n == 'bias' else 'linear'); a.set_xlabel(AXLAB[cols_n])
    for a in axs:
        a.axvline(1.0 if cols_n == 'bias' else STIG['fieldScreenSize'], color=INK2, ls=':', lw=1)
        for s in ('top', 'right'): a.spines[s].set_visible(False)
        a.grid(alpha=0.2)
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].text(1.03, 0.95, 'Stigmergic value', transform=axs[0].transAxes, fontsize=7, color=INK2, rotation=90, va='top')
    fig.suptitle(f'{rows_n} = {rv[ri]:g}, {n}x{n} tissue (dotted line: Stigmergic value)', fontsize=10)
    fig.tight_layout(); fig.savefig(fname, dpi=140); plt.close(fig); print('wrote', fname)


def fine_scan_plot(sweep, fname, xlabel, xvals_fn, ref_x):
    """Mean +/- sd over the 3 jitter replicates (last axis) of novelty, activity and hyperpolarised fraction, per model."""
    D = load(sweep, 11)
    if not D: return
    cols = {'lattice': INK, 'potentia2': '#1baf7a', 'potentia4': '#2a78d6'}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    for tag, R in D.items():
        shape = R['shape']
        names = list(R['axes'])
        x = xvals_fn(np.array(R['axes'][names[0]], float))
        for a, key, yl in zip(axs, ('novelty', 'act_last', 'fh'), ('novelty score (mV)', 'late-time activity (mV)', 'hyperpolarised fraction')):
            Y = np.array([r['desc'][key] for r in R['results']]).reshape(shape)
            mu, sd = Y.mean(axis=1), Y.std(axis=1, ddof=1)
            a.errorbar(x, mu, yerr=sd, marker='o', ms=3.5, lw=1.6, capsize=2, color=cols[tag], label=MODEL_LABEL[tag])
            a.set_ylabel(yl); a.set_xlabel(xlabel)
    for a in axs:
        a.axvline(ref_x, color=INK2, ls=':', lw=1)
        for sp in ('top', 'right'): a.spines[sp].set_visible(False)
        a.grid(alpha=0.2)
        if sweep == 'F1': a.set_xscale('log')
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'{sweep}: mean $\\pm$ sd over 3 jitter replicates (dotted line: Stigmergic value), 11x11 tissue', fontsize=10)
    fig.tight_layout(); fig.savefig(fname, dpi=140); plt.close(fig); print('wrote', fname)


if __name__ == '__main__':
    for sweep in ('S1', 'S2', 'S3'):
        regime_maps(sweep, 11)
    A, B = pick_sets('S1', 11)
    side_by_side('S1', 11, A, 'results/fig_side_by_side_A.png', 'Same parameter sets, original vs Potentia (11x11, homogeneous start); bias* = Stigmergic bias; last-column border = regime')
    side_by_side('S1', 11, B, 'results/fig_side_by_side_B.png', 'Agreement and disagreement cases, original vs Potentia (11x11); bias* = Stigmergic bias')
    novelty_vs_bias('S1', 11, 'results/fig_novelty_vs_bias_S1_11.png')
    if os.path.exists('results/atlas_S1sub_25_lattice.pkl'):
        D = load('S1sub', 25)
        ks = [0, 7, 12, 17]   # R_s=2 / bias .25; R_s=4 / bias .5; R_s=6 / bias 1; R_s=8 / bias 2 (row-major over (R_s, bias))
        ks = [int(np.ravel_multi_index((1, 2), D['lattice']['shape'])), int(np.ravel_multi_index((2, 3), D['lattice']['shape'])), int(np.ravel_multi_index((0, 2), D['lattice']['shape']))]
        side_by_side('S1sub', 25, [(ks[0], 'Stigmergic operating point, 25x25 tissue'), (ks[1], 'R_s=6, bias 2x*'), (ks[2], 'R_s=2, bias 1x*')],
                     'results/fig_side_by_side_25.png', 'Original vs Potentia on a 25x25 tissue (same parameter sets, homogeneous start)', tags_wanted=('lattice', 'potentia2'), tag_n=25)
    atlas_gallery(25, 'results/fig_atlas_25.png')
    for sw, xl, fn, ref in (('F1', 'threshold  bias / bias$_*$', lambda v: v / STIG['bias'], 1.0), ('F2', 'field range $R_s$', lambda v: v, STIG['fieldScreenSize'])):
        fine_scan_plot(sw, f'results/fig_fine_{sw}.png', xl, fn, ref)
    regime_maps('F3', 11)
