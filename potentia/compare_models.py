"""Same parameter sets, original (lattice) vs Potentia: how similar are the patterns?

Compares, for every parameter set of a sweep, three levels:
  1. early deterministic snapshots (t = 3, 10, 30, 100 s): rms difference of cell-averaged Vmem (mV)
  2. late-time statistics (fh, Vmean, sd_x, act, edge, novelty): rank correlation across parameter sets
  3. regime label agreement
and uses Potentia(m=2) vs Potentia(m=4) as the *control* for how much disagreement is just sensitivity of
the dynamics (as opposed to lattice-vs-continuum discretisation).
usage: python compare_models.py S1 11
"""
import sys, pickle, itertools
import numpy as np
from scipy.stats import spearmanr
from patterns import _cell_blocks

COARSE = {'settled': 's', 'recurrent': 'r', 'perpetual novelty': 'n', 'diverged': 'x'}
coarse = lambda r: COARSE[r.split(':')[0]]
STATS = ['fh', 'Vmean', 'sd_x', 'act', 'edge', 'novelty']
KEEP = [3, 10, 30, 100]


def load(sweep, n):
    out = {}
    for tag, fn in (('lattice', f'results/atlas_{sweep}_{n}_lattice.pkl'), ('potentia2', f'results/atlas_{sweep}_{n}_potentia2.pkl'),
                    ('potentia4', f'results/atlas_{sweep}_{n}_potentia4.pkl')):
        try:
            out[tag] = pickle.load(open(fn, 'rb'))
        except FileNotFoundError:
            pass
    return out


def cellsnap(r, t, m):
    X = r['keep'][t].astype(np.float64)
    return _cell_blocks(X, m)


def pair_stats(A, B, ma, mb, label):
    ra, rb = A['results'], B['results']
    n = len(ra)
    exact = np.mean([x['regime'] == y['regime'] for x, y in zip(ra, rb)])
    coarse_agree = np.mean([coarse(x['regime']) == coarse(y['regime']) for x, y in zip(ra, rb)])
    print(f'\n--- {label}')
    print(f'  regime agreement: exact {exact:.2f}   coarse (settled/recurrent/novel) {coarse_agree:.2f}')
    row = []
    for t in KEEP:
        d = np.array([np.sqrt(((cellsnap(x, t, ma) - cellsnap(y, t, mb)) ** 2).mean()) for x, y in zip(ra, rb)])
        # only parameter sets where something happens: reference field has cell-to-cell contrast or has moved off V_INIT
        row.append(d)
        print(f'  snapshot t={t:>3}s: median rms diff {np.median(d):6.2f} mV   90th pct {np.percentile(d, 90):6.2f} mV   '
              f'(fraction of sets within 2 mV: {np.mean(d < 2):.2f})')
    corr = {}
    for s in STATS:
        xa = np.array([x['desc'][s] for x in ra]); xb = np.array([y['desc'][s] for y in rb])
        ok = np.isfinite(xa) & np.isfinite(xb)
        corr[s] = spearmanr(xa[ok], xb[ok])[0]
    print('  Spearman rank correlation of late-time statistics: ' + '  '.join(f'{s} {corr[s]:.2f}' for s in STATS))
    return dict(exact=exact, coarse=coarse_agree, snap=row, corr=corr)


if __name__ == '__main__':
    sweep, n = sys.argv[1], int(sys.argv[2])
    D = load(sweep, n)
    print(f'sweep {sweep}, {n}x{n} tissue, {len(next(iter(D.values()))["results"])} parameter sets, T={next(iter(D.values()))["T"]:.0f}s')
    M = {'lattice': 1, 'potentia2': 2, 'potentia4': 4}
    out = {}
    for a, b in (('lattice', 'potentia2'), ('lattice', 'potentia4'), ('potentia2', 'potentia4')):
        if a in D and b in D:
            out[(a, b)] = pair_stats(D[a], D[b], M[a], M[b], f'{a} vs {b}' + ('   <-- control (continuum vs continuum)' if a == 'potentia2' else ''))
    pickle.dump(out, open(f'results/compare_{sweep}_{n}.pkl', 'wb'))
