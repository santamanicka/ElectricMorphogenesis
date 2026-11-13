"""Anatomy search with feature filters (anatomy_features.py): sweep tissue size, proportions and field range, score every simulated
second, keep the best-matching moment of each run.

    python anatomy_feature_search.py coarse      # size x range at the reference proportions (range 10-40% of the body width)
    python anatomy_feature_search.py props       # proportions round the best four of each animal
    python anatomy_feature_search.py controls    # the best parameters in other outlines and at m = 4
Results: anatomy/feat/results.jsonl, anatomy/feat/best/<tag>.npz.

Each run also records the match of the same frames to the *other* animal's target (`cross`), as a specificity check, and the
contrast gate of anatomy_search.py is kept (std of G_pol over the tissue >= MIN_STD).
"""
import os, sys, json, time, itertools
import numpy as np
from multiprocessing import Pool
from anatomy_search import build, tag, MIN_STD, REF_ASPECT, SNAP_T
from anatomy_targets import TAIL_BASE
import anatomy_features as AF

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'anatomy', 'feat')
RES = os.path.join(D, 'results.jsonl')
T, SCORE_DT = 240.0, 1.0
OTHER = {'planarian': 'tadpole', 'tadpole': 'planarian'}


def full_tag(c):
    return tag(c) + ('_' + c['outline'] if c.get('outline') else '')


def run(c):
    t0 = time.time()
    P, mask, _, R_um, R_s = build(c)
    sh = c['shape']
    rect = AF.rectifier(sh, P.Np, c.get('tail', TAIL_BASE), mask)      # always the animal's own body frame; samples off the tissue are dropped
    Ft, Fo = AF.target_features(sh), AF.target_features(OTHER[sh])
    V, g = P.initial()
    Tend = c.get('T', T)
    per = int(round(SCORE_DT / P.dt)); n = int(round(Tend / SCORE_DT))
    tr = np.zeros((n + 1, 7)); best = None; snaps = {}
    for k in range(1, n + 1):
        for _ in range(per):
            V, g = P.step(V, g)
        sd = g[mask].std()
        Fp = AF.features(AF.rectify(g, rect))
        m = AF.match(Fp, Ft); mo = AF.match(Fp, Fo)
        tr[k] = [m['score']] + [m[ch] for ch in AF.CHANNELS] + [mo['score'], sd]
        tt = k * SCORE_DT
        if sd >= MIN_STD and (best is None or m['score'] > best[0]['score']):
            best = (m, tt, g.copy(), V.copy(), mo['score'], sd)
        if tt in SNAP_T:
            snaps[tt] = g.astype(np.float32)
    if best is None:
        best = ({k: 0.0 for k in AF.CHANNELS + ('score',)}, 0.0, g.copy(), V.copy(), 0.0, 0.0)
    m, tb, gb, Vb, cross, sd = best
    ok = tr[:, 6] >= MIN_STD
    row = dict(c, tag=full_tag(c), score=m['score'], channels={k: m[k] for k in AF.CHANNELS}, cross=float(cross), std_best=float(sd), t_best=float(tb),
               range_um=float(R_um), R_s=float(R_s), cols=int(P.Np[1] // P.m), score_median=float(np.median(tr[ok, 0])) if ok.any() else 0.0,
               score_p90=float(np.quantile(tr[ok, 0], 0.9)) if ok.any() else 0.0, cross_max=float(tr[ok, 5].max()) if ok.any() else 0.0, wall=time.time() - t0)
    os.makedirs(os.path.join(D, 'best'), exist_ok=True)
    np.savez_compressed(os.path.join(D, 'best', row['tag'] + '.npz'), g=gb.astype(np.float32), V=Vb.astype(np.float32), mask=mask, trace=tr,
                        snap_t=np.array(sorted(snaps)), snaps=np.array([snaps[k] for k in sorted(snaps)]))
    return row


def done():
    if not os.path.exists(RES):
        return {}
    return {r['tag']: r for r in map(json.loads, open(RES))}


def sweep(configs, procs=4):
    os.makedirs(D, exist_ok=True)
    have = done()
    todo = [c for c in configs if full_tag(c) not in have]
    todo.sort(key=lambda c: -c['rows'] ** 2 * c['aspect'] * c.get('m', 2) ** 2)
    print(len(configs), 'configs,', len(todo), 'to run', flush=True)
    with Pool(procs) as pool, open(RES, 'a') as f:
        for i, row in enumerate(pool.imap_unordered(run, todo)):
            f.write(json.dumps(row) + '\n'); f.flush()
            print(f"{i + 1}/{len(todo)} {row['tag']}  score={row['score']:.3f} at t={row['t_best']:g}s  cross={row['cross']:.3f}  ({row['wall']:.0f}s)", flush=True)
    print('STAGEDONE', flush=True)


def top(shape, n):
    rows = [r for r in done().values() if r['shape'] == shape and not r.get('outline') and 'm' not in r]
    return sorted(rows, key=lambda r: -r['score'])[:n]


RANGES = (10, 12.5, 15, 17.5, 20, 22.5, 25, 30, 40)
SIZES = (8, 12, 16, 20, 24, 32)

if __name__ == '__main__':
    stage = sys.argv[1]
    cfg = []
    keys = lambda sh: ('shape', 'rows', 'range', 'aspect') + (('tail',) if sh == 'tadpole' else ())
    if stage == 'coarse':
        for sh in ('planarian', 'tadpole'):
            for r, f in itertools.product(SIZES, RANGES):
                c = dict(shape=sh, rows=r, aspect=REF_ASPECT[sh], range=f)
                if sh == 'tadpole':
                    c['tail'] = TAIL_BASE
                cfg.append(c)
    elif stage == 'props':
        for sh in ('planarian', 'tadpole'):
            for b in top(sh, 4):
                base = {k: b[k] for k in keys(sh)}
                for a in ((3.4, 4.0, 5.2, 5.8) if sh == 'planarian' else (3.9, 5.1)):
                    cfg.append(dict(base, aspect=a))
                if sh == 'tadpole':
                    for w in (0.3, 0.5):
                        for a in (3.9, 4.5, 5.1):
                            cfg.append(dict(base, aspect=a, tail=w))
    elif stage == 'controls':
        for sh in ('planarian', 'tadpole'):
            b = top(sh, 1)[0]
            base = {k: b[k] for k in keys(sh)}
            for o in ('planarian', 'tadpole', 'stadium', 'ellipse', 'rectangle'):
                if o != sh:
                    cfg.append(dict(base, outline=o))
            cfg.append(dict(base, m=4))
    sweep(list({full_tag(c): c for c in cfg}.values()))
