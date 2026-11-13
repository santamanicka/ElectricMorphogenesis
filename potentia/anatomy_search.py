"""Anatomy search: run Potentia inside an animal outline from a homogeneous start, score G_pol against the schematic target
throughout the run, and keep the best-matching moment.

Free parameters: tissue size (rows = width in cells), proportions (aspect = length / width; for the tadpole also the tail width)
and the field range (% of the width).  Everything else is the Stigmergic set (threshold bias included).

Score (every SCORE_DT simulated seconds, over the tissue pixels): the share of the variance of G_pol that the target layers of
anatomy_targets.py (three per animal, blurred by BLUR pixels) explain *beyond* a generic rim-to-centre gradient (a constant, the
distance to the outline and its square): a partial R2.  The layers' coefficients are free in sign and size, so a layer may be marked
by high or by low G_pol.  The best moment is the time of the highest score among the moments at which the pattern has contrast
(std of G_pol over the tissue >= MIN_STD; G_pol runs from 0 to 2).
Why not the plain R2 of g on the layers (kept as `r2_full`): it picks the first seconds of every run, when the only structure is a
smooth depolarised band along the outline, which any anatomy with lateral or central organs matches (seen in the first two versions
of this script: best moments at t = 0.5 to 4.5 s).

    python anatomy_search.py coarse          # stage 1: size x range at the reference proportions
    python anatomy_search.py props           # stage 2: proportions round the best of stage 1
    python anatomy_search.py fine            # stage 3: finer range and size round the best so far
Results: anatomy/results.jsonl (one line per run) and anatomy/best/<tag>.npz (the best frame, the score trace, a few snapshots).
"""
import os, sys, json, time, itertools
import numpy as np
from scipy import ndimage as ndi
from multiprocessing import Pool
from potentia import Potentia
from shapes import shape_mask
from anatomy_targets import targets_on_grid, TAIL_BASE

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'anatomy')
RES = os.path.join(D, 'results.jsonl')
M, T, SCORE_DT, BLUR, MIN_STD = 2, 240.0, 0.5, 0.75, 0.25
REF_ASPECT = {'planarian': 4.6, 'tadpole': 4.5}
SNAP_T = (10, 20, 40, 60, 90, 120, 160, 200, 240)


def tag(c):
    s = f"{c['shape']}_r{c['rows']:g}_a{c['aspect']:g}_f{c['range']:g}"
    if c['shape'] == 'tadpole':
        s += f"_w{c['tail']:g}"
    for k in ('m', 'T', 'seed'):
        if k in c:
            s += f"_{k}{c[k]:g}"
    return s


def build(c):
    m = int(c.get('m', M))
    rows, cols = c['rows'], int(round(c['rows'] * c['aspect']))
    Np = (int(round(rows * m)), cols * m)
    tail = c.get('tail', TAIL_BASE)
    layers, mask = targets_on_grid(c['shape'], Np, tail)
    if c['shape'] == 'planarian':
        assert (mask == shape_mask('planarian', Np)).all(), 'planarian target grid differs from the preset outline'
    if c.get('outline'):                                      # control: the same parameters in another outline (the target stays the animal's)
        mask = shape_mask(c['outline'], Np)
    R_um = c['range'] / 100.0 * rows * 10.0
    R_s = R_um * 1e-6 / (Potentia.a * np.sqrt(2)) - 0.001
    P = Potentia(size_cells=(rows, cols), m=m, fieldScreenSize=R_s, mask=mask)
    return P, mask, layers, R_um, R_s


def design(layers, mask, blur=BLUR):
    X = [np.ones(int(mask.sum()))]
    for k in layers:
        X.append(ndi.gaussian_filter(layers[k], blur * 1.0)[mask])
    X = np.stack(X, 1)
    return X, np.linalg.pinv(X)


def r2(g, mask, X, Xp):
    y = g[mask]
    c = Xp @ y
    res = y - X @ c
    tot = ((y - y.mean()) ** 2).sum()
    return (1.0 - (res ** 2).sum() / tot if tot > 1e-12 else 0.0), c


def run(c):
    t0 = time.time()
    P, mask, layers, R_um, R_s = build(c)
    tmask, tl = mask, layers
    if c.get('outline'):                                      # score on the pixels common to the animal outline and the control outline
        _, amask = targets_on_grid(c['shape'], P.Np, c.get('tail', TAIL_BASE))
        tmask = mask & amask
    X, Xp = design(tl, tmask)
    dist = ndi.distance_transform_edt(mask)[tmask]; dist = dist / dist.max()          # generic rim-to-centre gradient: 1, d, d^2
    N = np.stack([np.ones_like(dist), dist, dist ** 2], 1); Np_ = np.linalg.pinv(N)
    Xr = X[:, 1:] - N @ (Np_ @ X[:, 1:]); Xrp = np.linalg.pinv(Xr)                    # the layers with that gradient taken out

    def scores(g):
        y = g[tmask]
        full, co = r2(g, tmask, X, Xp)
        yr = y - N @ (Np_ @ y)                                # what the rim-to-centre gradient leaves
        tot = (yr ** 2).sum()
        res = yr - Xr @ (Xrp @ yr)
        part = 1.0 - (res ** 2).sum() / tot if tot > 1e-12 else 0.0
        null = 1.0 - tot / max(1e-12, ((y - y.mean()) ** 2).sum())
        return part, full, null, co, y.std()

    V, g = P.initial(noise_g=c.get('noise', 0.0), seed=int(c.get('seed', 0)))
    Tend = c.get('T', T)
    per = int(round(SCORE_DT / P.dt)); n = int(round(Tend / SCORE_DT))
    tr = np.zeros((n + 1, 4)); best = None
    snaps = {}
    for k in range(1, n + 1):
        for _ in range(per):
            V, g = P.step(V, g)
        part, full, null, co, sd = scores(g)
        tr[k] = part, full, null, sd
        tt = k * SCORE_DT
        if sd >= MIN_STD and (best is None or part > best[0]):
            best = (part, tt, g.copy(), V.copy(), co, full, null, sd)
        if tt in SNAP_T:
            snaps[tt] = g.astype(np.float32)
    if best is None:
        best = (0.0, 0.0, g.copy(), V.copy(), np.zeros(X.shape[1]), 0.0, 0.0, 0.0)
    part, tb, gb, Vb, co, full, null, sd = best
    each = {}
    for i, k in enumerate(tl):                                # partial R2 of each layer alone at the best moment
        yr = gb[tmask] - N @ (Np_ @ gb[tmask]); xi = Xr[:, [i]]
        each[k] = float(1.0 - ((yr - xi @ (np.linalg.pinv(xi) @ yr)) ** 2).sum() / max(1e-12, (yr ** 2).sum()))
    ok = tr[:, 3] >= MIN_STD
    row = dict(c, tag=tag(c) + ('_' + c['outline'] if c.get('outline') else ''), score=float(part), r2_full=float(full), r2_null=float(null), std_best=float(sd),
               t_best=float(tb), range_um=float(R_um), R_s=float(R_s), cols=int(P.Np[1] // P.m), coef={k: float(v) for k, v in zip(tl, co[1:])}, score_each=each,
               frac_contrast=float(ok[1:].mean()), score_median=float(np.median(tr[ok, 0])) if ok.any() else 0.0,
               score_p90=float(np.quantile(tr[ok, 0], 0.9)) if ok.any() else 0.0, wall=time.time() - t0)
    os.makedirs(os.path.join(D, 'best'), exist_ok=True)
    np.savez_compressed(os.path.join(D, 'best', row['tag'] + '.npz'), g=gb.astype(np.float32), V=Vb.astype(np.float32), mask=mask, trace=tr,
                        snap_t=np.array(sorted(snaps)), snaps=np.array([snaps[k] for k in sorted(snaps)]), **{'L_' + k: tl[k].astype(np.float32) for k in tl})
    return row


def done():
    if not os.path.exists(RES):
        return {}
    return {r['tag']: r for r in map(json.loads, open(RES))}


def sweep(configs, procs=4):
    have = done()
    todo = [c for c in configs if tag(c) + ('_' + c['outline'] if c.get('outline') else '') not in have]
    todo.sort(key=lambda c: -c['rows'] ** 2 * c['aspect'])    # big ones first
    print(len(configs), 'configs,', len(todo), 'to run', flush=True)
    with Pool(procs) as pool, open(RES, 'a') as f:
        for i, row in enumerate(pool.imap_unordered(run, todo)):
            f.write(json.dumps(row) + '\n'); f.flush()
            print(f"{i + 1}/{len(todo)} {row['tag']}  score={row['score']:.3f} (full R2 {row['r2_full']:.3f}) at t={row['t_best']:g}s  ({row['wall']:.0f}s)", flush=True)


def top(shape, n, key='score'):
    rows = [r for r in done().values() if r['shape'] == shape and not r.get('outline') and 'm' not in r and 'seed' not in r]
    return sorted(rows, key=lambda r: -r[key])[:n]


RANGES = (8, 12, 16, 20, 25, 30, 40, 60)
SIZES = (8, 12, 16, 24, 32)

if __name__ == '__main__':
    stage = sys.argv[1]
    cfg = []
    if stage == 'coarse':
        for sh in ('planarian', 'tadpole'):
            for r, f in itertools.product(SIZES, RANGES):
                c = dict(shape=sh, rows=r, aspect=REF_ASPECT[sh], range=f)
                if sh == 'tadpole':
                    c['tail'] = TAIL_BASE
                cfg.append(c)
    elif stage == 'props':
        for sh, asps in (('planarian', (3.4, 4.0, 5.2, 5.8)), ('tadpole', (3.9, 5.1))):
            for b in top(sh, 4):
                for a in asps:
                    cfg.append({**{k: b[k] for k in ('shape', 'rows', 'range') + (('tail',) if sh == 'tadpole' else ())}, 'aspect': a})
                if sh == 'tadpole':
                    for w in (0.3, 0.5):
                        for a in (3.9, 4.5, 5.1):
                            cfg.append(dict(shape=sh, rows=b['rows'], range=b['range'], aspect=a, tail=w))
    elif stage == 'fine':
        for sh in ('planarian', 'tadpole'):
            for b in top(sh, 3):
                for dr in (-2, 0, 2):
                    for df in (0.8, 0.9, 1.1, 1.2):
                        c = {k: b[k] for k in ('shape', 'aspect') + (('tail',) if sh == 'tadpole' else ())}
                        c['rows'] = b['rows'] + dr; c['range'] = round(b['range'] * df, 1)
                        cfg.append(c)
    uniq = {tag(c): c for c in cfg}
    sweep(list(uniq.values()))
