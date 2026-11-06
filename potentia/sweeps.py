"""Parameter-sweep driver for Potentia (and, for comparison, the lattice model).

Everything here uses *finite, non-periodic* tissue and a *homogeneous start*
(Vmem = -9.2 mV, G_pol = 1.0 G_ref, eV = 0), as in the original model's autonomous mode.
No clamping.  Parameter names follow the repo's field parameters.
"""
import os, json, time
import numpy as np
from multiprocessing import Pool
from potentia import Potentia
from lattice import Lattice
import patterns

HERE = os.path.dirname(os.path.abspath(__file__))
import torchfree_io as _tio
_p = _tio.load(os.path.join(HERE, '..', 'data', 'StigmergicModelParameters.dat'))
_f = _p['fieldParameters']
# the Stigmergic operating point, read from the repo's trained-model file (bias is 5.30867e-4, not 5.3e-4)
STIG = dict(fieldScreenSize=float(_f['fieldScreenSize']), fieldStrength=float(_f['fieldStrength']),
            gain=float(_f['fieldTransductionGain']), weight=float(_f['fieldTransductionWeight'][0]),
            bias=float(_f['fieldTransductionBias'][0]), tau=float(_f['fieldTransductionTimeConstant'][0]),
            GJStrength=float(_p['GJParameters']['GJStrength']), fieldVector=bool(_f['fieldVector']))


def run_point(cfg):
    """cfg: dict(size=(n0,n1), m, T, rec_dt, params={...}, keep=[times], model='potentia'|'lattice', noise=0, seed=0)"""
    params = {**STIG, **cfg.get('params', {})}
    size, m, T, rec_dt = tuple(cfg['size']), cfg.get('m', 2), cfg['T'], cfg.get('rec_dt', 0.5)
    model = cfg.get('model', 'potentia')
    if model == 'lattice':
        L = Lattice(dims=size, fieldScreenSize=params['fieldScreenSize'], fieldStrength=params['fieldStrength'],
                    gain=params['gain'], weight=params['weight'], bias=params['bias'], tau=params['tau'],
                    GJStrength=params['GJStrength'], fieldVector=params['fieldVector'])
        V, G = L.initial()
        dt, step = L.dt, (lambda V, G: L.step(V, G))
        shape, Gscale, m = size, L.G_ref, 1
        V = V.copy()
    else:
        P = Potentia(size_cells=size, m=m, **params)
        V, g = P.initial(noise_V=cfg.get('noise', 0.0), seed=cfg.get('seed', 0))
        dt, shape = P.dt, P.Np
        G = g
        step = P.step
    nsteps = int(round(T / dt)); rec = max(1, int(round(rec_dt / dt)))
    Vt, tt, gm = [], [], []
    keep = {}
    keep_steps = {int(round(k / dt)): k for k in cfg.get('keep', [])}
    t0 = time.time()
    for it in range(nsteps + 1):
        if it % rec == 0:
            Vt.append((V.reshape(shape) * 1e3).astype(np.float32)); tt.append(it * dt)
            gm.append(float(np.mean(G / (Gscale if model == 'lattice' else 1.0))))
        if it in keep_steps:
            keep[keep_steps[it]] = (V.reshape(shape) * 1e3).astype(np.float16)
        if it < nsteps:
            V, G = step(V, G)
    Vt, tt = np.array(Vt), np.array(tt)
    if not np.isfinite(Vt).all():
        d = {'diverged': True}
        return {'cfg': cfg, 'desc': d, 'regime': 'diverged', 'runtime': time.time() - t0, 'keep': keep}
    d = patterns.describe(Vt, tt, m, size)
    d['g_mean_late'] = float(np.mean(gm[len(gm) // 2:]))
    out = {'cfg': cfg, 'desc': d, 'regime': patterns.classify(d), 'runtime': time.time() - t0,
           'keep': keep, 'final': (Vt[-1]).astype(np.float16)}
    if cfg.get('save_series'):
        out['series_cell'] = patterns._cell_blocks(Vt, m).astype(np.float16)
        out['t'] = tt
    return out


def run_grid(cfgs, procs=4, label=''):
    t0 = time.time()
    with Pool(procs) as pool:
        res = []
        for i, r in enumerate(pool.imap(run_point, cfgs, chunksize=1)):
            res.append(r)
            if (i + 1) % max(1, len(cfgs) // 10) == 0:
                print(f'  [{label}] {i+1}/{len(cfgs)}  {time.time()-t0:.0f}s', flush=True)
    return res


def grid_cfgs(axes, base, **kw):
    """axes: ordered dict {param: values}; returns cfg list (row-major over the axes) and the shape."""
    names = list(axes)
    shape = tuple(len(axes[n]) for n in names)
    cfgs = []
    for idx in np.ndindex(*shape):
        p = dict(base.get('params', {}))
        for n, i in zip(names, idx):
            p[n] = float(axes[n][i])
        cfgs.append({**base, 'params': p, **kw})
    return cfgs, shape
