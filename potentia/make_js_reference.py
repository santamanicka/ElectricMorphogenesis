"""Generate reference states from the Python Potentia model for checking the JavaScript core (visualizer/core.js).
Writes visualizer/core_reference.json.  Run:  python make_js_reference.py   then   node visualizer/core_test.js"""
import json, os
import numpy as np
from sweeps import STIG
from potentia import Potentia
from shapes import shape_mask, SHAPES
from development import Development, grow_state, dev_update, A_UM

CONFIGS = [
    dict(name='lattice m=1, 5x6 cells, Stigmergic', size=(5, 6), m=1, params={}),
    dict(name='m=2, 4x5, Stigmergic', size=(4, 5), m=2, params={}),
    dict(name='m=3 (odd), 4x3, R_s=2.5', size=(4, 3), m=3, params=dict(fieldScreenSize=2.5)),
    dict(name='m=4, 3x4, R_s=6, GJ=1.0 (dt reduced)', size=(3, 4), m=4, params=dict(fieldScreenSize=6.0, GJStrength=1.0)),
    dict(name='m=2, scalar potential, R_s=3', size=(4, 4), m=2, params=dict(fieldVector=False, fieldScreenSize=3.0)),
    dict(name='m=2, consistent leak=1, gain=-2, weight=500, bias x2', size=(5, 4), m=2, params=dict(leak=1.0, gain=-2.0, weight=500.0, bias=2 * STIG['bias'])),
    dict(name='m=1, R_s=1 (short range), strength 3', size=(6, 5), m=1, params=dict(fieldScreenSize=1.0, fieldStrength=3.0)),
    # periodic tissue (torus): sources and gap junctions wrap around; R_s=6 on a 4x5-cell tissue makes the kernel wrap more than once
    dict(name='periodic, m=1, 5x6 cells, Stigmergic', size=(5, 6), m=1, params={}, periodic=True),
    dict(name='periodic, m=2, 4x5, R_s=6 (kernel wraps several times)', size=(4, 5), m=2, params=dict(fieldScreenSize=6.0), periodic=True),
    dict(name='periodic, m=3 (odd), 4x3, R_s=2.5', size=(4, 3), m=3, params=dict(fieldScreenSize=2.5), periodic=True),
    dict(name='periodic, m=2, scalar potential, R_s=3', size=(4, 4), m=2, params=dict(fieldVector=False, fieldScreenSize=3.0), periodic=True),
    dict(name='periodic, m=4, 3x4, GJ=1.0 (dt reduced)', size=(3, 4), m=4, params=dict(GJStrength=1.0), periodic=True),
    # tissue shapes (masks on the pixel grid): the JavaScript core must follow Potentia(mask=...)
    dict(name='shape disc, m=2, 8x8 cells', size=(8, 8), m=2, params={}, shape=('disc', None)),
    dict(name='shape annulus (0.4), m=3 (odd), 6x6, R_s=3', size=(6, 6), m=3, params=dict(fieldScreenSize=3.0), shape=('annulus', 0.4)),
    dict(name='shape tadpole, m=2, 5x12, scalar potential', size=(5, 12), m=2, params=dict(fieldVector=False), shape=('tadpole', None)),
    dict(name='shape random mask (30% of pixels removed), m=2, 5x6, GJ=1.0', size=(5, 6), m=2, params=dict(GJStrength=1.0), shape=('random', 0.3)),
]
STEPS = 300
out = []
for k, c in enumerate(CONFIGS):
    params = {**STIG, **c['params']}
    rng = np.random.default_rng(100 + k)
    mask = None
    if 'shape' in c:
        npx = (c['size'][0] * c['m'], c['size'][1] * c['m'])
        mask = (rng.uniform(size=npx) > c['shape'][1]) if c['shape'][0] == 'random' else shape_mask(c['shape'][0], npx, c['shape'][1])
    P = Potentia(size_cells=c['size'], m=c['m'], periodic=c.get('periodic', False), mask=mask, **params)
    V0 = rng.uniform(-55e-3, -5e-3, P.Np); g0 = rng.uniform(0.2, 1.8, P.Np)
    if mask is not None: V0 = V0 * mask; g0 = g0 * mask
    V, g = V0.copy(), g0.copy()
    V40 = g40 = None
    for it in range(STEPS):
        if it == 40: V40, g40 = V.copy(), g.copy()
        V, g = P.step(V, g)
    out.append(dict(name=c['name'], sizeCells=list(c['size']), m=c['m'], periodic=bool(c.get('periodic', False)), params={k_: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k_, v in params.items()},
                    mask=(None if mask is None else mask.astype(int).ravel().tolist()), steps=STEPS, dt=P.dt, Np=list(P.Np), V0=V0.ravel().tolist(), g0=g0.ravel().tolist(),
                    V40=V40.ravel().tolist(), g40=g40.ravel().tolist(), Vend=V.ravel().tolist(), gend=g.ravel().tolist()))
# equilibrium (G_pol -> Vmem) reference: window edges, exact table nodes, and random g
P0 = Potentia(size_cells=(3, 3), m=2)
rng = np.random.default_rng(7)
gq = np.concatenate([[0.0, 0.3, 0.8, 0.805, 0.81, 1.0, 1.2, 1.43, 1.435, 1.44, 1.5, 2.0], rng.uniform(0, 2, 60)])
_gs, _dep, _hyp = P0._equilibrium_table(); _m = np.abs(_dep - _hyp) > 1e-6
eq = dict(g=gq.tolist(), dep=P0.equilibrium_V(gq, 'dep').tolist(), hyp=P0.equilibrium_V(gq, 'hyp').tolist(), window=[float(_gs[_m].min()), float(_gs[_m].max())])
print('bistable window (g, table resolution 5e-4):', eq['window'])
# shape generator reference: every shape on several grids, default and one other parameter value
shape_ref = []
for name, (_lab, plab, pdef, rng_) in SHAPES.items():
    for Np in [(24, 24), (30, 50), (50, 30), (21, 33), (64, 64), (9, 40)]:
        for p in ([None] if plab is None else [None, round((rng_[0] + rng_[1]) / 2 + 0.07, 3)]):
            mk = shape_mask(name, Np, p)
            shape_ref.append(dict(name=name, R=Np[0], C=Np[1], p=p, bits=''.join('1' if v else '0' for v in mk.ravel())))
# development reference: masks along the sequence, state inheritance, range schedule, and a complete development run (mask and range changing while the model runs)
def hexmask(mk):
    bits = ''.join('1' if v else '0' for v in np.asarray(mk).ravel()); bits += '0' * (-len(bits) % 4)
    return '%0*x' % (len(bits) // 4, int(bits, 2))
dev_masks = []
for Np in [(100, 260), (40, 108), (60, 160), (50, 50), (30, 200)]:
    d = Development(Np, 100.0); ts = [0.0, 0.05, 0.18, 0.3, 0.36, 0.45, 0.52, 0.6, 0.74, 0.9, 1.0]
    dev_masks.append(dict(R=Np[0], C=Np[1], taus=ts, masks=[hexmask(d.mask_at(t)) for t in ts], stages=[d.stage_at(t)['name'] for t in ts]))
grow = []
for seed, (R, C), isl in [(1, (9, 11), False), (2, (12, 8), False), (3, (10, 10), True)]:
    r = np.random.default_rng(seed); old = r.uniform(size=(R, C)) > 0.6; new = old | (r.uniform(size=(R, C)) > 0.8)
    if isl: new[R - 3:, C - 3:] = True; old[R - 4:, C - 4:] = False
    V0 = r.uniform(-0.055, -0.005, (R, C)) * old; g0 = r.uniform(0, 2, (R, C)) * old
    Vo, go = grow_state(V0, g0, old, new, (-0.0092, 0.7))
    grow.append(dict(R=R, C=C, old=hexmask(old), new=hexmask(new), V=V0.ravel().tolist(), g=g0.ravel().tolist(), Vo=Vo.ravel().tolist(), go=go.ravel().tolist(), fill=[-0.0092, 0.7]))
dd = Development((100, 260), 100.0); rng_ref = []
for mode in ('const', 'prop', 'stage'):
    for t in [0.0, 0.1, 0.2, 0.36, 0.5, 0.6, 0.8, 1.0]:
        rng_ref.append(dict(mode=mode, tau=t, um=dd.range_um(mode, t, dd.mask_at(t), 5.0, 28.3, 25.0, [10, 20, 30, 40, 50, 60]), mask=hexmask(dd.mask_at(t))))
# a complete run: 10 x 27 cells, m = 2, 20 s development with the range proportional to the tissue (30 %), updates every 0.5 s
runcfg = dict(size=(10, 27), m=2, duration=20.0, update=0.5, rng=dict(mode='prop', pct=30.0), steps=2000, fill=(-9.24e-3, 1.0))
dv = Development((runcfg['size'][0] * runcfg['m'], runcfg['size'][1] * runcfg['m']), runcfg['duration'], runcfg['update'])
mk0 = dv.mask_at(0.0); r0 = dv.range_um('prop', 0.0, mk0, 5.0, 28.3, 30.0, None); Rs0 = r0 / (A_UM * np.sqrt(2)) - 0.001
Pr = Potentia(size_cells=runcfg['size'], m=runcfg['m'], mask=mk0, **{**STIG, 'fieldScreenSize': Rs0})
Vr, gr = Pr.initial_from_g(1.0, 'dep'); Kst = int(round(runcfg['update'] / Pr.dt)); upd = []
for n in range(1, runcfg['steps'] + 1):
    Vr, gr = Pr.step(Vr, gr)
    if n % Kst == 0:
        Vr, gr, stt = dev_update(Pr, dv, n, Vr, gr, dict(mode='prop', pct=30.0), runcfg['fill']); upd.append([stt['tissue'], stt['range_um'], stt['stage']])
run_ref = dict(cfg=dict(size=list(runcfg['size']), m=runcfg['m'], duration=runcfg['duration'], update=runcfg['update'], pct=30.0, steps=runcfg['steps'], fill=list(runcfg['fill']), Rs0=Rs0, mask0=hexmask(mk0)),
               V=Vr.ravel().tolist(), g=gr.ravel().tolist(), final_mask=hexmask(Pr.mask), updates=upd)
out = dict(states=out, eq=eq, shapes=shape_ref, dev_masks=dev_masks, grow=grow, dev_range=rng_ref, dev_run=run_ref)
fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizer', 'core_reference.json')
json.dump(out, open(fn, 'w'))
print('wrote', fn, f'{os.path.getsize(fn)/1024:.0f} KB,', len(out), 'configurations')
