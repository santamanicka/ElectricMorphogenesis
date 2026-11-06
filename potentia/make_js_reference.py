"""Generate reference states from the Python Potentia model for checking the JavaScript core (visualizer/core.js).
Writes visualizer/core_reference.json.  Run:  python make_js_reference.py   then   node visualizer/core_test.js"""
import json, os
import numpy as np
from sweeps import STIG
from potentia import Potentia

CONFIGS = [
    dict(name='lattice m=1, 5x6 cells, Stigmergic', size=(5, 6), m=1, params={}),
    dict(name='m=2, 4x5, Stigmergic', size=(4, 5), m=2, params={}),
    dict(name='m=3 (odd), 4x3, R_s=2.5', size=(4, 3), m=3, params=dict(fieldScreenSize=2.5)),
    dict(name='m=4, 3x4, R_s=6, GJ=1.0 (dt reduced)', size=(3, 4), m=4, params=dict(fieldScreenSize=6.0, GJStrength=1.0)),
    dict(name='m=2, scalar potential, R_s=3', size=(4, 4), m=2, params=dict(fieldVector=False, fieldScreenSize=3.0)),
    dict(name='m=2, consistent leak=1, gain=-2, weight=500, bias x2', size=(5, 4), m=2, params=dict(leak=1.0, gain=-2.0, weight=500.0, bias=2 * STIG['bias'])),
    dict(name='m=1, R_s=1 (short range), strength 3', size=(6, 5), m=1, params=dict(fieldScreenSize=1.0, fieldStrength=3.0)),
]
STEPS = 300
out = []
for k, c in enumerate(CONFIGS):
    params = {**STIG, **c['params']}
    P = Potentia(size_cells=c['size'], m=c['m'], **params)
    rng = np.random.default_rng(100 + k)
    V0 = rng.uniform(-55e-3, -5e-3, P.Np); g0 = rng.uniform(0.2, 1.8, P.Np)
    V, g = V0.copy(), g0.copy()
    V40 = g40 = None
    for it in range(STEPS):
        if it == 40: V40, g40 = V.copy(), g.copy()
        V, g = P.step(V, g)
    out.append(dict(name=c['name'], sizeCells=list(c['size']), m=c['m'], params={k_: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k_, v in params.items()},
                    steps=STEPS, dt=P.dt, Np=list(P.Np), V0=V0.ravel().tolist(), g0=g0.ravel().tolist(),
                    V40=V40.ravel().tolist(), g40=g40.ravel().tolist(), Vend=V.ravel().tolist(), gend=g.ravel().tolist()))
fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizer', 'core_reference.json')
json.dump(out, open(fn, 'w'))
print('wrote', fn, f'{os.path.getsize(fn)/1024:.0f} KB,', len(out), 'configurations')
