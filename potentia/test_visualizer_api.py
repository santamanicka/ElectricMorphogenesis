"""API-level tests for visualizer.py (no browser): the server must drive exactly the validated Potentia model."""
import base64, io, json, sys, threading, http.client
import numpy as np
import visualizer as viz
from potentia import Potentia

srv = viz.make_server('127.0.0.1', 0)
threading.Thread(target=srv.serve_forever, daemon=True).start()
PORT = srv.server_address[1]

def call(path, body=None, raw=False):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=60)
    if body is None: c.request('GET', path)
    else: c.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
    r = c.getresponse(); data = r.read(); c.close()
    return r.status, (data if raw else json.loads(data))

dec = lambda s, shape: np.frombuffer(base64.b64decode(s), '<f4').reshape(shape).astype(float)
enc = lambda a: base64.b64encode(np.ascontiguousarray(a, '<f4').tobytes()).decode()
ok = True
def check(name, cond, extra=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name + (f'   {extra}' if extra else ''))

code, D = call('/api/defaults')
check('defaults are the Stigmergic set from the .dat file', code == 200 and D['params']['fieldScreenSize'] == 4.0 and abs(D['params']['bias'] - 5.308665553936392e-4) < 1e-15
      and D['params']['gain'] == -1.0 and D['params']['weight'] == 1000.0 and D['params']['GJStrength'] == 0.05 and D['params']['fieldVector'])
cfg = dict(size_cells=[11, 11], m=2, params={**D['params'], 'leak': None}, ic=dict(mode='homogeneous', V0_mV=-9.2, g0=1.0))
code, r = call('/api/init', cfg)
check('init: grid 22x22, homogeneous -9.2 mV', code == 200 and r['info']['Np'] == [22, 22] and np.allclose(dec(r['frame']['V'], (22, 22)), -9.2, atol=1e-5),
      f"Np={r['info']['Np']} dt={r['info']['dt']} R={r['info']['R_um']:.1f}um")

# stepping through the API == direct Potentia
code, r = call('/api/step', dict(steps=500, max_ms=1e9))
P = Potentia(size_cells=(11, 11), m=2, **{k: v for k, v in D['params'].items()})
V, g = P.initial(V0=-9.2e-3, g0=1.0)
for _ in range(500): V, g = P.step(V, g)
Va = dec(r['frame']['V'], (22, 22)); Ga = dec(r['frame']['g'], (22, 22))
check('500 API steps match a direct Potentia run (float32 transport)', np.abs(Va - V * 1e3).max() < 1e-4 and np.abs(Ga - g).max() < 1e-6,
      f"max|dV|={np.abs(Va - V*1e3).max():.1e} mV, t={r['frame']['t']:.2f}")

# until clamps the run time exactly
code, r = call('/api/step', dict(steps=10**6, until=7.0, max_ms=1e9))
check('until=7.0 stops at t=7.0 s', abs(r['frame']['t'] - 7.0) < 1e-9, f"t={r['frame']['t']}")

# max_ms cap returns early without error
code, r = call('/api/step', dict(steps=10**6, max_ms=20))
check('max_ms cap returns early', code == 200 and 0 < r['frame']['steps_done'] < 10**6, f"steps_done={r['frame']['steps_done']}")

# manual painting round trip
paint_V = np.full((22, 22), -9.2); paint_V[8:14, 8:14] = -55.0
paint_g = np.full((22, 22), 1.0); paint_g[8:14, 8:14] = 2.0
call('/api/init', cfg)
code, r = call('/api/state', dict(V=enc(paint_V), g=enc(paint_g)))
check('painted state is stored exactly', np.allclose(dec(r['frame']['V'], (22, 22)), paint_V, atol=1e-5) and np.allclose(dec(r['frame']['g'], (22, 22)), paint_g, atol=1e-6))
code, r = call('/api/step', dict(steps=200, max_ms=1e9))
Pp = Potentia(size_cells=(11, 11), m=2, **D['params'])
Vd, gd = paint_V * 1e-3, paint_g.copy()
for _ in range(200): Vd, gd = Pp.step(Vd, gd)
check('stepping from a painted state matches direct Potentia', np.abs(dec(r['frame']['V'], (22, 22)) - Vd * 1e3).max() < 1e-3)
code, bad = call('/api/state', dict(V=enc(np.zeros((5, 5))), g=enc(np.zeros((5, 5)))))
check('wrong-size painted state rejected (400)', code == 400, bad.get('error', '')[:60])

# random IC: seeded + right statistics
rc = {**cfg, 'ic': dict(mode='random', V0_mV=-20.0, g0=1.0, sigmaV_mV=4.0, sigmaG=0.1, seed=3)}
_, r1 = call('/api/init', rc); _, r2 = call('/api/init', rc)
v1 = dec(r1['frame']['V'], (22, 22)); g1 = dec(r1['frame']['g'], (22, 22))
check('random IC reproducible for a seed', np.array_equal(v1, dec(r2['frame']['V'], (22, 22))))
check('random IC statistics (mean -20 mV, sd 4 mV; g sd 0.1)', abs(v1.mean() + 20) < 1.0 and abs(v1.std() - 4) < 0.8 and abs(g1.std() - 0.1) < 0.03, f'V {v1.mean():.2f} +- {v1.std():.2f}, g sd {g1.std():.3f}')
_, ru = call('/api/init', {**cfg, 'ic': dict(mode='random', uniform=True, seed=1)})
vu = dec(ru['frame']['V'], (22, 22)); check('uniform random IC within [-55,-5] mV', vu.min() >= -55.01 and vu.max() <= -4.99)

# sensed-field view
_, re_ = call('/api/init', {**cfg, 'want_ebar': True}); call('/api/step', dict(steps=300, max_ms=1e9))
_, r = call('/api/step', dict(steps=0, want_ebar=True))
Pe = Potentia(size_cells=(11, 11), m=2, **D['params']); Ve, ge = Pe.initial()
for _ in range(300): Ve, ge = Pe.step(Ve, ge)
eb = dec(r['frame']['ebar'], (22, 22))
check('sensed-field/bias view equals perceived_field/bias', np.allclose(eb, Pe.perceived_field(Ve) / Pe.bias, rtol=1e-4, atol=1e-6))

# limits and validation
code, e = call('/api/init', {**cfg, 'size_cells': [200, 200], 'm': 8}); check('oversize grid rejected (400)', code == 400, e.get('error', '')[:70])
code, e = call('/api/init', {**cfg, 'm': 0}); check('invalid m rejected (400)', code == 400)
# a different tissue / non-square / m=1 lattice mode works
code, r = call('/api/init', {**cfg, 'size_cells': [9, 13], 'm': 1}); check('non-square 9x13, m=1 (lattice mode)', code == 200 and r['info']['Np'] == [9, 13])

# export
call('/api/init', cfg); call('/api/step', dict(steps=50, max_ms=1e9))
code, raw = call('/api/export', raw=True); z = np.load(io.BytesIO(raw), allow_pickle=False)
check('export .npz has V_mV, g, t', code == 200 and z['V_mV'].shape == (22, 22) and abs(float(z['t']) - 0.5) < 1e-9, f"t={float(z['t'])}")
code, page = call('/', raw=True); check('index page served', code == 200 and b'Potentia' in page and b'__VIRIDIS__' not in page)
srv.shutdown()
print('\nALL PASS' if ok else '\nFAILURES'); sys.exit(0 if ok else 1)
