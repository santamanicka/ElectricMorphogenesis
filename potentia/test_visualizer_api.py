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
cfg = dict(size_cells=[11, 11], m=2, params={**D['params'], 'leak': None}, ic=dict(mode='homogeneous', g0=1.0, branch='dep'))
code, r = call('/api/init', cfg)
EQ1 = Potentia(size_cells=(3, 3), m=2).equilibrium_V(1.0, 'dep') * 1e3
check('init: grid 22x22, G_pol=1 with Vmem slaved to its depolarised equilibrium (-9.24 mV, the repo start)', code == 200 and r['info']['Np'] == [22, 22] and np.allclose(dec(r['frame']['V'], (22, 22)), EQ1, atol=1e-4) and abs(EQ1 + 9.24) < 0.01,
      f"Np={r['info']['Np']} dt={r['info']['dt']} R={r['info']['R_um']:.1f}um")

# stepping through the API == direct Potentia
code, r = call('/api/step', dict(steps=500, max_ms=1e9))
P = Potentia(size_cells=(11, 11), m=2, **{k: v for k, v in D['params'].items()})
V, g = P.initial_from_g(1.0, 'dep')
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

# initial conditions on G_pol: random start, branch choice, Vmem follows G_pol
rc = {**cfg, 'ic': dict(mode='random', g0=1.0, sigmaG=0.3, seed=3, branch='dep')}
_, r1 = call('/api/init', rc); _, r2 = call('/api/init', rc)
g1 = dec(r1['frame']['g'], (22, 22)); v1 = dec(r1['frame']['V'], (22, 22))
check('random IC reproducible for a seed', np.array_equal(g1, dec(r2['frame']['g'], (22, 22))))
check('random IC acts on G_pol (mean 1.0, sd ~0.3, clipped to [0,2])', abs(g1.mean() - 1.0) < 0.05 and abs(g1.std() - 0.3) < 0.05 and g1.min() >= 0 and g1.max() <= 2, f'g {g1.mean():.3f} +- {g1.std():.3f}')
Pq = Potentia(size_cells=(3, 3), m=2)
eqv = Pq.equilibrium_V(g1.astype(np.float32).astype(float), 'dep') * 1e3
check('Vmem follows G_pol pixel by pixel (isolated-cell equilibrium)', np.mean(np.abs(v1 - eqv) < 0.05) > 0.99, f'max|dV|={np.abs(v1 - eqv).max():.3f} mV')
_, ru = call('/api/init', {**cfg, 'ic': dict(mode='random', uniform=True, seed=1, branch='dep')})
gu = dec(ru['frame']['g'], (22, 22)); vu = dec(ru['frame']['V'], (22, 22))
check('uniform random IC: g uniform on [0,2]; Vmem hyperpolarised exactly where g > 1.44', gu.min() >= 0 and gu.max() <= 2 and np.all((vu < -40) == (gu > 1.44)) or np.mean((vu < -40) == (gu > 1.44)) > 0.995)
_, rh = call('/api/init', {**cfg, 'ic': dict(mode='homogeneous', g0=1.0, branch='hyp')})
check('branch=hyp at g=1 gives the hyperpolarised state (-49.54 mV)', np.allclose(dec(rh['frame']['V'], (22, 22)), -49.54, atol=0.01))
_, rm = call('/api/init', {**cfg, 'ic': dict(mode='homogeneous', g0=0.3, branch='hyp')})
check('outside the bistable window the branch choice is irrelevant (g=0.3 -> -5.95 mV either way)', np.allclose(dec(rm['frame']['V'], (22, 22)), -5.946, atol=0.01))
code, e = call('/api/init', {**cfg, 'ic': dict(mode='homogeneous', g0=1.0, branch='sideways')}); check('invalid branch rejected (400)', code == 400)

# the point of initialising on G_pol: a painted Vmem is overwritten where G_pol is monostable, but kept where it is bistable
def vmem_after(g0, steps=300):
    call('/api/init', {**cfg, 'ic': dict(mode='homogeneous', g0=g0, branch='dep')})
    Vp = np.full((22, 22), -55.0); gp = np.full((22, 22), g0)                  # whole tissue forced to -55 mV
    call('/api/state', dict(V=enc(Vp), g=enc(gp)))
    _, r = call('/api/step', dict(steps=steps, max_ms=1e9))
    return dec(r['frame']['V'], (22, 22))[8:14, 8:14].mean()                  # interior pixels, away from the field-sensing edge
v_mono = vmem_after(0.3); v_bist = vmem_after(1.0); v_hyp = vmem_after(2.0)
check('Vmem forced to -55 mV is overwritten within 3 s where G_pol is monostable-depolarised (g=0.3)', v_mono > -10, f'{v_mono:.2f} mV')
check('... and stays hyperpolarised where G_pol is bistable (g=1.0)', v_bist < -45, f'{v_bist:.2f} mV')
check('... and is unchanged where G_pol is monostable-hyperpolarised (g=2.0)', v_hyp < -50, f'{v_hyp:.2f} mV')

# sensed-field view
_, re_ = call('/api/init', {**cfg, 'want_ebar': True}); call('/api/step', dict(steps=300, max_ms=1e9))
_, r = call('/api/step', dict(steps=0, want_ebar=True))
Pe = Potentia(size_cells=(11, 11), m=2, **D['params']); Ve, ge = Pe.initial_from_g(1.0, 'dep')
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
