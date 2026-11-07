#!/usr/bin/env python3
"""Interactive Potentia visualizer: a local web UI backed by the validated Potentia class.

    python visualizer.py [--port 8765] [--host 127.0.0.1] [--open]

then open http://127.0.0.1:8765 .  Needs only numpy + scipy (no torch, no web framework).

Inputs in the UI: tissue size (cells), pixel resolution m, field and GJ parameters (default = the
Stigmergic set read from data/StigmergicModelParameters.dat), run time, and initial conditions on G_pol
(homogeneous, random, or manual painting). Vmem is not an independent input: it follows G_pol (the isolated-cell
equilibrium); only where G_pol is bistable (about 0.81 < g < 1.44) a branch choice remains (default: depolarised).  Pure bioelectric; the tissue is finite with
no-flux edges by default, or periodic (a torus: field and gap junctions wrap around) with the "periodic" option, or any shape on the grid ("mask": disc, ring, ...; see shapes.py; no-flux edge, nothing outside);
painted pixels can optionally be clamped (held) for a number of steps.

Single-user local tool: one simulation session is held by the server (a second browser tab shares it).
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from potentia import Potentia            # noqa: E402
import torchfree_io                      # noqa: E402

MAX_SIDE_PX = 512                        # per side
MAX_TOTAL_PX = 200_000
V_BIST_MV = -30.0


def stigmergic_defaults():
    p = torchfree_io.load(os.path.join(HERE, '..', 'data', 'StigmergicModelParameters.dat'))
    f = p['fieldParameters']
    sc = lambda x: float(np.asarray(x).reshape(-1)[0])
    return dict(fieldScreenSize=float(f['fieldScreenSize']), fieldStrength=sc(f['fieldStrength']),
                gain=sc(f['fieldTransductionGain']), weight=sc(f['fieldTransductionWeight']),
                bias=sc(f['fieldTransductionBias']), tau=sc(f['fieldTransductionTimeConstant']),
                GJStrength=float(p['GJParameters']['GJStrength']), fieldVector=bool(f['fieldVector']))


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype='<f4').tobytes()).decode('ascii')


def unb64u8(s, shape):
    a = np.frombuffer(base64.b64decode(s), dtype=np.uint8)
    if a.size != shape[0] * shape[1]:
        raise ValueError(f'mask has {a.size} values, expected {shape[0] * shape[1]}')
    return a.reshape(shape).astype(bool)


def unb64(s, shape):
    a = np.frombuffer(base64.b64decode(s), dtype='<f4')
    if a.size != shape[0] * shape[1]:
        raise ValueError(f'state array has {a.size} values, expected {shape[0] * shape[1]}')
    return a.reshape(shape).astype(np.float64)


class Session:
    """One running Potentia simulation."""

    def __init__(self):
        self.lock = threading.Lock()
        self.P = None
        self.V = self.g = None
        self.n = 0
        self.cfg = None
        self.diverged = False
        self.clamp = None                       # painted pixels held for a number of steps (see set_state)

    # ---- construction ----------------------------------------------------------------
    def init(self, cfg):
        size = tuple(int(x) for x in cfg['size_cells'])
        m = int(cfg['m'])
        if not (1 <= m <= 16):
            raise ValueError('m must be between 1 and 16')
        if min(size) < 2 or max(size) * m > MAX_SIDE_PX or size[0] * size[1] * m * m > MAX_TOTAL_PX:
            raise ValueError(f'grid {size[0] * m}x{size[1] * m} pixels is outside the allowed range '
                             f'(each side <= {MAX_SIDE_PX}, total <= {MAX_TOTAL_PX}); lower m or the tissue size')
        pr = cfg['params']
        leak = pr.get('leak', None)
        periodic = bool(cfg.get('periodic', False))
        mask = unb64u8(cfg['mask'], (size[0] * m, size[1] * m)) if cfg.get('mask') else None       # tissue shape: base64 uint8, 1 = tissue
        P = Potentia(size_cells=size, m=m, periodic=periodic, mask=mask, fieldScreenSize=float(pr['fieldScreenSize']),
                     fieldStrength=float(pr['fieldStrength']), gain=float(pr['gain']), weight=float(pr['weight']),
                     bias=float(pr['bias']), tau=float(pr['tau']), GJStrength=float(pr['GJStrength']),
                     fieldVector=bool(pr['fieldVector']), leak=None if leak is None else float(leak))
        # initial conditions are set on G_pol; Vmem follows it (G_pol directly determines Vmem)
        ic = cfg.get('ic', {'mode': 'homogeneous'})
        g0 = float(ic.get('g0', 1.0))
        branch = ic.get('branch', 'dep')
        if branch not in ('dep', 'hyp'):
            raise ValueError("branch must be 'dep' or 'hyp'")
        mode = ic.get('mode', 'homogeneous')
        if mode == 'random':
            rng = np.random.default_rng(int(ic.get('seed', 0)))
            if ic.get('uniform', False):
                g = rng.uniform(0.0, 2.0, P.Np)
            else:
                g = g0 + float(ic.get('sigmaG', 0.3)) * rng.standard_normal(P.Np)
        else:                                  # 'homogeneous' and 'paint' (paint starts from the homogeneous background)
            g = np.full(P.Np, g0)
        V, g = P.initial_from_g(g, branch)
        self.P, self.V, self.g, self.n, self.cfg, self.diverged, self.clamp = P, V, g, 0, cfg, False, None
        return self.info()

    def info(self):
        P = self.P
        return dict(Np=list(P.Np), size_cells=list(P.size_cells), m=P.m, periodic=bool(P.periodic), shaped=P.mask is not None, tissue_px=int(P.mask.sum()) if P.mask is not None else int(P.Np[0] * P.Np[1]), dt=P.dt, h_um=P.h * 1e6, R_um=P.R * 1e6,
                    tissue_um=[P.Np[0] * P.h * 1e6, P.Np[1] * P.h * 1e6], kernel_px=P.R / P.h)

    # ---- state -----------------------------------------------------------------------
    def set_state(self, V_mV, g, clamp=None):
        """Replace the state (a painted state). clamp = {steps, maskV, maskG} (masks base64 uint8, same shape as the grid) holds the painted
        values of the masked pixels for `steps` steps: they are re-imposed after every step c = 1 .. steps-1 counted from now, i.e. the painted
        values are the input of each of the `steps` steps, and the state after step `steps` is free.  A new state replaces any earlier clamp."""
        self.V = V_mV * 1e-3
        self.g = np.clip(g, 0.0, 2.0)
        if self.P.mask is not None:
            self.V, self.g = self.V * self.P.maskf, self.g * self.P.maskf                # nothing outside the tissue
        self.diverged = False
        self.clamp = None
        if clamp:
            steps = int(clamp.get('steps', 0))
            mV = unb64u8(clamp['maskV'], self.P.Np) if clamp.get('maskV') else np.zeros(self.P.Np, bool)
            mG = unb64u8(clamp['maskG'], self.P.Np) if clamp.get('maskG') else np.zeros(self.P.Np, bool)
            if steps > 0 and (mV.any() or mG.any()):
                self.clamp = dict(start=self.n, steps=steps, mV=mV, mG=mG, valV=self.V.copy(), valG=self.g.copy())

    def clamp_left(self):
        c = self.clamp
        return 0 if c is None else max(0, c['steps'] - (self.n - c['start']))

    def advance(self, steps, until=None, max_ms=150.0):
        P = self.P
        steps = int(max(0, steps))
        if until is not None:
            steps = min(steps, max(0, int(round(until / P.dt)) - self.n))
        t0 = time.perf_counter()
        done = 0
        cl = self.clamp
        while done < steps:
            for _ in range(min(25, steps - done)):
                self.V, self.g = P.step(self.V, self.g)
                done += 1
                if cl is not None and 0 < self.n + done - cl['start'] < cl['steps']:     # hold the painted values for the next step
                    self.V[cl['mV']] = cl['valV'][cl['mV']]
                    self.g[cl['mG']] = cl['valG'][cl['mG']]
            if (time.perf_counter() - t0) * 1e3 > max_ms:
                break
        self.n += done
        if cl is not None and self.n - cl['start'] >= cl['steps']:
            self.clamp = None                                                              # released
        if not (np.isfinite(self.V).all() and np.isfinite(self.g).all()):
            self.diverged = True
        return done

    def frame(self, want_ebar=False, steps_done=0, wall_ms=0.0):
        P = self.P
        V = self.V * 1e3
        Vt, gt = (V, self.g) if P.mask is None else (V[P.mask], self.g[P.mask])            # statistics over the tissue only
        out = dict(t=self.n * P.dt, step=self.n, Np=list(P.Np), V=b64(V), g=b64(self.g), steps_done=steps_done,
                   wall_ms=wall_ms, diverged=self.diverged, clamp_left=self.clamp_left(),
                   stats=dict(meanV=float(Vt.mean()), minV=float(Vt.min()), maxV=float(Vt.max()),
                              fh=float((Vt < V_BIST_MV).mean()), meang=float(gt.mean())))
        if want_ebar:
            eb = P.perceived_field(self.V) / P.bias
            out['ebar'] = b64(eb if P.mask is None else eb * P.maskf)
        return out

    def export_npz(self):
        buf = io.BytesIO()
        cfg = {k: v for k, v in self.cfg.items() if k != 'mask'}
        extra = {} if self.P.mask is None else dict(mask=self.P.mask)
        np.savez(buf, V_mV=self.V * 1e3, g=self.g, t=self.n * self.P.dt, step=self.n, config=json.dumps(cfg), **extra)
        return buf.getvalue()


SESSION = Session()
DEFAULTS = None


class Handler(BaseHTTPRequestHandler):
    server_version = 'PotentiaViz/1.0'

    def log_message(self, fmt, *args):         # quiet
        if os.environ.get('POTENTIA_VIZ_LOG'):
            super().log_message(fmt, *args)

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self):
        n = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(n) or b'{}')

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            with open(os.path.join(HERE, 'visualizer', 'index.html'), 'rb') as f:
                return self._send(200, f.read(), 'text/html; charset=utf-8')
        if path == '/api/defaults':
            return self._send(200, dict(params=DEFAULTS, cell_um=10.0, max_side_px=MAX_SIDE_PX, max_total_px=MAX_TOTAL_PX,
                                        ic=dict(g0=1.0, branch='dep', sigmaG=0.3)))
        if path == '/api/export':
            with SESSION.lock:
                if SESSION.P is None:
                    return self._send(409, dict(error='nothing to export: initialise first'))
                data = SESSION.export_npz()
            return self._send(200, data, 'application/octet-stream')
        return self._send(404, dict(error='not found'))

    def do_POST(self):
        try:
            req = self._json()
            path = self.path.split('?')[0]
            with SESSION.lock:
                if path == '/api/init':
                    info = SESSION.init(req)
                    return self._send(200, dict(info=info, frame=SESSION.frame(want_ebar=bool(req.get('want_ebar')))))
                if SESSION.P is None:
                    return self._send(409, dict(error='initialise first'))
                if path == '/api/state':
                    SESSION.set_state(unb64(req['V'], SESSION.P.Np), unb64(req['g'], SESSION.P.Np), req.get('clamp'))
                    return self._send(200, dict(frame=SESSION.frame(want_ebar=bool(req.get('want_ebar')))))
                if path == '/api/step':
                    t0 = time.perf_counter()
                    done = SESSION.advance(req.get('steps', 1), req.get('until'), float(req.get('max_ms', 150.0)))
                    return self._send(200, dict(frame=SESSION.frame(want_ebar=bool(req.get('want_ebar')), steps_done=done,
                                                                     wall_ms=(time.perf_counter() - t0) * 1e3)))
            return self._send(404, dict(error='not found'))
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, dict(error=str(e)))


def make_server(host='127.0.0.1', port=8765):
    global DEFAULTS
    DEFAULTS = stigmergic_defaults()
    return ThreadingHTTPServer((host, port), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--host', default='127.0.0.1', help='bind address (default: local only)')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--open', action='store_true', help='open the browser')
    a = ap.parse_args()
    srv = make_server(a.host, a.port)
    url = f'http://{a.host}:{srv.server_address[1]}/'
    print(f'Potentia visualizer on {url}   (Ctrl-C to stop)', flush=True)
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
