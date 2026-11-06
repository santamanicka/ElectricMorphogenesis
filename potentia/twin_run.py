"""Is the 'never settles, never repeats' behaviour at the Stigmergic point genuinely non-repeating (chaotic)?

A recurrence score can't distinguish a very long period from non-repetition in finite time, so we also test
sensitivity: run a reference trajectory and a twin whose initial state differs by a tiny amount, and track the
rms separation d(t) of the cell-averaged Vmem.  Exponential growth of d(t) up to saturation = positive Lyapunov
exponent = non-repeating; d(t) staying tiny or decaying = stable orbit/fixed point.
 (a) symmetric kick   : +1 uV on the central cell(s)   (stays inside the D4-symmetric invariant subspace)
 (b) asymmetric kick  : 1 nV random noise on all cells (tests stability against symmetry breaking)
"""
import sys, numpy as np
from sweeps import STIG
from lattice import Lattice
from potentia import Potentia

T = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
N = 11
rng = np.random.default_rng(0)


def make(model, m):
    if model == 'lattice':
        L = Lattice(dims=(N, N), fieldScreenSize=STIG['fieldScreenSize'], bias=STIG['bias'], weight=STIG['weight'], tau=STIG['tau'],
                    GJStrength=STIG['GJStrength'], fieldVector=True)
        return (lambda V, g: (lambda r: (r[0], r[1] / L.G_ref))(L.step(V.ravel(), g.ravel() * L.G_ref))), L.dt, (lambda V: V.reshape(N, N)), 1, L
    P = Potentia(size_cells=(N, N), m=m, **{k: v for k, v in STIG.items() if k != 'fieldStrength'}, fieldStrength=STIG['fieldStrength'])
    return P.step, P.dt, (lambda V: P.cell_average(V)), m, P


def run(model, m, kick):
    step, dt, cells, mm, obj = make(model, m)
    shape = (N, N) if model == 'lattice' else obj.Np
    V0 = np.full(shape, -9.2e-3); g0 = np.ones(shape)
    V1 = V0.copy(); g1 = g0.copy()
    if kick == 'sym':
        c = N * mm // 2
        sl = (slice(c, c + 1), slice(c, c + 1)) if mm % 2 == 0 and model == 'lattice' else (slice(c - (mm % 2 == 0) * 1 + (mm % 2 == 0) * 0, c + 1 + 0), slice(c - (mm % 2 == 0) * 1, c + 1))
        if model == 'lattice':
            V1[N // 2, N // 2] += 1e-6
        else:
            h = N * mm // 2
            V1[h - 1:h + 1, h - 1:h + 1] += 1e-6 if mm % 2 == 0 else 0.0
            if mm % 2 == 1:
                V1[h, h] += 1e-6
    else:
        V1 += 1e-9 * rng.standard_normal(shape)
    nsteps = int(round(T / dt)); rec = int(round(1.0 / dt))
    ts, ds = [], []
    for it in range(nsteps):
        if it % rec == 0:
            ts.append(it * dt); ds.append(np.sqrt(((cells(V0) - cells(V1)) ** 2).mean()) * 1e3)
        V0, g0 = step(V0, g0); V1, g1 = step(V1, g1)
    return np.array(ts), np.array(ds)


if __name__ == '__main__':
    print(f'twin-run divergence at the Stigmergic point, {N}x{N} finite tissue, homogeneous start, T={T:.0f}s')
    print('d(t) = rms separation of cell-averaged Vmem (mV); initial separations: sym 1e-6 V on centre cell(s) ~ 1e-3 mV; asym 1e-9 V ~ 1e-6 mV')
    res = {}
    for model, m in (('lattice', 1), ('potentia', 2)):
        for kick in ('sym', 'asym'):
            ts, ds = run(model, m, kick)
            res[(model, m, kick)] = (ts, ds)
            sat = np.median(ds[-100:])
            # exponent: fit log d over the window where 1e-2*sat < d < 0.1*sat, first crossing
            ok = (ds > 1e-2 * sat) & (ds < 0.1 * sat)
            idx = np.where(ok)[0]
            lam = np.nan
            if len(idx) > 5:
                i0, i1 = idx[0], idx[min(len(idx) - 1, idx[0] + 60)]
                seg = slice(i0, i1 + 1)
                lam = np.polyfit(ts[seg], np.log(ds[seg]), 1)[0]
            def at(t): return ds[min(len(ds) - 1, int(t))]
            print(f'{model:8s} m={m} {kick:5s}: d(t=10s)={at(10):.2e}  d(100s)={at(100):.2e}  d(300s)={at(300):.2e}  d(end)={ds[-1]:.2e} mV  '
                  f'saturation(median last 100 s) {sat:.2f} mV   growth rate ~ {lam:.3f} /s' if np.isfinite(lam) else
                  f'{model:8s} m={m} {kick:5s}: d(t=10s)={at(10):.2e}  d(100s)={at(100):.2e}  d(300s)={at(300):.2e}  d(end)={ds[-1]:.2e} mV  saturation {sat:.2f} mV   growth rate: no clean exponential window')
    np.save('results/twin_run.npy', {f'{k[0]}_m{k[1]}_{k[2]}': v for k, v in res.items()}, allow_pickle=True)
