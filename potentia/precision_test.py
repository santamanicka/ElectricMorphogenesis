"""Does a homogeneous start break its symmetry because of numerical precision?   python precision_test.py   (writes results/precision_test.txt)

A homogeneous start in a disc on a square grid is exactly D4-symmetric, and every term of the update commutes with D4 (operator_symmetry.py), so in exact arithmetic the solution
would stay symmetric for ever.  In floating point two things can seed an asymmetry of about 1e-16: (1) the kernel of the field sum is not exactly (anti)symmetric (the order in which
the 8 x 8 sub-samples of a source pixel are averaged differs between mirror images), (2) the rounding in the sums and FFTs differs between mirror images.  The dynamics then amplify it.
Runs (disc, 30 x 30 cells, m = 2, Stigmergic, homogeneous start; symmetry error = rms(V - mirror V) / spread over the tissue, the maximum over the D4 operations):
  A  float64, the model as it is
  B  float64, the kernel made exactly (anti)symmetric                          -> removes seed (1)
  C  long double (80-bit, 64-bit mantissa), exactly symmetric kernel and its spectrum in long double -> also lowers seed (2) by a factor of about 2000
  D  float64, G_pol perturbed by 1e-12 / 1e-9 / 1e-6                             -> seeds of known size
If precision (the size of the seed) is what matters, the time at which the error reaches order one should shift by ln(seed ratio) / lambda, with lambda the growth rate of the error.
"""
import os
import time
import numpy as np
from scipy import fft as sfft

from potentia import Potentia
from shapes import shape_mask
from sweeps import STIG

lines = []
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def log(s=''):
    print(s, flush=True)
    lines.append(s)


class PotentiaSym(Potentia):
    """The same model with the field kernel made exactly odd/even (the average of the kernel and its mirror image)."""

    def _kernel_arrays(self):
        n, (Kx, Ky) = super()._kernel_arrays()
        Kx = 0.5 * (Kx - Kx[::-1, :]); Kx = 0.5 * (Kx + Kx[:, ::-1])
        Ky = 0.5 * (Ky - Ky[:, ::-1]); Ky = 0.5 * (Ky + Ky[::-1, :])
        return n, (Kx, Ky)


class PotentiaLD(PotentiaSym):
    """Exactly symmetric kernel; the state, the sums and the kernel spectrum carried in long double (the kernel values themselves stay float64: exactly symmetric, not more precise)."""

    def _build_kernel(self):
        """As Potentia._build_kernel (finite tissue) but the kernel spectrum is computed in long double: the float64 FFT of an exactly odd kernel is not exactly odd in the spectral domain."""
        n, ks = self._kernel_arrays()
        P = int(np.ceil(self.m / 2)) + 2
        need = max(self.Np) + self.m + len(n) + 4 + 2 * P
        M = (sfft.next_fast_len(need),) * 2
        self.pad = P; self.M = M
        spec = []
        for k in ks:
            arr = np.zeros(M, dtype=np.longdouble)
            ix = n % M[0]; iy = n % M[1]
            np.add.at(arr, (ix[:, None] + 0 * iy[None, :], 0 * ix[:, None] + iy[None, :]), k.astype(np.longdouble))
            spec.append(sfft.rfft2(arr))
        self.Kspec = spec

    def field(self, V):
        Vp = np.zeros(self.M, dtype=V.dtype)
        Pd = self.pad
        Vp[Pd:Pd + self.Np[0], Pd:Pd + self.Np[1]] = V
        Vh = sfft.rfft2(Vp)
        ex = sfft.irfft2(Vh * self.Kspec[0], s=self.M)
        ey = sfft.irfft2(Vh * self.Kspec[1], s=self.M)
        return np.sqrt(ex * ex + ey * ey + self.epsilon)


Np = (60, 60)
mk = shape_mask('disc', Np)
ops = [lambda a: a[::-1, :], lambda a: a[:, ::-1], np.rot90, lambda a: a.T]


def sym_error(V):
    V = np.asarray(V, dtype=np.float64) if V.dtype != np.longdouble else V
    sel = mk
    ref = np.sqrt(np.mean((V[sel] - V[sel].mean()) ** 2))
    return float(max(np.sqrt(np.mean((V[sel] - op(V)[sel]) ** 2)) for op in ops) / ref)


def run(cls, noise, T, dtype=np.float64, label=''):
    P = cls(size_cells=(30, 30), m=2, mask=mk, **STIG)
    g0 = 1.0 + noise * np.random.default_rng(0).standard_normal(P.Np)
    V, g = P.initial_from_g(g0, 'dep')
    V = V.astype(dtype); g = g.astype(dtype)
    rec = {}; t0 = time.time()
    for it in range(int(round(T / P.dt)) + 1):
        if it % 1000 == 0:
            rec[it // 100] = sym_error(V)
        if it < int(round(T / P.dt)):
            V, g = P.step(V, g)
    log(f'    {label:<44} ({time.time() - t0:5.0f} s wall)')
    return rec


def first_above(rec, level=0.1):
    for t in sorted(rec):
        if rec[t] > level:
            return t
    return None


times = [10, 100, 200, 300, 400, 500, 600, 700, 800]
log('Symmetry error versus time (homogeneous disc start, 30 x 30 cells, m = 2)')
runs = {}
for label, cls, noise, T, dt_ in (('A  float64, as is', Potentia, 0.0, 800.0, np.float64), ('B  float64, exactly symmetric kernel', PotentiaSym, 0.0, 800.0, np.float64),
                                  ('D1 float64, G_pol perturbed by 1e-12', Potentia, 1e-12, 800.0, np.float64), ('D2 float64, G_pol perturbed by 1e-9', Potentia, 1e-9, 800.0, np.float64),
                                  ('D3 float64, G_pol perturbed by 1e-6', Potentia, 1e-6, 800.0, np.float64), ('C  long double, exactly symmetric kernel', PotentiaLD, 0.0, 800.0, np.longdouble)):
    runs[label] = run(cls, noise, T, dt_, label)
log()
log(f'    {"run":<44}' + ''.join(f'{t:>10d} s' for t in times) + '   first time above 0.1')
for label, rec in runs.items():
    fa = first_above(rec)
    log(f'    {label:<44}' + ''.join(f'{rec.get(t, float("nan")):>12.1e}' for t in times) + f'   {fa if fa is not None else "> 800"} s')
log()
# growth rate of the error in the exponential range (1e-12 .. 1e-3) for the runs that have it
for label, rec in runs.items():
    ts = np.array(sorted(rec)); es = np.array([rec[t] for t in ts])
    sel = (es > 1e-11) & (es < 1e-3)
    if sel.sum() >= 4:
        lam = np.polyfit(ts[sel], np.log(es[sel]), 1)[0]
        log(f'    growth rate of the error while it is between 1e-11 and 1e-3, {label[:30]:<30}: {lam:.3f} per s  ({sel.sum()} samples)')
print('\n'.join(lines))
with open(os.path.join(OUT, 'precision_test.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
