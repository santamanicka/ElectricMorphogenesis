"""Potentia: a continuous-space generalisation of the bioelectric cellular field network.

The lattice model has continuous *states* (Vmem, G_pol) but a discrete *space* (cells on a
grid).  Potentia keeps the states and equations, and makes space continuous, the way Lenia
does for the Game of Life: the tissue is a field V(x,t) (membrane potential) with a slow
channel-state field g(x,t) = G_pol/G_ref, integrated on a pixel grid of pitch h that is
*independent* of the cell size.  Cells survive only as length scales.

    C dV/dt = I_ion(V,g) + div( D * gate(p*|dV|/V0) grad V )          gap junctions -> gated diffusion
    dg/dt   = (10/tau) * ( -leak*g + w * tanh( (gamma*ebar + beta)/2 ) ),  g in [0,2]
    E(x)    = fc * (C/p^2) * INT V(y) K(x-y) d^2y,   K(r) = r/|r|^2 for |r|<=R   (2D Coulomb, screened)
    ebar(x) = mean over s in {+-1}^2 of |E(x + a*s)|                           perception at cell corners

Lengths: a = cell radius (5 um), p = 2a = cell pitch, R = a*sqrt(2)*(R_s+.001) = field range.
`m` = pixels per cell pitch (h = p/m).  The lattice model is the special case m=1,
source='point' (charge at cell centres, probes at cell corners); `m>=2, source='pixel'` is the
continuum regime (charge spread over its pixel).  tests/validate_potentia.py checks both claims.

Pure-bioelectric version: no GRN is loaded and `geneRegulatoryNetwork` is never imported.  The original's
provision is kept as an inert hook (`grn=None`): if a dict is supplied it is applied exactly where the original
applies it (gene -> G_pol coupling *before* the field update; gene state advanced from Vmem after the Vmem update).
Ligand signalling is not implemented either (it is disabled in data/StigmergicModelParameters.dat).

Defaults reproduce the repo's *as-implemented* equations, including the fact that G_pol is
in siemens inside `-G_pol + w*tanh()`, which makes the leak coefficient 1e-9 (see README note).
"""
import numpy as np
from scipy import fft as sfft


class Potentia:
    Z = 3; V_th = -27e-3; V_T = 27e-3; V_0 = 12e-3
    C = 0.1e-9; E_pol = -55e-3; E_dep = -5e-3; G_ref = 1.0e-9
    a = 5.0e-6; k_e = 8.987e9; eps_r = 1e7; dt = 0.01; epsilon = 1e-15

    def __init__(self, size_cells=(11, 11), m=2, periodic=False, fieldScreenSize=4.0,
                 fieldStrength=1.0, gain=-1.0, weight=1000.0, bias=5.3e-4, tau=10.0,
                 GJStrength=0.05, fieldVector=True, leak=None, G_dep=1.5,
                 source=None, n_sub=8, dt=None, workers=1, grn=None):
        self.size_cells = tuple(size_cells)
        self.m = int(m)
        self.periodic = periodic
        self.R_s = float(fieldScreenSize)
        self.gain, self.weight, self.bias, self.tau = gain, weight, bias, tau
        self.fs = fieldStrength
        self.GJ = GJStrength
        self.vector = fieldVector
        self.leak_g = self.G_ref if leak is None else leak
        self.G_dep = G_dep
        self.workers = workers
        self.grn = grn            # provision only; None = pure bioelectric (default, nothing loaded)
        self.p = 2 * self.a
        self.h = self.p / self.m
        self.Np = (self.size_cells[0] * self.m, self.size_cells[1] * self.m)
        self.R = self.a * np.sqrt(2) * (self.R_s + .001)
        self.source = source or ('point' if self.m == 1 else 'pixel')
        self.n_sub = 1 if self.source == 'point' else n_sub
        assert self.n_sub == 1 or self.n_sub % 2 == 0, 'n_sub must be even (avoid r=0 in sub-sampling)'
        self.fc = self.fs * (self.k_e / self.eps_r)
        self.half = (self.m % 2 == 1)          # odd m: probes on the corner grid
        # perception shifts (probe-index offsets) for s = -1,+1
        self.dk = ((1 - self.m) // 2, (1 + self.m) // 2) if self.half else (-self.m // 2, self.m // 2)
        self._build_kernel()
        # GJ coefficient per neighbour pair, in 1/s:  (G_0/C) * m^2   (gate argument uses m*dV)
        self.gj_coef = self.GJ * (self.G_ref / self.C) * self.m ** 2
        self.k_ion = self.G_ref / self.C
        # explicit-Euler stability of the gated diffusion: dt*gj_coef <= ~0.2 (default dt=0.01 is the repo's)
        if dt is not None:
            self.dt = dt
        elif self.gj_coef > 0:
            self.dt = min(Potentia.dt, 0.2 / self.gj_coef)

    # ------------------------------------------------------------------ kernel
    def _kernel_arrays(self):
        """Kernel on probe-minus-source offsets, window [-W, W] pixels; returns (Kx, Ky) or Ks, plus index n."""
        h, m = self.h, self.m
        W = int(np.ceil(self.R / h)) + 3
        n = np.arange(-W, W + 1)
        off = (n - 0.5) if self.half else n.astype(float)          # probe - source, in units of h
        u = (((np.arange(self.n_sub) + 0.5) / self.n_sub) - 0.5) * h if self.n_sub > 1 else np.zeros(1)
        # displacement probe - (source centre + u)
        dX = off[:, None, None, None] * h - u[None, None, :, None]
        dY = off[None, :, None, None] * h - u[None, None, None, :]
        dX, dY = np.broadcast_arrays(dX, dY)
        r2 = dX ** 2 + dY ** 2
        inside = (np.sqrt(r2) <= self.R)
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.vector:
                Kx = np.where(inside, dX / r2, 0.0).mean(axis=(2, 3))
                Ky = np.where(inside, dY / r2, 0.0).mean(axis=(2, 3))
                ks = (Kx, Ky)
            else:
                ks = (np.where(inside, 1.0 / np.sqrt(r2), 0.0).mean(axis=(2, 3)),)
        scale = self.fc * self.C * (self.h ** 2 / self.p ** 2)       # charge per pixel = C*V*h^2/p^2
        return n, tuple(k * scale for k in ks)

    def _build_kernel(self):
        n, ks = self._kernel_arrays()
        if self.periodic:
            M = self.Np
            self.pad = 0
        else:
            P = int(np.ceil(self.m / 2)) + 2
            need = max(self.Np) + self.m + len(n) + 4 + 2 * P
            M = (sfft.next_fast_len(need),) * 2
            self.pad = P
        self.M = M
        spec = []
        for k in ks:
            arr = np.zeros(M)
            ix = n % M[0]
            iy = n % M[1]
            np.add.at(arr, (ix[:, None] + 0 * iy[None, :], 0 * ix[:, None] + iy[None, :]), k)
            spec.append(sfft.rfft2(arr, workers=self.workers))
        self.Kspec = spec

    # ------------------------------------------------------------------ operators
    def field(self, V):
        """|E| (vector) or scalar potential on the probe grid, shape M (padded if finite)."""
        if self.periodic:
            Vp = V
        else:
            Vp = np.zeros(self.M)
            P = self.pad
            Vp[P:P + self.Np[0], P:P + self.Np[1]] = V
        Vh = sfft.rfft2(Vp, workers=self.workers)
        if self.vector:
            ex = sfft.irfft2(Vh * self.Kspec[0], s=self.M, workers=self.workers)
            ey = sfft.irfft2(Vh * self.Kspec[1], s=self.M, workers=self.workers)
            return np.sqrt(ex * ex + ey * ey + self.epsilon)
        return sfft.irfft2(Vh * self.Kspec[0], s=self.M, workers=self.workers)

    def perceive(self, A):
        """Mean of the field at the four perception probes (cell corners) of every pixel."""
        n0, n1 = self.Np
        acc = 0.0
        for sx in self.dk:
            for sy in self.dk:
                if self.periodic:
                    acc = acc + np.roll(np.roll(A, -sx, axis=0), -sy, axis=1)
                else:
                    P = self.pad
                    acc = acc + A[P + sx:P + sx + n0, P + sy:P + sy + n1]
        return acc / 4.0

    def gj(self, V):
        m, V0 = self.m, self.V_0
        out = np.zeros_like(V)
        for ax in (0, 1):
            if self.periodic:
                d = np.roll(V, -1, axis=ax) - V
                f = self.gj_coef * d * (2.0 / (1.0 + np.cosh(m * d / V0)))
                out += f - np.roll(f, 1, axis=ax)
            else:
                sl_a = [slice(None)] * 2; sl_b = [slice(None)] * 2
                sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None)
                d = V[tuple(sl_b)] - V[tuple(sl_a)]
                f = self.gj_coef * d * (2.0 / (1.0 + np.cosh(m * d / V0)))
                out[tuple(sl_a)] += f
                out[tuple(sl_b)] -= f
        return out

    def ion(self, V, g):
        e_in = np.exp(self.Z * (V - self.V_th) / self.V_T)
        Pin = 1.0 / (1.0 + e_in)
        Pout = e_in / (1.0 + e_in)                  # = 1/(1+exp(-z)), without overflow
        return self.k_ion * (-g * (V - self.E_pol) * Pin - self.G_dep * (V - self.E_dep) * Pout)

    def _grn_to_conductance(self, g):
        """Original provision (cellularFieldNetwork.updateIonChannelConductance, inputSource='gene'):
              dp = -G_pol + (2*sigmoid(gene + GRNBiases) - 1) @ GRNtoVmemWeights
        grn = {'state': (Np0,Np1,nG), 'biases': (nG,), 'toVmemWeights': (nG,), 'target': 'Vmem'|'VmemAndLigand'|'Ligand',
               'advance': optional callable(V) -> new state}.  Weights are expected already divided by their time constant."""
        G = self.grn
        if G.get('target', 'Vmem') not in ('Vmem', 'VmemAndLigand'):
            return g
        drive = (2.0 / (1.0 + np.exp(-(G['state'] + G['biases']))) - 1.0) @ G['toVmemWeights']
        dp = -self.leak_g * g + drive
        return np.clip(g + self.dt * dp, 0.0, 2.0)

    def step(self, V, g):
        if self.grn is not None:
            g = self._grn_to_conductance(g)
        ebar = self.perceive(self.field(V))
        x = self.gain * ebar + self.bias
        dg = (10.0 / self.tau) * (-self.leak_g * g + self.weight * np.tanh(0.5 * x))
        g = np.clip(g + self.dt * dg, 0.0, 2.0)
        V = V + self.dt * (self.ion(V, g) + self.gj(V))
        if self.grn is not None and self.grn.get('advance') is not None:
            self.grn['state'] = self.grn['advance'](V)
        return V, g

    # ------------------------------------------------------------------ utilities
    def initial(self, V0=-9.2e-3, g0=1.0, noise_V=0.0, noise_g=0.0, seed=None):
        V = np.full(self.Np, V0)
        g = np.full(self.Np, g0)
        if noise_V or noise_g:
            rng = np.random.default_rng(seed)
            V = V + noise_V * rng.standard_normal(self.Np)
            g = np.clip(g + noise_g * rng.standard_normal(self.Np), 0, 2)
        return V, g

    def perceived_field(self, V):
        return self.perceive(self.field(V))

    def cell_average(self, X):
        """Block-mean of m x m pixels -> one value per cell (comparable with the lattice model)."""
        m = self.m
        n0, n1 = self.size_cells
        return X.reshape(n0, m, n1, m).mean(axis=(1, 3))

    def run(self, nsteps, V=None, g=None, snap_every=0, snap_fields=('V',)):
        if V is None:
            V, g = self.initial()
        snaps = {k: [] for k in snap_fields}
        times = []
        for it in range(nsteps):
            if snap_every and it % snap_every == 0:
                times.append(it)
                if 'V' in snaps: snaps['V'].append(V.astype(np.float32))
                if 'g' in snaps: snaps['g'].append(g.astype(np.float32))
            V, g = self.step(V, g)
        out = {k: np.array(v) for k, v in snaps.items()}
        out['t'] = np.array(times) * self.dt
        return V, g, out


    @classmethod
    def from_repo_parameters(cls, parameters, m=2, size_cells=None, **overrides):
        """Build a Potentia from a repo parameter dict (e.g. tio.load('data/StigmergicModelParameters.dat')).
        Mirrors cellularFieldNetwork.loadParameters for the GJ and field blocks.  GRNParameters / ligandParameters are
        *read but not loaded*: Potentia is the pure-bioelectric version, so GRNEnabled is ignored (provision kept)."""
        fp = parameters['fieldParameters']
        scal = lambda x: float(np.asarray(x).reshape(-1)[0])
        kw = dict(fieldScreenSize=fp['fieldScreenSize'], fieldStrength=scal(fp['fieldStrength']),
                  gain=scal(fp.get('fieldTransductionGain', 1.0)), weight=scal(fp['fieldTransductionWeight']),
                  bias=scal(fp['fieldTransductionBias']), tau=scal(fp['fieldTransductionTimeConstant']),
                  fieldVector=fp.get('fieldVector', False),
                  GJStrength=(parameters.get('GJParameters') or {}).get('GJStrength', 0.05))
        kw.update(overrides)
        obj = cls(size_cells=size_cells or tuple(parameters['latticeDims']), m=m, **kw)
        grnp = parameters.get('GRNParameters')
        obj.GRNEnabled = False
        obj.GRNParametersIgnored = bool(grnp) and bool(grnp.get('GRNEnabled', False))
        return obj
