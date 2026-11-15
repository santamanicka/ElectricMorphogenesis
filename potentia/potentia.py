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
                 source=None, n_sub=8, dt=None, workers=1, grn=None, size_um=None, mask=None, range_map=None):
        self.m = int(m)
        if size_um is not None:
            # physical tissue size in micrometres (the continuum way to say 'extent'); must be a whole number of pixels
            h_um = 2 * self.a * 1e6 / self.m
            npx = [int(round(L / h_um)) for L in size_um]
            if any(abs(n * h_um - L) > 1e-6 * L for n, L in zip(npx, size_um)):
                raise ValueError(f'size_um={tuple(size_um)} is not a multiple of the pixel pitch h={h_um:g} um (m={self.m})')
            size_cells = tuple(n / self.m for n in npx)
        self.size_cells = tuple(size_cells)
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
        self.Np = (int(round(self.size_cells[0] * self.m)), int(round(self.size_cells[1] * self.m)))
        # optional tissue shape: boolean mask on the pixel grid (True = tissue), see shapes.py.  Outside pixels hold no charge, have no gap junctions and stay at V = g = 0.
        self.mask = None
        self._lv = None                                                  # position-dependent range (set_range_map), None = one range everywhere
        if mask is not None:
            self._apply_mask(mask, periodic)
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
        if range_map is not None:
            self.set_range_map(range_map)

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

    def _kernel_spec(self, R, M):
        """Spectrum (one array per kernel component) of the field kernel of range R (metres) folded onto the M grid, as _build_kernel does for self.R."""
        old = self.R
        self.R = R
        try:
            n, ks = self._kernel_arrays()
        finally:
            self.R = old
        spec = []
        for k in ks:
            arr = np.zeros(M)
            ix = n % M[0]
            iy = n % M[1]
            np.add.at(arr, (ix[:, None] + 0 * iy[None, :], 0 * ix[:, None] + iy[None, :]), k)
            spec.append(sfft.rfft2(arr, workers=self.workers))
        return spec

    def set_range_map(self, R_map, level_step_px=1.5, max_levels=12):
        """Position-dependent field range: R_map = the range in metres at every pixel (array of the pixel grid; only tissue pixels count), or None for the single range of the model.
        Each cell senses the field of the charges within *its own* range.  The field is computed at a few ranges R_j (evenly spaced between the smallest and the largest of the map, about
        `level_step_px` pixels apart, at most `max_levels`), the field is perceived at each, and a cell takes the linear interpolation between the two levels that enclose its range
        (a cell whose range is one of the levels senses exactly that level's field).  The cost of a step is that of `len(levels)` fields.  See localrange.py for a map from the local width."""
        if R_map is None:
            self._lv = None
            return
        if self.periodic:
            raise ValueError('a position-dependent range needs periodic=False')
        Rm = np.asarray(R_map, float)
        if Rm.shape != tuple(self.Np):
            raise ValueError(f'range map has shape {Rm.shape}, expected the pixel grid {tuple(self.Np)}')
        sel = Rm[self.mask] if self.mask is not None else Rm.ravel()
        lo, hi = float(sel.min()), float(sel.max())
        h = self.h
        n = 1 if hi - lo < 0.5 * h else int(min(max_levels, max(2, int(np.ceil((hi - lo) / (level_step_px * h))) + 1)))
        if n == 1:
            Rl = [0.5 * (lo + hi)]
            ws = [np.ones(self.Np)]
        else:
            dR = (hi - lo) / (n - 1)
            Rl = [lo + j * dR for j in range(n)]
            f = np.minimum(np.maximum((Rm - lo) / dR, 0.0), float(n - 1))
            idx0 = np.minimum(np.floor(f), float(n - 2))
            alpha = f - idx0
            ws = [np.where(idx0 == j, 1.0 - alpha, 0.0) + np.where(idx0 + 1.0 == j, alpha, 0.0) for j in range(n)]
        W = int(np.ceil(max(Rl) / h)) + 3
        P = int(np.ceil(self.m / 2)) + 2
        need = max(self.Np) + self.m + (2 * W + 1) + 4 + 2 * P
        M = (sfft.next_fast_len(need),) * 2
        self._lv = dict(R=Rl, spec=[self._kernel_spec(R, M) for R in Rl], M=M, w=ws, lo=lo, hi=hi, Rmap=Rm)

    def _perceive_local(self, V):
        """Perceived field with a position-dependent range: the interpolation, level by level, of the field perceived at each range."""
        lv = self._lv
        if self.mask is not None:
            V = V * self.maskf
        M = lv['M']
        P = self.pad
        Vp = np.zeros(M)
        Vp[P:P + self.Np[0], P:P + self.Np[1]] = V
        Vh = sfft.rfft2(Vp, workers=self.workers)
        acc = 0.0
        for spec, w in zip(lv['spec'], lv['w']):
            if self.vector:
                ex = sfft.irfft2(Vh * spec[0], s=M, workers=self.workers)
                ey = sfft.irfft2(Vh * spec[1], s=M, workers=self.workers)
                A = np.sqrt(ex * ex + ey * ey + self.epsilon)
            else:
                A = sfft.irfft2(Vh * spec[0], s=M, workers=self.workers)
            acc = acc + w * self.perceive(A)
        return acc

    # ------------------------------------------------------------------ operators
    def field(self, V):
        """|E| (vector) or scalar potential on the probe grid, shape M (padded if finite)."""
        if self.mask is not None:
            V = V * self.maskf                                        # only tissue pixels carry charge
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
                if self.mask is not None:
                    f = f * self._gj_pairs[ax]
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
        ebar = self.perceived_field(V)
        x = self.gain * ebar + self.bias
        dg = (10.0 / self.tau) * (-self.leak_g * g + self.weight * np.tanh(0.5 * x))
        g = np.clip(g + self.dt * dg, 0.0, 2.0)
        V = V + self.dt * (self.ion(V, g) + self.gj(V))
        if self.mask is not None:
            V = V * self.maskf
            g = g * self.maskf
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
        if self.mask is not None:
            V, g = V * self.maskf, g * self.maskf
        return V, g

    def _apply_mask(self, mask, periodic=None):
        """Install a tissue shape (an all-True mask is the plain rectangle, stored as None)."""
        periodic = self.periodic if periodic is None else periodic
        mask = np.asarray(mask, bool)
        if mask.shape != self.Np:
            raise ValueError(f'mask has shape {mask.shape}, expected the pixel grid {self.Np}')
        if periodic:
            raise ValueError('a tissue shape (mask) needs periodic=False: the boundary of the shape is a no-flux edge')
        if not mask.any():
            raise ValueError('the mask contains no tissue')
        if mask.all():
            self.mask = None
            return
        self.mask = mask
        self.maskf = mask.astype(float)
        self._gj_pairs = (mask[:-1, :] & mask[1:, :], mask[:, :-1] & mask[:, 1:])      # a gap junction exists only between two tissue pixels

    def set_mask(self, mask, V, g, fill=(0.0, 0.0)):
        """Change the tissue shape while the simulation runs (growth or shrinkage). Returns the new (V, g): pixels that stay keep their state, new pixels inherit the mean of their
        already-assigned neighbours (development.grow_state), removed pixels are zeroed; `fill` = (V, g) is the state of a new island that touches no tissue."""
        if self._lv is not None:
            raise ValueError('the tissue shape cannot change while a position-dependent range is set (set_range_map(None) first)')
        from development import grow_state
        old = self.mask if self.mask is not None else np.ones(self.Np, bool)
        new = np.asarray(mask, bool)
        self._apply_mask(new)
        return grow_state(V, g, old, new, fill)

    def set_range(self, fieldScreenSize):
        """Change the field range (R_s, in cell half-diagonals) and rebuild the kernel."""
        if self._lv is not None:
            raise ValueError("the single field range cannot change while a position-dependent range is set (set_range_map(None) first)")
        self.R_s = float(fieldScreenSize)
        self.R = self.a * np.sqrt(2) * (self.R_s + .001)
        self._build_kernel()

    # ---- initial conditions are set on G_pol; Vmem follows it -----------------------------------------------
    _EQ_CACHE = {}

    def _equilibrium_table(self):
        """Stable equilibria of an isolated cell (no gap junctions, no field) on a g grid (step 5e-4), as two tables:
        'dep' = the most depolarised stable root, 'hyp' = the most hyperpolarised one (they coincide outside the bistable window)."""
        key = (self.G_dep, self.Z, self.V_th, self.V_T, self.E_pol, self.E_dep)
        if key not in Potentia._EQ_CACHE:
            gs = np.linspace(0.0, 2.0, 4001)
            Vs = np.linspace(-0.08, 0.02, 2001)
            def f(V, g):                                                     # same channel terms as ion(), without the 1/tau factor
                e = np.exp(self.Z * (V - self.V_th) / self.V_T)
                return -g * (V - self.E_pol) / (1.0 + e) - self.G_dep * (V - self.E_dep) * e / (1.0 + e)
            F = f(Vs[None, :], gs[:, None])
            dep = np.full(gs.size, np.nan); hyp = np.full(gs.size, np.nan)
            sc = (F[:, :-1] > 0) & (F[:, 1:] < 0)                           # + to - crossing = stable root
            for k in range(gs.size):
                idx = np.where(sc[k])[0]
                if idx.size == 0: continue
                roots = []
                for i in idx:                                                # refine each bracket by bisection
                    lo, hi = Vs[i], Vs[i + 1]
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        if f(mid, gs[k]) > 0: lo = mid
                        else: hi = mid
                    roots.append(0.5 * (lo + hi))
                hyp[k], dep[k] = min(roots), max(roots)
            Potentia._EQ_CACHE[key] = (gs, dep, hyp)
        return Potentia._EQ_CACHE[key]

    def equilibrium_V(self, g, branch='dep'):
        """Vmem (volts) that an isolated cell with conductance g relaxes to; where two stable states exist (about 0.81 < g < 1.44)
        'dep' (the repo's default start, -9.2 mV at g=1) picks the depolarised one and 'hyp' the hyperpolarised one."""
        gs, dep, hyp = self._equilibrium_table()
        tab = {'dep': dep, 'hyp': hyp}[branch]
        i = np.clip(np.rint((np.asarray(g, float) - gs[0]) / (gs[1] - gs[0])).astype(int), 0, gs.size - 1)
        return tab[i]

    def initial_from_g(self, g, branch='dep'):
        """State for a given G_pol field: Vmem is the isolated-cell equilibrium of each pixel (G_pol directly determines Vmem)."""
        g = np.clip(np.asarray(g, float), 0.0, 2.0) * np.ones(self.Np)
        V = self.equilibrium_V(g, branch)
        if self.mask is not None:
            V, g = V * self.maskf, g * self.maskf
        return V, g

    def perceived_field(self, V):
        if self._lv is not None:
            return self._perceive_local(V)
        return self.perceive(self.field(V))

    def cell_average(self, X):
        """Block-mean of m x m pixels -> one value per cell (comparable with the lattice model)."""
        m = self.m
        if self.Np[0] % m or self.Np[1] % m:
            raise ValueError('cell_average needs a tissue that is a whole number of cells')
        n0, n1 = self.Np[0] // m, self.Np[1] // m
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
