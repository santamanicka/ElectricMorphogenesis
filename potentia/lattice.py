"""Reference numpy port of the discrete bioelectric 'cellular field network'.

This is a *direct* translation of cellularFieldNetwork.py (field-enabled, no ligand,
no GRN) so that Potentia can be validated against, and compared with, the lattice model
without needing torch.  It deliberately keeps the original structure (dense distance
matrices, cell-corner field grid, same update order, SI units) rather than being clever.

Update order per iteration (cellularFieldNetwork.simulate):
    1. eV  <- field computed from the current Vmem          (updateExtracellularVoltage)
    2. G_pol <- G_pol + dt * dG_pol(eV)                      (updateIonChannelConductance, 'field')
    3. I   <- ion-channel + gap-junction current             (updateCurrent)
    4. Vmem <- Vmem + dt * I / C                             (updateVmem)

`periodic=True` is an extension used only for comparison with periodic Potentia: GJ adjacency
wraps (as in the repo's latticePeriodicBoundary) and field distances use the minimum-image
convention (valid while the screening radius is below half the domain size).
"""
import numpy as np


class Lattice:
    # constants copied from cellularFieldNetwork.__init__
    Z = 3
    V_th = -27e-3
    V_T = 27e-3
    V_0 = 12e-3
    C = 0.1e-9
    E_pol = -55e-3
    E_dep = -5e-3
    G_ref = 1.0e-9
    a = 5.0e-6                 # cell radius
    k_e = 8.987e9
    eps_r = 1e7
    dt = 0.01
    epsilon = 1e-15

    def __init__(self, dims=(11, 11), fieldScreenSize=4, fieldStrength=1.0, gain=-1.0,
                 weight=1000.0, bias=5.3e-4, tau=10.0, GJStrength=0.05, fieldVector=True,
                 fieldRangeSymmetric=False, periodic=False, G_dep=1.5, leak=None):
        self.dims = tuple(dims)
        self.R_s = fieldScreenSize
        self.fs, self.gain, self.weight, self.bias, self.tau = fieldStrength, gain, weight, bias, tau
        self.G_0 = GJStrength * self.G_ref
        self.vector = fieldVector
        self.periodic = periodic
        self.G_dep = G_dep * self.G_ref
        # Leak coefficient in units where g = G_pol/G_ref.  The repo computes
        #   dp = 10*(-G_pol[SI] + tanh*w)/tau ;  G_pol[SI] += dt*dp*G_ref
        # i.e. in g-units  dg/dt = (10/tau)*(-G_ref*g + w*tanh(x/2)), so the effective leak
        # coefficient is G_ref = 1e-9 (negligible next to w*tanh ~ O(1e-1..1e3)).
        # leak=None reproduces the repo; leak=1.0 is the dimensionally consistent version.
        self.leak_g = self.G_ref if leak is None else leak
        nr, nc = self.dims
        self.nC = nr * nc
        p = 2 * self.a
        # cell centres, row-major (x = row index, y = column index) as in utilities.computeCellularCoordinates
        self.xc = np.repeat(np.arange(nr), nc) * p + self.a
        self.yc = np.tile(np.arange(nc), nr) * p + self.a
        # field points: cell corners (resolution = 1), x-major as in computeExtracellularCoordinates
        if periodic:
            nfr, nfc = nr, nc
        else:
            nfr, nfc = nr + 1, nc + 1
        self.fdims = (nfr, nfc)
        self.xf = np.repeat(np.arange(nfr), nfc) * p
        self.yf = np.tile(np.arange(nfc), nfr) * p
        dx = self.xf[:, None] - self.xc[None, :]
        dy = self.yf[:, None] - self.yc[None, :]
        if periodic:
            Lx, Ly = nr * p, nc * p
            dx = (dx + Lx / 2) % Lx - Lx / 2
            dy = (dy + Ly / 2) % Ly - Ly / 2
        self.dx, self.dy = dx, dy
        dist = np.sqrt(dx ** 2 + dy ** 2)
        self.dist = dist
        thr_out = self.a * np.sqrt(2) * (self.R_s + .001)
        screen = (dist <= thr_out) * 1.0
        d_scr = np.where(screen > 0, dist, np.inf)
        self.rinv = 1.0 / d_scr
        thr_in = self.a * np.sqrt(2) * ((self.R_s if fieldRangeSymmetric else 1) + .001)
        if periodic:
            # corner (i,j) of the periodic field grid sits at (i p, j p): perceived by cells whose centre is within a*sqrt2
            pass
        self.screenIn = (dist <= thr_in) * 1.0
        self.numFieldNeighbors = self.screenIn.sum(0)[0]
        self.fc = self.fs * (self.k_e / self.eps_r)
        # gap-junction adjacency
        A = np.zeros((self.nC, self.nC))
        idx = np.arange(self.nC).reshape(nr, nc)
        for (di, dj) in ((1, 0), (0, 1)):
            if periodic:
                a_ = idx
                b_ = np.roll(np.roll(idx, -di, axis=0), -dj, axis=1)
            else:
                a_ = idx[:nr - di, :nc - dj]
                b_ = idx[di:, dj:]
            A[a_.ravel(), b_.ravel()] = 1
            A[b_.ravel(), a_.ravel()] = 1
        self.A = A
        self.iu, self.ju = np.where(np.triu(A) != 0)

    # ---- pieces -------------------------------------------------------------------------
    def field(self, V):
        """|E| (vector) or scalar potential at the field points, from cell charges Q=C*V."""
        Q = self.C * V
        if self.vector:
            r2 = self.rinv ** 2
            ex = self.fc * (r2 * self.dx) @ Q
            ey = self.fc * (r2 * self.dy) @ Q
            return np.sqrt(ex ** 2 + ey ** 2 + self.epsilon)
        return self.fc * self.rinv @ Q

    def perceive(self, eV):
        s = self.screenIn.T @ eV                 # (nC,)
        return s / self.numFieldNeighbors        # 'average' aggregation

    def update_G(self, G, ebar):
        x = self.gain * ebar + self.bias
        drive = (2.0 / (1.0 + np.exp(-x)) - 1.0) * self.weight
        dp = 10.0 * (-self.leak_g * (G / self.G_ref) + drive) / self.tau
        G = G + self.dt * dp * self.G_ref
        return np.clip(G, 0.0, 2.0 * self.G_ref)

    def current(self, V, G):
        Vm = V
        Pin = 1.0 / (1.0 + np.exp(self.Z * (Vm - self.V_th) / self.V_T))
        Pout = 1.0 / (1.0 + np.exp(-self.Z * (Vm - self.V_th) / self.V_T))
        I = -G * (Vm - self.E_pol) * Pin - self.G_dep * (Vm - self.E_dep) * Pout
        Vi, Vj = V[self.iu], V[self.ju]
        gij = 2 * self.G_0 / (1 + np.cosh((Vi - Vj) / self.V_0))
        flux = gij * (Vj - Vi)                   # current into i from j
        gj = np.zeros_like(V)
        np.add.at(gj, self.iu, flux)
        np.add.at(gj, self.ju, -flux)
        return I + gj

    def step(self, V, G, return_eV=False):
        eV = self.field(V)
        G = self.update_G(G, self.perceive(eV))
        V = V + self.dt * self.current(V, G) / self.C
        return (V, G, eV) if return_eV else (V, G)

    def initial(self, V0=-9.2e-3, G0=1.0):
        return np.full(self.nC, V0), np.full(self.nC, G0 * self.G_ref)

    def run(self, nsteps, V=None, G=None, record_every=0):
        if V is None:
            V, G = self.initial()
        traj = []
        for it in range(nsteps):
            if record_every and it % record_every == 0:
                traj.append(V.copy())
            V, G = self.step(V, G)
        return V, G, (np.array(traj) if record_every else None)
