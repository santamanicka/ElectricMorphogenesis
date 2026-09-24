"""One iteration of the trained 11x11 model as a pure, differentiable function of the state (Vmem, G_pol).

The model's own step (cellularFieldNetwork.simulate, then embryo's clamp) mutates tensors in place and assumes
gradients are off, so it cannot be differentiated. This re-implements exactly the same arithmetic, in the same
order, as a pure function, and is checked against the model to machine precision before anything relies on it:

    field    E = L @ Vmem (linear in Vmem, L built from the model itself), eV = sqrt(|E|^2 + epsilon)
    drive    G_pol += dt * G_ref * 10/tau * (-G_pol + (2*sigmoid(gain*<eV>_screen + bias) - 1) * weight), clipped
    voltage  Vmem  += dt/C * (I_ion(Vmem, G_pol_new) + I_gap(Vmem))
    clamp    during the hold, ring G_pol := code * G_ref, then one more Vmem update with the clamped G_pol

The start-of-step Vmem enters in three roles -- the field it generates, the gap-junction current, and the cell's
own membrane current -- and each role takes its own argument. At a real state all three are the same Vmem, so
dF/dVmem is exactly the sum of the three role Jacobians; off the diagonal, the field role is the field channel
and the gap-junction role is the contact channel.
"""
import numpy as np
import torch

import boundaryCodeUtilities as boundary
from embryo import model


class Step:
    def __init__(self, ringCode=None):
        torch.set_grad_enabled(False)
        reference = boundary.loadCheckpoint(1888)
        parameters = dict(reference)
        parameters['latticePeriodicBoundaryGJ'] = False
        parameters['ATPParameters'] = None
        initial = reference['simParameters']['initialValues']
        batchInitial = {name: initial[name] for name in ('Vmem', 'eV', 'ligandConc')}
        batchInitial['G_pol'] = dict(cells=[initial['G_pol']['cells'][0]], values=[initial['G_pol']['values'][0]])
        batchInitial['G_dep'] = initial['G_dep']
        system = model(parameters, 1)
        system.setExperimentalConditions((batchInitial, 1))
        circuit = system.electricNetwork
        self.reference, self.system, self.circuit = reference, system, circuit
        # the starting state, captured before _fieldOperator probes the model by overwriting its Vmem
        self.initialVmem = circuit.Vmem[0, :, 0].clone()
        self.initialGpol = circuit.G_pol[0, :, 0].clone()

        self.C = float(circuit.C)
        self.dt = float(circuit.timestep)
        self.Gref = float(circuit.G_ref)
        self.Gdep = float(circuit.G_dep.flatten()[0])
        self.Epol, self.Edep = float(circuit.E_pol), float(circuit.E_dep)
        self.Vth, self.VT, self.Z = float(circuit.V_th), float(circuit.V_T), float(circuit.Z)
        self.G0, self.Gres, self.V0 = float(circuit.G_0), float(circuit.G_res), float(circuit.V_0)
        self.minG, self.maxG = float(circuit.min_Gpol), float(circuit.max_Gpol)
        self.epsilon = float(circuit.epsilon)
        self.gain = float(circuit.fieldTransductionGain)
        self.weight = float(circuit.fieldTransductionWeight)
        self.bias = float(circuit.fieldTransductionBias)
        self.tau = float(circuit.fieldTransductionTimeConstant)
        self.screen = circuit.fieldScreenMatrixIn[0].to(torch.double)            # (gridPoints, cells)
        self.numFieldNeighbours = float(circuit.numFieldNeighbors)
        self.adjacency = circuit.Adjacency.to(torch.double)                         # (cells, cells)
        self.numCells = self.adjacency.shape[0]
        self.L = self._fieldOperator()                                              # (2, gridPoints, cells)

        self.ring = torch.tensor(np.array(boundary.boundaryRingCells), dtype=torch.long)
        self.ringMask = torch.zeros(self.numCells, dtype=torch.double)
        self.ringMask[self.ring] = 1.0
        self.ringCode = None
        if ringCode is not None:
            self.ringCode = torch.zeros(self.numCells, dtype=torch.double)
            self.ringCode[self.ring] = torch.as_tensor(ringCode, dtype=torch.double) * self.Gref

    def _fieldOperator(self):
        """E = L @ Vmem exactly: the model's field vector is linear in Vmem, so read L off one cell at a time."""
        circuit = self.circuit
        columns = []
        for cell in range(self.numCells):
            unit = torch.zeros(1, self.numCells, 1, dtype=torch.double)
            unit[0, cell, 0] = 1.0
            circuit.Vmem = unit
            circuit.updateExtracellularVoltage(source='Vmem')
            columns.append(torch.stack([circuit.eVforceVector[0][0, :, 0], circuit.eVforceVector[1][0, :, 0]]).clone())
        circuit.Vmem = self.initialVmem.reshape(1, -1, 1).clone()           # put the model back as it was
        circuit.updateExtracellularVoltage(source='Vmem')
        return torch.stack(columns, dim=-1).to(torch.double)

    # ---------------------------------------------------------------- the pieces
    def fieldRead(self, vmem):
        E = torch.einsum('agc,...c->...ag', self.L, vmem)
        magnitude = torch.sqrt((E ** 2).sum(-2) + self.epsilon)
        return magnitude @ self.screen / self.numFieldNeighbours

    def ionCurrent(self, vmem, gpol):
        inward = 1.0 / (1.0 + torch.exp(self.Z * (vmem - self.Vth) / self.VT))
        outward = 1.0 / (1.0 + torch.exp(-self.Z * (vmem - self.Vth) / self.VT))
        return -gpol * (vmem - self.Epol) * inward - self.Gdep * (vmem - self.Edep) * outward

    def gapCurrent(self, vmem):
        difference = vmem.unsqueeze(-2) - vmem.unsqueeze(-1)                         # [i, j] = V_j - V_i
        conductance = (self.Gres + 2.0 * self.G0 / (1.0 + torch.cosh(difference / self.V0))) * self.adjacency
        return (conductance * difference).sum(-1)

    # ---------------------------------------------------------------- one iteration
    def roles(self, vField, vGap, vSelf, gpol, clamped):
        """The step with the start-of-step Vmem split into its three roles. Returns (Vmem', G_pol')."""
        drive = 10.0 * (-gpol + (2.0 * torch.sigmoid(self.gain * self.fieldRead(vField) + self.bias) - 1.0)
                        * self.weight) / self.tau
        gNew = torch.clamp(gpol + self.dt * drive * self.Gref, self.minG, self.maxG)
        vNew = vSelf + self.dt * (self.ionCurrent(vSelf, gNew) + self.gapCurrent(vGap)) / self.C
        if clamped:
            gNew = gNew * (1.0 - self.ringMask) + self.ringCode * self.ringMask
            vNew = vNew + self.dt * (self.ionCurrent(vNew, gNew) + self.gapCurrent(vNew)) / self.C
        return vNew, gNew

    def __call__(self, vmem, gpol, clamped):
        return self.roles(vmem, vmem, vmem, gpol, clamped)

    def jacobians(self, vmem, gpol, clamped):
        """Role-split Jacobian at one state: dict of (242 x 121) blocks for vField, vGap, vSelf and G."""
        def flat(vField, vGap, vSelf, g):
            vNew, gNew = self.roles(vField, vGap, vSelf, g, clamped)
            return torch.cat([vNew, gNew], dim=-1)
        with torch.enable_grad():
            blocks = torch.func.jacrev(flat, argnums=(0, 1, 2, 3))(vmem, vmem, vmem, gpol)
        return dict(field=blocks[0], gap=blocks[1], self_=blocks[2], g=blocks[3])

    # ---------------------------------------------------------------- the same Jacobian, by hand
    def _firstUpdate(self, vmem, gpol):
        """Role-split Jacobian of the unclamped update, batched over a leading axis. Also returns (Vmem', G_pol')."""
        dt, C, Gref = self.dt, self.C, self.Gref
        E = torch.einsum('agc,qc->qag', self.L, vmem)                                  # (q, 2, grid)
        magnitude = torch.sqrt((E ** 2).sum(1) + self.epsilon)                          # (q, grid)
        # d|E_g| / dV_c = (E_x L_x + E_y L_y) / |E|
        dMagnitude = torch.einsum('qag,agc->qgc', E / magnitude.unsqueeze(1), self.L)  # (q, grid, cells)
        fieldRead = magnitude @ self.screen / self.numFieldNeighbours                   # (q, cells)
        dFieldRead = torch.einsum('gi,qgc->qic', self.screen, dMagnitude) / self.numFieldNeighbours
        sigma = torch.sigmoid(self.gain * fieldRead + self.bias)
        drive = 10.0 * (-gpol + (2.0 * sigma - 1.0) * self.weight) / self.tau
        dDrive = 10.0 / self.tau * self.weight * 2.0 * sigma * (1.0 - sigma) * self.gain
        gPre = gpol + dt * drive * Gref
        inside = ((gPre >= self.minG) & (gPre <= self.maxG)).to(torch.double)
        gNew = torch.clamp(gPre, self.minG, self.maxG)
        gFromField = (inside * dt * Gref * dDrive).unsqueeze(-1) * dFieldRead            # (q, cells, cells)
        gFromSelf = inside * (1.0 - dt * Gref * 10.0 / self.tau)                        # (q, cells)

        ionG, ionV, current = self._ionTerms(vmem, gNew)
        gapJ, gapCurrent = self._gapTerms(vmem)
        vNew = vmem + dt * (current + gapCurrent) / C

        q, n = vmem.shape
        eye = torch.eye(n, dtype=torch.double)
        field = torch.zeros(q, 2 * n, n, dtype=torch.double)
        field[:, :n] = dt / C * ionG.unsqueeze(-1) * gFromField                          # V' through the new G
        field[:, n:] = gFromField
        gap = torch.zeros(q, 2 * n, n, dtype=torch.double)
        gap[:, :n] = dt / C * gapJ
        self_ = torch.zeros(q, 2 * n, n, dtype=torch.double)
        self_[:, :n] = eye * (1.0 + dt / C * ionV).unsqueeze(-1)
        g = torch.zeros(q, 2 * n, n, dtype=torch.double)
        g[:, :n] = eye * (dt / C * ionG * gFromSelf).unsqueeze(-1)
        g[:, n:] = eye * gFromSelf.unsqueeze(-1)
        return dict(field=field, gap=gap, self_=self_, g=g), vNew, gNew

    def _ionTerms(self, vmem, gpol):
        inward = 1.0 / (1.0 + torch.exp(self.Z * (vmem - self.Vth) / self.VT))
        outward = 1.0 / (1.0 + torch.exp(-self.Z * (vmem - self.Vth) / self.VT))
        dInward = -(self.Z / self.VT) * inward * (1.0 - inward)
        dOutward = (self.Z / self.VT) * outward * (1.0 - outward)
        current = -gpol * (vmem - self.Epol) * inward - self.Gdep * (vmem - self.Edep) * outward
        ionG = -(vmem - self.Epol) * inward
        ionV = (-gpol * (inward + (vmem - self.Epol) * dInward)
                - self.Gdep * (outward + (vmem - self.Edep) * dOutward))
        return ionG, ionV, current

    def _gapTerms(self, vmem):
        difference = vmem.unsqueeze(-2) - vmem.unsqueeze(-1)                                     # [i, j] = V_j - V_i
        scaled = difference / self.V0
        conductance = self.Gres + 2.0 * self.G0 / (1.0 + torch.cosh(scaled))
        dConductance = -2.0 * self.G0 * torch.sinh(scaled) / (self.V0 * (1.0 + torch.cosh(scaled)) ** 2)
        h = (conductance + difference * dConductance) * self.adjacency                             # d I_i / d V_j
        jacobian = h - torch.diag_embed(h.sum(-1))
        return jacobian, (conductance * difference * self.adjacency).sum(-1)

    def analyticJacobians(self, vmem, gpol, clamped):
        """Role-split Jacobians at a batch of states (q, cells). Same blocks as jacobians(), batched."""
        blocks, vNew, gNew = self._firstUpdate(vmem, gpol)
        if not clamped:
            return blocks
        # the clamp: ring G_pol := code, then one more Vmem update with the clamped G_pol, on every cell
        n = self.numCells
        free = 1.0 - self.ringMask
        gClamped = gNew * free + self.ringCode * self.ringMask
        ionG, ionV, _ = self._ionTerms(vNew, gClamped)
        gapJ, _ = self._gapTerms(vNew)
        eye = torch.eye(n, dtype=torch.double)
        A = torch.zeros(vmem.shape[0], 2 * n, 2 * n, dtype=torch.double)
        A[:, :n, :n] = eye * (1.0 + self.dt / self.C * ionV).unsqueeze(-1) + self.dt / self.C * gapJ
        A[:, :n, n:] = eye * (self.dt / self.C * ionG * free).unsqueeze(-1)
        A[:, n:, n:] = eye * free
        return {role: A @ block for role, block in blocks.items()}
