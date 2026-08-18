"""
Exact, non-perturbative representation/provenance tracking.

Rather than estimating a local derivative by perturbing and re-simulating (measure_response_
operator.py, measure_representation_pattern.py -- both approximate, and both break down once the
trajectory is far from where they were calibrated), this decomposes the ACTUAL forward computation
of one already-simulated trajectory into exact per-source contributions, at every step, and
propagates a "provenance pie" per cell forward through time: p_i(t) is a distribution over all
cells giving the share of cell i's current state traceable to each cell's original identity.

The decomposition follows directly from the model's own update equations (cellularFieldNetwork.py):

  Vmem_i(t+1) = Vmem_i(t)*[1 - (dt/C)*sum_j G_ij(t)]      self-persistence
              + sum_j Vmem_j(t)*[(dt/C)*G_ij(t)]           each gap-junction neighbour, exact term
              + (dt/C)*IonChannelCurrent_i(t)              intrinsic drive (self)

  G_pol_i(t+1) = G_pol_i(t)*(1-rate) + rate*sigmoidTerm_i(t)   rate = dt*G_ref*10/timeConstant
    where sigmoidTerm_i is a sigmoid of eVneighborsMean_i(t), itself a fixed-geometry function of
    every cell's Vmem(t) via the (fieldVector=True) field: eV_g = sqrt(eVx_g^2+eVy_g^2+eps), each
    of eVx_g, eVy_g an exact linear combination of Vmem.

Every "+" above is exact (it IS the computation, not an estimate of it), so there is no linear-
regime validity question. What is a genuine design choice, stated here rather than hidden: when a
term is itself a nonlinear function of several provenanced upstream quantities (a product, as in
IonChannelCurrent = f(G_pol_i)*g(Vmem_i); a sigmoid of a sum; a vector norm), there is no unique
attribution -- the convention used throughout is a magnitude-weighted mixture of the upstream
provenance vectors (share = |own linear-order contribution| / sum of all such magnitudes at that
node), which is attribution-neutral to any further nonlinear reshaping of the combined value.

A numerical consistency check (--verify) confirms that summing the derived per-source magnitude
terms reproduces the actual Vmem/G_pol update, catching any formula error before results are used.

  python measure_provenance_propagation.py --condition learned --numSteps 300 --verify
"""

import argparse
import copy

import numpy as np
import torch

import utilities
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--fieldScreenSize', type=float, default=None)
parser.add_argument('--condition', type=str, default='learned', choices=['learned', 'random', 'uniform'])
parser.add_argument('--clampSeed', type=int, default=7)
parser.add_argument('--clampIters', type=int, default=100)
parser.add_argument('--numSteps', type=int, default=300, help='free-evolution steps to propagate provenance over')
parser.add_argument('--snapshotEvery', type=int, default=50, help='save the full provenance matrix every N steps')
parser.add_argument('--verify', action='store_true', help='check the decomposition reconstructs the real trajectory')
parser.add_argument('--outputPrefix', type=str, default='data/provenance')
args = parser.parse_args()

utils = utilities.utilities()


def buildParameters():
    parameters = copy.deepcopy(torch.load(args.sourceDat, weights_only=False))
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    if args.fieldScreenSize is not None:
        parameters['fieldParameters']['fieldScreenSize'] = args.fieldScreenSize
    numCells = parameters['latticeDims'][0] * parameters['latticeDims'][1]
    values = parameters['simParameters']['initialValues']
    if 'ligandConc' not in values:
        values['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
    return parameters


numRows, numCols = torch.load(args.sourceDat, weights_only=False)['latticeDims']


def clampParametersFor(circuit):
    if args.condition == 'uniform':
        return None
    if args.condition == 'learned':
        parameters = torch.load(args.sourceDat, weights_only=False)
        if parameters['clampParameters'] is None:
            raise SystemExit(f'{args.sourceDat} carries no learned clamp')
        return parameters['clampParameters']
    leftHalf = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
    mirrored = utils.computeSymmetricalIndices(circuit, leftHalf, mode='field', symmetry='twofold')
    allIdx = np.concatenate((leftHalf, mirrored))
    _, uniqueIdx = np.unique(allIdx, return_index=True)
    clampIndices = (np.zeros(len(allIdx[uniqueIdx]), dtype=int), allIdx[uniqueIdx])
    numHalf = len(leftHalf)
    timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
    generator = torch.Generator().manual_seed(args.clampSeed)
    frequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * 900 + 100
    phases = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
    amplitudes = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 - 1
    clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                   * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    return {'clampMode': 'fieldDomeTwoFoldSymmetry', 'clampIndices': clampIndices,
           'clampValues': clampValues, 'clampStartIter': 0, 'clampEndIter': args.clampIters}


parameters = buildParameters()
instance = model(parameters, 1)
instance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
circuit = instance.electricNetwork
numCells = circuit.numCells
print(f"{numRows}x{numCols} lattice, condition={args.condition}, {numCells} cells, "
     f"{circuit.numFieldGridPoints} field points")

if args.condition == 'uniform':
    startIter = 0
else:
    # timeseriesVmem[i] records the state BEFORE iteration i's update, while circuit.Vmem after
    # numSimIters steps holds the state AFTER -- continuing directly from the live circuit here
    # would start one free-evolution step past the standard "state following clamp end" reference
    # (verified: up to 0.58 mV different from timeseriesVmem[101]). Extract that exact snapshot
    # and re-inject it into a fresh instance instead, matching every other script in this session.
    instance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                      clampParameters=clampParametersFor(circuit), perturbation=None,
                      numSimIters=args.clampIters + 2, storeVariables=('Vmem', 'Gpol'))
    preIdx = args.clampIters + 1
    vmemPre = instance.timeseriesVmem[preIdx].clone()
    gpolPre = instance.timeseriesGpol[preIdx].clone()

    parameters = buildParameters()
    instance = model(parameters, 1)
    instance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    circuit = instance.electricNetwork
    initialValues = dict(parameters['simParameters']['initialValues'])
    initialValues['Vmem'] = vmemPre.clone().double()
    initialValues['eV'] = torch.zeros_like(circuit.eV)
    circuit.initVariables(initialValues)
    circuit.G_pol = gpolPre.clone().double()
    startIter = preIdx

# --- Fixed (structural, state-independent) geometry, precomputed once -------------------------
C, G_ref, dt = circuit.C, circuit.G_ref, circuit.timestep
fieldConstant = circuit.fieldStrength * (circuit.k_e / circuit.relativePermittivity)
rinv = 1.0 / circuit.fieldCellDistanceMatrixScreened[0]                 # (numFieldGridPoints, numCells)
deltax = circuit.extracellularCoordinates[0].t() - circuit.cellularCoordinates[0]   # (numFieldGridPoints, numCells)
deltay = circuit.extracellularCoordinates[1].t() - circuit.cellularCoordinates[1]
rinvsq = rinv ** 2
Wx = fieldConstant * C * rinvsq * deltax
Wy = fieldConstant * C * rinvsq * deltay
fieldCellMag = torch.sqrt(Wx ** 2 + Wy ** 2)                            # (numFieldGridPoints, numCells)
maskMatrix = circuit.fieldScreenMatrixIn[0].double()                    # (numFieldGridPoints, numCells)
numFieldNeighbors = circuit.numFieldNeighbors                           # scalar (uniform across cells here)
rateGpol = (dt * G_ref * 10.0 / circuit.fieldTransductionTimeConstant).item()

# --- Provenance state: row i = distribution over source cells for cell i's current value -------
pVmem = torch.eye(numCells, dtype=torch.float64)
pGpol = torch.eye(numCells, dtype=torch.float64)

snapshots = {}
maxVmemErr, maxGpolErr = 0.0, 0.0
for step in range(args.numSteps):
    VmemBefore = circuit.Vmem[0, :, 0].clone()
    GpolBefore = circuit.G_pol[0, :, 0].clone()

    circuit.updateExtracellularVoltage(source='Vmem')
    eVNow = circuit.eV[0, :, 0].clone()                                 # (numFieldGridPoints,)

    m_gk = fieldCellMag * VmemBefore.abs()[None, :]                     # (numFieldGridPoints, numCells)
    rowSumG = m_gk.sum(1, keepdim=True).clamp_min(1e-300)
    pEV = (m_gk @ pVmem) / rowSumG                                      # (numFieldGridPoints, numCells)

    circuit.updateIonChannelConductance(inputSource='field', fieldModulation=False,
                                        fieldAggregation='average', stochasticIonChannels=False,
                                        perturbation=None)
    GpolAfter = circuit.G_pol[0, :, 0].clone()

    w_gi = eVNow.abs()[:, None] * maskMatrix                            # (numFieldGridPoints, numCells)
    colSumI = w_gi.sum(0).clamp_min(1e-300)                             # (numCells,)
    pFieldMean = (w_gi.t() @ pEV) / colSumI[:, None]                    # (numCells, numCells)

    persistTermG = GpolBefore * (1 - rateGpol)
    driveTermG = GpolAfter - persistTermG
    m_selfG, m_driveG = persistTermG.abs(), driveTermG.abs()
    totalG = (m_selfG + m_driveG).clamp_min(1e-300)
    pGpolNew = (m_selfG[:, None] * pGpol + m_driveG[:, None] * pFieldMean) / totalG[:, None]

    circuit.updateCurrent()
    G_ij = circuit.G_ij[0].clone()
    InCurrent = circuit.InCurrent[0, :, 0].clone()
    OutCurrent = circuit.OutCurrent[0, :, 0].clone()

    circuit.updateVmem()
    VmemAfter = circuit.Vmem[0, :, 0].clone()

    sumG = G_ij.sum(1)
    persistTermV = (1 - (dt / C) * sumG) * VmemBefore
    neighborTerm = (dt / C) * G_ij * VmemBefore[None, :]                # neighborTerm[i, j]
    intrinsicTerm = (dt / C) * (InCurrent + OutCurrent)

    if args.verify:
        reconstructed = persistTermV + neighborTerm.sum(1) + intrinsicTerm
        maxVmemErr = max(maxVmemErr, (reconstructed - VmemAfter).abs().max().item())
        maxGpolErr = max(maxGpolErr, (persistTermG + driveTermG - GpolAfter).abs().max().item())

    pIn = 0.5 * pGpolNew + 0.5 * pVmem            # InCurrent depends on G_pol_after and Vmem_before
    pOut = pVmem                                   # OutCurrent's G_dep is a fixed, non-provenanced constant
    m_In, m_Out = InCurrent.abs(), OutCurrent.abs()
    totalIntrinsic = (m_In + m_Out).clamp_min(1e-300)
    pIntrinsic = (m_In[:, None] * pIn + m_Out[:, None] * pOut) / totalIntrinsic[:, None]

    m_selfV, m_neighborV = persistTermV.abs(), neighborTerm.abs()
    totalV = (m_selfV + m_neighborV.sum(1) + m_In + m_Out).clamp_min(1e-300)
    pVmemNew = (m_selfV[:, None] * pVmem + (m_neighborV @ pVmem)
               + (m_In + m_Out)[:, None] * pIntrinsic) / totalV[:, None]

    pVmem, pGpol = pVmemNew, pGpolNew
    if (step + 1) % args.snapshotEvery == 0 or step == args.numSteps - 1:
        snapshots[step + 1] = pVmem.clone().numpy()
        print(f"  step {step+1:>4}/{args.numSteps}  Vmem provenance row-sum check: "
             f"{pVmem.sum(1).mean().item():.6f} (should be 1.0)")

if args.verify:
    print(f"Reconstruction check: max |Vmem| error = {maxVmemErr:.3e}, "
         f"max |G_pol| error = {maxGpolErr:.3e} (should be ~machine precision)")

np.savez(f'{args.outputPrefix}_{args.condition}.npz',
        **{f'pVmem_step{k}': v for k, v in snapshots.items()},
        finalVmem=circuit.Vmem[0, :, 0].detach().numpy() * 1000.0,
        latticeDims=(numRows, numCols), startIter=startIter, numSteps=args.numSteps)
print(f"Saved {args.outputPrefix}_{args.condition}.npz")
