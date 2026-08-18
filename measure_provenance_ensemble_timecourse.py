"""
Ensemble-averaged version of the unfolding-over-time provenance analysis: instead of comparing
learned against a handful of individual random seeds (whose per-seed idiosyncrasies can look like
"the same as learned" or "different from learned" depending on which seed you happen to look at),
this runs the exact provenance propagation for many random seeds and averages the facial-feature
block-pooled provenance row at each of the same checkpoint steps used for the single-seed timecourse
(every 10 steps, 10..890, plus the final step). This makes the unfolding analysis consistent with
the "learned vs 100-seed ensemble average" standard already used for the static (step-899) block
comparison in measure_provenance_ensemble.py / visualize_provenance_blocks_top5.py.

Only the three facial-feature blocks' pooled provenance rows are tracked at each checkpoint (not
the full 121x121 matrix at every checkpoint for every seed), since that's all the downstream
unfolding/top-5-filmstrip figures need, and it keeps the running-sum memory footprint trivial
(3 blocks x 90 checkpoints x 121 floats) regardless of ensemble size.

  python measure_provenance_ensemble_timecourse.py --numSeeds 100 --numSteps 899 --snapshotEvery 10
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
parser.add_argument('--seedStart', type=int, default=1)
parser.add_argument('--numSeeds', type=int, default=100)
parser.add_argument('--clampIters', type=int, default=100)
parser.add_argument('--numSteps', type=int, default=899)
parser.add_argument('--snapshotEvery', type=int, default=10)
parser.add_argument('--outputPrefix', type=str, default='data/provenanceTimecourse')
args = parser.parse_args()

utils = utilities.utilities()

blocks = {
    'eye':   np.array([24, 25, 35, 36, 29, 30, 40, 41]),
    'nose':  np.array([49, 60, 71]),
    'mouth': np.array([92, 93, 94]),
}


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
refInstance = model(buildParameters(), 1)
refCircuit = refInstance.electricNetwork
numCells = refCircuit.numCells
snapshotSteps = sorted(set(list(range(args.snapshotEvery, args.numSteps, args.snapshotEvery))
                          + [args.numSteps]))
print(f"{numRows}x{numCols} lattice, {numCells} cells, {args.numSeeds} seeds x {args.numSteps} steps, "
     f"{len(snapshotSteps)} checkpoints/seed")

# --- Fixed (structural, state-independent) geometry, built once and reused across all seeds ----
C, G_ref, dt = refCircuit.C, refCircuit.G_ref, refCircuit.timestep
fieldConstant = refCircuit.fieldStrength * (refCircuit.k_e / refCircuit.relativePermittivity)
rinv = 1.0 / refCircuit.fieldCellDistanceMatrixScreened[0]
deltax = refCircuit.extracellularCoordinates[0].t() - refCircuit.cellularCoordinates[0]
deltay = refCircuit.extracellularCoordinates[1].t() - refCircuit.cellularCoordinates[1]
rinvsq = rinv ** 2
Wx = fieldConstant * C * rinvsq * deltax
Wy = fieldConstant * C * rinvsq * deltay
fieldCellMag = torch.sqrt(Wx ** 2 + Wy ** 2)
maskMatrix = refCircuit.fieldScreenMatrixIn[0].double()
rateGpol = (dt * G_ref * 10.0 / refCircuit.fieldTransductionTimeConstant).item()
leftHalf = utils.computeDomeIndices(refCircuit, mode='field', region='leftHalf')
mirrored = utils.computeSymmetricalIndices(refCircuit, leftHalf, mode='field', symmetry='twofold')
allIdx = np.concatenate((leftHalf, mirrored))
_, uniqueIdx = np.unique(allIdx, return_index=True)
clampIndices = (np.zeros(len(allIdx[uniqueIdx]), dtype=int), allIdx[uniqueIdx])
numHalf = len(leftHalf)
timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
del refInstance, refCircuit


def clampParametersFor(seed):
    generator = torch.Generator().manual_seed(seed)
    frequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * 900 + 100
    phases = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
    amplitudes = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 - 1
    clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                   * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    return {'clampMode': 'fieldDomeTwoFoldSymmetry', 'clampIndices': clampIndices,
           'clampValues': clampValues, 'clampStartIter': 0, 'clampEndIter': args.clampIters}


# Running sums of block-pooled provenance rows, keyed by (block name, checkpoint step) -> 121-vec
sumRows = {(name, s): torch.zeros(numCells, dtype=torch.float64)
          for name in blocks for s in snapshotSteps}
worstVmemErr, worstGpolErr = 0.0, 0.0
seeds = range(args.seedStart, args.seedStart + args.numSeeds)

for seedIdx, seed in enumerate(seeds):
    parameters = buildParameters()
    prepInstance = model(parameters, 1)
    prepInstance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    prepInstance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                          clampParameters=clampParametersFor(seed), perturbation=None,
                          numSimIters=args.clampIters + 2, storeVariables=('Vmem', 'Gpol'))
    preIdx = args.clampIters + 1
    vmemPre = prepInstance.timeseriesVmem[preIdx].clone()
    gpolPre = prepInstance.timeseriesGpol[preIdx].clone()
    del prepInstance

    parameters = buildParameters()
    instance = model(parameters, 1)
    instance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    circuit = instance.electricNetwork
    initialValues = dict(parameters['simParameters']['initialValues'])
    initialValues['Vmem'] = vmemPre.clone().double()
    initialValues['eV'] = torch.zeros_like(circuit.eV)
    circuit.initVariables(initialValues)
    circuit.G_pol = gpolPre.clone().double()

    pVmem = torch.eye(numCells, dtype=torch.float64)
    pGpol = torch.eye(numCells, dtype=torch.float64)
    maxVmemErr, maxGpolErr = 0.0, 0.0
    snapshotSet = set(snapshotSteps)

    for step in range(args.numSteps):
        VmemBefore = circuit.Vmem[0, :, 0].clone()
        GpolBefore = circuit.G_pol[0, :, 0].clone()

        circuit.updateExtracellularVoltage(source='Vmem')
        eVNow = circuit.eV[0, :, 0].clone()

        m_gk = fieldCellMag * VmemBefore.abs()[None, :]
        rowSumG = m_gk.sum(1, keepdim=True).clamp_min(1e-300)
        pEV = (m_gk @ pVmem) / rowSumG

        circuit.updateIonChannelConductance(inputSource='field', fieldModulation=False,
                                            fieldAggregation='average', stochasticIonChannels=False,
                                            perturbation=None)
        GpolAfter = circuit.G_pol[0, :, 0].clone()

        w_gi = eVNow.abs()[:, None] * maskMatrix
        colSumI = w_gi.sum(0).clamp_min(1e-300)
        pFieldMean = (w_gi.t() @ pEV) / colSumI[:, None]

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
        neighborTerm = (dt / C) * G_ij * VmemBefore[None, :]
        intrinsicTerm = (dt / C) * (InCurrent + OutCurrent)

        reconstructed = persistTermV + neighborTerm.sum(1) + intrinsicTerm
        maxVmemErr = max(maxVmemErr, (reconstructed - VmemAfter).abs().max().item())
        maxGpolErr = max(maxGpolErr, (persistTermG + driveTermG - GpolAfter).abs().max().item())

        pIn = 0.5 * pGpolNew + 0.5 * pVmem
        pOut = pVmem
        m_In, m_Out = InCurrent.abs(), OutCurrent.abs()
        totalIntrinsic = (m_In + m_Out).clamp_min(1e-300)
        pIntrinsic = (m_In[:, None] * pIn + m_Out[:, None] * pOut) / totalIntrinsic[:, None]

        m_selfV, m_neighborV = persistTermV.abs(), neighborTerm.abs()
        totalV = (m_selfV + m_neighborV.sum(1) + m_In + m_Out).clamp_min(1e-300)
        pVmemNew = (m_selfV[:, None] * pVmem + (m_neighborV @ pVmem)
                   + (m_In + m_Out)[:, None] * pIntrinsic) / totalV[:, None]

        pVmem, pGpol = pVmemNew, pGpolNew

        if (step + 1) in snapshotSet:
            for name, idxs in blocks.items():
                sumRows[(name, step + 1)] += pVmem[idxs].mean(dim=0)

    worstVmemErr = max(worstVmemErr, maxVmemErr)
    worstGpolErr = max(worstGpolErr, maxGpolErr)
    del instance
    if (seedIdx + 1) % 10 == 0 or seedIdx == 0:
        print(f"  seed {seed:>4} ({seedIdx+1}/{args.numSeeds}) done, "
             f"worst reconstruction error so far: Vmem {worstVmemErr:.2e}, G_pol {worstGpolErr:.2e}")

print(f"\nAll {args.numSeeds} seeds done. Worst reconstruction error across the whole ensemble: "
     f"Vmem {worstVmemErr:.3e}, G_pol {worstGpolErr:.3e} (should be ~machine precision)")

saveDict = {'blocks': blocks, 'latticeDims': (numRows, numCols), 'numSeeds': args.numSeeds,
           'snapshotSteps': np.array(snapshotSteps)}
for (name, s), total in sumRows.items():
    saveDict[f'{name}_pVmem_step{s}'] = (total / args.numSeeds).numpy()

outPath = f'{args.outputPrefix}_randomEnsemble{args.numSeeds}.npz'
np.savez(outPath, **saveDict)
print(f"Saved {outPath}")
