"""
Average the signed-value provenance decomposition (measure_provenance_propagation_signed.py) over
many random clamp seeds, exactly mirroring measure_provenance_ensemble.py's treatment of the
fractional-share method: each seed's vVmem row sums exactly to that seed's own Vmem_i, so the
seed-average of vVmem[i,:] sums exactly to the seed-average of Vmem_i -- the ensemble-mean pattern's
own signed-value decomposition, no renormalisation needed.

  python measure_provenance_ensemble_signed.py --numSeeds 100 --numSteps 899
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
parser.add_argument('--outputPrefix', type=str, default='data/provenanceSigned')
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
refInstance = model(buildParameters(), 1)
refCircuit = refInstance.electricNetwork
numCells = refCircuit.numCells
print(f"{numRows}x{numCols} lattice, {numCells} cells, {args.numSeeds} seeds x {args.numSteps} steps "
     "(signed-value method)")

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


sumV = torch.zeros(numCells, numCells, dtype=torch.float64)
sumFinalVmem = torch.zeros(numCells, dtype=torch.float64)
worstReconErr, worstMaxEntry = 0.0, 0.0
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
    vVmem = torch.diag(circuit.Vmem[0, :, 0].clone())
    vGpol = torch.diag(circuit.G_pol[0, :, 0].clone())
    maxReconErr, maxEntry = 0.0, 0.0

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
        vGpolNew = (1 - rateGpol) * vGpol + driveTermG[:, None] * pFieldMean

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

        pIn = 0.5 * pGpolNew + 0.5 * pVmem
        pOut = pVmem
        m_In, m_Out = InCurrent.abs(), OutCurrent.abs()
        totalIntrinsic = (m_In + m_Out).clamp_min(1e-300)
        pIntrinsic = (m_In[:, None] * pIn + m_Out[:, None] * pOut) / totalIntrinsic[:, None]

        m_selfV, m_neighborV = persistTermV.abs(), neighborTerm.abs()
        totalV = (m_selfV + m_neighborV.sum(1) + m_In + m_Out).clamp_min(1e-300)
        pVmemNew = (m_selfV[:, None] * pVmem + (m_neighborV @ pVmem)
                   + (m_In + m_Out)[:, None] * pIntrinsic) / totalV[:, None]

        vVmemNew = ((1 - (dt / C) * sumG)[:, None] * vVmem
                   + (dt / C) * (G_ij @ vVmem)
                   + intrinsicTerm[:, None] * pIntrinsic)

        valueRecon = vVmemNew.sum(1)
        maxReconErr = max(maxReconErr, (valueRecon - VmemAfter).abs().max().item())
        maxEntry = max(maxEntry, vVmemNew.abs().max().item())

        pVmem, pGpol = pVmemNew, pGpolNew
        vVmem, vGpol = vVmemNew, vGpolNew

    sumV += vVmem
    sumFinalVmem += circuit.Vmem[0, :, 0]
    worstReconErr = max(worstReconErr, maxReconErr)
    worstMaxEntry = max(worstMaxEntry, maxEntry)
    del instance
    if (seedIdx + 1) % 10 == 0 or seedIdx == 0:
        print(f"  seed {seed:>4} ({seedIdx+1}/{args.numSeeds}) done, worst value-recon error so far: "
             f"{worstReconErr*1000.0:.2e} mV, worst |entry| so far: {worstMaxEntry*1000.0:.2e} mV")

meanV = (sumV / args.numSeeds).numpy() * 1000.0
meanFinalVmem = (sumFinalVmem / args.numSeeds).numpy() * 1000.0
print(f"\nAll {args.numSeeds} seeds done. Worst value-recon error: {worstReconErr*1000.0:.3e} mV "
     f"(should be ~machine precision). Worst |entry|: {worstMaxEntry*1000.0:.2f} mV.")
print(f"Mean row-sum vs mean finalVmem, max diff: "
     f"{np.abs(meanV.sum(1) - meanFinalVmem).max():.3e} mV")

np.savez(f'{args.outputPrefix}_randomEnsemble{args.numSeeds}.npz',
        **{f'vVmem_step{args.numSteps}': meanV},
        finalVmem=meanFinalVmem, latticeDims=(numRows, numCols),
        startIter=args.clampIters + 1, numSteps=args.numSteps, numSeeds=args.numSeeds)
print(f"Saved {args.outputPrefix}_randomEnsemble{args.numSeeds}.npz")
