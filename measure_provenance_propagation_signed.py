"""
Signed-value extension of measure_provenance_propagation.py: an EXACT decomposition of Vmem_i(t) in
mV, not as a fraction (row summing to 1, always non-negative), but as a signed value (row summing
exactly to Vmem_i(t) itself, entries free to be positive or negative). This is a genuinely different
quantity from the existing share pVmem, not a rescaling of it -- see the docstring discussion below
for why, and Appendix 17 for why it was attempted.

Why this needs new machinery, not just multiplying the finished share matrix by Vmem_i(t): the share
method uses |neighborTerm| as its per-neighbour weight, discarding sign, so two neighbours pulling
Vmem_i in OPPOSITE directions get credited as if they were both "helping" -- their true, partially
cancelling net effect is invisible. A signed decomposition should let opposing contributions actually
cancel in the numbers, which requires carrying real (not absolute) values through the recursion, not
just relabelling the final share matrix with the final value.

Where this is well-posed without any new convention (exact linear algebra, no attribution ambiguity):

  vVmem_i(t+1) += [1 - (dt/C)*sum_j G_ij(t)] * vVmem_i(t)[S]           self-persistence
  vVmem_i(t+1) += sum_j (dt/C)*G_ij(t) * vVmem_j(t)[S]                 each gap-junction neighbour

Both are literally linear rescalings of an already-exact previous-step decomposition by a real,
signed, already-known coefficient -- no normalisation, no magnitude weighting, no convention needed.

Where a convention is still unavoidable: the intrinsic current (a product of provenanced G_pol and a
provenanced voltage-dependent factor) and G_pol's own drive term (downstream of a sigmoid of a vector
norm of the field -- a nonlinearity with no unique inverse). Those two cases reuse the EXISTING,
already-validated fractional shares (pIntrinsic, pFieldMean) from measure_provenance_propagation.py
unchanged, applied as a multiplier on the REAL SIGNED scalar term rather than its absolute value:

  vVmem_i(t+1) += intrinsicTerm_i(t) * pIntrinsic_i(t)[S]     (intrinsicTerm_i is signed; pIntrinsic sums to 1)
  vGpol_i(t+1) += driveTermG_i(t) * pFieldMean_i(t)[S]        (driveTermG_i is signed; pFieldMean sums to 1)

This keeps every part of the decomposition that has a real, unambiguous linear-algebra answer fully
exact and signed, while the two genuinely nonlinear/ambiguous parts keep using the same magnitude-
weighted convention as the original method (no new ambiguity invented, none avoided either).

Both the row-sum reconstruction (sum_S vVmem_i[S] should equal Vmem_i(t) exactly) and the magnitude
of the individual entries (which could in principle blow up via cancellation even while the sum stays
bounded) are monitored at every snapshot with --verify.

  python measure_provenance_propagation_signed.py --condition learned --numSteps 300 --verify
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
parser.add_argument('--numSteps', type=int, default=300)
parser.add_argument('--snapshotEvery', type=int, default=50)
parser.add_argument('--verify', action='store_true')
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
print(f"{numRows}x{numCols} lattice, condition={args.condition}, {numCells} cells")

if args.condition == 'uniform':
    startIter = 0
else:
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

C, G_ref, dt = circuit.C, circuit.G_ref, circuit.timestep
fieldConstant = circuit.fieldStrength * (circuit.k_e / circuit.relativePermittivity)
rinv = 1.0 / circuit.fieldCellDistanceMatrixScreened[0]
deltax = circuit.extracellularCoordinates[0].t() - circuit.cellularCoordinates[0]
deltay = circuit.extracellularCoordinates[1].t() - circuit.cellularCoordinates[1]
rinvsq = rinv ** 2
Wx = fieldConstant * C * rinvsq * deltax
Wy = fieldConstant * C * rinvsq * deltay
fieldCellMag = torch.sqrt(Wx ** 2 + Wy ** 2)
maskMatrix = circuit.fieldScreenMatrixIn[0].double()
rateGpol = (dt * G_ref * 10.0 / circuit.fieldTransductionTimeConstant).item()

# --- fractional shares (existing, unchanged method) AND signed values (new), tracked in parallel --
pVmem = torch.eye(numCells, dtype=torch.float64)
pGpol = torch.eye(numCells, dtype=torch.float64)
vVmem = torch.diag(circuit.Vmem[0, :, 0].clone())          # row i sums exactly to Vmem_i(t)
vGpol = torch.diag(circuit.G_pol[0, :, 0].clone())          # row i sums exactly to G_pol_i(t)

snapshots = {}
maxVmemErr, maxGpolErr = 0.0, 0.0
maxValueReconErr, maxAbsEntry = 0.0, 0.0
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

    if args.verify:
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

    # --- signed value recursion: exact for persistence/neighbour, convention-reused for intrinsic ---
    vVmemNew = ((1 - (dt / C) * sumG)[:, None] * vVmem
               + (dt / C) * (G_ij @ vVmem)
               + intrinsicTerm[:, None] * pIntrinsic)

    if args.verify:
        valueRecon = vVmemNew.sum(1)
        maxValueReconErr = max(maxValueReconErr, (valueRecon - VmemAfter).abs().max().item())
        maxAbsEntry = max(maxAbsEntry, vVmemNew.abs().max().item())

    pVmem, pGpol = pVmemNew, pGpolNew
    vVmem, vGpol = vVmemNew, vGpolNew
    if (step + 1) % args.snapshotEvery == 0 or step == args.numSteps - 1:
        snapshots[step + 1] = (vVmem.clone() * 1000.0).numpy()   # store in mV for downstream scripts
        print(f"  step {step+1:>4}/{args.numSteps}  value row-sum recon err so far: "
             f"{maxValueReconErr*1000.0:.3e} mV, max |entry| so far: {maxAbsEntry*1000.0:.3e} mV "
             f"(Vmem itself ranges roughly [-55,-5] mV)")

if args.verify:
    print(f"\nShare-method reconstruction check (unchanged): max |Vmem| error = {maxVmemErr*1000.0:.3e} mV, "
         f"max |G_pol| error = {maxGpolErr:.3e}")
    print(f"Signed-value reconstruction check (new): max |sum_S vVmem[i,S] - Vmem_i| error = "
         f"{maxValueReconErr*1000.0:.3e} mV (should be ~machine precision)")
    print(f"Largest individual |value contribution| seen at any step: {maxAbsEntry*1000.0:.3e} mV "
         f"(compare to the real trajectory's own range, roughly -55 to -5 mV -- if this is orders of "
         f"magnitude larger, individual entries are cancelling rather than staying interpretable)")

np.savez(f'{args.outputPrefix}_{args.condition}.npz',
        **{f'vVmem_step{k}': v for k, v in snapshots.items()},
        finalVmem=circuit.Vmem[0, :, 0].detach().numpy() * 1000.0,
        latticeDims=(numRows, numCols), startIter=startIter, numSteps=args.numSteps)
print(f"Saved {args.outputPrefix}_{args.condition}.npz")
