"""
Sanity checks for the exact provenance propagation method (measure_provenance_propagation.py),
independent of the reconstruction check already built into that script.

  1. Locality at step 1: mass should be exactly zero outside the true one-step reachable set
     (gap-junction neighbours union field-screening reach), not just "small far away."
  2. Non-negativity: every entry is a share of a pie chart; none should ever go negative.
  3. Row sums stay exactly 1 at every step, not just the last one.
  4. Monotonic spatial spreading: the number of cells with nonzero provenance for a fixed target
     should never shrink as steps increase (accumulation only adds sources).
  5. D4 symmetry: for the fully symmetric uniform condition, two mirror-image cells' entire
     provenance rows should be exact mirror images of each other, not just their aggregates.
  6. Cross-validation against the independent finite-difference method (measure_representation_
     pattern.py) at horizons short enough for both to be valid.

  python sanity_check_provenance.py
"""

import copy

import numpy as np
import torch

import utilities
from embryo import model

torch.set_grad_enabled(False)
utils = utilities.utilities()
SOURCE_DAT = 'data/StigmergicModelParameters.dat'
TOL = 1e-9


def buildParameters():
    parameters = copy.deepcopy(torch.load(SOURCE_DAT, weights_only=False))
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    numCells = parameters['latticeDims'][0] * parameters['latticeDims'][1]
    values = parameters['simParameters']['initialValues']
    if 'ligandConc' not in values:
        values['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
    return parameters


numRows, numCols = torch.load(SOURCE_DAT, weights_only=False)['latticeDims']
refInstance = model(buildParameters(), 1)
circuit = refInstance.electricNetwork
numCells = circuit.numCells

C, G_ref, dt = circuit.C, circuit.G_ref, circuit.timestep
fieldConstant = circuit.fieldStrength * (circuit.k_e / circuit.relativePermittivity)
rinv = 1.0 / circuit.fieldCellDistanceMatrixScreened[0]
deltax = circuit.extracellularCoordinates[0].t() - circuit.cellularCoordinates[0]
deltay = circuit.extracellularCoordinates[1].t() - circuit.cellularCoordinates[1]
rinvsq = rinv ** 2
Wx = fieldConstant * C * rinvsq * deltax
Wy = fieldConstant * C * rinvsq * deltay
fieldCellMag = torch.sqrt(Wx ** 2 + Wy ** 2)          # (numFieldGridPoints, numCells): cell k's write reach
maskMatrix = circuit.fieldScreenMatrixIn[0].double()   # (numFieldGridPoints, numCells): cell i's read reach
Adjacency = circuit.Adjacency.clone()
rateGpol = (dt * G_ref * 10.0 / circuit.fieldTransductionTimeConstant).item()


def propagate(vmemRef, gpolRef, numSteps, snapshotSteps):
    parameters = buildParameters()
    instance = model(parameters, 1)
    instance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    localCircuit = instance.electricNetwork
    initialValues = dict(parameters['simParameters']['initialValues'])
    initialValues['Vmem'] = vmemRef.clone().double()
    initialValues['eV'] = torch.zeros_like(localCircuit.eV)
    localCircuit.initVariables(initialValues)
    localCircuit.G_pol = gpolRef.clone().double()

    pVmem = torch.eye(numCells, dtype=torch.float64)
    pGpol = torch.eye(numCells, dtype=torch.float64)
    snapshots = {}
    for step in range(numSteps):
        VmemBefore = localCircuit.Vmem[0, :, 0].clone()
        GpolBefore = localCircuit.G_pol[0, :, 0].clone()
        localCircuit.updateExtracellularVoltage(source='Vmem')
        eVNow = localCircuit.eV[0, :, 0].clone()
        m_gk = fieldCellMag * VmemBefore.abs()[None, :]
        rowSumG = m_gk.sum(1, keepdim=True).clamp_min(1e-300)
        pEV = (m_gk @ pVmem) / rowSumG
        localCircuit.updateIonChannelConductance(inputSource='field', fieldModulation=False,
                                                  fieldAggregation='average', stochasticIonChannels=False,
                                                  perturbation=None)
        GpolAfter = localCircuit.G_pol[0, :, 0].clone()
        w_gi = eVNow.abs()[:, None] * maskMatrix
        colSumI = w_gi.sum(0).clamp_min(1e-300)
        pFieldMean = (w_gi.t() @ pEV) / colSumI[:, None]
        persistTermG = GpolBefore * (1 - rateGpol)
        driveTermG = GpolAfter - persistTermG
        m_selfG, m_driveG = persistTermG.abs(), driveTermG.abs()
        totalG = (m_selfG + m_driveG).clamp_min(1e-300)
        pGpolNew = (m_selfG[:, None] * pGpol + m_driveG[:, None] * pFieldMean) / totalG[:, None]
        localCircuit.updateCurrent()
        G_ij = localCircuit.G_ij[0].clone()
        InCurrent = localCircuit.InCurrent[0, :, 0].clone()
        OutCurrent = localCircuit.OutCurrent[0, :, 0].clone()
        localCircuit.updateVmem()
        sumG = G_ij.sum(1)
        persistTermV = (1 - (dt / C) * sumG) * VmemBefore
        neighborTerm = (dt / C) * G_ij * VmemBefore[None, :]
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
        if (step + 1) in snapshotSteps:
            snapshots[step + 1] = pVmem.clone()
    return snapshots


uniformVmem = torch.full((1, numCells, 1), -9.2e-3, dtype=torch.float64)
uniformGpol = torch.full((1, numCells, 1), G_ref, dtype=torch.float64)

print("=" * 70)
print("CHECK 1: locality at step 1")
print("=" * 70)
snap1 = propagate(uniformVmem, uniformGpol, 1, {1})[1]
# One-step reachable set: self, GJ neighbours, and field-mediated (any cell k sharing a grid
# point g that is within BOTH k's write reach and target i's read reach).
writeReach = (fieldCellMag > 0).double()     # (numFieldGridPoints, numCells): who writes to g
readReach = (maskMatrix > 0).double()        # (numFieldGridPoints, numCells): who reads from g
fieldReachable = (readReach.t() @ writeReach) > 0   # (numCells_i, numCells_k)
gjReachable = Adjacency > 0
selfReachable = torch.eye(numCells, dtype=torch.bool)
predictedReachable = selfReachable | gjReachable | fieldReachable

violations = ((snap1.abs() > TOL) & (~predictedReachable)).sum().item()
actualSupport = (snap1.abs() > TOL).sum().item()
predictedSupportSize = predictedReachable.sum().item()
print(f"  entries with nonzero provenance outside the predicted reachable set: {violations} "
     f"(should be 0)")
print(f"  actual nonzero entries: {actualSupport}, predicted-reachable entries: {predictedSupportSize} "
     f"(actual should be <= predicted)")
print(f"  {'PASS' if violations == 0 else 'FAIL'}")

print()
print("=" * 70)
print("CHECK 2 & 3: non-negativity and row sums, at several horizons")
print("=" * 70)
snaps = propagate(uniformVmem, uniformGpol, 100, {1, 5, 20, 50, 100})
for step, P in sorted(snaps.items()):
    minVal = P.min().item()
    rowSumErr = (P.sum(1) - 1).abs().max().item()
    print(f"  step {step:>4}: min entry = {minVal:+.2e} (should be >= 0), "
         f"max |row sum - 1| = {rowSumErr:.2e}")
allNonneg = all(P.min().item() >= -1e-12 for P in snaps.values())
allRowSums = all((P.sum(1) - 1).abs().max().item() < 1e-8 for P in snaps.values())
print(f"  {'PASS' if allNonneg and allRowSums else 'FAIL'}")

print()
print("=" * 70)
print("CHECK 4: monotonic spatial spreading (support size never shrinks)")
print("=" * 70)
targetCell = 60   # centre
supportSizes = []
for step in sorted(snaps.keys()):
    support = (snaps[step][targetCell].abs() > TOL).sum().item()
    supportSizes.append((step, support))
    print(f"  step {step:>4}: support size for centre cell = {support}")
monotonic = all(supportSizes[i][1] <= supportSizes[i + 1][1] for i in range(len(supportSizes) - 1))
print(f"  {'PASS' if monotonic else 'FAIL'}")

print()
print("=" * 70)
print("CHECK 5: D4 mirror symmetry (uniform condition, exact provenance rows)")
print("=" * 70)


def mirrorIndex(idx):
    r, c = idx // numCols, idx % numCols
    return r * numCols + (numCols - 1 - c)


pairs = [(0, mirrorIndex(0)), (5 * numCols, 5 * numCols + (numCols - 1)), (16, mirrorIndex(16))]
mirrorRelErrs = []
for a, b in pairs:
    rowA, rowB = snaps[100][a], snaps[100][b]
    mirroredRowA = torch.stack([rowA[mirrorIndex(j)] for j in range(numCells)])
    absErr = (mirroredRowA - rowB).abs()
    # Entries span ~0.45 down to ~1e-71 (see check 2/3), so an absolute tolerance is meaningless
    # here; a fixed floor avoids dividing by ~0 on the deep-underflow tail.
    relErr = (absErr / rowB.abs().clamp_min(1e-12)).max().item()
    mirrorRelErrs.append(relErr)
    print(f"  cell {a} vs mirror {b}: max relative |P[{a}, mirror(j)] - P[{b}, j]| / P[{b}, j] "
         f"= {relErr:.2e} (on entries >= 1e-12)")
print(f"  {'PASS' if max(mirrorRelErrs) < 1e-6 else 'FAIL'}")

print()
print("=" * 70)
print("CHECK 6: cross-validation against the independent finite-difference method")
print("=" * 70)
learnedParams = torch.load(SOURCE_DAT, weights_only=False)
paramsForClamp = buildParameters()
prepInstance = model(paramsForClamp, 1)
prepInstance.setExperimentalConditions((paramsForClamp['simParameters']['initialValues'], 1))
prepInstance.simulate(externalInputs=paramsForClamp['simParameters']['externalInputs'],
                      clampParameters=learnedParams['clampParameters'], perturbation=None,
                      numSimIters=102, storeVariables=('Vmem', 'Gpol'))
vmemPre = prepInstance.timeseriesVmem[101].clone()
gpolPre = prepInstance.timeseriesGpol[101].clone()

learnedSnaps = propagate(vmemPre, gpolPre, 300, {1, 10, 50, 150, 300})
fd = np.load('data/representationPattern_learned.npz', allow_pickle=True)
probeCells = {'corner_TL': 0, 'corner_TR': 10, 'edgeMid_top': 5, 'nearBoundary': 16, 'centre': 60}

print("  Pearson correlation between the two independent methods' response fields "
     "(finite-difference response magnitude vs. exact provenance column):")
for probeName, probeCell in probeCells.items():
    for T in [1, 10, 50]:   # short horizons, where the finite-difference method is known reliable
        fdResponse = np.abs(fd[f'{probeName}_T{T}'])            # (numCells,) mV per G_ref
        provColumn = learnedSnaps[T][:, probeCell].numpy()      # (numCells,) share of probeCell's identity
        corr = np.corrcoef(fdResponse, provColumn)[0, 1]
        print(f"    probe={probeName:<12} T={T:<4} r={corr:.3f}")
