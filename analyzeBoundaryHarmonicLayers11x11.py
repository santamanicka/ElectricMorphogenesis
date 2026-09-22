"""Do the bulk's layers belong to the code's orders? (PolyPatterning_Sim.md, Section 12).

The predictions and the layer basis are fixed beforehand in --predictionsPath. Layer n is the exact discrete harmonic
extension of the ring mode cos(n theta): the solution of the 4-neighbour discrete Laplace equation on the 81 interior
cells with the ring held at cos(n theta_j). Passive diffusion would make the interior the harmonic extension of the
ring's values, so perturbing coefficient a_k would move layer k alone; the response matrix would be diagonal and
nothing would be left over. This script measures how far the tissue departs from that.

Each of a_0..a_3 is moved by +-epsilon for several epsilon, the three single knockouts are run alongside, and every
code is replayed as in training. At each probe iteration the interior response is fitted onto the layers, giving the
response matrix, its diagonal share, and the residual share that no boundary data could produce.

Writes data/boundaryHarmonicLayers<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--trainedRunPath', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz')
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicLayerPredictions1888Hold301FaceMinus60Minus5.json')
args = parser.parse_args()

predictions = json.load(open(args.predictionsPath))
run = dict(np.load(args.trainedRunPath))
outputPath = f"data/boundaryHarmonicLayers{int(run['referenceCheckpoint'])}Hold{int(run['holdIterations'])}{run['targetName']}.json"
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
reference = boundary.loadCheckpoint(int(run['referenceCheckpoint']))
hold, numIterations = int(run['holdIterations']), int(run['numIterations'])
code, trainedMoment = run['bestCoefficients'], int(run['bestIteration'])
angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, np.arange(len(code))))
layerOrders = predictions['layerBasis']['orders']
probeIterations = predictions['method']['probeIterations']
steps = [0.005, 0.015, 0.05, 0.15]

# ------------------------------------------------------------------------------------------- the layers
interiorCells = boundary.interiorCellIndices
interiorPosition = {cell: index for index, cell in enumerate(interiorCells)}
ringPosition = {cell: index for index, cell in enumerate(boundary.boundaryRingCells)}


def harmonicExtension(ringData):
    """Interior values solving the discrete Laplace equation with the ring held at `ringData`."""
    operator = np.zeros((len(interiorCells), len(interiorCells)))
    source = np.zeros(len(interiorCells))
    for index, cell in enumerate(interiorCells):
        row, col = divmod(cell, boundary.latticeCols)
        operator[index, index] = -4.0
        for neighbourRow, neighbourCol in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            neighbour = neighbourRow * boundary.latticeCols + neighbourCol
            if neighbour in interiorPosition:
                operator[index, interiorPosition[neighbour]] = 1.0
            else:
                source[index] -= ringData[ringPosition[neighbour]]
    return np.linalg.solve(operator, source)


layers = np.column_stack([harmonicExtension(np.cos(order * angles)) for order in layerOrders])
print('layer basis: condition number %.1f, RMS by order %s' % (np.linalg.cond(layers), np.array2string(np.sqrt((layers ** 2).mean(0)), precision=3)), flush=True)


shellBasis, shellLabels = [], []
for shell in range(1, 6):
    cells = boundary.shellCells(shell)
    shellAngles = boundary.ringAngles(cells)
    for order in layerOrders:
        if order > len(cells) // 2:
            continue
        vector = np.zeros(len(interiorCells))
        vector[[interiorPosition[cell] for cell in cells]] = np.cos(order * shellAngles)
        shellBasis.append(vector)
        shellLabels.append((shell, order))
shellBasis = np.column_stack(shellBasis)
shellOrderColumns = [[index for index, (shell, order) in enumerate(shellLabels) if order == wanted] for wanted in layerOrders]


def shellDecompose(interiorValues):
    """Exploratory second basis, added after the pre-registration: cos(n theta) fitted on each interior square shell.
    Layer n's size is the norm of the field it accounts for across shells, so these amplitudes are magnitudes."""
    amplitudes, *_ = np.linalg.lstsq(shellBasis, interiorValues, rcond=None)
    sizes = [float(np.sqrt(((shellBasis[:, columns] * amplitudes[columns]).sum(1)) ** 2).sum() ** 0.5) for columns in shellOrderColumns]
    residual = interiorValues - shellBasis @ amplitudes
    total = np.sqrt((interiorValues ** 2).sum())
    return np.array(sizes), float(np.sqrt((residual ** 2).sum()) / (total + 1e-12))


def decompose(interiorValues):
    """Least-squares amplitudes on the layers, and the share of the response left over."""
    amplitudes, *_ = np.linalg.lstsq(layers, interiorValues, rcond=None)
    residual = interiorValues - layers @ amplitudes
    total = np.sqrt((interiorValues ** 2).sum())
    return amplitudes, float(np.sqrt((residual ** 2).sum()) / (total + 1e-12))


# --------------------------------------------------------------------------------------------- the runs
rows, labels = [code], [('base', None, 0.0)]
for order in range(len(code)):
    for step in steps:
        for sign in (+1, -1):
            perturbed = code.copy()
            perturbed[order] += sign * step
            rows.append(perturbed)
            labels.append(('step', order, sign * step))
for order in range(1, len(code)):
    knocked = code.copy()
    knocked[order] = 0.0
    rows.append(knocked)
    labels.append(('knockout', order, -code[order]))
ringValues = np.clip(np.array(rows) @ basis.T, 0, 2)
clipped = np.abs(ringValues - np.array(rows) @ basis.T).max()
print(f'{len(rows)} codes; largest clipping {clipped:.4f}; ring values {ringValues.min():.3f} to {ringValues.max():.3f}', flush=True)

patterns = {}
featureOnset = {'iteration': None}
featureCells = boundary.featureCellIndices


def onIteration(iteration, vmem):
    values = vmem.numpy()
    if featureOnset['iteration'] is None and values[0, featureCells].mean() < boundary.hyperpolarizedThresholdMilliVolts:
        featureOnset['iteration'] = iteration
    if iteration in probeIterations:
        patterns[iteration] = values.copy()


boundary.ringHoldBatchReplay(reference, ringValues, hold, numIterations, onIteration)
print('feature onset iteration', featureOnset['iteration'], flush=True)

# ------------------------------------------------------------------------------------------ the response
stepIndex = {(order, step): index for index, (kind, order, step) in enumerate(labels) if kind == 'step'}
knockIndex = {order: index for index, (kind, order, step) in enumerate(labels) if kind == 'knockout'}
result = dict(predictions=predictions, code=code.tolist(), hold=hold, trainedMoment=trainedMoment,
              layerOrders=layerOrders, steps=steps, probeIterations=probeIterations,
              featureOnsetIteration=featureOnset['iteration'], layerConditionNumber=float(np.linalg.cond(layers)),
              base={}, response={}, knockoutResponse={}, knockoutResponseShell={},
              shellBasisNote='square-shell cos(n theta) fits, added after the pre-registration as a second basis')

for iteration in probeIterations:
    values = patterns[iteration]
    baseAmplitudes, baseResidual = decompose(values[0, interiorCells])
    result['base'][str(iteration)] = dict(amplitudes=np.round(baseAmplitudes, 4).tolist(), residualShare=round(baseResidual, 4),
                                          featureMeanMilliVolts=round(float(values[0, featureCells].mean()), 2))
    perStep = {}
    for step in steps:
        matrix, residuals = np.zeros((len(layerOrders), len(code))), []
        for order in range(len(code)):
            difference = values[stepIndex[(order, step)], interiorCells] - values[stepIndex[(order, -step)], interiorCells]
            amplitudes, residualShare = decompose(difference / (2 * step))
            matrix[:, order] = amplitudes
            residuals.append(residualShare)
        shares = [float(abs(matrix[layerOrders.index(order), order]) / (np.abs(matrix[:, order]).sum() + 1e-12)) for order in range(len(code))]
        shellMatrix, shellResiduals = np.zeros((len(layerOrders), len(code))), []
        for order in range(len(code)):
            difference = values[stepIndex[(order, step)], interiorCells] - values[stepIndex[(order, -step)], interiorCells]
            sizes, residualShare = shellDecompose(difference / (2 * step))
            shellMatrix[:, order] = sizes
            shellResiduals.append(residualShare)
        shellShares = [float(shellMatrix[layerOrders.index(order), order] / (shellMatrix[:, order].sum() + 1e-12)) for order in range(len(code))]
        perStep[str(step) + 'Shell'] = dict(responseMatrix=np.round(shellMatrix, 4).tolist(), diagonalShare=[round(share, 4) for share in shellShares],
                                            meanDiagonalShare=round(float(np.mean(shellShares)), 4), residualShare=[round(value, 4) for value in shellResiduals])
        perStep[str(step)] = dict(responseMatrix=np.round(matrix, 4).tolist(), diagonalShare=[round(share, 4) for share in shares],
                                  meanDiagonalShare=round(float(np.mean(shares)), 4), residualShare=[round(value, 4) for value in residuals],
                                  responseSize=[round(float(np.abs(matrix[:, order]).sum()), 4) for order in range(len(code))])
    result['response'][str(iteration)] = perStep
    matrix, residuals = np.zeros((len(layerOrders), len(code))), []
    for order in range(1, len(code)):
        difference = values[knockIndex[order], interiorCells] - values[0, interiorCells]
        amplitudes, residualShare = decompose(difference / -code[order])
        matrix[:, order] = amplitudes
        residuals.append(residualShare)
    shares = [float(abs(matrix[layerOrders.index(order), order]) / (np.abs(matrix[:, order]).sum() + 1e-12)) for order in range(1, len(code))]
    shellMatrix, shellResiduals = np.zeros((len(layerOrders), len(code))), []
    for order in range(1, len(code)):
        difference = values[knockIndex[order], interiorCells] - values[0, interiorCells]
        sizes, residualShare = shellDecompose(difference / -code[order])
        shellMatrix[:, order] = sizes
        shellResiduals.append(residualShare)
    shellShares = [float(shellMatrix[layerOrders.index(order), order] / (shellMatrix[:, order].sum() + 1e-12)) for order in range(1, len(code))]
    result['knockoutResponseShell'][str(iteration)] = dict(responseMatrix=np.round(shellMatrix[:, 1:], 4).tolist(), diagonalShare=[round(share, 4) for share in shellShares],
                                                           meanDiagonalShare=round(float(np.mean(shellShares)), 4), residualShare=[round(value, 4) for value in shellResiduals])
    result['knockoutResponse'][str(iteration)] = dict(responseMatrix=np.round(matrix[:, 1:], 4).tolist(), diagonalShare=[round(share, 4) for share in shares],
                                                      meanDiagonalShare=round(float(np.mean(shares)), 4), residualShare=[round(value, 4) for value in residuals])

json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath)
