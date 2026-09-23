"""Causal test of the latch: clamp one cell's conductance and see what the rest of the tissue does.

Predictions, fixed before running:
  P1  clamping a feature cell below G_up during the selective rise leaves it light at the scored moment (>=12 of 14)
  P2  blocking a nucleator changes more other cells than blocking a recruit
  P3  forcing a background cell above G_up, then releasing, makes it latch dark
  P4  a no-op clamp, holding a cell at its own recorded conductance, changes nothing
"""
import sys
import json
import numpy as np
import torch
import boundaryCodeUtilities as boundary
from embryo import model

outPath = sys.argv[1]
recorded = np.load(sys.argv[2])
summary = json.load(open('data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json'))
reference = boundary.loadCheckpoint(1888)
hold, numIterations = int(summary['hold']), int(summary['numIterations'])
best = 2173
winner = summary['orders']['3']['best']
coefficients = np.asarray(np.load(f"{summary['trainingDirs'][winner['round']]}/order3_restart{winner['restart']:02d}.npz")['bestCoefficients'], float)
angles = boundary.ringAngles(boundary.boundaryRingCells)
ringValues = np.cos(np.outer(angles, np.arange(len(coefficients)))) @ coefficients

orderIndex = list(recorded['orders']).index(3)
baselineGpol = recorded['gpol'][orderIndex].astype(float)
baselineVmem = recorded['vmem'][orderIndex].astype(float)
featureCells = sorted(set(boundary.featureCellIndices.tolist()))
interior = np.array(boundary.interiorCellIndices)
darkAt = baselineVmem[best] < boundary.hyperpolarizedThresholdMilliVolts
backgroundLight = [int(c) for c in interior if not darkAt[c] and c not in set(featureCells)]

WINDOW = (1250, best)
BLOCK_VALUE = 1.25     # inside the window for every neighbourhood, so the cell cannot be forced dark
FORCE_VALUE = 1.65     # above G_up for every neighbourhood, so the cell is forced dark

runs = [dict(kind='baseline', cell=None)]
for cell in featureCells:
    runs.append(dict(kind='block', cell=int(cell)))
for cell in backgroundLight[:14]:
    runs.append(dict(kind='force', cell=int(cell)))
for cell in featureCells[:4]:
    runs.append(dict(kind='noop', cell=int(cell)))
numRuns = len(runs)
print(f'{numRuns} runs: 1 baseline, {len(featureCells)} blocks, {min(14, len(backgroundLight))} forces, 4 no-ops', flush=True)

torch.set_grad_enabled(False)
parameters = dict(reference)
parameters['latticePeriodicBoundaryGJ'] = False
parameters['ATPParameters'] = None
initial = reference['simParameters']['initialValues']
batchInitial = {n: initial[n].repeat(numRuns, 1, 1) for n in ('Vmem', 'eV', 'ligandConc')}
batchInitial['G_pol'] = dict(cells=[initial['G_pol']['cells'][0]] * numRuns, values=[initial['G_pol']['values'][0]] * numRuns)
batchInitial['G_dep'] = initial['G_dep']
system = model(parameters, numRuns)
system.setExperimentalConditions((batchInitial, numRuns))
circuit = system.electricNetwork
clamp = dict(reference['clampParameters'])
clamp['clampIndices'] = (np.repeat(np.arange(numRuns), len(boundary.boundaryRingCells)), np.tile(boundary.boundaryRingCells, numRuns))
clamp['clampValues'] = torch.tensor(np.tile(np.tile(ringValues, numRuns).reshape(1, -1), (hold, 1)), dtype=torch.double)
clamp['clampStartIter'], clamp['clampEndIter'] = 0, hold - 1

vmem = np.zeros((numRuns, best + 1, boundary.numCells), dtype=np.float32)
for iteration in range(best + 1):
    system.simulate(clampParameters=clamp if iteration < hold else None, numSimIters=1, outerIter=iteration, fieldModulation=False)
    if WINDOW[0] <= iteration <= WINDOW[1]:
        for index, spec in enumerate(runs):
            if spec['kind'] == 'block':
                circuit.G_pol[index, spec['cell'], 0] = BLOCK_VALUE * circuit.G_ref
            elif spec['kind'] == 'force':
                circuit.G_pol[index, spec['cell'], 0] = FORCE_VALUE * circuit.G_ref
            elif spec['kind'] == 'noop':
                circuit.G_pol[index, spec['cell'], 0] = baselineGpol[iteration, spec['cell']] * circuit.G_ref
    vmem[:, iteration] = circuit.Vmem[:, :, 0].numpy() * 1000.0

np.savez_compressed(outPath, vmem=vmem, kinds=np.array([r['kind'] for r in runs]),
                    cells=np.array([-1 if r['cell'] is None else r['cell'] for r in runs]),
                    window=np.array(WINDOW), blockValue=BLOCK_VALUE, forceValue=FORCE_VALUE, best=best)
print('done', vmem.shape, flush=True)
