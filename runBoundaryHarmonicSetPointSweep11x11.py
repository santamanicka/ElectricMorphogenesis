"""Move the field transduction's set point and record what the whole tissue does.

The interior's mean conductance obeys d<G>/dt = -500*(<field> - fieldTransductionBias), and <field> rises as the
tissue polarises. That makes the tissue an integral controller whose set point is the bias. This sweeps the bias
with everything else held fixed -- same trained orders 0-3 code in every run -- so the prediction is about the
trajectory as a whole, not about any one cell.

Predictions and criteria: data/boundaryHarmonicSetPointPredictions1888Hold301FaceMinus60Minus5.json, committed
before this file.

    python3 runBoundaryHarmonicSetPointSweep11x11.py <outPath.npz>
"""
import sys
import json

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from embryo import model

outPath = sys.argv[1]
summary = json.load(open('data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json'))
reference = boundary.loadCheckpoint(1888)
hold, numIterations = int(summary['hold']), int(summary['numIterations'])

winner = summary['orders']['3']['best']
coefficients = np.asarray(np.load(
    f"{summary['trainingDirs'][winner['round']]}/order3_restart{winner['restart']:02d}.npz")['bestCoefficients'], float)
angles = boundary.ringAngles(boundary.boundaryRingCells)
ringValues = np.cos(np.outer(angles, np.arange(len(coefficients)))) @ coefficients

MULTIPLIERS = (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4)
trainedBias = float(reference['fieldParameters']['fieldTransductionBias'])
biases = np.array([m * trainedBias for m in MULTIPLIERS])
numRuns = len(biases)
print(f'trained bias {trainedBias:.3e}; sweeping {[f"{b:.3e}" for b in biases]}', flush=True)

torch.set_grad_enabled(False)
parameters = dict(reference)
parameters['latticePeriodicBoundaryGJ'] = False
parameters['ATPParameters'] = None
initial = reference['simParameters']['initialValues']
batchInitial = {name: initial[name].repeat(numRuns, 1, 1) for name in ('Vmem', 'eV', 'ligandConc')}
batchInitial['G_pol'] = dict(cells=[initial['G_pol']['cells'][0]] * numRuns,
                             values=[initial['G_pol']['values'][0]] * numRuns)
batchInitial['G_dep'] = initial['G_dep']
system = model(parameters, numRuns)
system.setExperimentalConditions((batchInitial, numRuns))
circuit = system.electricNetwork

# one bias per run: the transduction reads eVneighborsMean of shape (runs, cells, 1), so a (runs, 1, 1) bias
# broadcasts over cells and leaves every other constant untouched.
circuit.fieldTransductionBias = torch.tensor(biases.reshape(numRuns, 1, 1), dtype=torch.double)

clamp = dict(reference['clampParameters'])
clamp['clampIndices'] = (np.repeat(np.arange(numRuns), len(boundary.boundaryRingCells)),
                         np.tile(boundary.boundaryRingCells, numRuns))
clamp['clampValues'] = torch.tensor(np.tile(np.tile(ringValues, numRuns).reshape(1, -1), (hold, 1)), dtype=torch.double)
clamp['clampStartIter'], clamp['clampEndIter'] = 0, hold - 1

vmem = np.zeros((numRuns, numIterations, boundary.numCells), dtype=np.float32)
gpol = np.zeros((numRuns, numIterations, boundary.numCells), dtype=np.float32)
field = None
for iteration in range(numIterations):
    system.simulate(clampParameters=clamp if iteration < hold else None, numSimIters=1,
                    outerIter=iteration, fieldModulation=False)
    vmem[:, iteration] = circuit.Vmem[:, :, 0].numpy() * 1000.0
    gpol[:, iteration] = (circuit.G_pol[:, :, 0] / circuit.G_ref).numpy()
    value = circuit.eV.detach().numpy()
    if field is None:
        field = np.zeros((numRuns, numIterations) + value.shape[1:], dtype=np.float32)
    field[:, iteration] = value
    if iteration % 500 == 0:
        print(f'  iteration {iteration}', flush=True)

np.savez_compressed(outPath, vmem=vmem, gpol=gpol, field=field, biases=biases,
                    multipliers=np.array(MULTIPLIERS), trainedBias=trainedBias, hold=hold)
print('done', vmem.shape, flush=True)
