"""Replay ring codes and record the state the reduced analyses need (PolyPatterning_Sim.md, Section 12).

Two modes. --mode trained replays the best code of every order and stores Vmem, G_pol and the extracellular
field at every iteration, which is what the switch-rule analysis needs. --mode ensemble replays an ensemble of
codes and stores the hyperpolarised configuration (bit-packed), the conductance every --gpolStride iterations and
its running means, which is what the program analysis needs; --sliceWidth draws the ensemble around the trained
order-3 code instead of across the whole feasible region.

The outputs are npz files, written wherever --outputPath says; they are intermediates, not report data.
"""
import argparse
import json

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=('trained', 'ensemble'))
parser.add_argument('--outputPath', type=str, required=True)
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--trainedRunPath', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz')
parser.add_argument('--numCodes', type=int, default=1024)
parser.add_argument('--sliceWidth', type=float, default=0.0, help='>0 draws the ensemble around the trained code')
parser.add_argument('--gpolStride', type=int, default=10)
parser.add_argument('--seed', type=int, default=17)
args = parser.parse_args()

summary = json.load(open(args.summaryPath))
reference = boundary.loadCheckpoint(1888)
hold, numIterations = int(summary['hold']), int(summary['numIterations'])
angles = boundary.ringAngles(boundary.boundaryRingCells)


def trainedCoefficients(name):
    winner = summary['orders'][name]['best']
    path = f"{summary['trainingDirs'][winner['round']]}/order{name}_restart{winner['restart']:02d}.npz"
    return np.asarray(np.load(path)['bestCoefficients'], dtype=float)


if args.mode == 'trained':
    orderNames = sorted(summary['orders'], key=int)
    coefficients = [trainedCoefficients(name) for name in orderNames]
    ringValues = np.array([np.cos(np.outer(angles, np.arange(len(c)))) @ c for c in coefficients])
    labels = np.array([int(name) for name in orderNames])
else:
    trained = np.asarray(np.load(args.trainedRunPath)['bestCoefficients'], dtype=float)
    basis = np.cos(np.outer(angles, np.arange(len(trained))))
    generator = np.random.default_rng(23 if args.sliceWidth > 0 else args.seed)
    codes = [trained.copy()] if args.sliceWidth > 0 else []
    while len(codes) < args.numCodes:
        candidate = (trained + generator.uniform(-args.sliceWidth, args.sliceWidth, len(trained)) if args.sliceWidth > 0
                     else np.concatenate([generator.uniform(0.3, 1.3, 1), generator.uniform(-0.6, 0.6, len(trained) - 1)]))
        if (basis @ candidate).min() >= 0.02 and (basis @ candidate).max() <= 1.98:
            codes.append(candidate)
    codes = np.array(codes)
    ringValues = codes @ basis.T

numRuns = len(ringValues)
print(f'{args.mode}: {numRuns} runs, hold {hold}, {numIterations} iterations', flush=True)

torch.set_grad_enabled(False)
parameters = dict(reference)
parameters['latticePeriodicBoundaryGJ'] = False
parameters['ATPParameters'] = None
initial = reference['simParameters']['initialValues']
batchInitial = {name: initial[name].repeat(numRuns, 1, 1) for name in ('Vmem', 'eV', 'ligandConc')}
batchInitial['G_pol'] = dict(cells=[initial['G_pol']['cells'][0]] * numRuns, values=[initial['G_pol']['values'][0]] * numRuns)
batchInitial['G_dep'] = initial['G_dep']
system = model(parameters, numRuns)
system.setExperimentalConditions((batchInitial, numRuns))
circuit = system.electricNetwork
clamp = dict(reference['clampParameters'])
clamp['clampIndices'] = (np.repeat(np.arange(numRuns), len(boundary.boundaryRingCells)),
                         np.tile(boundary.boundaryRingCells, numRuns))
clamp['clampValues'] = torch.tensor(np.tile(ringValues.reshape(1, -1), (hold, 1)), dtype=torch.double)
clamp['clampStartIter'], clamp['clampEndIter'] = 0, hold - 1

interior = np.array(boundary.interiorCellIndices)
featureCells = np.array(sorted(set(boundary.featureCellIndices.tolist())))
isFeature = np.isin(interior, featureCells)

if args.mode == 'trained':
    vmem = np.zeros((numRuns, numIterations, boundary.numCells), dtype=np.float32)
    gpol = np.zeros((numRuns, numIterations, boundary.numCells), dtype=np.float32)
    field = None
else:
    packed = np.zeros((numRuns, numIterations, 16), dtype=np.uint8)
    gpolFrames = np.zeros((numRuns, numIterations // args.gpolStride + 1, boundary.numCells), dtype=np.float32)
    interiorMean = np.zeros((numRuns, numIterations), dtype=np.float32)
    featureMean = np.zeros((numRuns, numIterations), dtype=np.float32)
    backgroundMean = np.zeros((numRuns, numIterations), dtype=np.float32)

for iteration in range(numIterations):
    system.simulate(clampParameters=clamp if iteration < hold else None, numSimIters=1,
                    outerIter=iteration, fieldModulation=False)
    voltage = circuit.Vmem[:, :, 0].numpy() * 1000.0
    conductance = (circuit.G_pol[:, :, 0] / circuit.G_ref).numpy()
    if args.mode == 'trained':
        vmem[:, iteration] = voltage
        gpol[:, iteration] = conductance
        value = circuit.eV.detach().numpy()
        if field is None:
            field = np.zeros((numRuns, numIterations) + value.shape[1:], dtype=np.float32)
        field[:, iteration] = value
    else:
        packed[:, iteration] = np.packbits(voltage < boundary.hyperpolarizedThresholdMilliVolts, axis=1)
        interiorMean[:, iteration] = conductance[:, interior].mean(1)
        featureMean[:, iteration] = conductance[:, interior][:, isFeature].mean(1)
        backgroundMean[:, iteration] = conductance[:, interior][:, ~isFeature].mean(1)
        if iteration % args.gpolStride == 0:
            gpolFrames[:, iteration // args.gpolStride] = conductance
    if iteration % 500 == 0:
        print('  iteration', iteration, flush=True)

if args.mode == 'trained':
    np.savez_compressed(args.outputPath, vmem=vmem, gpol=gpol, field=field, orders=labels, hold=hold,
                        bestIterations=np.array([int(summary['orders'][n]['best']['iteration']) for n in orderNames]))
else:
    np.savez_compressed(args.outputPath, packed=packed, gpolFrames=gpolFrames, gpolStride=args.gpolStride,
                        interiorMean=interiorMean, featureMean=featureMean, backgroundMean=backgroundMean,
                        codes=codes, hold=hold)
print('wrote', args.outputPath, flush=True)
