"""How much of the conductance's rise does the clamp actually drive? (PolyPatterning_Sim.md, Section 12).

The reduced story says phase 1 is the clamp driving G_pol up across the tissue. That is only a claim about the
clamp if the same rise does not happen without one. This replays the same model under four conditions - no clamp
at all, a uniform ring hold at the trained code's mean, a uniform hold at the order-0 ceiling, and the trained
orders 0-3 code - and compares what the conductance does, during the hold and after it.

Writes data/boundaryHarmonicClampContribution<rest of the summary's name> (never overwriting).
"""
import argparse
import json
import os

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from embryo import model

import importlib.util
_spec = importlib.util.spec_from_file_location('branches', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                       'computeBoundaryHarmonicBranches11x11.py'))
branches = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(branches)

parser = argparse.ArgumentParser()
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--traceStride', type=int, default=5)
args = parser.parse_args()

outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicClampContribution')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

summary = json.load(open(args.summaryPath))
reference = boundary.loadCheckpoint(1888)
hold, numIterations = int(summary['hold']), int(summary['numIterations'])
angles = boundary.ringAngles(boundary.boundaryRingCells)
winner = summary['orders']['3']['best']
coefficients = np.asarray(np.load(f"{summary['trainingDirs'][winner['round']]}/order3_restart{winner['restart']:02d}.npz")['bestCoefficients'], float)
trainedRing = np.cos(np.outer(angles, np.arange(len(coefficients)))) @ coefficients

conditions = [
    dict(name='free', ring=None, note='no clamp at any iteration'),
    dict(name='uniformMean', ring=np.full(40, float(trainedRing.mean())), note="uniform hold at the trained code's mean"),
    dict(name='uniformCeiling', ring=np.full(40, 1.3), note="uniform hold at the order-0 best (the ceiling)"),
    dict(name='trained', ring=trainedRing, note='the trained orders 0-3 code'),
]
numRuns = len(conditions)
clamped = [i for i, c in enumerate(conditions) if c['ring'] is not None]

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

# only the clamped samples get clamp indices, so the free run is untouched
clamp = dict(reference['clampParameters'])
clamp['clampIndices'] = (np.repeat(np.array(clamped), len(boundary.boundaryRingCells)),
                         np.tile(boundary.boundaryRingCells, len(clamped)))
clamp['clampValues'] = torch.tensor(np.tile(np.concatenate([conditions[i]['ring'] for i in clamped]).reshape(1, -1), (hold, 1)), dtype=torch.double)
clamp['clampStartIter'], clamp['clampEndIter'] = 0, hold - 1

interior = np.array(boundary.interiorCellIndices)
featureCells = np.array(sorted(set(boundary.featureCellIndices.tolist())))
ring = np.array(boundary.boundaryRingCells)
isFeature = np.isin(interior, featureCells)

conductanceTrace = np.zeros((numRuns, numIterations, 3), dtype=np.float32)   # interior, feature, background
ringTrace = np.zeros((numRuns, numIterations), dtype=np.float32)
darkTrace = np.zeros((numRuns, numIterations), dtype=np.int16)
voltageTrace = np.zeros((numRuns, numIterations, boundary.numCells), dtype=np.float32)
conductanceFull = np.zeros((numRuns, numIterations, boundary.numCells), dtype=np.float32)
for iteration in range(numIterations):
    system.simulate(clampParameters=clamp if iteration < hold else None, numSimIters=1,
                    outerIter=iteration, fieldModulation=False)
    conductance = (circuit.G_pol[:, :, 0] / circuit.G_ref).numpy()
    voltage = circuit.Vmem[:, :, 0].numpy() * 1000.0
    conductanceTrace[:, iteration, 0] = conductance[:, interior].mean(1)
    conductanceTrace[:, iteration, 1] = conductance[:, interior][:, isFeature].mean(1)
    conductanceTrace[:, iteration, 2] = conductance[:, interior][:, ~isFeature].mean(1)
    ringTrace[:, iteration] = conductance[:, ring].mean(1)
    darkTrace[:, iteration] = (voltage[:, interior] < boundary.hyperpolarizedThresholdMilliVolts).sum(1)
    voltageTrace[:, iteration] = voltage
    conductanceFull[:, iteration] = conductance

# the same branch label the switch-rule analysis uses, so the two can be drawn in one panel: a cell is dark when
# the stable root of its own dV/dt that it sits on is hyperpolarised, not merely when its voltage is below a line
grid = np.linspace(-0.070, 0.005, 1201)
branchDark = np.zeros((numRuns, numIterations), dtype=np.int16)
for index in range(numRuns):
    _, low, high, _ = branches.branchesOf(grid, conductanceFull[index], voltageTrace[index])
    voltage = voltageTrace[index].astype(float)
    seat = np.where(np.abs(voltage - low) <= np.abs(voltage - high), low, high)
    branchDark[index] = (seat < branches.SEPARATRIX if hasattr(branches, 'SEPARATRIX') else seat < -30.0)[:, interior].sum(1)
    print(f"  {conditions[index]['name']}: branch-labelled dark cells peak at {branchDark[index].max()}", flush=True)

times = list(range(0, numIterations, args.traceStride))
result = dict(hold=hold, numIterations=numIterations, times=times, conditions=[])
for index, condition in enumerate(conditions):
    interiorMean = conductanceTrace[index, :, 0].astype(float)
    trough = 302 + int(interiorMean[302:1300].argmin())
    peak = trough + int(interiorMean[trough:].argmax())
    result['conditions'].append(dict(
        name=condition['name'], note=condition['note'],
        ringHeld=None if condition['ring'] is None else round(float(np.mean(condition['ring'])), 3),
        start=round(float(interiorMean[0]), 4),
        holdPeak=round(float(interiorMean[:hold].max()), 4),
        holdPeakAt=int(interiorMean[:hold].argmax()),
        atRelease=round(float(interiorMean[hold]), 4),
        trough=trough, troughValue=round(float(interiorMean[trough]), 4),
        peak=peak, peakValue=round(float(interiorMean[peak]), 4),
        gapAtPeak=round(float(conductanceTrace[index, peak, 1] - conductanceTrace[index, peak, 2]), 4),
        darkAtHoldEnd=int(darkTrace[index, hold - 1]), darkMax=int(darkTrace[index].max()),
        darkAt2173=int(darkTrace[index, 2173]),
        branchDarkMax=int(branchDark[index].max()), branchDarkAt2173=int(branchDark[index, 2173]),
        interior=[round(float(v), 4) for v in conductanceTrace[index, times, 0]],
        feature=[round(float(v), 4) for v in conductanceTrace[index, times, 1]],
        background=[round(float(v), 4) for v in conductanceTrace[index, times, 2]],
        ring=[round(float(v), 4) for v in ringTrace[index, times]],
        dark=[int(v) for v in darkTrace[index, times]],
        branchDark=[int(v) for v in branchDark[index, times]]))

json.dump(result, open(outputPath, 'w'))
print('  %-15s %8s %9s %9s %9s %9s %9s' % ('condition', 'start', 'holdPeak', 'release', 'trough', '2nd peak', 'gap'), flush=True)
for c in result['conditions']:
    print('  %-15s %8.3f %9.3f %9.3f %9.3f %9.3f %+9.3f' % (
        c['name'], c['start'], c['holdPeak'], c['atRelease'], c['troughValue'], c['peakValue'], c['gapAtPeak']), flush=True)
print('\n  dark interior cells: ' + ' · '.join(f"{c['name']} max {c['darkMax']}, at 2173 {c['darkAt2173']}" for c in result['conditions']), flush=True)
print('wrote', outputPath, flush=True)
