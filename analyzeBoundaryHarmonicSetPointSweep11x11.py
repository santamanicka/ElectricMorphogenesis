"""Score the set-point sweep against its registered predictions.

Criteria: data/boundaryHarmonicSetPointPredictions1888Hold301FaceMinus60Minus5.json, committed before the runner.

    python3 analyzeBoundaryHarmonicSetPointSweep11x11.py --runPath <setpoint.npz>
"""
import argparse
import json

import numpy as np
import torch
from scipy import stats

import boundaryCodeUtilities as boundary
from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--runPath', type=str, required=True)
parser.add_argument('--predictionsPath', type=str,
                    default='data/boundaryHarmonicSetPointPredictions1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str,
                    default='data/boundaryHarmonicSetPoint1888Hold301FaceMinus60Minus5.json')
args = parser.parse_args()

run = np.load(args.runPath)
vmem, gpol, fieldGrid = run['vmem'], run['gpol'], run['field']
biases, multipliers = run['biases'], run['multipliers']
hold = int(run['hold'])

# the screen that turns field grid points into what each cell reads
torch.set_grad_enabled(False)
reference = boundary.loadCheckpoint(1888)
parameters = dict(reference)
parameters['latticePeriodicBoundaryGJ'] = False
parameters['ATPParameters'] = None
initial = reference['simParameters']['initialValues']
batchInitial = {name: initial[name] for name in ('Vmem', 'eV', 'ligandConc')}
batchInitial['G_pol'] = dict(cells=[initial['G_pol']['cells'][0]], values=[initial['G_pol']['values'][0]])
batchInitial['G_dep'] = initial['G_dep']
system = model(parameters, 1)
system.setExperimentalConditions((batchInitial, 1))
circuit = system.electricNetwork
screen = circuit.fieldScreenMatrixIn[0].numpy().astype(float)
numFieldNeighbours = float(circuit.numFieldNeighbors)

interior = np.array(boundary.interiorCellIndices)
featureCells = np.array(sorted(set(boundary.featureCellIndices.tolist())))
isFeature = np.isin(interior, featureCells)

runs = []
for index in range(len(biases)):
    conductance = gpol[index].astype(float)
    voltage = vmem[index].astype(float)
    meanG = conductance[:, interior].mean(1)
    # the screen sums over grid points; the transduction reads their AVERAGE
    meanField = ((fieldGrid[index][:, :, 0].astype(float) @ screen) / numFieldNeighbours)[:, interior].mean(1)

    firstPeakAt = int(meanG[:hold + 1].argmax())
    # the registered definition searches only the hold; a low enough set point does not turn until after it,
    # so the true first turning point is recorded beside it
    falling = np.where(np.diff(meanG) < 0)[0]
    trueTurningPoint = int(falling[0]) + 1 if len(falling) else None
    trough = 302 + int(meanG[302:1300].argmin())
    secondPeakAt = trough + int(meanG[trough:].argmax())
    darkCount = (voltage[:, interior] < boundary.hyperpolarizedThresholdMilliVolts).sum(1)

    # where does this run's own field first cross this run's own bias?
    sign = np.sign(meanField - biases[index])
    crossings = np.where(np.diff(sign))[0] + 1
    firstCrossing = int(crossings[0]) if len(crossings) else None

    runs.append(dict(
        multiplier=float(multipliers[index]), bias=float(biases[index]),
        firstPeak=round(float(meanG[:hold + 1].max()), 4), firstPeakAt=firstPeakAt,
        trough=trough, troughValue=round(float(meanG[trough]), 4),
        secondPeak=secondPeakAt, secondPeakValue=round(float(meanG[secondPeakAt]), 4),
        phase3Duration=secondPeakAt - trough,
        selectivity=round(float(conductance[secondPeakAt][interior][isFeature].mean()
                                - conductance[secondPeakAt][interior][~isFeature].mean()), 4),
        maxDarkCount=int(darkCount.max()), darkAtFirstPeak=int(darkCount[firstPeakAt]),
        firstFieldCrossing=firstCrossing, trueTurningPoint=trueTurningPoint,
        crossingOffsetAtTrueTurn=(abs(firstCrossing - trueTurningPoint)
                                  if firstCrossing is not None and trueTurningPoint is not None else None),
        quality=None,
        crossingOffset=(abs(firstCrossing - firstPeakAt) if firstCrossing is not None else None)))

# face quality at each run's own best moment, the same readout the ensemble analysis uses
interiorList = [int(c) for c in interior]
featureSet = set(int(c) for c in featureCells)
for index, entry in enumerate(runs):
    voltage = vmem[index].astype(float)
    best, bestQuality = None, -1.0
    for t in range(302, voltage.shape[0], 5):
        got = {c for c in interiorList if voltage[t, c] < boundary.hyperpolarizedThresholdMilliVolts}
        if not got:
            continue
        quality = len(got & featureSet) / len(got | featureSet)
        if quality > bestQuality:
            bestQuality, best = quality, t
    entry['quality'] = round(bestQuality, 3)
    entry['qualityMoment'] = best

order = [r['multiplier'] for r in runs]


def monotone(key):
    rho, p = stats.spearmanr(order, [r[key] for r in runs])
    return dict(rho=round(float(rho), 3), p=round(float(p), 4))


trained = next(r for r in runs if abs(r['multiplier'] - 1.0) < 1e-9)
lowest = min(runs, key=lambda r: r['multiplier'])
offsets = [r['crossingOffset'] for r in runs]
p4rho, p4p = stats.spearmanr([r['phase3Duration'] for r in runs], [r['selectivity'] for r in runs])

result = dict(
    predictions=json.load(open(args.predictionsPath))['predictions'],
    trainedBias=float(run['trainedBias']), runs=runs,
    verdicts=dict(
        P1=dict(**monotone('firstPeak'), holds=bool(monotone('firstPeak')['rho'] >= 0.9 and monotone('firstPeak')['p'] < 0.05)),
        P2=dict(**monotone('firstPeakAt'), holds=bool(monotone('firstPeakAt')['rho'] >= 0.9 and monotone('firstPeakAt')['p'] < 0.05)),
        P3=dict(offsets=offsets, worst=max(o for o in offsets if o is not None),
                offsetsAtTrueTurningPoint=[r['crossingOffsetAtTrueTurn'] for r in runs],
                holdsAtTrueTurningPoint=bool(all(r['crossingOffsetAtTrueTurn'] is not None
                                                and r['crossingOffsetAtTrueTurn'] <= 2 for r in runs)),
                holds=bool(all(o is not None and o <= 2 for o in offsets))),
        P4=dict(rho=round(float(p4rho), 3), p=round(float(p4p), 4), holds=bool(p4rho >= 0.6)),
        P5=dict(**monotone('maxDarkCount'), lowest=lowest['maxDarkCount'], trained=trained['maxDarkCount'],
                holds=bool(monotone('maxDarkCount')['rho'] >= 0.9 and monotone('maxDarkCount')['p'] < 0.05
                           and lowest['maxDarkCount'] < trained['maxDarkCount'])),
        P6=dict(firstPeak=trained['firstPeak'], trough=trained['trough'], secondPeak=trained['secondPeak'],
                holds=bool(abs(trained['firstPeak'] - 1.421) < 5e-4 and trained['trough'] == 585
                           and trained['secondPeak'] == 1765))))
json.dump(result, open(args.outputPath, 'w'), indent=1)

print(f"{'bias x':>7} {'1st peak':>9} {'at':>6} {'turns':>6} {'crossing':>9} {'off':>4} "
      f"{'2nd peak':>9} {'select':>8} {'maxDark':>8} {'quality':>8}")
for r in runs:
    print(f"{r['multiplier']:>7.1f} {r['firstPeak']:>9.3f} {r['firstPeakAt']:>6} "
          f"{r['trueTurningPoint']:>6} {r['firstFieldCrossing']:>9} {r['crossingOffsetAtTrueTurn']:>4} "
          f"{r['secondPeakValue']:>9.3f} {r['selectivity']:>8.3f} {r['maxDarkCount']:>8} {r['quality']:>8.3f}")
print()
for name, verdict in result['verdicts'].items():
    print(f"{name}: {'holds' if verdict['holds'] else 'FAILS'}  {verdict}", flush=True)
print('wrote', args.outputPath, flush=True)
