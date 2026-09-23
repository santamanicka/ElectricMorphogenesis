"""The reduced rule behind face formation: every cell is a latching switch whose window its neighbours shift
(PolyPatterning_Sim.md, Section 12).

Each cell's membrane voltage relaxes fast compared with its polarising conductance G_pol and with its neighbours,
so at every moment the cell sits at a stable root of its own dV/dt. Solving that equation with the neighbours held
at their own branch voltages gives two critical conductances: above G_up the depolarised root ceases to exist and
the cell is forced dark; below G_down the hyperpolarised root ceases to exist and it is forced light; between them
both roots exist and the cell keeps whichever it is on. Both thresholds move by about 0.035 per neighbour, up for a
light neighbour and down for a dark one, which is what lets a darkening cell recruit the cells beside it.

The script derives the threshold table from the model's own constants, measures how well the resulting rule predicts
the recorded runs, and writes the trace, snapshots and events the report figure needs.

Writes data/boundaryHarmonicSwitchRule<rest of the summary's name> (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--recordPath', type=str, required=True, help='npz with vmem, gpol, orders, hold, bestIterations')
parser.add_argument('--branchPath', type=str, required=True, help='npz of effective branches from the same record')
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--traceStride', type=int, default=5)
args = parser.parse_args()

outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicSwitchRule')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

# --------------------------------------------------------------------------- the model's own constants
conductanceReference, depolarisingConductance = 1e-9, 1.5e-9
polarisingReversal, depolarisingReversal = -0.055, -0.005
gateMidpoint, gateWidth, gateSteepness = -0.027, 0.027, 3.0
gapJunctionStrength, gapJunctionWidth = 5e-11, 0.012
darkArchetype, lightArchetype = -0.050, -0.010
separatrix = -30.0

rows = columns = boundary.latticeRows
numCells = boundary.numCells
neighbourIndex = np.zeros((numCells, 4), dtype=int)
neighbourMask = np.zeros((numCells, 4), dtype=bool)
for cell in range(numCells):
    row, column = divmod(cell, columns)
    for slot, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        r, c = row + dr, column + dc
        if 0 <= r < rows and 0 <= c < columns:
            neighbourIndex[cell, slot], neighbourMask[cell, slot] = r * columns + c, True
degree = neighbourMask.sum(1)


def stableRootCount(conductance, darkNeighbours, totalNeighbours, grid):
    gate = 1.0 / (1.0 + np.exp(gateSteepness * (grid - gateMidpoint) / gateWidth))
    current = (-conductance * (grid - polarisingReversal) * gate
               - depolarisingConductance * (grid - depolarisingReversal) * (1.0 - gate))
    for slot in range(totalNeighbours):
        neighbour = darkArchetype if slot < darkNeighbours else lightArchetype
        current = current + 2.0 * gapJunctionStrength / (1.0 + np.cosh((grid - neighbour) / gapJunctionWidth)) * (neighbour - grid)
    return int(((current[:-1] > 0) & (current[1:] <= 0)).sum())


def thresholdTable():
    """Critical G_pol / G_ref either side of the bistable window, for every neighbourhood."""
    grid = np.linspace(-0.070, 0.005, 4001)
    conductances = np.linspace(0.0, 2.0, 4001)
    table = {}
    for total in range(5):
        for dark in range(total + 1):
            counts = np.array([stableRootCount(value * conductanceReference, dark, total, grid) for value in conductances])
            bistable = np.where(counts == 2)[0]
            table[(total, dark)] = (float(conductances[bistable[0]]), float(conductances[bistable[-1]])) if len(bistable) else None
    return table


table = thresholdTable()
lowerFit = np.array([[table[(total, dark)][0] for dark in range(total + 1)] for total in range(5)], dtype=object)
print('threshold table (G_pol / G_ref):', flush=True)
for total in range(5):
    for dark in range(total + 1):
        low, high = table[(total, dark)]
        print(f'  {total} neighbours, {dark} dark:  down {low:.3f}  up {high:.3f}', flush=True)

# a linear reading of the table, in light- and dark-neighbour counts
design, lows, highs = [], [], []
for total in range(5):
    for dark in range(total + 1):
        design.append([1.0, total - dark, dark])
        lows.append(table[(total, dark)][0])
        highs.append(table[(total, dark)][1])
design = np.array(design)
lowCoefficients = np.linalg.lstsq(design, np.array(lows), rcond=None)[0]
highCoefficients = np.linalg.lstsq(design, np.array(highs), rcond=None)[0]
print(f'G_down = {lowCoefficients[0]:.3f} + {lowCoefficients[1]:+.3f} * light {lowCoefficients[2]:+.3f} * dark', flush=True)
print(f'G_up   = {highCoefficients[0]:.3f} + {highCoefficients[1]:+.3f} * light {highCoefficients[2]:+.3f} * dark', flush=True)

# ----------------------------------------------------------------------------------- measure the rule
record, branchData = np.load(args.recordPath), np.load(args.branchPath)
orders = [int(v) for v in record['orders']]
interior = np.array(boundary.interiorCellIndices)
featureCells = sorted(set(boundary.featureCellIndices.tolist()))
isFeature = np.isin(interior, featureCells)
result = dict(orders=orders, hold=int(record['hold']), separatrix=separatrix,
              thresholdTable={f'{total},{dark}': table[(total, dark)] for total in range(5) for dark in range(total + 1)},
              lowerCoefficients=lowCoefficients.tolist(), upperCoefficients=highCoefficients.tolist(),
              featureCells=featureCells, interiorCells=interior.tolist(), runs={})

for index, order in enumerate(orders):
    conductance = record['gpol'][index].astype(float)
    voltage = record['vmem'][index].astype(float)
    best = int(record['bestIterations'][index])
    low = branchData[f'low{index}'].astype(float)
    high = branchData[f'high{index}'].astype(float)
    seat = np.where(np.abs(voltage - low) <= np.abs(voltage - high), low, high)
    dark = seat < separatrix

    darkNeighbours = np.where(neighbourMask[None], dark[:, neighbourIndex], False).sum(2)
    lightNeighbours = degree[None] - darkNeighbours
    lowerThreshold = lowCoefficients[0] + lowCoefficients[1] * lightNeighbours + lowCoefficients[2] * darkNeighbours
    upperThreshold = highCoefficients[0] + highCoefficients[1] * lightNeighbours + highCoefficients[2] * darkNeighbours
    predicted = np.where(conductance > upperThreshold, True, np.where(conductance < lowerThreshold, False, dark))
    correct = (predicted[:best + 1] == dark[:best + 1])[:, interior]
    margin = np.minimum(np.abs(conductance - lowerThreshold), np.abs(conductance - upperThreshold))[:best + 1][:, interior]

    events = []
    for time, cell in np.argwhere(dark[1:best + 1] != dark[:best]):
        if cell not in set(interior.tolist()):
            continue
        before = int(np.where(neighbourMask[cell], dark[time, neighbourIndex[cell]], False).sum())
        events.append(dict(iteration=int(time) + 1, cell=int(cell), toDark=bool(dark[time + 1, cell]),
                           darkNeighbours=before, conductance=round(float(conductance[time + 1, cell]), 3),
                           feature=bool(cell in set(featureCells))))

    interiorMean = conductance[:, interior].mean(1)
    trough = 302 + int(interiorMean[302:1300].argmin()) if best > 1300 else None
    secondPeak = trough + int(interiorMean[trough:best + 1].argmax()) if trough else None
    times = list(range(0, best + 1, args.traceStride))
    result['runs'][str(order)] = dict(
        order=order, bestIteration=best,
        accuracy=round(float(correct.mean()), 4),
        errorsWithinPoint05=round(float((margin[~correct] < 0.05).mean()), 3) if (~correct).any() else None,
        holdPeak=round(float(interiorMean[:302].max()), 3),
        trough=trough, troughValue=round(float(interiorMean[trough]), 3) if trough else None,
        secondPeak=secondPeak, secondPeakValue=round(float(interiorMean[secondPeak]), 3) if secondPeak else None,
        times=times,
        featureMean=[round(float(v), 4) for v in conductance[times][:, interior][:, isFeature].mean(1)],
        backgroundMean=[round(float(v), 4) for v in conductance[times][:, interior][:, ~isFeature].mean(1)],
        darkCount=[int(v) for v in dark[times][:, interior].sum(1)],
        darkTrace=[''.join('1' if v else '0' for v in row) for row in dark[times][:, interior]],
        featureDarkCount=[int(v) for v in dark[times][:, interior][:, isFeature].sum(1)],
        events=events,
        snapshots=[dict(iteration=int(t), vmem=[round(float(v), 1) for v in voltage[t]],
                        gpol=[round(float(v), 3) for v in conductance[t]],
                        dark=[bool(v) for v in dark[t]])
                   for t in ([0, 150, 301] + ([trough, secondPeak] if trough else []) + [best]) if t is not None])
    print(f"  orders 0-{order}: accuracy {result['runs'][str(order)]['accuracy']}, "
          f"trough {trough}, second peak {secondPeak}, {len(events)} interior events", flush=True)

json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath, os.path.getsize(outputPath) // 1024, 'KiB', flush=True)
