"""Causal test of the latch: clamp one cell's conductance during the selective rise and see what follows
(PolyPatterning_Sim.md, Section 12).

The reduced rule says a cell goes dark when its conductance is driven above G_up and stays dark once the
conductance falls back inside the bistable window. That is a claim about cause, and it is tested here by
overriding one cell's conductance in an otherwise untouched run of the trained orders 0-3 code.

Predictions, fixed before the runs:
  P1  holding a feature cell inside the window through the rise leaves it light at the scored moment (>= 12 of 14)
  P2  blocking a nucleator disturbs more of the rest of the tissue than blocking a recruit
  P3  holding a light background cell above G_up through the rise, then releasing it, leaves it latched dark
  P4  a no-op override, holding a cell at the conductance it had anyway, changes nothing

Writes data/boundaryHarmonicLatchIntervention<rest of the summary's name> (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--interventionPath', type=str, required=True, help='npz produced by the intervention replay')
parser.add_argument('--recordPath', type=str, required=True, help='npz of the untouched trained runs')
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
args = parser.parse_args()

outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicLatchIntervention')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

data = np.load(args.interventionPath)
record = np.load(args.recordPath)
voltage, kinds, cells = data['vmem'], [str(k) for k in data['kinds']], data['cells']
best = int(data['best'])
dark = voltage[:, best] < boundary.hyperpolarizedThresholdMilliVolts
baseline = dark[0]
interior = np.array(boundary.interiorCellIndices)
featureCells = sorted(set(boundary.featureCellIndices.tolist()))

neighbourIndex = np.zeros((boundary.numCells, 4), dtype=int)
neighbourMask = np.zeros((boundary.numCells, 4), dtype=bool)
for cell in range(boundary.numCells):
    row, column = divmod(cell, boundary.latticeCols)
    for slot, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        r, c = row + dr, column + dc
        if 0 <= r < boundary.latticeRows and 0 <= c < boundary.latticeCols:
            neighbourIndex[cell, slot], neighbourMask[cell, slot] = r * boundary.latticeCols + c, True

# nucleator or recruit, read off the untouched run
orderIndex = list(record['orders']).index(3)
untouched = record['vmem'][orderIndex].astype(float) < boundary.hyperpolarizedThresholdMilliVolts
flips = untouched[1:] != untouched[:-1]
role = {}
for cell in featureCells:
    times = [t + 1 for t in np.where(flips[:best, cell])[0] if untouched[t + 1, cell] and t + 1 > 1200]
    if not times:
        role[cell] = dict(role='none', lastDark=None)
        continue
    last = times[-1]
    around = int(np.where(neighbourMask[cell], untouched[last - 1, neighbourIndex[cell]], False).sum())
    role[cell] = dict(role='nucleator' if around == 0 else 'recruit', lastDark=int(last))

runs = []
for index, (kind, cell) in enumerate(zip(kinds, cells)):
    if kind == 'baseline':
        continue
    cell = int(cell)
    changed = [int(c) for c in interior if dark[index][c] != baseline[c] and c != cell]
    around = set(neighbourIndex[cell][neighbourMask[cell]].tolist())
    runs.append(dict(kind=kind, cell=cell, row=cell // boundary.latticeCols, column=cell % boundary.latticeCols,
                     cellDark=bool(dark[index][cell]), baselineDark=bool(baseline[cell]),
                     othersChanged=len(changed), adjacentChanged=sum(1 for c in changed if c in around),
                     role=role.get(cell, {}).get('role'), lastDark=role.get(cell, {}).get('lastDark')))

blocks = [r for r in runs if r['kind'] == 'block']
forces = [r for r in runs if r['kind'] == 'force']
noops = [r for r in runs if r['kind'] == 'noop']
nucleators = [r['othersChanged'] for r in blocks if r['role'] == 'nucleator']
recruits = [r['othersChanged'] for r in blocks if r['role'] == 'recruit']
totalChanged = sum(r['othersChanged'] for r in forces)
totalAdjacent = sum(r['adjacentChanged'] for r in forces)
chance = float(np.mean([len(set(neighbourIndex[r['cell']][neighbourMask[r['cell']]].tolist()) & set(interior.tolist()))
                        / (len(interior) - 1) for r in forces]))

result = dict(
    best=best, window=[int(v) for v in data['window']],
    blockValue=float(data['blockValue']), forceValue=float(data['forceValue']),
    runs=runs,
    verdicts=dict(
        P1=dict(passed=sum(1 for r in blocks if not r['cellDark']), of=len(blocks), threshold=12,
                holds=sum(1 for r in blocks if not r['cellDark']) >= 12),
        P2=dict(nucleatorMedian=float(np.median(nucleators)) if nucleators else None,
                recruitMedian=float(np.median(recruits)) if recruits else None,
                holds=bool(nucleators and recruits and np.median(nucleators) > np.median(recruits) * 1.5)),
        P3=dict(passed=sum(1 for r in forces if r['cellDark']), of=len(forces),
                holds=all(r['cellDark'] for r in forces)),
        P4=dict(maxChanged=max((r['othersChanged'] for r in noops), default=0), of=len(noops),
                holds=all(r['othersChanged'] == 0 for r in noops))),
    spread=dict(totalChanged=totalChanged, adjacentChanged=totalAdjacent,
                adjacentShare=round(totalAdjacent / max(totalChanged, 1), 4), chanceShare=round(chance, 4)))

json.dump(result, open(outputPath, 'w'))
for name, v in result['verdicts'].items():
    print(f'  {name}: {"holds" if v["holds"] else "FAILS"}  {v}', flush=True)
print(f"  knock-on changes adjacent to the clamped cell: {totalAdjacent} of {totalChanged} "
      f"({100 * totalAdjacent / max(totalChanged, 1):.0f}%), chance {100 * chance:.0f}%", flush=True)
print('wrote', outputPath, flush=True)
