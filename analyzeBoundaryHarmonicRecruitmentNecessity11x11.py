"""Is the nucleate-then-recruit ordering causal, or does one driver set it?

EXPLORATORY. This analysis was written after the choreography result, to answer a question the causal
intervention raised: blocking a cell does not mostly change its neighbours, so in what sense does a
nucleator recruit anything? No predictions were registered before it; nothing here is a confirmatory test.

Two measurements, both on the darkening events of the settled second rise:

  driver  Each cell's threshold moves by about 0.04 when a neighbour changes state, while the field swings
          its conductance by about 0.6. So a crossing on a *rising* conductance needs no neighbour --
          the drive alone reaches the bar. Only a crossing on a *falling* conductance requires the bar to
          have come down to meet it, which is the only thing a neighbour can do.

  links   For the trained orders 0-3 code, where one cell was blocked during the selective rise: take every
          pair (blocked cell, neighbour that crossed later with at least one dark neighbour) and ask whether
          removing the upstream cell actually changed the downstream one's state at the scored moment.
"""
import argparse
import json

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--recordPath', type=str, required=True, help='npz from recordBoundaryHarmonicRuns11x11.py --mode trained')
parser.add_argument('--interventionPath', type=str, required=True, help='npz from runBoundaryHarmonicLatchIntervention11x11.py')
parser.add_argument('--switchRulePath', type=str,
                    default='data/boundaryHarmonicSwitchRule1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str,
                    default='data/boundaryHarmonicRecruitmentNecessity1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--lateAfter', type=int, default=1200, help='same cut the choreography figure marks')
parser.add_argument('--slopeWindow', type=int, default=10, help='half-width, in iterations, of the dG/dt estimate')
args = parser.parse_args()

switchRule = json.load(open(args.switchRulePath))
record = np.load(args.recordPath)
interior = [int(c) for c in boundary.interiorCellIndices]
featureCells = set(switchRule['featureCells'])


def neighboursOf(cell):
    row, column = divmod(cell, boundary.latticeCols)
    return [(row + dr) * boundary.latticeCols + (column + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= row + dr < boundary.latticeRows and 0 <= column + dc < boundary.latticeCols]


# ------------------------------------------------------------------ what drives each crossing
byOrder, totals = {}, dict(late=0, rising=0, falling=0, fallingWithoutNeighbour=0, nucleators=0, nucleatorsRising=0)
for name in sorted(switchRule['runs'], key=int):
    run = switchRule['runs'][name]
    conductance = record['gpol'][list(record['orders']).index(int(name))].astype(float)
    late = [e for e in run['events'] if e['toDark'] and e['iteration'] > args.lateAfter]
    rows = []
    for event in late:
        time, cell = event['iteration'], event['cell']
        before = conductance[max(time - args.slopeWindow, 0), cell]
        after = conductance[min(time + args.slopeWindow, len(conductance) - 1), cell]
        rows.append(dict(iteration=time, cell=cell, darkNeighbours=event['darkNeighbours'],
                         conductance=event['conductance'], feature=cell in featureCells,
                         rising=bool(after > before)))
    nucleators = [r for r in rows if r['darkNeighbours'] == 0]
    falling = [r for r in rows if not r['rising']]
    byOrder[name] = dict(
        late=len(rows), nucleators=len(nucleators), recruits=len(rows) - len(nucleators),
        nucleatorsOnRising=sum(1 for r in nucleators if r['rising']),
        recruitsOnRising=sum(1 for r in rows if r['darkNeighbours'] > 0 and r['rising']),
        falling=len(falling), fallingWithoutNeighbour=sum(1 for r in falling if r['darkNeighbours'] == 0),
        fallingEvents=falling, secondPeak=run['secondPeak'],
        nucleatorsBeforePeak=sum(1 for r in nucleators if run['secondPeak'] and r['iteration'] < run['secondPeak']),
        recruitsAfterPeak=sum(1 for r in rows if r['darkNeighbours'] > 0
                              and run['secondPeak'] and r['iteration'] >= run['secondPeak']))
    totals['late'] += len(rows)
    totals['rising'] += sum(1 for r in rows if r['rising'])
    totals['falling'] += len(falling)
    totals['fallingWithoutNeighbour'] += sum(1 for r in falling if r['darkNeighbours'] == 0)
    totals['nucleators'] += len(nucleators)
    totals['nucleatorsRising'] += sum(1 for r in nucleators if r['rising'])

# ------------------------------------------------------------------ does removing the upstream cell matter
intervention = np.load(args.interventionPath)
kinds, cells = intervention['kinds'], intervention['cells']
best = int(intervention['best'])
darkAtBest = intervention['vmem'][:, best] < boundary.hyperpolarizedThresholdMilliVolts
baseline = darkAtBest[list(kinds).index('baseline')]

trained = switchRule['runs']['3']
crossing = {}
for event in trained['events']:
    if event['toDark'] and event['iteration'] > args.lateAfter:
        crossing[event['cell']] = event          # events are in time order, so the last one wins

links = []
for index, (kind, cell) in enumerate(zip(kinds, cells)):
    if str(kind) != 'block':
        continue
    cell = int(cell)
    upstream = crossing.get(cell)
    if upstream is None:
        continue
    changed = {c for c in interior if darkAtBest[index][c] != baseline[c] and c != cell}
    for other in neighboursOf(cell):
        downstream = crossing.get(other)
        if (other not in interior or downstream is None
                or downstream['iteration'] <= upstream['iteration'] or downstream['darkNeighbours'] == 0):
            continue
        links.append(dict(upstream=cell, downstream=other,
                          upstreamIteration=upstream['iteration'], downstreamIteration=downstream['iteration'],
                          upstreamRole='nucleator' if upstream['darkNeighbours'] == 0 else 'recruit',
                          broken=bool(other in changed)))

result = dict(
    exploratory=True,
    note='written after the result it examines; no predictions were registered beforehand',
    lateAfter=args.lateAfter, slopeWindow=args.slopeWindow,
    drive=dict(byOrder=byOrder, totals=totals),
    links=dict(items=links, total=len(links), broken=sum(1 for link in links if link['broken']),
               survived=sum(1 for link in links if not link['broken'])))
json.dump(result, open(args.outputPath, 'w'), indent=1)

print(f"late crossings over seven codes: {totals['late']}, "
      f"{totals['rising']} on a rising conductance ({100 * totals['rising'] / totals['late']:.0f}%)", flush=True)
print(f"on a falling conductance: {totals['falling']}, "
      f"of which {totals['fallingWithoutNeighbour']} had no dark neighbour", flush=True)
print(f"nucleators on a rising conductance: {totals['nucleatorsRising']} of {totals['nucleators']}", flush=True)
print(f"links: {result['links']['broken']} broken, {result['links']['survived']} survived, "
      f"of {len(links)}", flush=True)
print('wrote', args.outputPath, flush=True)
