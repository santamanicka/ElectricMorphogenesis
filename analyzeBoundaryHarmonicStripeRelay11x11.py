"""The stripe code's relay, scored against its registration (V1, V3 and R1-R4 of boundaryHarmonicStripeMechanismPredictions...json).

Input: the npz of computeBoundaryHarmonicRelay11x11.py --baseline extraUpdateOnly --target stripesInterior --order <size>. Reads it
through relayLoopNets (the ten-block layout, the phases from the run's own landmarks) and reports

  V1   the decomposition's closure and flux conservation
  V3   the left-right mirror symmetry of the ten-block net
  R1   the share of the injected selectivity that is the ring's held conductance
  R2   the field channel's share of the gross transfer from the release to the best moment
  R3   the largest field transfer of flood and of clear
  R4   the transfers that include ring left against those that include ring top
and, for the report, the net transfer of every pair in each phase and channel and the cell-level flux at a few moments.

    python3 analyzeBoundaryHarmonicStripeRelay11x11.py --relayPath <relay.npz> --outputPath data/boundaryHarmonicStripeRelay...json

Writes the output (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary
import relayLoopNets

parser = argparse.ArgumentParser()
parser.add_argument('--relayPath', type=str, required=True)
parser.add_argument('--outputPath', type=str, required=True)
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicStripeMechanismPredictions1888Hold301StripesInteriorMinus60Minus5.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')

raw = np.load(args.relayPath)
assert str(raw['baseline']) == 'extraUpdateOnly' and str(raw['targetName']) == 'stripesInterior'
layout = relayLoopNets.layoutFor('stripesInterior')
names = [str(x) for x in raw['readoutNames']]
primary = names.index('selectivity')
flux, times = raw['flux'], [int(t) for t in raw['fluxTimes']]
edges, gross, window = raw['edges'], raw['grossEdges'], int(raw['edgeWindow'])
source, hold = raw['sourceFlux'], int(raw['hold'])
primaryIteration, scoredIteration = int(raw['primaryIteration']), int(raw['scoredIteration'])
ring = np.array(boundary.boundaryRingCells)
n = boundary.numCells
phases = relayLoopNets.phaseWindows(raw, 'stripesInterior')
net = relayLoopNets.readNet(args.relayPath)
phaseNames = list(phases)

# ---------------------------------------------------------------------------------------------------------------- V1
closureWorst = float(raw['closure'].max())
conservation = {name: float(c) for name, c in zip(names, raw['conservation'])}
V1 = dict(closureWorst=closureWorst, conservation=conservation, holds=bool(closureWorst <= 1e-6 and max(conservation.values()) <= 1e-3))

# ---------------------------------------------------------------------------------------------------------------- V3
swap = np.arange(len(layout.NODE_ORDER))
for left, right in (('ringLeft', 'ringRight'), ('flankLeftUpper', 'flankRightUpper'), ('flankLeftLower', 'flankRightLower')):
    swap[layout.index[left]], swap[layout.index[right]] = layout.index[right], layout.index[left]
lastWindow = max(max(w) for w in phases.values() if len(w)) + 1
worstAsymmetry = 0.0
for channel in (0, 1):
    perWindow = np.array([layout.weight.T @ (edges[primary, channel, w] - edges[primary, channel, w].T) @ layout.weight for w in range(lastWindow)])
    worstAsymmetry = max(worstAsymmetry, float(np.abs(perWindow - perWindow[:, swap][:, :, swap]).max() / np.abs(perWindow).max()))
V3 = dict(relativeAsymmetry=worstAsymmetry, holds=bool(worstAsymmetry <= 1e-4))

# ---------------------------------------------------------------------------------------------------------------- R1
sourceOfPrimary = source[primary]                                          # (2, n): [voltage side effect, conductance injection] per cell
injected = float(sourceOfPrimary.sum())
ringConductance = float(sourceOfPrimary[1, ring].sum())
R1 = dict(total=injected, ringConductance=ringConductance, voltageSideEffect=float(sourceOfPrimary[0].sum()),
          otherConductance=float(sourceOfPrimary[1].sum() - ringConductance), share=ringConductance / injected)
R1['holds'] = bool(R1['share'] >= 0.95)

# ---------------------------------------------------------------------------------------------------------------- R2
first, last = (hold + 1) // window, primaryIteration // window
fieldGross, contactGross = float(gross[primary, 0, first:last + 1].sum()), float(gross[primary, 1, first:last + 1].sum())
R2 = dict(fieldShare=fieldGross / (fieldGross + contactGross), windows=[first, last], holds=bool(fieldGross / (fieldGross + contactGross) >= 0.5))

# ---------------------------------------------------------------------------------------------------------------- R3, R4
pairs = layout.PAIRS
fieldNet = dict(zip(phaseNames, net['field']))
contactNet = dict(zip(phaseNames, net['contact']))
largest = {}
for phase in ('flood', 'clear'):
    values = np.array(fieldNet[phase])
    k = int(np.argmax(np.abs(values)))
    largest[phase] = dict(pair=list(pairs[k]), value=float(values[k]),
                          isRingEndIntoStripe=bool(tuple(pairs[k]) in (('ringTop', 'stripeUpper'), ('ringBottom', 'stripeLower')) and values[k] > 0))
R3 = dict(largestFieldTransfer=largest, holds=bool(all(item['isRingEndIntoStripe'] for item in largest.values())))
absoluteSum = lambda node: float(sum(abs(fieldNet[p][k]) for p in ('flood', 'clear') for k, pair in enumerate(pairs) if node in pair))
R4 = dict(ringLeft=absoluteSum('ringLeft'), ringTop=absoluteSum('ringTop'))
R4['ratio'] = R4['ringLeft'] / R4['ringTop']
R4['holds'] = bool(R4['ratio'] < 0.5)

verdicts = dict(V1=V1, V3=V3, R1=R1, R2=R2, R3=R3, R4=R4)
for key, verdict in verdicts.items():
    print(f"{key}: {'holds' if verdict['holds'] else 'FAILS'}", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in verdict.items() if k != 'holds'}, flush=True)

at = lambda state: times.index(state - state % 5)
snapshotStates = sorted({hold + 1, primaryIteration // 2 + 1, primaryIteration + 1})
result = dict(
    note='CONFIRMATORY for V1, V3 and R1-R4 (criteria registered before this run); the tables are descriptive.',
    predictions=json.load(open(args.predictionsPath)), verdicts=verdicts, primaryIteration=primaryIteration, scoredIteration=scoredIteration,
    troughIteration=int(raw['troughIteration']), phases={name: [int(w.start), int(w.stop)] for name, w in phases.items()},
    nodeOrder=layout.NODE_ORDER, canonical=layout.CANONICAL, pairs=[list(p) for p in pairs], field=net['field'], contact=net['contact'],
    gap=net['gap'], overlapAtScoredMoment=net['faceOverlap'], difference={nm: float(d) for nm, d in zip(names, raw['difference'])},
    fluxStates=[s - 1 for s in snapshotStates],
    flux=[[round(float(v), 6) for v in flux[primary, at(s)]] for s in snapshotStates], injection=[round(float(v), 6) for v in sourceOfPrimary[1]])
json.dump(result, open(args.outputPath, 'w'), separators=(',', ':'))
print('wrote', args.outputPath, flush=True)
