"""Assembles the stripes report's Relay Loop data: the stripes' counterpart of assembleRelayLoopFiveLevelData.py, from one record per ring code
(runBoundaryHarmonicStripeRelayLoopRelays.sh -> extractRelayLoopSweepRecord11x11.py --withTrajectory, under data/relayLoopStripes/) and the placement
maps of buildBoundaryHarmonicRingCodeStripeLevels11x11.py.

For every code the record holds the whole causal net read at the stripe code's readout (iteration 504): the net transfer between every pair of the
seven canonical blocks (ring top, bottom, left; stripe upper, lower; left flank upper, lower; the right-hand blocks mirror them) in the flood (the
hold) and the clear (the release to 504), by field and by gap junction; the stripe's overlap and the dark cells of its two halves at 504; the
selectivity gap; the Vmem at the end of the flood and at 504; and the conductance curves. There is no write phase: the stripe is read before the
interior's conductance has turned, so its windows are empty.

Writes data/relayLoopStripesPageData<suffix>.json (never overwriting), the blocks the report's Relay Loop sections draw, in the shapes of the face
page's data:
  trainedTop3Phases / variantData / sliderData / gridData   each code's top three field transfers per phase, one side per mirror pair
  trainedEdges / trackedData   the trained code's own ten biggest transfers (the five biggest of each phase), and those same pairs read off every code
  vmemData / conductanceData   the tissue's Vmem at the phase ends, and the conductance curves, per code
  edgeLanes   how far, and to which side, each (pair, phase) edge bows, so the same pair in the same phase is the same curve in every panel

    python3 assembleRelayLoopStripesData11x11.py
"""
import argparse
import json
import os

import numpy as np

import relayLoopNets

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--placementPath', type=str, default=f'data/boundaryHarmonicRingCodeStripeLevels{SUFFIX}.json')
parser.add_argument('--recordDirectory', type=str, default='data/relayLoopStripes')
parser.add_argument('--outputPath', type=str, default=f'data/relayLoopStripesPageData{SUFFIX}.json')
parser.add_argument('--topPerPhase', type=int, default=3)
parser.add_argument('--trackedPerPhase', type=int, default=5)
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')

layout = relayLoopNets.layoutFor('stripesInterior')
PAIRS = layout.PAIRS
PHASE_NAMES = ['flood', 'clear']
MIRRORED = {'ringLeft', 'flankLeftUpper', 'flankLeftLower'}               # the nodes that have a twin on the other side
placement = json.load(open(args.placementPath))
records = {}
for variant in placement['variants']:
    path = f'{args.recordDirectory}/{variant["key"]}.json'
    if not os.path.exists(path):
        raise SystemExit(f'no record for {variant["key"]} yet ({path})')
    records[variant['key']] = json.load(open(path))
assert 'trained' in records
field = lambda key: np.array(records[key]['field'])                         # (phase, pair), positive = canonical direction a -> b


def phasesOf(key):
    """{phase: [{from, to, value, mirrored}]}: the top few field transfers of each phase, canonical side only, signed into a direction."""
    out = {}
    for pi, phase in enumerate(PHASE_NAMES):
        ranked = np.argsort(-np.abs(field(key)[pi]))[:args.topPerPhase]
        out[phase] = []
        for i in ranked:
            a, b = PAIRS[i]
            v = float(field(key)[pi, i])
            out[phase].append(dict(from_=a if v >= 0 else b, to=b if v >= 0 else a, value=round(abs(v), 5), mirrored=bool(a in MIRRORED or b in MIRRORED)))
    return out


# ------------------------------------------------------------------------------------------- the trained code's own tracked edges
trainedEdges = []
for pi, phase in enumerate(PHASE_NAMES):
    for i in np.argsort(-np.abs(field('trained')[pi]))[:args.trackedPerPhase]:
        a, b = PAIRS[i]
        v = float(field('trained')[pi, i])
        trainedEdges.append(dict(from_=a if v >= 0 else b, to=b if v >= 0 else a, phase=phase, value=round(abs(v), 5), pairIndex=int(i), phaseIndex=pi))
for e in trainedEdges:                                                          # a trend line from the trained code's own numbers
    other = 1 - e['phaseIndex']
    sign = 1.0 if (e['from_'], e['to']) == PAIRS[e['pairIndex']] else -1.0 if (e['to'], e['from_']) == PAIRS[e['pairIndex']] else None
    otherValue = sign * float(field('trained')[other, e['pairIndex']])
    e['trend'] = (f"{PHASE_NAMES[other]}: {abs(otherValue):.3f}" + ('' if otherValue >= 0 else ', running the other way')) if abs(otherValue) >= 5e-4 else f'nothing in {PHASE_NAMES[other]}'


def trackedOf(key):
    out = []
    for e in trainedEdges:
        sign = 1.0 if (e['from_'], e['to']) == PAIRS[e['pairIndex']] else -1.0
        v = sign * float(field(key)[e['phaseIndex'], e['pairIndex']])
        out.append(dict(value=round(abs(v), 6), reversed=bool(v < 0 and abs(v) > 1e-9)))
    return out


# ------------------------------------------------------------------------------------------- per-code scalars
def scalarsOf(key):
    r = records[key]
    return dict(overlap=round(float(r['faceOverlap']), 4), gap=round(float(r['gap']), 4), upperDark=int(r['upperDark']), lowerDark=int(r['lowerDark']))


clippedOf = {v['key']: v['cellsClipped'] for v in placement['variants']}
sliderData = {order: dict(points=[dict(offset=p['offset'], coefficient=p['coefficient'], key=p['key'], knockout=p['knockout'], phases=phasesOf(p['key']),
                                       clipped=clippedOf.get(p['key'], 0), **scalarsOf(p['key'])) for p in points]) for order, points in placement['slider'].items()}
gridData = {}
for pairKey, rows in placement['grid'].items():
    i, j = (int(x) for x in pairKey.split('_'))
    gridData[pairKey] = dict(orders=[i, j], cells=[[dict(key=key, phases=phasesOf(key), clipped=clippedOf.get(key, 0), **scalarsOf(key)) for key in row] for row in rows])
variantData = {}
for c in placement['curated']:
    if c['key'] == 'trained':
        continue
    variantData[c['key']] = dict(label=c['label'], kind=c['kind'], phases=phasesOf(c['key']), **scalarsOf(c['key']))

allKeys = set(records)
trackedData = {key: trackedOf(key) for key in sorted(allKeys - {'trained'})}
vmemData = {key: dict(flood=records[key]['trajectory']['vmem']['flood'], clear=records[key]['trajectory']['vmem']['clear'], best=records[key]['trajectory']['vmem']['best'])
            for key in sorted(allKeys)}
states = records['trained']['trajectory']['states']
assert all(records[k]['trajectory']['states'] == states for k in allKeys)
conductanceData = dict(states=states, curves={key: {f: records[key]['trajectory'][f] for f in ('featureMean', 'featureMin', 'featureMax', 'backgroundMean', 'backgroundMin', 'backgroundMax')}
                                              for key in sorted(allKeys)})

# ------------------------------------------------------------------------------------------- the edge lanes
# One bow per (pair, phase), set in the pair's canonical direction: the flood bows one way and the clear the other, as on the face page, and a pair whose
# straight line passes close to a third node (ring top to stripe lower runs through stripe upper) bows wide, away from it.
NODE_XY = dict(ringTop=(5.5, 0.5), ringBottom=(5.5, 10.5), ringLeft=(0.5, 5.5), ringRight=(10.5, 5.5), stripeUpper=(5.5, 3.39), stripeLower=(5.5, 7.61),
               flankLeftUpper=(2.5, 3.39), flankLeftLower=(2.5, 7.61), flankRightUpper=(8.5, 3.39), flankRightLower=(8.5, 7.61))
lanes = {}
for a, b in PAIRS:
    pa, pb = np.array(NODE_XY[a]), np.array(NODE_XY[b])
    direction = pb - pa
    length = np.linalg.norm(direction)
    blocker = None
    for name, p in NODE_XY.items():
        if name in (a, b):
            continue
        t = float(np.dot(p - pa, direction) / length ** 2)
        if 0.05 < t < 0.95 and np.linalg.norm(pa + t * direction - p) < 0.75:
            blocker = (name, np.cross(direction, p - pa))
    if blocker:
        side = -1.0 if blocker[1] > 0 else 1.0                                     # bow to the side the blocking node is not on
        lanes[f'{a}|{b}'] = dict(flood=round(2.4 * side, 2), clear=round(-2.4 * side, 2))
edgeLanes = dict(default=dict(flood=-0.9, clear=0.9, write=-2.2), lanes=lanes)

page = dict(trainedCoefficients=placement['trainedCoefficients'], pairs=[list(p) for p in PAIRS], phases=PHASE_NAMES,
            trainedTop3Phases=phasesOf('trained'), trainedEdges=trainedEdges, trackedData=trackedData, sliderData=sliderData, gridData=gridData, variantData=variantData,
            vmemData=vmemData, conductanceData=conductanceData, edgeLanes=edgeLanes, trained=scalarsOf('trained'),
            sliderOffsets=placement['sliderOffsets'], gridOffsets=placement['gridOffsets'])
text = json.dumps(page, separators=(',', ':')).replace('"from_"', '"from"')
open(args.outputPath, 'w').write(text)
print(f'wrote {args.outputPath} ({len(text) / 1e6:.2f} MB): {len(allKeys)} codes, {len(sliderData)} slider orders, {len(gridData)} grid pairs, '
      f'{len(trainedEdges)} tracked edges, {len(lanes)} custom lanes')
for e in trainedEdges:
    print(f"   {e['phase']:6s} {e['from_']} -> {e['to']} {e['value']:.4f}   {e['trend']}")
