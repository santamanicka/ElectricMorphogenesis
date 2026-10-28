"""Assembles the data blocks of the order-4 Relay Loop page ("The Fifth Order's Loop") from the order-4 page set: the 121 codes of
buildBoundaryHarmonicRingCodePageSet11x11.py (the trained code, one-order sliders and two-order grids) and their relay records in
data/relayLoopFullNets1888Hold301FaceMinus60Minus5Order4.json. EXPLORATORY: nothing here was registered.

What the order-3 page got from separate relay extractions is rebuilt here from those records and from a replay of each code on the reference tissue:
  top three transfers per phase, and the trained code's tracked edges read off every code   <- the record's field nets (28 canonical block pairs x 3 phases)
  tissue Vmem at the end of flood, clear and write, and at the scored moment                 <- the replay
  mean / lowest / highest G_pol / G_ref of the feature cells and of the other interior cells <- the replay
The grid is 4 x 4 (multipliers 0, 0.5, 1, 1.5 on each of two orders): the 1 rows and columns are the slider codes, the centre is the trained code.
The trained code's key is 'trained' here (it is 'pageTrained' in the page set) so the page's own conventions hold.

Run with BOUNDARY_FACE_ORDER=4. Writes one JSON, data/relayLoopOrder4PageData1888Hold301FaceMinus60Minus5Order4.json (never overwriting).

    BOUNDARY_FACE_ORDER=4 python3 assembleRelayLoopOrder4Data11x11.py
"""
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary
import canalizationTalkCommon as common

assert boundary.faceOrder == 4, 'set BOUNDARY_FACE_ORDER=4'
SUFFIX = '1888Hold301FaceMinus60Minus5'
outputPath = f'data/relayLoopOrder4PageData{SUFFIX}Order4.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

pageSet = json.load(open(boundary.orderPath(f'data/boundaryHarmonicRingCodePageSet{SUFFIX}.json')))
nets = json.load(open(boundary.orderPath(f'data/relayLoopFullNets{SUFFIX}.json')))
trained = np.asarray(pageSet['trainedCoefficients'], float)
numOrders = len(trained)
pairs = [tuple(p) for p in nets['pairs']]
phases = nets['phases']
MIRRORED = {'ringLeft', 'bgTL', 'bgBL'}                                   # canonical blocks that have a mirror twin (the right-hand blocks are not stored)
SLIDER_MULTIPLIERS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
GRID_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5]
label = lambda m: f'{int(round(m * 100)):03d}'
codes = {('trained' if v['key'] == 'pageTrained' else v['key']): v for v in pageSet['variants']}
record = {('trained' if k == 'pageTrained' else k): c for k, c in nets['codes'].items()}
assert set(codes) == set(record), set(codes) ^ set(record)


def sliderKey(order, multiplier):
    return 'trained' if multiplier == 1.0 else f'pageSlider_o{order}_m{label(multiplier)}'


def gridKey(i, j, mi, mj):
    if mi == 1.0 and mj == 1.0:
        return 'trained'
    if mi == 1.0:
        return sliderKey(j, mj)
    if mj == 1.0:
        return sliderKey(i, mi)
    return f'pageGrid_o{i}o{j}_m{label(mi)}_m{label(mj)}'


basis = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(numOrders)))


def clippedOf(multipliers):
    raw = basis @ (np.asarray(multipliers) * trained)
    return int((raw < 0).sum() + (raw > 2).sum())


def topEdges(key, count=3):
    """{phase: [{from, to, value, mirrored}]}: the biggest net field transfers of each phase, direction by sign."""
    out = {}
    for p, phase in enumerate(phases):
        values = np.array(record[key]['field'][p])
        edges = []
        for index in np.argsort(-np.abs(values))[:count]:
            a, b = pairs[index]
            sender, receiver = (a, b) if values[index] > 0 else (b, a)
            edges.append(dict(from_=sender, to=receiver, value=round(abs(float(values[index])), 5), mirrored=bool(a in MIRRORED or b in MIRRORED)))
        out[phase] = edges
    return out


# the trained code's tracked edges: its four biggest transfers of each phase, in phase order
TRACKED_PER_PHASE = 4
trainedField = [np.array(f) for f in record['trained']['field']]
tracked = []                                                              # (phase index, pair index, sender, receiver)
for p in range(len(phases)):
    for index in np.argsort(-np.abs(trainedField[p]))[:TRACKED_PER_PHASE]:
        a, b = pairs[index]
        tracked.append((p, int(index), a, b) if trainedField[p][index] > 0 else (p, int(index), b, a))


def trackedOf(key):
    out = []
    for p, index, sender, receiver in tracked:
        value = float(record[key]['field'][p][index])
        canonical = pairs[index]
        forward = (sender, receiver) == canonical
        out.append(dict(value=round(abs(value), 5), reversed=bool((value > 0) != forward)))
    return out


trainedEdges = []
for rank, (p, index, sender, receiver) in enumerate(tracked):
    others = []
    for q, other in enumerate(phases):
        if q == p:
            continue
        v = float(trainedField[q][index]) * (1 if pairs[index] == (sender, receiver) else -1)
        if abs(v) >= 0.02:
            others.append(f'{other} {v:+.2f}')
    within = [i for i, t in enumerate(tracked) if t[0] == p].index(rank) + 1
    trend = f'rank {within} of {phases[p]}' + (f'; the same pair in {", ".join(others)} (minus: it runs the other way)' if others else '; small in the other phases')
    trainedEdges.append(dict(from_=sender, to=receiver, phase=phases[p], value=round(abs(float(trainedField[p][index])), 5), bow=1, trend=trend))

# ---------------------------------------------------------------- replays: tissue snapshots and conductance curves
landmarks = boundary.faceLandmarkStates                                    # states; recorded iteration = state - 1
recorded = dict(flood=landmarks['release'] - 1, clear=landmarks['trough'] - 1, write=landmarks['peak'] - 1)
scored = landmarks['scored'] - 1
conductanceStates = list(range(0, landmarks['peak'], 10)) + [landmarks['peak']]
replayer = common.Replayer()
FEATURE = common.FACE_CELLS
OTHER = np.array([c for c in common.INTERIOR if c not in set(FEATURE.tolist())])
vmemData, conductanceCurves = {}, {}
for number, (key, variant) in enumerate(sorted(codes.items())):
    vmem, gpol = replayer.run(np.asarray(variant['ringValues'], float), scored + 1)
    vmemData[key] = {phase: np.round(vmem[it], 1).tolist() for phase, it in recorded.items()}
    vmemData[key]['best'] = np.round(vmem[scored], 1).tolist()
    rows = np.vstack([np.ones((1, gpol.shape[1])), gpol])                  # row s = after s steps (row 0 is the initial G_ref)
    states = rows[conductanceStates]
    curves = {}
    for group, cells in (('feature', FEATURE), ('background', OTHER)):
        block = states[:, cells]
        for stat in ('Mean', 'Min', 'Max'):
            curves[f'{group}{stat}'] = np.round(getattr(block, stat.lower())(1), 4).tolist()
    conductanceCurves[key] = curves
    if number % 20 == 0:
        print(f'replayed {number + 1} of {len(codes)}')

# ---------------------------------------------------------------- the page's blocks
sliderData = {str(o): dict(points=[dict(multiplier=m, coefficient=round(float(m * trained[o]), 6), key=sliderKey(o, m), phases=topEdges(sliderKey(o, m)),
                                        clipped=clippedOf([m if i == o else 1.0 for i in range(numOrders)])) for m in SLIDER_MULTIPLIERS]) for o in range(numOrders)}
gridData = {}
for i in range(numOrders):
    for j in range(i + 1, numOrders):
        cells = [[dict(key=gridKey(i, j, mi, mj), phases=topEdges(gridKey(i, j, mi, mj)),
                       clipped=clippedOf([mi if o == i else mj if o == j else 1.0 for o in range(numOrders)])) for mj in GRID_MULTIPLIERS] for mi in GRID_MULTIPLIERS]
        gridData[f'{i}_{j}'] = dict(orders=[i, j], cells=cells)
variantData = {}
for o in range(numOrders):
    for m, text, kind in ((0.0, 'knocked out', 'knockout'), (2.0, 'doubled', 'steering')):
        key = sliderKey(o, m)
        variantData[key] = dict(label=f'order {o} {text}', kind=kind, gap=round(record[key]['gap'], 4), phases=topEdges(key))
keysUsed = set(vmemData)
data = dict(
    config=dict(numOrders=numOrders, coefficients=[round(float(c), 6) for c in trained], phaseEnd=recorded, scored=scored, gridMultipliers=GRID_MULTIPLIERS, sliderMultipliers=SLIDER_MULTIPLIERS,
                trackedCount=len(tracked), trainedGap=record['trained']['gap'], trainedFaceOverlap=record['trained']['faceOverlap']),
    trainedEdges=trainedEdges, trainedTop3Phases=topEdges('trained'), sliderData=sliderData, gridData=gridData, variantData=variantData,
    trackedData={key: trackedOf(key) for key in sorted(keysUsed - {'trained'})}, vmemData=vmemData,
    conductanceData=dict(states=conductanceStates, curves=conductanceCurves),
    gapOf={key: record[key]['gap'] for key in sorted(keysUsed)}, faceOverlapOf={key: record[key]['faceOverlap'] for key in sorted(keysUsed)})
text = json.dumps(data, separators=(',', ':')).replace('"from_"', '"from"')
open(outputPath, 'w').write(text)
print(f'wrote {outputPath} ({len(text) / 1e6:.2f} MB): {len(sliderData)} slider orders x {len(SLIDER_MULTIPLIERS)} stops, {len(gridData)} grid pairs x {len(GRID_MULTIPLIERS)}^2, '
      f'{len(tracked)} tracked edges, {len(keysUsed)} codes; trained gap {record["trained"]["gap"]:.3f}, face overlap {record["trained"]["faceOverlap"]:.2f}')
