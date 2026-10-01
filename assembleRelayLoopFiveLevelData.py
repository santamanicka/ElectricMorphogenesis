"""Assembles the Relay Loop artifact's two-sided SLIDER_DATA and 5 x 5 GRID_DATA JS objects (plus the tracked-edge,
Vmem and conductance blocks for every ring code they reference) from buildBoundaryHarmonicRingCodeFiveLevelGrid11x11.py's
placement maps and the per-code outputs of processFiveLevelResults.sh (new codes) and of the earlier slider/grid and
variant pipelines (reused codes, which share the same file naming by key).

Writes sliderData.json, gridData.json, trackedData.json, vmemData.json, conductanceData.json, variantData.json (the eight
curated single-mode codes) and trainedTop3Phases.json into --outputDirectory, ready for buildRelayLoopArtifact.py to splice
into the page template.

    python3 assembleRelayLoopFiveLevelData.py --outputDirectory <dir>
"""
import argparse
import json
import os

SUFFIX = '1888Hold301FaceMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--outputDirectory', type=str, default='/tmp')
args = parser.parse_args()

nameToKey = {'ring top': 'ringTop', 'ring bottom': 'ringBottom', 'ring left': 'ringLeft', 'ring right': 'ringRight',
             'eyes': 'eyes', 'nose': 'nose', 'mouth': 'mouth', 'background top-left': 'bgTL', 'background top-right': 'bgTR',
             'background bottom-left': 'bgBL', 'background bottom-right': 'bgBR'}
# the trained code's own eleven (sender, receiver, phase), in TRAINED_EDGES' order in the artifact -- the fixed lens
# every other code's tracked values are read against.
TRAINED_PAIRS = [
    ('ringTop', 'bgTL', 'flood'), ('mouth', 'ringBottom', 'flood'),
    ('bgTL', 'eyes', 'clear'), ('bgTL', 'ringLeft', 'clear'), ('bgTL', 'nose', 'clear'),
    ('ringBottom', 'bgBL', 'write'), ('bgBL', 'bgTL', 'write'), ('ringLeft', 'bgBL', 'write'),
    ('eyes', 'bgTL', 'write'), ('nose', 'bgTL', 'write'), ('bgBL', 'mouth', 'write'),
]


def loadPhases(key):
    d = json.load(open(f'data/boundaryHarmonicRelayVariantPhaseSummary_{key}{SUFFIX}.json'))
    return {phase: [dict(from_=nameToKey[e['a']], to=nameToKey[e['b']], value=e['value'], mirrored=e['mirrored'])
                    for e in edges] for phase, edges in d['phases'].items()}


def loadTracked(key):
    """The eleven, in TRAINED_PAIRS' order, as {value, reversed} -- reversed marks a pair whose net flow runs the
    opposite way from the trained code's."""
    d = json.load(open(f'data/boundaryHarmonicRelayTrackedEdges_{key}{SUFFIX}.json'))
    out = []
    for (a, b, phase), entry in zip(TRAINED_PAIRS, d['tracked']):
        assert entry['phase'] == phase
        out.append(dict(value=entry['value'], reversed=(nameToKey[entry['a']], nameToKey[entry['b']]) != (a, b)))
    return out


placement = json.load(open(f'data/boundaryHarmonicRingCodeFiveLevel{SUFFIX}.json'))
trainedPhases = loadPhases('trained')
phasesOf = lambda key: trainedPhases if key == 'trained' else loadPhases(key)

clippedOf = {v['key']: v['cellsClipped'] for v in placement['variants']}         # ring cells the [0, 2] clip touched; a reused code was never clipped
sliderData = {order: dict(points=[dict(multiplier=p['multiplier'], coefficient=p['coefficient'], key=p['key'], phases=phasesOf(p['key']),
                                       clipped=clippedOf.get(p['key'], 0)) for p in points]) for order, points in placement['slider'].items()}
gridData = {}
for pairKey, rows in placement['grid'].items():
    i, j = (int(x) for x in pairKey.split('_'))
    gridData[pairKey] = dict(orders=[i, j], cells=[[dict(key=key, phases=phasesOf(key), clipped=clippedOf.get(key, 0)) for key in row] for row in rows])

# the single-mode dropdown's eight curated codes, knockouts first, each with its label, kind and trained-minus-baseline selectivity
variantList = {v['key']: v for v in json.load(open(f'data/boundaryHarmonicRingCodeVariants{SUFFIX}.json'))['variants']}
variantData = {}
for key in sorted(variantList, key=lambda k: (variantList[k]['kind'] != 'knockout', k)):
    movie = json.load(open(f'data/boundaryHarmonicRelayVariantMovie_{key}{SUFFIX}.json'))
    variantData[key] = dict(label=variantList[key]['label'], kind=variantList[key]['kind'],
                            gap=round(movie['difference']['selectivity'], 4), phases=loadPhases(key))

allKeys = {p['key'] for s in sliderData.values() for p in s['points']}
allKeys |= {c['key'] for g in gridData.values() for row in g['cells'] for c in row}
singleModeKeys = [v['key'] for v in json.load(open(f'data/boundaryHarmonicRingCodeVariants{SUFFIX}.json'))['variants']]
allKeys |= set(singleModeKeys)
trackedData = {key: loadTracked(key) for key in sorted(allKeys - {'trained'})}   # trained is TRAINED_EDGES itself


def merged(old, new, wanted):
    """Per-key blocks from the earlier file and the five-level file; a key that is in both takes the five-level one."""
    blocks = {**json.load(open(old))['curves' if 'Conductance' in old else 'snapshots'], **json.load(open(new))['curves' if 'Conductance' in new else 'snapshots']}
    missing = [k for k in wanted if k not in blocks]
    assert not missing, missing
    return {k: blocks[k] for k in sorted(wanted)}


vmemData = merged(f'data/boundaryHarmonicRelayVmemSnapshots{SUFFIX}.json', f'data/boundaryHarmonicRelayVmemSnapshotsFiveLevel{SUFFIX}.json', allKeys)
bestMoment = json.load(open(f'data/boundaryHarmonicRelayBestMomentVmem{SUFFIX}.json'))['snapshots']      # the frame at the trained code's scored iteration (2173)
assert all(k in bestMoment for k in vmemData), [k for k in vmemData if k not in bestMoment]
for key in vmemData:
    vmemData[key] = dict(vmemData[key], best=bestMoment[key])
conductance = json.load(open(f'data/boundaryHarmonicRelayConductanceCurves{SUFFIX}.json'))
conductanceData = dict(states=conductance['states'],
                       curves=merged(f'data/boundaryHarmonicRelayConductanceCurves{SUFFIX}.json',
                                     f'data/boundaryHarmonicRelayConductanceCurvesFiveLevel{SUFFIX}.json', allKeys))
assert conductanceData['states'] == json.load(open(f'data/boundaryHarmonicRelayConductanceCurvesFiveLevel{SUFFIX}.json'))['states']

os.makedirs(args.outputDirectory, exist_ok=True)
# json.dump can't use the bare word "from" as a Python dict key, so loadPhases() used "from_"; fix the key back on the way out
for name, payload in (('sliderData', sliderData), ('gridData', gridData), ('trackedData', trackedData), ('vmemData', vmemData), ('conductanceData', conductanceData),
                      ('variantData', variantData), ('trainedTop3Phases', trainedPhases)):
    open(f'{args.outputDirectory}/{name}.json', 'w').write(json.dumps(payload, separators=(',', ':')).replace('"from_"', '"from"'))
print(f'wrote {args.outputDirectory}: {len(sliderData)} slider orders x {len(next(iter(sliderData.values()))["points"])} stops, '
      f'{len(gridData)} grid pairs x 5x5, {len(trackedData)} tracked codes, {len(allKeys)} codes in all')
