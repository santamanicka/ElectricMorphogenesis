"""Assembles the Relay Loop artifact's SLIDER_DATA and GRID_DATA JS objects from every phase-summary JSON
(computeBoundaryHarmonicRelayVariantPhaseSummary11x11.py's output): the trained code, the three single-order
knockouts (reused as each order's own slider t=1, except order 0's, which has no prior knockout and uses the new
slider_o0_t100), and the 37 new slider/grid points. Grid row 0 / column 0 (one order still at t=0) is filled in
from that OTHER order's own slider, not recomputed -- see buildBoundaryHarmonicRingCodeSliderGrid11x11.py.

Writes /tmp/sliderData.json and /tmp/gridData.json, ready to splice into the artifact's __SLIDER_DATA__ /
__GRID_DATA__ placeholders.

    python3 assembleRelayLoopSliderGridData.py
"""
import json

nameToKey = {'ring top': 'ringTop', 'ring bottom': 'ringBottom', 'ring left': 'ringLeft', 'ring right': 'ringRight',
             'eyes': 'eyes', 'nose': 'nose', 'mouth': 'mouth', 'background top-left': 'bgTL', 'background top-right': 'bgTR',
             'background bottom-left': 'bgBL', 'background bottom-right': 'bgBR'}


def loadPhases(variantKey):
    d = json.load(open(f'data/boundaryHarmonicRelayVariantPhaseSummary_{variantKey}1888Hold301FaceMinus60Minus5.json'))
    return {phase: [dict(from_=nameToKey[e['a']], to=nameToKey[e['b']], value=e['value'], mirrored=e['mirrored'])
                    for e in edges] for phase, edges in d['phases'].items()}


def gapOf(variantKey):
    if variantKey == 'trained':
        d = json.load(open('data/boundaryHarmonicRingOnlyRelay1888Hold301FaceMinus60Minus5.json'))
    else:
        d = json.load(open(f'data/boundaryHarmonicRelayVariantMovie_{variantKey}1888Hold301FaceMinus60Minus5.json'))
    return round(d['difference']['selectivity'], 4)


def loadTracked(variantKey):
    """computeBoundaryHarmonicRelayTrackedEdges11x11.py's own eleven, in TRAINED_EDGES' fixed order, as
    {value, reversed} -- reversed marks a pair whose net flow runs the opposite way from the trained code's."""
    d = json.load(open(f'data/boundaryHarmonicRelayTrackedEdges_{variantKey}1888Hold301FaceMinus60Minus5.json'))
    out = []
    for (a, b, phase), entry in zip(TRAINED_PAIRS, d['tracked']):
        assert entry['phase'] == phase
        reversed_ = (nameToKey[entry['a']], nameToKey[entry['b']]) != (a, b)
        out.append(dict(value=entry['value'], reversed=reversed_))
    return out


# the trained code's own eleven (sender, receiver, phase), in TRAINED_EDGES' order in the artifact -- the fixed
# lens every other code's tracked values are read against.
TRAINED_PAIRS = [
    ('ringTop', 'bgTL', 'flood'), ('mouth', 'ringBottom', 'flood'),
    ('bgTL', 'eyes', 'clear'), ('bgTL', 'ringLeft', 'clear'), ('bgTL', 'nose', 'clear'),
    ('ringBottom', 'bgBL', 'write'), ('bgBL', 'bgTL', 'write'), ('ringLeft', 'bgBL', 'write'),
    ('eyes', 'bgTL', 'write'), ('nose', 'bgTL', 'write'), ('bgBL', 'mouth', 'write'),
]

trainedPhases = loadPhases('trained')
KNOCKOUT_REUSE = {1: 'knockoutOrder1', 2: 'knockoutOrder2', 3: 'knockoutOrder3'}   # order -> existing t=1 phase summary

# ---------------------------------------------------------------- slider: order -> sorted list of {t, coefficient, phases}
coefficients = json.load(open('data/boundaryHarmonicRingCodeSliderGrid1888Hold301FaceMinus60Minus5.json'))['trainedCoefficients']
sliderVariants = {v['key']: v for v in json.load(open('data/boundaryHarmonicRingCodeSliderGrid1888Hold301FaceMinus60Minus5.json'))['variants']}

sliderData = {}
for order in range(4):
    points = [dict(t=0.0, coefficient=round(coefficients[order], 6), phases=trainedPhases, key='trained')]
    for t in (0.25, 0.5, 0.75):
        key = f'slider_o{order}_t{int(round(t * 100)):03d}'
        v = sliderVariants[key]
        points.append(dict(t=t, coefficient=v['coefficient'], phases=loadPhases(key), key=key))
    if order == 0:
        v = sliderVariants['slider_o0_t100']
        points.append(dict(t=1.0, coefficient=v['coefficient'], phases=loadPhases('slider_o0_t100'), key='slider_o0_t100'))
    else:
        points.append(dict(t=1.0, coefficient=0.0, phases=loadPhases(KNOCKOUT_REUSE[order]), key=KNOCKOUT_REUSE[order]))
    sliderData[str(order)] = dict(points=points)

# ---------------------------------------------------------------- grid: pair -> 3x3 cells (trained / half / knock on each axis)
gridData = {}
for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
    pairKey = f'{i}_{j}'
    cells = [[None, None, None] for _ in range(3)]
    cells[0][0] = dict(phases=trainedPhases, gap=gapOf('trained'), key='trained')
    cells[1][0] = dict(phases=sliderData[str(i)]['points'][2]['phases'], gap=None, key=sliderData[str(i)]['points'][2]['key'])
    cells[2][0] = dict(phases=sliderData[str(i)]['points'][4]['phases'], gap=None, key=sliderData[str(i)]['points'][4]['key'])
    cells[0][1] = dict(phases=sliderData[str(j)]['points'][2]['phases'], gap=None, key=sliderData[str(j)]['points'][2]['key'])
    cells[0][2] = dict(phases=sliderData[str(j)]['points'][4]['phases'], gap=None, key=sliderData[str(j)]['points'][4]['key'])
    for ri, rlevel in ((1, 'half'), (2, 'knock')):
        for ci, clevel in ((1, 'half'), (2, 'knock')):
            key = f'grid_o{i}o{j}_r{rlevel}_c{clevel}'
            cells[ri][ci] = dict(phases=loadPhases(key), gap=gapOf(key), key=key)
    # fill in the gaps for the reused row0/col0 cells too, from the movie each already has
    cells[1][0]['gap'] = gapOf(f'slider_o{i}_t050')
    cells[2][0]['gap'] = gapOf('slider_o0_t100') if i == 0 else gapOf(KNOCKOUT_REUSE[i])
    cells[0][1]['gap'] = gapOf(f'slider_o{j}_t050')
    cells[0][2]['gap'] = gapOf('slider_o0_t100') if j == 0 else gapOf(KNOCKOUT_REUSE[j])
    gridData[pairKey] = dict(orders=[i, j], cells=cells)

# ---------------------------------------------------------------- tracked: every dataset key's eleven, fixed order
allKeys = set()
for order, s in sliderData.items():
    for p in s['points']:
        allKeys.add(p['key'])
for pairKey, g in gridData.items():
    for row in g['cells']:
        for c in row:
            allKeys.add(c['key'])
allKeys.update(['knockoutOrder1', 'knockoutOrder2', 'knockoutOrder3', 'knockoutOrders123',
                'steerOrder0', 'steerOrder1', 'steerOrder2', 'steerOrder3'])   # the single-mode dropdown's own 8
allKeys.discard('trained')                                     # trained is TRAINED_EDGES itself; no file to load
trackedData = {key: loadTracked(key) for key in sorted(allKeys)}

# json.dump can't use the bare word "from" as a Python dict key, so loadPhases() used "from_"; fix the key back
# on the way out, exactly as the single-variant assembly does for VARIANT_DATA.
open('/tmp/sliderData.json', 'w').write(json.dumps(sliderData, separators=(',', ':')).replace('"from_"', '"from"'))
open('/tmp/gridData.json', 'w').write(json.dumps(gridData, separators=(',', ':')).replace('"from_"', '"from"'))
open('/tmp/trackedData.json', 'w').write(json.dumps(trackedData, separators=(',', ':')))
print('wrote /tmp/sliderData.json, /tmp/gridData.json and /tmp/trackedData.json')
print('slider orders:', list(sliderData.keys()))
print('grid pairs:', list(gridData.keys()))
print('tracked keys:', len(trackedData))
