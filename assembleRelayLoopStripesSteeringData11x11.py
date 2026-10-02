"""The data behind the stripes report's steering lab: the stripes' counterpart of assembleRelayLoopSteeringData11x11.py. EXPLORATORY.

Every simulated ring code that is allowed to inform the lab (the page's own codes, data/relayLoopStripes/, and the sweep, data/relayLoopStripeSweep/,
merged by mergeRelayLoopSweep11x11.py --target stripesInterior) is boiled down to what the lab shows for a setting of the three orders: where the
ring's four region levels are, which signed edges are in the code's top three field transfers per phase, and how the code ends (the stripe's overlap at
iteration 504, how many cells of its upper and lower halves are dark, the selectivity gap). The page estimates a new setting by a Gaussian-kernel
average of these simulated codes in region-level space, so nothing is a fitted model: it is what the nearest simulations did.

The kernel width is chosen here by held-out blocks of region-level space (8 k-means blocks), maximising the mean AUC over the edges that occur in at
least 8% of codes; the held-out skill is stored so the page can state how far to trust it. The page reproduces the ring values exactly from the stored
cosine basis, so only the estimate, not the ring, is approximate.

The file name carries the number of codes, so adding results makes a new file and never overwrites.

    python3 assembleRelayLoopStripesSteeringData11x11.py
"""
import argparse
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import boundaryCodeUtilities as boundary
import relayLoopNets

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--placementPath', type=str, default=f'data/boundaryHarmonicRingCodeStripeLevels{SUFFIX}.json')
parser.add_argument('--pageRecordDirectory', type=str, default='data/relayLoopStripes')
parser.add_argument('--sweepNetsPath', type=str, default=f'data/relayLoopSweepNets{SUFFIX}.json')
parser.add_argument('--outputDirectory', type=str, default='data')
args = parser.parse_args()

placement = json.load(open(args.placementPath))
trainedCoefficients = np.asarray(placement['trainedCoefficients'], float)
NUM_ORDERS = len(trainedCoefficients)
ringAngles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(ringAngles, np.arange(NUM_ORDERS)))                           # ring cell, order
foldedAngle = np.abs(np.angle(np.exp(1j * ringAngles)))
ringBin = np.digitize(foldedAngle, [np.pi / 4, np.pi / 2, 3 * np.pi / 4])
REGIONS = ['top', 'upper sides', 'lower sides', 'bottom']
BANDWIDTHS = (0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25)
MIN_PREVALENCE, FORMED = 0.08, 0.9
layout = relayLoopNets.layoutFor('stripesInterior')
PHASE_NAMES = ['flood', 'clear']

# ---------------------------------------------------------------- the codes
sweep = json.load(open(args.sweepNetsPath))
assert [tuple(p) for p in sweep['pairs']] == [tuple(p) for p in layout.PAIRS]
codeList = []
for variant in placement['variants']:
    r = json.load(open(f'{args.pageRecordDirectory}/{variant["key"]}.json'))
    codeList.append(dict(key=r['key'], source='page', coefficients=r['multipliers'], overlap=r['faceOverlap'], gap=r['gap'], upper=r['upperDark'], lower=r['lowerDark'],
                         field=np.array(r['field'])))
for key in sorted(sweep['codes']):
    c = sweep['codes'][key]
    codeList.append(dict(key=key, source='sweep', coefficients=c['multipliers'], overlap=c['faceOverlap'], gap=c['gap'], upper=c['upperDark'], lower=c['lowerDark'],
                         field=np.array(c['field'])))
count = len(codeList)
coefficients = np.array([c['coefficients'] for c in codeList])


def levelsOf(rows):
    values = np.clip(basis @ np.atleast_2d(rows).T, 0.0, 2.0).T
    return np.array([[row[ringBin == b].mean() for b in range(4)] for row in values])


levels = levelsOf(coefficients)


def topEdges(field):
    chosen = set()
    for phase in range(len(PHASE_NAMES)):
        for index in np.argsort(-np.abs(field[phase]))[:3]:
            a, b = layout.PAIRS[index]
            chosen.add((PHASE_NAMES[phase], a, b) if field[phase, index] > 0 else (PHASE_NAMES[phase], b, a))
    return chosen


tops = [topEdges(c['field']) for c in codeList]
edgeCounts = {}
for chosen in tops:
    for edge in chosen:
        edgeCounts[edge] = edgeCounts.get(edge, 0) + 1
edges = sorted(e for e, n in edgeCounts.items() if n / count >= 0.04)
edgeId = {e: i for i, e in enumerate(edges)}
overlap = np.array([c['overlap'] for c in codeList])
gap = np.array([c['gap'] for c in codeList])
upper = np.array([c['upper'] for c in codeList])
lower = np.array([c['lower'] for c in codeList])
formed = (overlap >= FORMED).astype(int)
PUSH = [('clear', 'ringTop', 'stripeUpper'), ('clear', 'ringBottom', 'stripeLower')]
LEAK = [('clear', 'stripeUpper', 'flankLeftUpper'), ('clear', 'stripeLower', 'flankLeftLower')]
moduleFlag = lambda names: np.array([any(e in t for e in names) for t in tops]).astype(int)
pushFlag, leakFlag = moduleFlag(PUSH), moduleFlag(LEAK)

# ---------------------------------------------------------------- the kernel width, by held-out blocks of region-level space
blocks = KMeans(8, n_init=10, random_state=0).fit_predict(levels)
distanceSquared = ((levels[:, None, :] - levels[None, :, :]) ** 2).sum(2)
labels = {e: np.array([e in t for t in tops]).astype(int) for e in edges if edgeCounts[e] / count >= MIN_PREVALENCE}


def heldOut(bandwidth, label):
    weight = np.exp(-distanceSquared / (2 * bandwidth ** 2)) * (blocks[:, None] != blocks[None, :])
    total = weight.sum(1)
    return np.where(total > 1e-9, (weight @ label) / np.maximum(total, 1e-9), label.mean())


auc = lambda y, p: roc_auc_score(y, p) if 5 <= y.sum() <= len(y) - 5 else float('nan')
skillByBandwidth = {h: float(np.nanmean([auc(label, heldOut(h, label)) for label in labels.values()])) for h in BANDWIDTHS}
bandwidth = max(skillByBandwidth, key=skillByBandwidth.get)
edgeSkill = {f'{e[0]} {e[1]}->{e[2]}': float(auc(label, heldOut(bandwidth, label))) for e, label in labels.items()}
skill = dict(meanEdgeAuc=skillByBandwidth[bandwidth], edgeAuc=edgeSkill, formed=float(auc(formed, heldOut(bandwidth, formed))),
             pushPresent=float(auc(pushFlag, heldOut(bandwidth, pushFlag))), leakPresent=float(auc(leakFlag, heldOut(bandwidth, leakFlag))), blocks=8)
print(f'{count} codes; kernel width {bandwidth} G_pol/G_ref (mean held-out edge AUC by width: ' + ', '.join(f'{h}: {s:.3f}' for h, s in skillByBandwidth.items()) + ')')
print(f'held-out AUC: stripe formed {skill["formed"]:.2f}, push {skill["pushPresent"]:.2f}, leak {skill["leakPresent"]:.2f}; {int(formed.sum())} of {count} codes form the stripe')

# ---------------------------------------------------------------- presets and story numbers
trainedIndex = next(i for i, c in enumerate(codeList) if c['key'] == 'trained')
# the ring's level at the top (angle 0), the bottom (angle pi) and the sides (angle pi/2), from the coefficients: T = a0 + a1 + a2, a0 - a1 + a2 and S = a0 - a2, the
# levels the stripe's window is stated in (the region levels above are means over 45-degree bins, which is what the lab's kernel works in)
topLevel, bottomLevel, sideLevel = coefficients[:, 0] + coefficients[:, 1] + coefficients[:, 2], coefficients[:, 0] - coefficients[:, 1] + coefficients[:, 2], coefficients[:, 0] - coefficients[:, 2]
farFormed = [int(i) for i in np.where(formed == 1)[0] if np.linalg.norm(levels[i] - levels[trainedIndex]) > 0.05 and codeList[i]['key'] != 'trained']
gapBest = int(np.argmax(gap))
offsetsOf = lambda i: [round(float(x), 4) for x in coefficients[i] - trainedCoefficients]
presets = [dict(name='trained', offsets=[0, 0, 0]), dict(name='order 0 up 0.01', offsets=[0.01, 0, 0]), dict(name='order 0 down 0.01', offsets=[-0.01, 0, 0]),
           dict(name='order 2 up 0.02', offsets=[0, 0, 0.02]), dict(name='order 2 down 0.05', offsets=[0, 0, -0.05]), dict(name='order 1 up 0.2', offsets=[0, 0.2, 0]),
           dict(name='order 2 knocked out', offsets=[0, 0, round(-float(trainedCoefficients[2]), 4)]),
           dict(name=f'highest-gap simulated code (gap {gap[gapBest]:+.2f})', offsets=offsetsOf(gapBest))]
presets += [dict(name=f'a distant stripe-maker (overlap {overlap[i]:.2f})', offsets=offsetsOf(i)) for i in farFormed[:2]]
inWindow = (topLevel >= 1.485) & (topLevel <= 1.495) & (bottomLevel >= 1.485) & (bottomLevel <= 1.495)
story = dict(codes=count, formed=int(formed.sum()), formedInWindow=int((formed & inWindow).sum()), inWindow=int(inWindow.sum()),
             formedOutsideWindow=int((formed & ~inWindow).sum()), window=[1.485, 1.495],
             formedLevels=dict(top=[round(float(topLevel[formed == 1].min()), 3), round(float(topLevel[formed == 1].max()), 3)] if formed.any() else None,
                               bottom=[round(float(bottomLevel[formed == 1].min()), 3), round(float(bottomLevel[formed == 1].max()), 3)] if formed.any() else None,
                               side=[round(float(sideLevel[formed == 1].min()), 3), round(float(sideLevel[formed == 1].max()), 3)] if formed.any() else None),
             upperFormedOnly=int(((upper >= 11) & (lower < 11)).sum()), lowerFormedOnly=int(((lower >= 11) & (upper < 11)).sum()), bothHalves=int(((upper >= 11) & (lower >= 11)).sum()))
print('story:', story)
loadings = [[float(basis[ringBin == b, o].mean()) for o in range(NUM_ORDERS)] for b in range(4)]


def pack(values, digits=4):
    return [round(float(v), digits) for v in values]


data = dict(
    suffix=SUFFIX, codes=count, sources={s: sum(c['source'] == s for c in codeList) for s in ('page', 'sweep')},
    trainedCoefficients=pack(trainedCoefficients, 6), ringBasis=[pack(row, 5) for row in basis], ringBin=ringBin.tolist(), regionNames=REGIONS,
    regionLoadings=[pack(row, 4) for row in loadings], trainedLevels=pack(levels[trainedIndex]),
    coefficientLowest=placement['coefficientLowest'], coefficientHighest=placement['coefficientHighest'],
    edges=[list(e) for e in edges], edgePrevalence=pack([edgeCounts[e] / count for e in edges]),
    bandwidth=bandwidth, bandwidthSkill={str(h): s for h, s in skillByBandwidth.items()}, skill=skill,
    modules=dict(push=[list(e) for e in PUSH], leak=[list(e) for e in LEAK]),
    trainedTopEdges=sorted(edgeId[e] for e in tops[trainedIndex] if e in edgeId),
    columns=dict(coefficients=[pack(c, 5) for c in coefficients], levels=[pack(l, 4) for l in levels], edgeIds=[sorted(edgeId[e] for e in t if e in edgeId) for t in tops],
                 overlap=pack(overlap), gap=pack(gap), upperDark=upper.tolist(), lowerDark=lower.tolist(), push=pushFlag.tolist(), leak=leakFlag.tolist(),
                 source=[('page', 'sweep').index(c['source']) for c in codeList], key=[c['key'] for c in codeList]),
    presets=presets, story=story)
outputPath = f'{args.outputDirectory}/relayLoopSteeringData{SUFFIX}Codes{count}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
json.dump(data, open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath} ({os.path.getsize(outputPath) // 1024} KB); {len(edges)} signed edges kept; presets: ' + '; '.join(p['name'] for p in presets))
