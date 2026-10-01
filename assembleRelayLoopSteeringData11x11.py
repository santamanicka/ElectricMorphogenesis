"""The data behind the Relay Loop page's steering lab: every simulated ring code that is allowed to inform it (the page's own 126
codes, the 400 sweep codes, and -- with --includeConfirmation -- the confirmatory test's codes), boiled down to what the lab shows
for a setting of the four orders: where the ring's region levels are, which signed edges are in the code's top three per phase, which
modules are present, and how the code ends (face overlap, selectivity gap). The page estimates a new setting by a Gaussian-kernel
average of these simulated codes in region-level space, so nothing is a fitted model: it is what the nearest simulations did.

The kernel width is chosen here by held-out blocks of region-level space (8 k-means blocks), maximising the mean AUC over the edges that
occur in at least 8% of codes; the held-out skill is stored so the page can state how far to trust it. The page reproduces the ring
values exactly from the stored cosine basis, so only the estimate, not the ring, is approximate.

The file name carries the number of codes, so adding results (a new sweep, the confirmation) makes a new file and never overwrites;
buildRelayLoopArtifact.py uses the one with the most codes. Story numbers are read from the exploratory analyses' JSONs.

    python3 assembleRelayLoopSteeringData11x11.py [--includeConfirmation]
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301FaceMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--includeConfirmation', action='store_true', help='also use the confirmatory test\'s codes (only after it has been scored)')
args = parser.parse_args()

trainedCoefficients = np.asarray(json.load(open(f'data/boundaryHarmonicRingCodeFiveLevel{SUFFIX}.json'))['trainedCoefficients'])
ringAngles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(ringAngles, np.arange(4)))                                    # ring cell, order
foldedAngle = np.abs(np.angle(np.exp(1j * ringAngles)))
ringBin = np.digitize(foldedAngle, [np.pi / 4, np.pi / 2, 3 * np.pi / 4])
REGIONS = ['top', 'upper sides', 'lower sides', 'bottom']
BANDWIDTHS = (0.05, 0.08, 0.12, 0.18, 0.25, 0.35)
MIN_PREVALENCE, FACE_LIKE = 0.08, 0.3

# ---------------------------------------------------------------- the codes
sources = [('page', json.load(open(f'data/relayLoopFullNets{SUFFIX}.json'))), ('sweep', json.load(open(f'data/relayLoopSweepNets{SUFFIX}.json')))]
codeList = []
pairs = phases = None
for name, nets in sources:
    pairs, phases = [tuple(p) for p in nets['pairs']], nets['phases']
    for key in sorted(nets['codes']):
        c = nets['codes'][key]
        codeList.append(dict(key=key, source=name, multipliers=c['multipliers'], faceOverlap=c['faceOverlap'], gap=c['gap'], field=np.array(c['field'])))
if args.includeConfirmation:
    variants = {v['key']: v for v in json.load(open(f'data/boundaryHarmonicRingCodeModulesConfirmation{SUFFIX}.json'))['variants']}
    for path in sorted(glob.glob('data/relayLoopModulesConfirmation/*.json')):
        r = json.load(open(path))
        codeList.append(dict(key=r['key'], source='confirmation', multipliers=r['multipliers'], faceOverlap=r['faceOverlap'], gap=r['gap'], field=np.array(r['field'])))
count = len(codeList)
multipliers = np.array([c['multipliers'] for c in codeList])


def levelsOf(multiplierRows):
    values = np.clip(basis @ (np.atleast_2d(multiplierRows) * trainedCoefficients).T, 0.0, 2.0).T
    return np.array([[row[ringBin == b].mean() for b in range(4)] for row in values])


levels = levelsOf(multipliers)


def topEdges(field):
    chosen = set()
    for phase in range(3):
        for index in np.argsort(-np.abs(field[phase]))[:3]:
            a, b = pairs[index]
            chosen.add((phases[phase], a, b) if field[phase, index] > 0 else (phases[phase], b, a))
    return chosen


tops = [topEdges(c['field']) for c in codeList]
edgeCounts = {}
for chosen in tops:
    for edge in chosen:
        edgeCounts[edge] = edgeCounts.get(edge, 0) + 1
edges = sorted(e for e, n in edgeCounts.items() if n / count >= 0.04)
edgeId = {e: i for i, e in enumerate(edges)}
LOWER = [('write', 'ringBottom', 'bgBL'), ('write', 'ringLeft', 'bgBL')]
PUSH = [('flood', 'ringTop', 'bgTL')]
REVERSED = [('write', 'bgBL', 'ringBottom'), ('clear', 'ringBottom', 'mouth'), ('flood', 'ringBottom', 'bgBL')]
moduleFlag = lambda names: np.array([any(e in t for e in names) for t in tops]).astype(int)
lowerFlag, pushFlag, reversedFlag = moduleFlag(LOWER), moduleFlag(PUSH), moduleFlag(REVERSED)
face = np.array([c['faceOverlap'] for c in codeList])
gap = np.array([c['gap'] for c in codeList])

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
moduleSkill = dict(lowerChannel=float(auc(lowerFlag, heldOut(bandwidth, lowerFlag))), floodPush=float(auc(pushFlag, heldOut(bandwidth, pushFlag))),
                   reversedLower=float(auc(reversedFlag, heldOut(bandwidth, reversedFlag))))
faceLikeSkill = float(auc((face >= FACE_LIKE).astype(int), heldOut(bandwidth, (face >= FACE_LIKE).astype(int))))
print(f'{count} codes; kernel width {bandwidth} G_pol/G_ref (mean held-out edge AUC by width: ' + ', '.join(f'{h}: {s:.3f}' for h, s in skillByBandwidth.items()) + ')')
print(f'held-out AUC: modules {moduleSkill}, face-like {faceLikeSkill:.2f}')

# ---------------------------------------------------------------- presets and story numbers
trainedIndex = next(i for i, c in enumerate(codeList) if c['key'] == 'trained')
gapBest = int(np.argmax(gap))
farFaces = [i for i in np.where(face >= 0.5)[0] if np.linalg.norm(levels[i] - levels[trainedIndex]) > 0.2]
presets = [dict(name='trained', multipliers=[1, 1, 1, 1]), dict(name='order 0 up 20%', multipliers=[1.2, 1, 1, 1]), dict(name='order 0 down 20%', multipliers=[0.8, 1, 1, 1]),
           dict(name='order 1 up 50%', multipliers=[1, 1.5, 1, 1]), dict(name='order 1 down 25%', multipliers=[1, 0.75, 1, 1]), dict(name='order 3 knocked out', multipliers=[1, 1, 1, 0]),
           dict(name='highest-gap simulated code', multipliers=[round(float(x), 3) for x in multipliers[gapBest]])]
presets += [dict(name=f'a distant face-maker (overlap {face[i]:.2f})', multipliers=[round(float(x), 3) for x in multipliers[i]]) for i in farFaces[:2]]
story = {name: json.load(open(f'data/relayLoopOrderEdge{name}{SUFFIX}.json')) for name in ('Map', 'Story') if os.path.exists(f'data/relayLoopOrderEdge{name}{SUFFIX}.json')}
confirmationPath = f'data/relayLoopModulesConfirmation{SUFFIX}.json'
confirmation = json.load(open(confirmationPath)) if args.includeConfirmation and os.path.exists(confirmationPath) else None
loadings = [[float(basis[ringBin == b, o].mean()) for o in range(4)] for b in range(4)]


def pack(values, digits=4):
    return [round(float(v), digits) for v in values]


data = dict(
    suffix=SUFFIX, codes=count, sources={s: sum(c['source'] == s for c in codeList) for s in ('page', 'sweep', 'confirmation')},
    trainedCoefficients=pack(trainedCoefficients, 6), ringBasis=[pack(row, 5) for row in basis], ringBin=ringBin.tolist(), regionNames=REGIONS,
    regionLoadings=[pack(row, 4) for row in loadings], trainedLevels=pack(levels[trainedIndex]),
    edges=[list(e) for e in edges], edgePrevalence=pack([edgeCounts[e] / count for e in edges]),
    bandwidth=bandwidth, bandwidthSkill={str(h): s for h, s in skillByBandwidth.items()},
    skill=dict(meanEdgeAuc=skillByBandwidth[bandwidth], edgeAuc=edgeSkill, **moduleSkill, faceLike=faceLikeSkill, blocks=8),
    modules=dict(lower=[list(e) for e in LOWER], push=[list(e) for e in PUSH], reversed=[list(e) for e in REVERSED]),
    trainedTopEdges=sorted(edgeId[e] for e in tops[trainedIndex] if e in edgeId),
    columns=dict(multipliers=[pack(m, 4) for m in multipliers], levels=[pack(l, 4) for l in levels], edgeIds=[sorted(edgeId[e] for e in t if e in edgeId) for t in tops],
                 faceOverlap=pack(face), gap=pack(gap), lower=lowerFlag.tolist(), push=pushFlag.tolist(), reversed=reversedFlag.tolist(),
                 source=[('page', 'sweep', 'confirmation').index(c['source']) for c in codeList], key=[c['key'] for c in codeList]),
    presets=presets, story=story, confirmation=confirmation)
outputPath = f'data/relayLoopSteeringData{SUFFIX}Codes{count}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
json.dump(data, open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath} ({os.path.getsize(outputPath) // 1024} KB); {len(edges)} signed edges kept; presets: ' + '; '.join(p['name'] for p in presets))
