"""EXPLORATORY search for how the causal net changes as the ring code's orders are steered. Nothing here was predicted or
registered beforehand; every number is a description of the 126 codes the Relay Loop page shows (the trained code, the
eight curated codes, the one- and two-order slider and grid codes), read from data/relayLoopFullNets...json
(extractRelayLoopFullNets11x11.py), and the choice of what to look at followed what the first look showed. With 126
codes, 84 edge-phase features and many looks, treat individual edges and small gaps as leads, not findings.

What is measured, and why two metrics. A code's net is its field-channel net transfer between every pair of the eight
canonical blocks (left and centre blocks; the right-hand ones mirror them), in each of flood, clear and write: 28 pairs x 3
phases. Two ways of comparing nets are used because the first one misled:
  shape      each phase's 28 transfers divided by its largest, compared as vectors. Counts every tiny edge, so two nets that
             share their big edges still look far apart; this made the net look jumpy under steering.
  top-3      the set of the three biggest transfers per phase (the page's own top-3 lens), compared by Jaccard similarity.
             Follows the edges a reader would see, and shows the graded structure the shape metric hid.
An edge is "on" in a phase when it is at least 10% of that phase's largest transfer.

Writes data/relayLoopSteeringStory1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 analyzeRelayLoopSteeringStory11x11.py
"""
import json
import os

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

SUFFIX = '1888Hold301FaceMinus60Minus5'
outputPath = f'data/relayLoopSteeringStory{SUFFIX}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

nets = json.load(open(f'data/relayLoopFullNets{SUFFIX}.json'))
placement = json.load(open(f'data/boundaryHarmonicRingCodeFiveLevel{SUFFIX}.json'))
pairs = [tuple(p) for p in nets['pairs']]
phaseNames = nets['phases']
keys = sorted(nets['codes'])
count = len(keys)
position = {key: i for i, key in enumerate(keys)}
trained = position['trained']
transfer = np.array([nets['codes'][k]['field'] for k in keys])                       # code, phase, pair
gap = np.array([nets['codes'][k]['gap'] for k in keys])
faceOverlap = np.array([nets['codes'][k]['faceOverlap'] for k in keys])
multiplier = np.array([nets['codes'][k]['multipliers'] for k in keys])                # code, order
steered = np.array([k.startswith('steer') for k in keys])                             # the four +-0.30 steps are not multiples of the trained value
ON_FRACTION = 0.10
shape = transfer / np.abs(transfer).max(2, keepdims=True)
isOn = np.abs(shape) >= ON_FRACTION
result = dict(status='EXPLORATORY: no predictions or decision criteria were registered; see the module docstring',
              codes=count, pairs=[list(p) for p in pairs], phases=phaseNames)


def edgeName(phase, pair, value):
    sender, receiver = (pair[0], pair[1]) if value > 0 else (pair[1], pair[0])
    return f'{phaseNames[phase]} {sender}->{receiver}'


def topEdges(code):
    chosen = set()
    for phase in range(3):
        for index in np.argsort(-np.abs(transfer[code, phase]))[:3]:
            a, b = pairs[index]
            chosen.add((phaseNames[phase], a, b) if transfer[code, phase, index] > 0 else (phaseNames[phase], b, a))
    return chosen


topSets = [topEdges(i) for i in range(count)]
retention = lambda a, b: len(topSets[a] & topSets[b]) / len(topSets[a] | topSets[b])
retentionOfTrained = np.array([retention(trained, i) for i in range(count)])

# ---------------------------------------------------------------- outcomes
result['outcomes'] = dict(
    faceOverlapAtLeast0_5=[dict(key=keys[i], multipliers=multiplier[i].round(3).tolist(), faceOverlap=float(faceOverlap[i]))
                           for i in np.where(faceOverlap >= 0.5)[0]],
    faceOverlapAtLeast0_3=int((faceOverlap >= 0.3).sum()), gapAtLeast0_25=int((gap >= 0.25).sum()),
    highestGapCodes=[dict(key=keys[i], multipliers=multiplier[i].round(3).tolist(), gap=float(gap[i]), faceOverlap=float(faceOverlap[i]))
                     for i in np.argsort(-gap)[:6]],
    note='the conductance gap at the write peak is necessary for a face but not sufficient; the best-gap code is not a face')

# ---------------------------------------------------------------- how far, by each metric, along each order's slider
sliderStops = {}
shapeDistance = np.linalg.norm(shape.reshape(count, -1) - shape[trained].reshape(1, -1), axis=1)
for order in range(4):
    stops = sorted([i for i in range(count) if not steered[i] and all(abs(multiplier[i][q] - 1) < 1e-9 for q in range(4) if q != order)],
                   key=lambda i: multiplier[i][order])
    sliderStops[str(order)] = [dict(multiplier=float(multiplier[i][order]), key=keys[i], topThreeRetention=float(retentionOfTrained[i]),
                                    shapeDistance=float(shapeDistance[i]), gap=float(gap[i])) for i in stops]
result['sliders'] = sliderStops

# ---------------------------------------------------------------- retention over every two-order grid
result['gridRetention'] = {pairKey: dict(rows=[[float(retentionOfTrained[position[key]]) for key in row] for row in rows],
                                         gap=[[float(gap[position[key]]) for key in row] for row in rows],
                                         faceOverlap=[[float(faceOverlap[position[key]]) for key in row] for row in rows])
                           for pairKey, rows in placement['grid'].items()}
down = [i for i in range(count) if i != trained and (multiplier[i] <= 1 + 1e-9).all()]
up = [i for i in range(count) if i != trained and (multiplier[i] >= 1 - 1e-9).all()]
mixed = [i for i in range(count) if i != trained and i not in down and i not in up]
totalFlow = np.abs(transfer).sum((1, 2))
result['byDirection'] = {name: dict(codes=len(group), meanTopThreeRetention=float(retentionOfTrained[group].mean()),
                                    medianTotalFlow=float(np.median(totalFlow[group])), p90TotalFlow=float(np.percentile(totalFlow[group], 90)))
                         for name, group in (('allOrdersAtOrBelowTrained', down), ('allAtOrAboveTrained', up), ('someUpSomeDown', mixed))}
result['byDirection']['trainedTotalFlow'] = float(totalFlow[trained])

# ---------------------------------------------------------------- which of the trained code's top edges survive
survival = []
for edge in sorted(topSets[trained]):
    survival.append(dict(edge=' '.join(str(x) for x in edge[:1]) + f' {edge[1]}->{edge[2]}',
                         keptInDownSteered=float(np.mean([edge in topSets[i] for i in down])),
                         keptInUpSteered=float(np.mean([edge in topSets[i] for i in up])),
                         keptInMixed=float(np.mean([edge in topSets[i] for i in mixed]))))
result['trainedTopEdgesKept'] = survival

# ---------------------------------------------------------------- edges whose direction does not depend on the steering
consistency = []
for phase in range(3):
    for index, (a, b) in enumerate(pairs):
        active = isOn[:, phase, index]
        if active.sum() < 30:
            continue
        positive = (shape[active, phase, index] > 0).mean()
        sender, receiver = (a, b) if positive >= 0.5 else (b, a)
        consistency.append(dict(phase=phaseNames[phase], edge=f'{sender}->{receiver}', onInShareOfCodes=float(active.mean()),
                                sameDirectionShare=float(max(positive, 1 - positive)),
                                trainedValue=float(transfer[trained, phase, index] * (1 if positive >= 0.5 else -1))))
consistency.sort(key=lambda row: -row['sameDirectionShare'])
faceNodes = {'eyes', 'nose', 'mouth'}
involvesFace = lambda edge: any(n in faceNodes for n in edge.split('->'))
faceShare = [r['sameDirectionShare'] for r in consistency if involvesFace(r['edge'])]
otherShare = [r['sameDirectionShare'] for r in consistency if not involvesFace(r['edge'])]
result['directionConsistency'] = dict(mostConsistent=consistency[:12], leastConsistent=consistency[-6:],
                                      medianFaceEdges=float(np.median(faceShare)), medianOtherEdges=float(np.median(otherShare)),
                                      mannWhitneyP=float(stats.mannwhitneyu(faceShare, otherShare)[1]),
                                      note='edges that involve the eyes, nose or mouth are NOT less direction-consistent than the rest')

# ---------------------------------------------------------------- predictability from the multipliers, and additivity
design = np.hstack([multiplier, multiplier ** 2] + [multiplier[:, [i]] * multiplier[:, [j]] for i in range(4) for j in range(i + 1, 4)])
folds = KFold(10, shuffle=True, random_state=0)
rSquared = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
perPhase = {}
for phase in range(3):
    scores = [rSquared(shape[:, phase, j], cross_val_predict(Ridge(alpha=1.0), design, shape[:, phase, j], cv=folds))
              for j in range(len(pairs)) if shape[:, phase, j].std() > 1e-9]
    perPhase[phaseNames[phase]] = dict(medianCrossValidatedRSquared=float(np.median(scores)), edgesAbove0_3=int(np.sum(np.array(scores) > 0.3)),
                                       edges=len(scores))
result['predictabilityFromMultipliers'] = dict(perPhase=perPhase, gapCrossValidatedRSquared=rSquared(gap, cross_val_predict(Ridge(alpha=1.0), design, gap, cv=folds)),
                                               note='ridge on the four multipliers, their squares and pair products, 10-fold')
generator = np.random.default_rng(0)
errors, nulls = [], []
for pairKey, rows in placement['grid'].items():
    for r in (0, 1, 3, 4):
        for c in (0, 1, 3, 4):
            a, b, actual = (position[rows[r][2]], position[rows[2][c]], position[rows[r][c]])
            effect = transfer[actual] - transfer[trained]
            additive = transfer[trained] + (transfer[a] - transfer[trained]) + (transfer[b] - transfer[trained])
            wrong = position[rows[2][int(generator.choice([x for x in (0, 1, 3, 4) if x != c]))]]
            mismatched = transfer[trained] + (transfer[a] - transfer[trained]) + (transfer[wrong] - transfer[trained])
            errors.append(np.linalg.norm(transfer[actual] - additive) / np.linalg.norm(effect))
            nulls.append(np.linalg.norm(transfer[actual] - mismatched) / np.linalg.norm(effect))
result['additivity'] = dict(cells=len(errors), medianRelativeError=float(np.median(errors)), mismatchedLevelNull=float(np.median(nulls)),
                            cellsWithErrorBelow0_5=int(np.sum(np.array(errors) < 0.5)),
                            note='two-order cells predicted by adding the two single-order effects; the null takes one of them from the wrong level')

# ---------------------------------------------------------------- the shape metric's apparent jumpiness
distances = squareform(pdist(shape.reshape(count, -1)))
others = [i for i in range(count) if i != trained]
random = np.random.default_rng(1)
result['metricComparison'] = dict(
    shapeDistanceTrainedToOthersMedian=float(np.median(distances[trained, others])),
    shapeDistanceAmongOthersMedian=float(np.median(distances[np.ix_(others, others)][np.triu_indices(count - 1, 1)])),
    topThreeRetentionTrainedToOthersMean=float(retentionOfTrained[others].mean()),
    topThreeRetentionRandomPairsMean=float(np.mean([retention(*random.choice(count, 2, replace=False)) for _ in range(2000)])))

# ---------------------------------------------------------------- the second code that makes a face
second = [i for i in np.where(faceOverlap >= 0.5)[0] if i != trained]
result['secondFaceMaker'] = [dict(key=keys[i], multipliers=multiplier[i].round(3).tolist(), shapeDistanceToTrained=float(shapeDistance[i]),
                                  topThreeRetention=float(retentionOfTrained[i]), topEdges=sorted(topSets[i]),
                                  trainedTopEdges=sorted(topSets[trained])) for i in second]

json.dump(result, open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}')
