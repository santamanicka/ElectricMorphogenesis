"""The ring codes of the confirmatory test of the lower-channel and flood-push modules, selected exactly as registered in
data/boundaryHarmonicRelayModulesConfirmationPredictions1888Hold301FaceMinus60Minus5.json (committed before this script existed).

A frozen model -- two random forests fitted to the 400 sweep codes only, from the ring's four region levels to whether the lower
channel / the flood push is among the top three transfers of its phase -- scores a pool of new, legal ring codes. Four cells of 80
are drawn from it (both modules predicted, lower only, push only, neither), the three control cells matched one-to-one to the
BOTH cell on how far the ring's region levels are from the trained code's. Every ring value is asserted to lie inside [0, 2]
G_pol / G_ref with no clipping, and every code is checked not to repeat a sweep or page code.

Writes data/boundaryHarmonicRingCodeModulesConfirmation1888Hold301FaceMinus60Minus5.json (never overwriting), ready for
computeBoundaryHarmonicRelay11x11.py --ringCodeVariantsPath / --ringCodeKey.

    python3 buildBoundaryHarmonicRingCodeModulesConfirmation11x11.py
"""
import json
import os

import numpy as np
from scipy.stats import qmc
from sklearn.ensemble import RandomForestClassifier

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301FaceMinus60Minus5'
outputPath = f'data/boundaryHarmonicRingCodeModulesConfirmation{SUFFIX}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

POOL_EXPONENT, SOBOL_SEED, SELECTION_SEED = 17, 20251101, 20251102                  # 2**17 = 131072 points
MIN_DISTANCE, PER_CELL, HIGH, LOW, MATCH_TOLERANCE = 0.20, 80, 0.7, 0.3, 0.05
trainedCoefficients = np.asarray(json.load(open(f'data/boundaryHarmonicRingCodeFiveLevel{SUFFIX}.json'))['trainedCoefficients'])
basis = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(4)))
foldedAngle = np.abs(np.angle(np.exp(1j * boundary.ringAngles(boundary.boundaryRingCells))))
bins = np.digitize(foldedAngle, [np.pi / 4, np.pi / 2, 3 * np.pi / 4])


def rawRingValues(multipliers):
    return basis @ (np.atleast_2d(multipliers) * trainedCoefficients).T                  # ring cell, code


def regionLevels(multipliers):
    values = np.clip(rawRingValues(multipliers), 0.0, 2.0).T
    return np.array([[row[bins == b].mean() for b in range(4)] for row in values])


# ---------------------------------------------------------------- the frozen model, from the sweep alone
sweep = json.load(open(f'data/relayLoopSweepNets{SUFFIX}.json'))
pairs, phaseNames = [tuple(p) for p in sweep['pairs']], sweep['phases']
sweepKeys = sorted(sweep['codes'])
sweepMultipliers = np.array([sweep['codes'][k]['multipliers'] for k in sweepKeys])
sweepTransfer = np.array([sweep['codes'][k]['field'] for k in sweepKeys])


def topEdges(code):
    chosen = set()
    for phase in range(3):
        for index in np.argsort(-np.abs(code[phase]))[:3]:
            a, b = pairs[index]
            chosen.add((phaseNames[phase], a, b) if code[phase, index] > 0 else (phaseNames[phase], b, a))
    return chosen


tops = [topEdges(code) for code in sweepTransfer]
anyOf = lambda *edges: np.array([any(e in t for e in edges) for t in tops])
lowerLabel = anyOf(('write', 'ringBottom', 'bgBL'), ('write', 'ringLeft', 'bgBL')).astype(int)
pushLabel = anyOf(('flood', 'ringTop', 'bgTL')).astype(int)
sweepLevels = regionLevels(sweepMultipliers)
forest = lambda: RandomForestClassifier(200, min_samples_leaf=3, random_state=0, n_jobs=4)
lowerModel, pushModel = forest().fit(sweepLevels, lowerLabel), forest().fit(sweepLevels, pushLabel)

# ---------------------------------------------------------------- the pool
trainedLevels = regionLevels(np.ones(4))[0]
points = qmc.Sobol(4, scramble=True, seed=SOBOL_SEED).random_base2(POOL_EXPONENT) * 2.0
ringValues = rawRingValues(points)
legal = (ringValues.min(0) >= 0.0) & (ringValues.max(0) <= 2.0)
points = points[legal]
levels = regionLevels(points)
distance = np.linalg.norm(levels - trainedLevels, axis=1)
far = distance >= MIN_DISTANCE
points, levels, distance = points[far], levels[far], distance[far]
pLower, pPush = lowerModel.predict_proba(levels)[:, 1], pushModel.predict_proba(levels)[:, 1]
print(f'pool: {2 ** POOL_EXPONENT} points, {int(legal.sum())} legal, {len(points)} at least {MIN_DISTANCE} from the trained region levels')
candidates = dict(both=(pLower >= HIGH) & (pPush >= HIGH), lowerOnly=(pLower >= HIGH) & (pPush <= LOW),
                  pushOnly=(pLower <= LOW) & (pPush >= HIGH), neither=(pLower <= LOW) & (pPush <= LOW))
print('candidates per cell:', {name: int(mask.sum()) for name, mask in candidates.items()})

# ---------------------------------------------------------------- BOTH at random, the other cells matched to it
generator = np.random.default_rng(SELECTION_SEED)
both = np.sort(generator.choice(np.where(candidates['both'])[0], min(PER_CELL, int(candidates['both'].sum())), replace=False))
both = both[np.argsort(distance[both])]
matched = {}
for name in ('lowerOnly', 'pushOnly', 'neither'):
    available = set(np.where(candidates[name])[0].tolist())
    matched[name] = {}
    for a in both:
        if not available:
            break
        best = min(available, key=lambda c: abs(distance[c] - distance[a]))
        if abs(distance[best] - distance[a]) <= MATCH_TOLERANCE:
            matched[name][int(a)] = int(best)
            available.discard(best)
kept = [int(a) for a in both if all(int(a) in matched[name] for name in matched)]
print(f'BOTH codes with a match in all three cells: {len(kept)} of {len(both)}')
if len(kept) < PER_CELL:
    print(f'cut every cell to {len(kept)}, as registered')

# ---------------------------------------------------------------- legality, newness, output
existing = np.vstack([sweepMultipliers, np.array([c['multipliers'] for c in json.load(open(f'data/relayLoopFullNets{SUFFIX}.json'))['codes'].values()])])
variants = []
for index, a in enumerate(kept):
    for cell, source in (('both', a), ('lowerOnly', matched['lowerOnly'][a]), ('pushOnly', matched['pushOnly'][a]), ('neither', matched['neither'][a])):
        multipliers = points[source]
        values = rawRingValues(multipliers)[:, 0]
        assert values.min() >= 0.0 and values.max() <= 2.0, (cell, index)
        assert np.linalg.norm(existing - multipliers, axis=1).min() > 0.01, (cell, index)
        variants.append(dict(key=f'modules{cell[0].upper()}{cell[1:]}{index:03d}', kind='modulesConfirmation', cell=cell, matchedIndex=index,
                             multipliers=[round(float(m), 6) for m in multipliers], coefficients=[round(float(m * c), 6) for m, c in zip(multipliers, trainedCoefficients)],
                             ringValues=[round(float(v), 6) for v in values], ringMin=round(float(values.min()), 4), ringMax=round(float(values.max()), 4),
                             regionLevels=dict(zip(['top', 'upperSides', 'lowerSides', 'bottom'], levels[source].round(4).tolist())),
                             regionDistance=round(float(distance[source]), 4), pLower=round(float(pLower[source]), 4), pPush=round(float(pPush[source]), 4)))
meanDistance = {cell: float(np.mean([v['regionDistance'] for v in variants if v['cell'] == cell])) for cell in ('both', 'lowerOnly', 'pushOnly', 'neither')}
spread = max(meanDistance.values()) - min(meanDistance.values())
json.dump(dict(registeredIn=f'boundaryHarmonicRelayModulesConfirmationPredictions{SUFFIX}.json', trainedCoefficients=[round(float(c), 6) for c in trainedCoefficients],
               codesPerCell=len(kept), meanRegionDistanceByCell=meanDistance, variants=variants), open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}: {len(variants)} codes ({len(kept)} per cell); mean distance by cell {({c: round(d, 3) for c, d in meanDistance.items()})}; spread {spread:.3f} (V2 needs < 0.03)')
