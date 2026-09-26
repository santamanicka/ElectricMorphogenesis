"""Tests the untested links of the causal story: the wall's halves at the face level, retiming, re-phasing, and the hold's length.

EXPLORATORY. Criteria: data/boundaryHarmonicCausalStoryPredictions1888Hold301FaceMinus60Minus5.json, committed before
any run here. Every run is the pure re-implementation of the model step (boundaryHarmonicStep.Step) with only part of
the ring held (the trained orders 0-3 code, iterations 0-300, with the clamp's tissue-wide extra Vmem update on every
hold iteration in every run), against the same extraUpdateOnly baseline as the relay, but taken on to state 2174
(recorded iteration 2173, the scored moment) so that the latch phase is included.

    python3 analyzeBoundaryHarmonicCausalStory11x11.py
"""
import argparse
import json
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from boundaryHarmonicStep import Step

parser = argparse.ArgumentParser()
parser.add_argument('--counterfactualPath', type=str, default='data/boundaryHarmonicWallCounterfactual1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicCausalStoryPredictions1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicCausalStory1888Hold301FaceMinus60Minus5.json')
args = parser.parse_args()
torch.set_grad_enabled(False)
started = time.time()

counterfactual = json.load(open(args.counterfactualPath))
summary = json.load(open(args.summaryPath))
hold = int(summary['hold'])
winner = summary['orders']['3']['best']
coefficients = np.asarray(np.load(
    f"{summary['trainingDirs'][winner['round']]}/order3_restart{winner['restart']:02d}.npz")['bestCoefficients'], float)
ringValues = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(len(coefficients)))) @ coefficients
step = Step(ringCode=ringValues)
n, Gref = step.numCells, step.Gref
LAST = 2174                                                          # state index of recorded iteration 2173, the scored moment
PEAK = 1766                                                          # state index of the relay's readout
RELEASE_STATE, TROUGH_WINDOW_END = 302, 1300
ring = [int(c) for c in counterfactual['singleCells']['ring']]
fullMask = step.ringMask.clone()

featureCells = np.array(sorted(set(boundary.featureCellIndices.tolist())))
interior = np.array(boundary.interiorCellIndices)
backgroundCells = np.array([c for c in interior if c not in set(featureCells.tolist())])
leftEye, rightEye, nose, mouth = (list(part) for part in boundary.featureParts)
groups = {'eyes': np.array(leftEye + rightEye), 'nose': np.array(nose), 'mouth': np.array(mouth), 'background': backgroundCells}
assert [len(g) for g in groups.values()] == [8, 3, 3, 67]
targetInterior = np.isin(interior, featureCells)

sideCells = lambda rowRange: [r * 11 + c for r in rowRange for c in (0, 10)]
subsets = dict(
    fullRing=ring,
    upperWall=counterfactual['upperWall']['cells'],
    lowerWall=counterfactual['lowerWall']['cells'],
    topRow=list(range(0, 11)),
    upperSides=sideCells(range(1, 7)),
    lowerSides=sideCells(range(7, 10)),
    bottomRow=list(range(110, 121)),
)
assert [len(subsets[k]) for k in ('upperWall', 'lowerWall', 'topRow', 'upperSides', 'lowerSides', 'bottomRow')] == [23, 17, 11, 12, 6, 11]
assert set(subsets['upperWall']) | set(subsets['lowerWall']) == set(ring) and not set(subsets['upperWall']) & set(subsets['lowerWall'])


def run(held, releaseAt=hold):
    """Vmem (volts) and G_pol (siemens) at every state 0..LAST; the ring cells in `held` are held for iterations
    below releaseAt, and the tissue-wide extra update runs for every hold iteration whatever is held."""
    mask = torch.zeros(n, dtype=torch.double)
    if held:
        mask[list(held)] = 1.0
    step.ringMask = mask
    V, G = step.initialVmem.clone(), step.initialGpol.clone()
    vTrajectory, gTrajectory = [V.numpy().copy()], [G.numpy().copy()]
    for m in range(LAST):
        V, G = step(V, G, bool(held) and m < releaseAt, m < hold)
        vTrajectory.append(V.numpy().copy())
        gTrajectory.append(G.numpy().copy())
    step.ringMask = fullMask
    return np.array(vTrajectory), np.array(gTrajectory)


def selectivity(G):
    return float((G[featureCells].mean() - G[backgroundCells].mean()) / Gref)


def groupParts(G, reference):
    """Each group's part of the selectivity difference (the face groups add their mean change over the fourteen face
    cells, the background subtracts its mean change over its sixty-seven), as in the wall counterfactual."""
    change = (G - reference) / Gref
    return {name: float(change[cells].sum() / len(featureCells)) if name != 'background' else float(-change[cells].sum() / len(backgroundCells))
            for name, cells in groups.items()}


def darkCounts(V):
    dark = V * 1000.0 < boundary.hyperpolarizedThresholdMilliVolts
    counts = {name: int(dark[cells].sum()) for name, cells in groups.items()}
    counts['face'] = counts['eyes'] + counts['nose'] + counts['mouth']
    return counts


def overlap(V):
    return boundary.structuralIntersectionOverUnion(V * 1000.0)


def trough(series):
    """Recorded iteration (state - 1) of the minimum from the release to iteration 1300, and the second peak after it."""
    window = series[RELEASE_STATE:TROUGH_WINDOW_END + 1]
    trough = RELEASE_STATE + int(np.argmin(window))
    peak = trough + int(np.argmax(series[trough:LAST + 1]))
    return trough - 1, peak - 1


results, trajectories = {}, {}
started_runs = time.time()
baselineV, baselineG = run(None)
trajectories['baseline'] = (baselineV, baselineG)
print(f'[{time.time() - started:4.0f}s] baseline selectivity at {PEAK}: {selectivity(baselineG[PEAK]):+.5f}', flush=True)
for name, held in subsets.items():
    trajectories[name] = run(held)
    print(f'[{time.time() - started:4.0f}s] {name}', flush=True)
for releaseAt in (100, 150, 200, 250):
    trajectories[f'fullRingReleasedAt{releaseAt}'] = run(ring, releaseAt=releaseAt)
    print(f'[{time.time() - started:4.0f}s] fullRingReleasedAt{releaseAt}', flush=True)

baselineSelectivity = selectivity(baselineG[PEAK])
for name, (V, G) in trajectories.items():
    interiorMean = G[:, interior].mean(1) / Gref
    troughIteration, peakIteration = trough(interiorMean)
    groupTrough = {g: trough(G[:, cells].mean(1) / Gref)[0] for g, cells in groups.items()}
    results[name] = dict(
        selectivityDifference=selectivity(G[PEAK]) - baselineSelectivity,
        parts=groupParts(G[PEAK], baselineG[PEAK]),
        interiorTrough=troughIteration, interiorSecondPeak=peakIteration, interiorTroughValue=float(interiorMean[troughIteration + 1]),
        groupTrough=groupTrough,
        darkAtScored=darkCounts(V[LAST]), overlapAtScored=overlap(V[LAST]), overlapAtState2173=overlap(V[LAST - 1]),
        bestOverlap=max((overlap(V[s]), s - 1) for s in range(RELEASE_STATE, LAST + 1)))
    r = results[name]
    print(f"{name:24s} diff {r['selectivityDifference']:+.4f}  trough {r['interiorTrough']:5d} peak {r['interiorSecondPeak']:5d}  "
          f"dark eyes/nose/mouth/bg {r['darkAtScored']['eyes']}/{r['darkAtScored']['nose']}/{r['darkAtScored']['mouth']}/{r['darkAtScored']['background']}  "
          f"overlap {r['overlapAtScored']:.3f}  group troughs {r['groupTrough']}", flush=True)

# ------------------------------------------------------------------ time courses of the four main runs, every 5 states to LAST
sampled = list(range(0, LAST + 1, 5)) + [LAST]
courses = {}
for name in ('baseline', 'fullRing', 'upperWall', 'lowerWall'):
    V, G = trajectories[name]
    courses[name] = dict(
        interiorG=[round(float(G[s, interior].mean() / Gref), 5) for s in sampled],
        groupG={g: [round(float(G[s, cells].mean() / Gref), 5) for s in sampled] for g, cells in groups.items()},
        dark={g: [int((V[s, cells] * 1000.0 < boundary.hyperpolarizedThresholdMilliVolts).sum()) for s in sampled] for g, cells in groups.items()},
        parts=None if name == 'baseline' else {g: [] for g in groups})
    if name != 'baseline':
        for s in sampled:
            for g, value in groupParts(G[s], baselineG[s]).items():
                courses[name]['parts'][g].append(round(value, 5))

# ------------------------------------------------------------------ validity and verdicts
full, upper, lower, base = results['fullRing'], results['upperWall'], results['lowerWall'], results['baseline']
V1 = dict(fullRing=full['selectivityDifference'], relay=0.35588658, upper=upper['selectivityDifference'], lower=lower['selectivityDifference'],
          wallCounterfactual=[counterfactual['upperWall']['alone'], counterfactual['lowerWall']['alone']])
V1['holds'] = bool(abs(V1['fullRing'] - counterfactual['fullRing']) < 1e-6 and abs(V1['upper'] - V1['wallCounterfactual'][0]) < 1e-6
                   and abs(V1['lower'] - V1['wallCounterfactual'][1]) < 1e-6 and abs(V1['fullRing'] - V1['relay']) < 1e-6)
V2 = dict(overlapAtState2174=full['overlapAtScored'], overlapAtState2173=full['overlapAtState2173'], report=0.933)
V2['holds'] = bool(min(abs(V2['overlapAtState2174'] - 0.933), abs(V2['overlapAtState2173'] - 0.933)) < 0.01)
noseMouth = lambda r: r['darkAtScored']['nose'] + r['darkAtScored']['mouth']
E1a = dict(upperEyesDark=upper['darkAtScored']['eyes'], lowerEyesDark=lower['darkAtScored']['eyes'],
           holds=bool(upper['darkAtScored']['eyes'] >= 6 and lower['darkAtScored']['eyes'] <= 2))
E1b = dict(upper=upper['overlapAtScored'], lower=lower['overlapAtScored'], full=full['overlapAtScored'], baseline=base['overlapAtScored'],
           holds=bool(upper['overlapAtScored'] < 0.5 and lower['overlapAtScored'] < 0.5 and full['overlapAtScored'] >= 0.8 and base['overlapAtScored'] < 0.3))
E1c = dict(upperNoseMouthDark=noseMouth(upper), lowerNoseMouthDark=noseMouth(lower), fullNoseMouthDark=noseMouth(full),
           holds=bool(noseMouth(upper) < 4 and noseMouth(lower) < 4 and noseMouth(full) >= 5))
E2a = dict(fullTrough=full['interiorTrough'], upperTrough=upper['interiorTrough'], lowerTrough=lower['interiorTrough'], baselineTrough=base['interiorTrough'],
           holds=bool(abs(upper['interiorTrough'] - full['interiorTrough']) <= 100 and abs(lower['interiorTrough'] - base['interiorTrough']) <= 100))
separation = lambda r: abs(r['groupTrough']['eyes'] - r['groupTrough']['background'])
E2b = dict(baselineSeparation=separation(base), fullSeparation=separation(full), baselineTroughs=base['groupTrough'], fullTroughs=full['groupTrough'],
           holds=bool(separation(base) < 40 and separation(full) >= 40))
halfEyes = 0.5 * upper['parts']['eyes']
E3 = dict(upperEyes=upper['parts']['eyes'], topRowEyes=results['topRow']['parts']['eyes'], upperSidesEyes=results['upperSides']['parts']['eyes'], half=halfEyes,
          holds=bool(results['topRow']['parts']['eyes'] < halfEyes and results['upperSides']['parts']['eyes'] < halfEyes))
E4 = dict(gapAt200=results['fullRingReleasedAt200']['selectivityDifference'], overlapAt200=results['fullRingReleasedAt200']['overlapAtScored'],
          gapAt100=results['fullRingReleasedAt100']['selectivityDifference'], threshold=0.7 * full['selectivityDifference'],
          holds=bool(results['fullRingReleasedAt200']['selectivityDifference'] >= 0.7 * full['selectivityDifference']
                     and results['fullRingReleasedAt200']['overlapAtScored'] >= 0.5
                     and results['fullRingReleasedAt100']['selectivityDifference'] < 0.7 * full['selectivityDifference']))
verdicts = dict(V1=V1, V2=V2, E1a=E1a, E1b=E1b, E1c=E1c, E2a=E2a, E2b=E2b, E3=E3, E4=E4)
for key, verdict in verdicts.items():
    print(f"{key}: {'holds' if verdict['holds'] else 'FAILS'}", {k: v for k, v in verdict.items() if k != 'holds'}, flush=True)

json.dump(dict(note='EXPLORATORY; criteria in the predictions file, registered before any run.', predictions=json.load(open(args.predictionsPath)),
               subsets={k: [int(c) for c in v] for k, v in subsets.items()}, results=results, sampledStates=sampled, courses=courses, verdicts=verdicts),
          open(args.outputPath, 'w'))
print(f'wrote {args.outputPath} ({time.time() - started:.0f}s)', flush=True)
