"""CONFIRMATORY test of the lower-channel and flood-push modules, scored against the criteria registered in
data/boundaryHarmonicRelayModulesConfirmationPredictions1888Hold301FaceMinus60Minus5.json (committed before the ring codes, the
runs or this script existed). Each criterion is evaluated as written and reported with its number, passed or failed; nothing else
is folded in. The only additions, labelled as such, are counts and rates that describe the cells.

Definitions (as registered, and as in the exploratory analyses): a module is present when its edge is among the three biggest
transfers of its phase in the code's field-channel net -- lower channel: in write, ring bottom -> bg bottom-left or ring left ->
bg bottom-left; flood push: in flood, ring top -> bg top-left; reversed lower channel: in write bg bottom-left -> ring bottom,
or in clear ring bottom -> mouth, or in flood ring bottom -> bg bottom-left. A face-like end state is structural overlap >= 0.3.

Writes data/relayLoopModulesConfirmation1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 analyzeRelayLoopModulesConfirmation11x11.py
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

import relayLoopNets

SUFFIX = '1888Hold301FaceMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--variantsPath', type=str, default=f'data/boundaryHarmonicRingCodeModulesConfirmation{SUFFIX}.json')
parser.add_argument('--recordsDirectory', type=str, default='data/relayLoopModulesConfirmation')
parser.add_argument('--predictionsPath', type=str, default=f'data/boundaryHarmonicRelayModulesConfirmationPredictions{SUFFIX}.json')
parser.add_argument('--outputPath', type=str, default=f'data/relayLoopModulesConfirmation{SUFFIX}.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')

CELLS = ('both', 'lowerOnly', 'pushOnly', 'neither')
FACE_LIKE, FACE = 0.3, 0.5
registered = json.load(open(args.predictionsPath))
variants = {v['key']: v for v in json.load(open(args.variantsPath))['variants']}
records = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(f'{args.recordsDirectory}/*.json'))}
missing = [k for k in variants if k not in records]
if missing:
    raise SystemExit(f'{len(missing)} of {len(variants)} codes have no record yet, e.g. {missing[:5]}')
keys = sorted(variants)
cell = np.array([variants[k]['cell'] for k in keys])
faceOverlap = np.array([records[k]['faceOverlap'] for k in keys])
gap = np.array([records[k]['gap'] for k in keys])
pairs = [tuple(p) for p in relayLoopNets.PAIRS]
phaseNames = list(relayLoopNets.PHASES)


def topEdges(code):
    chosen = set()
    for phase in range(3):
        for index in np.argsort(-np.abs(code[phase]))[:3]:
            a, b = pairs[index]
            chosen.add((phaseNames[phase], a, b) if code[phase, index] > 0 else (phaseNames[phase], b, a))
    return chosen


tops = [topEdges(np.array(records[k]['field'])) for k in keys]
anyOf = lambda *edges: np.array([any(e in t for e in edges) for t in tops])
lowerChannel = anyOf(('write', 'ringBottom', 'bgBL'), ('write', 'ringLeft', 'bgBL'))
floodPush = anyOf(('flood', 'ringTop', 'bgTL'))
reversedLower = anyOf(('write', 'bgBL', 'ringBottom'), ('clear', 'ringBottom', 'mouth'), ('flood', 'ringBottom', 'bgBL'))
faceLike = faceOverlap >= FACE_LIKE
rate = lambda mask: float(faceLike[mask].mean()) if mask.any() else float('nan')
inCell = {c: cell == c for c in CELLS}
fisher = lambda a, b: float(stats.fisher_exact([[int(faceLike[a].sum()), int((~faceLike[a]).sum())], [int(faceLike[b].sum()), int((~faceLike[b]).sum())]], alternative='greater')[1])
rest = ~inCell['both']
report = []


def record(name, passed, **numbers):
    entry = next(p for p in registered['predictions'] if p['name'] == name)
    report.append(dict(name=name, claim=entry['claim'], criterion=entry['criterion'], passed=bool(passed), **numbers))


# ---------------------------------------------------------------- validity
ringLegal = all(0.0 <= v['ringMin'] and v['ringMax'] <= 2.0 for v in variants.values())
conductanceBreaches = [k for k in keys if records[k]['gpolMin'] < 0.0 or records[k]['gpolMax'] > 2.0]
distanceByCell = {c: float(np.mean([variants[k]['regionDistance'] for k in np.array(keys)[inCell[c]]])) for c in CELLS}
sweepMultipliers = np.array([c['multipliers'] for c in json.load(open(f'data/relayLoopSweepNets{SUFFIX}.json'))['codes'].values()]
                            + [c['multipliers'] for c in json.load(open(f'data/relayLoopFullNets{SUFFIX}.json'))['codes'].values()])
nearest = min(float(np.linalg.norm(sweepMultipliers - np.array(variants[k]['multipliers']), axis=1).min()) for k in keys)
validity = dict(V1=dict(ringValuesLegal=ringLegal, conductanceBreaches=conductanceBreaches, passed=ringLegal and not conductanceBreaches),
                V2=dict(meanRegionDistanceByCell=distanceByCell, spread=max(distanceByCell.values()) - min(distanceByCell.values()),
                        passed=max(distanceByCell.values()) - min(distanceByCell.values()) < 0.03),
                V3=dict(nearestExistingCodeInMultipliers=nearest, passed=nearest > 0.01))

# ---------------------------------------------------------------- the registered criteria
lowerHigh, lowerLow = inCell['both'] | inCell['lowerOnly'], inCell['pushOnly'] | inCell['neither']
pushHigh, pushLow = inCell['both'] | inCell['pushOnly'], inCell['lowerOnly'] | inCell['neither']
a, b = float(lowerChannel[lowerHigh].mean()), float(lowerChannel[lowerLow].mean())
record('C1-lowerChannelIsPredictable', a >= 0.65 and b <= 0.30, observedAmongPredictedHigh=a, observedAmongPredictedLow=b)
a, b = float(floodPush[pushHigh].mean()), float(floodPush[pushLow].mean())
record('C2-floodPushIsPredictable', a >= 0.60 and b <= 0.25, observedAmongPredictedHigh=a, observedAmongPredictedLow=b)
rateBoth, rateRest = rate(inCell['both']), rate(rest)
p = fisher(inCell['both'], rest)
record('C3-bothModulesGoWithTheFace', rateBoth >= 0.30 and rateBoth - rateRest >= 0.15 and p < 0.01, rateBoth=rateBoth, rateOtherThreePooled=rateRest, difference=rateBoth - rateRest, fisherP=p)
rates = {c: rate(inCell[c]) for c in CELLS}
record('C4-noOtherCellMatchesBoth', all(rates['both'] > rates[c] for c in CELLS[1:]), ratesByCell=rates)
difference = rates['both'] - rates['lowerOnly'] - rates['pushOnly'] + rates['neither']
generator = np.random.default_rng(20251103)
draws = []
for _ in range(2000):
    resampled = {c: faceLike[inCell[c]][generator.integers(0, inCell[c].sum(), inCell[c].sum())].mean() for c in CELLS}
    draws.append(resampled['both'] - resampled['lowerOnly'] - resampled['pushOnly'] + resampled['neither'])
interval = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
record('C5-theModulesInteract', difference >= 0.10 and (interval[0] > 0 or interval[1] < 0), differenceOfDifferences=float(difference), bootstrapInterval95=interval)
meanGaps = {c: float(gap[inCell[c]].mean()) for c in CELLS}
record('C6-gapFollowsTheModules', meanGaps['both'] - np.mean([meanGaps[c] for c in CELLS[1:]]) >= 0.08, meanGapByCell=meanGaps,
       difference=float(meanGaps['both'] - np.mean([meanGaps[c] for c in CELLS[1:]])))
withReversed = reversedLower
table = [[int((faceLike & withReversed).sum()), int((~faceLike & withReversed).sum())], [int((faceLike & ~withReversed).sum()), int((~faceLike & ~withReversed).sum())]]
p = float(stats.fisher_exact(table, alternative='less')[1])
record('C7-theReversedLowerChannelIsAgainstTheFace', rate(withReversed) < rate(~withReversed) and p < 0.05, rateWithReversed=rate(withReversed), rateWithout=rate(~withReversed),
       codesWithReversed=int(withReversed.sum()), fisherP=p)
observedBoth = lowerChannel & floodPush
p = fisher(observedBoth, ~observedBoth)
record('C8-observedModulesNotJustPredicted', rate(observedBoth) >= 0.30 and rate(observedBoth) - rate(~observedBoth) >= 0.15 and p < 0.01,
       rateObservedBoth=rate(observedBoth), rateRest=rate(~observedBoth), codesWithBothObserved=int(observedBoth.sum()), fisherP=p)
record('C9-neitherCellIsQuiet', rates['neither'] <= 0.15, rateNeither=rates['neither'])

describe = {c: dict(codes=int(inCell[c].sum()), faceLike=int(faceLike[inCell[c]].sum()), faceAtLeast0_5=int((faceOverlap[inCell[c]] >= FACE).sum()),
                    observedLowerChannel=float(lowerChannel[inCell[c]].mean()), observedFloodPush=float(floodPush[inCell[c]].mean()),
                    observedReversedLower=float(reversedLower[inCell[c]].mean())) for c in CELLS}
result = dict(status='CONFIRMATORY: scored against the criteria registered before the runs', registeredIn=os.path.basename(args.predictionsPath), codes=len(keys),
              validity=validity, criteria=report, passedCount=sum(r['passed'] for r in report), criteriaCount=len(report),
              cellDescription=dict(note='descriptive counts, not registered criteria', cells=describe))
json.dump(result, open(args.outputPath, 'w'), indent=1)
print(f'wrote {args.outputPath}')
for k, v in validity.items():
    print(f'  {k}: {"PASS" if v["passed"] else "FAIL"}')
for r in report:
    print(f'  {r["name"]:48s} {"PASS" if r["passed"] else "FAIL"}  ' + ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items() if k not in ('name', 'claim', 'criterion', 'passed')))
print(f'{result["passedCount"]} of {result["criteriaCount"]} registered criteria passed')
