"""Scores the re-canalization runs of simulateCanalizationTalkRecanalization11x11.py against the predictions registered in
data/canalizationTalkRecanalizationPredictions1888Hold301.json (committed before the simulation existed). Read that file for the protocol, the measures and the criteria.

Three targets: the stripe and the order-3 face (from the order-3 trajectory file) and the order-4 face (the Order4 file; skipped if absent). For every code and every age T:
  reapplied    the branch: the same code held again for 301 iterations at T, then released (tau = iteration - T)
  continued    the original run's own frames T to T + 3299, nothing applied
  age 0        the original run's first 3300 iterations
measured as in the registration (exact IoU with the target; family share over tau 301-2999). Also checks the pipeline (V1: age-0 flags equal the stored family flags) and, as
EXPLORATORY additions, the stripe counts and deformed faces in each condition, the patterns a reapplied branch shows that its own free run never does, and the distance of each reapplied
branch from the age-0 trajectory.

    python3 analyzeCanalizationTalkRecanalization11x11.py

Writes data/canalizationTalkRecanalizationScores1888Hold301.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import canalizationTalkFamilies as fam

parser = argparse.ArgumentParser()
parser.add_argument('--order3Path', type=str, default='data/canalizationTalkRecanalizationTrajectories1888Hold301.npz')
parser.add_argument('--order4Path', type=str, default='data/canalizationTalkRecanalizationTrajectories1888Hold301Order4.npz')
parser.add_argument('--outputPath', type=str, default='data/canalizationTalkRecanalizationScores1888Hold301.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')

PRIMARY, SECONDARY = [3000, 5000], [1000, 10000, 20000]
RETURNED = 0.9
FAMILY_SHARE_FLOOR = 0.15
TIMING_TOLERANCE = 100
STORED_FLAGS = {3: 'data/canalizationTalkFamilyFlags1888Hold301.npz', 4: 'data/canalizationTalkFamilyFlags1888Hold301Order4.npz'}
STORED_TRAJECTORIES = {3: 'data/canalizationTalkFamilyTrajectories1888Hold301.npz', 4: 'data/canalizationTalkFamilyTrajectories1888Hold301Order4.npz'}


def unpack(packed):
    return np.unpackbits(packed, axis=-1)[..., :81].astype(bool)


class Runs:
    """One trajectory file: its codes, original runs and branches, with the flags of every frame computed on demand."""

    def __init__(self, path):
        store = np.load(path, allow_pickle=True)
        self.order, self.stride, self.hold = int(store['faceOrder']), int(store['stride']), int(store['hold'])
        self.names, self.groups = [str(n) for n in store['names']], [str(g) for g in store['groups']]
        self.ages, self.branchIterations = [int(a) for a in store['ages']], int(store['branchIterations'])
        self.original = unpack(store['original'])
        self.heldNames = [n for n, g in zip(self.names, self.groups) if g != 'noRing']
        self.branch = {age: unpack(store[f'branch{age}']) for age in self.ages}
        self.frames = self.branchIterations // self.stride
        self.cache = {}

    def flags(self, key, bits):
        if key not in self.cache:
            self.cache[key] = fam.frameFlags(bits)
        return self.cache[key]

    def condition(self, name, condition, age=0):
        """Dark sets (frames x 81) of one code under one condition: 'ageZero', 'continued' or 'reapplied' at `age`."""
        row = self.names.index(name)
        if condition == 'reapplied':
            return self.branch[age][self.heldNames.index(name)]
        start = 0 if condition == 'ageZero' else age // self.stride
        return self.original[row][start:start + self.frames]

    def measure(self, name, condition, age, target):
        bits = self.condition(name, condition, age)
        bars, faceBest, exact = self.flags((name, condition, age), bits)
        tau = np.arange(len(bits)) * self.stride
        window = (tau >= 301) & (tau < 3000)
        reach = np.where(exact[target] >= RETURNED)[0]
        family = (bars > 0) if target == 'stripe' else (faceBest >= 0.8)
        return dict(peakExact=float(exact[target].max()), firstExact=int(tau[reach[0]]) if len(reach) else None, familyShare=float(family[window].mean()),
                    exactAtLeast0p5Share=float((exact[target][window] >= 0.5).mean()), stripeCounts={str(n): int((bars[window] == n).sum() if n < 4 else (bars[window] >= 4).sum()) for n in (1, 2, 3, 4)},
                    deformedFaceFrames=int((faceBest[window] >= 0.8).sum()), distinctPatterns=int(len({b.tobytes() for b in bits[window]})))


def scoreTarget(runs, key, target, trainedName, controlGroup):
    """All measures for one target: the trained code and its controls under every condition."""
    controls = [n for n, g in zip(runs.names, runs.groups) if g == controlGroup]
    table = {'ageZero': {}, 'continued': {}, 'reapplied': {}}
    for name in [trainedName] + controls:
        table['ageZero'][name] = runs.measure(name, 'ageZero', 0, target)
        for age in runs.ages:
            table['continued'].setdefault(age, {})[name] = runs.measure(name, 'continued', age, target)
            table['reapplied'].setdefault(age, {})[name] = runs.measure(name, 'reapplied', age, target)
    return dict(trained=trainedName, controls=controls, table=table)


def explore(runs, key, target, trainedName, controls):
    """EXPLORATORY: patterns of the reapplied trained branch that its own free run (iteration 301 on) and the no-ring run never show, and the distance to the age-0 trajectory."""
    out = {}
    row = runs.names.index(trainedName)
    free = {b.tobytes() for b in runs.original[row][301 // runs.stride:]} | {b.tobytes() for b in runs.original[runs.names.index('noRing')][301 // runs.stride:]}
    ageZero = runs.condition(trainedName, 'ageZero')
    unrelated = [runs.condition(c, 'ageZero') for c in controls]
    edges = [(0, 300), (301, 1000), (1000, 2000), (2000, 3300)]
    for age in runs.ages:
        branch = runs.condition(trainedName, 'reapplied', age)
        late = branch[301 // runs.stride:]
        patterns = {b.tobytes() for b in late}
        tau = np.arange(len(branch)) * runs.stride
        fidelity, baseline = {}, {}
        for low, high in edges:
            mask = (tau >= low) & (tau < high)
            fidelity[f'{low}-{high}'] = float((branch[mask] != ageZero[mask]).mean())
            baseline[f'{low}-{high}'] = float(np.mean([(u[mask] != ageZero[mask]).mean() for u in unrelated]))
        out[str(age)] = dict(distinctAfterRelease=len(patterns), neverInFreeRun=len(patterns - free), shareNeverInFreeRun=float(len(patterns - free) / max(len(patterns), 1)),
                             distanceToAgeZero=fidelity, distanceOfUnrelatedCodesToAgeZero=baseline)
    return out


def check(runs, order):
    """V1: the age-0 flags of every code equal the stored family flags over the same frames (stripe family, face family, exact at 0.5)."""
    stored = np.load(STORED_FLAGS[order], allow_pickle=True)
    storedNames = [str(n) for n in stored['names']]
    frames, mismatches, compared = runs.frames, [], 0
    for name in runs.names:
        if name not in storedNames:
            continue
        bars, faceBest, exact = runs.flags((name, 'ageZero', 0), runs.condition(name, 'ageZero'))
        k = storedNames.index(name)
        same = (np.array_equal(bars > 0, stored['bars'][k][:frames] > 0) and np.array_equal(faceBest >= 0.8, stored['faceBest'][k][:frames].astype(np.float32) >= 0.8)
                and np.array_equal(exact['stripe'] >= 0.5, stored['exactStripe'][k][:frames].astype(np.float32) >= 0.5)
                and np.array_equal(exact['face'] >= 0.5, stored['exactFace'][k][:frames].astype(np.float32) >= 0.5))
        compared += 1
        if not same:
            mismatches.append(name)
    trajectories = np.load(STORED_TRAJECTORIES[order], allow_pickle=True)                   # EXPLORATORY addition: the stored IoUs are float16, so an IoU of exactly 0.8 can fall either side of the face-family
    trajectoryNames = [str(n) for n in trajectories['names']]                              # threshold; the dark sets themselves are compared as well
    different = [name for name in runs.names if name in trajectoryNames and name != 'noRing' and not np.array_equal(runs.original[runs.names.index(name)][:frames],
                 unpack(trajectories['packed'][trajectoryNames.index(name)])[:frames])]
    return dict(codesCompared=compared, mismatches=mismatches, holds=compared > 0 and not mismatches, darkSetsIdenticalInEveryFrame=not different, codesWithDifferentDarkSets=different,
                note='noRing is left out of the dark-set comparison: this simulation holds no ring AND skips the hold\'s tissue-wide extra update, the family analysis\'s no-ring run keeps it; the no-ring run only enters the exploratory novelty library')


# ------------------------------------------------------------------------------------------------------------ run
sources = {3: Runs(args.order3Path)}
if os.path.exists(args.order4Path):
    sources[4] = Runs(args.order4Path)
targets = {'stripe': (3, 'stripe', 'stripeTrained', 'stripeControl'), 'face order 3': (3, 'face', 'faceTrained', 'faceControl')}
if 4 in sources:
    targets['face order 4'] = (4, 'face', 'faceTrained', 'faceControl')
results = dict(note='EXPLORATORY; scored against data/canalizationTalkRecanalizationPredictions1888Hold301.json', validation={}, targets={}, predictions={}, exploratory={})
for order, runs in sources.items():
    results['validation'][f'V1 order {order}'] = check(runs, order)
for label, (order, target, trained, controlGroup) in targets.items():
    runs = sources[order]
    results['targets'][label] = scoreTarget(runs, label, target, trained, controlGroup)
    results['exploratory'][label] = explore(runs, label, target, trained, results['targets'][label]['controls'])


def trainedAt(label, condition, age):
    table = results['targets'][label]['table'][condition]
    return table[age][results['targets'][label]['trained']] if condition != 'ageZero' else table[results['targets'][label]['trained']]


def controlsAt(label, condition, age):
    table = results['targets'][label]['table'][condition][age]
    return [table[c] for c in results['targets'][label]['controls']]


allAges = PRIMARY + SECONDARY
faces = [label for label in targets if label.startswith('face')]
P = results['predictions']
p1a = {age: trainedAt('stripe', 'reapplied', age)['peakExact'] >= RETURNED for age in PRIMARY}
p1b = {age: trainedAt('stripe', 'reapplied', age)['familyShare'] >= FAMILY_SHARE_FLOOR
       and trainedAt('stripe', 'reapplied', age)['familyShare'] >= max(c['familyShare'] for c in controlsAt('stripe', 'reapplied', age)) for age in PRIMARY}
P['P1'] = dict(holds=all(p1a.values()) and all(p1b.values()), peakExactAtLeast0p9=p1a, familyShareCriterion=p1b,
               observed={str(age): dict(peakExact=trainedAt('stripe', 'reapplied', age)['peakExact'], familyShare=trainedAt('stripe', 'reapplied', age)['familyShare'],
                                        maxControlFamilyShare=max(c['familyShare'] for c in controlsAt('stripe', 'reapplied', age))) for age in PRIMARY})
P['P2'] = dict(holds=all(trainedAt(f, 'reapplied', age)['peakExact'] < RETURNED for f in faces for age in PRIMARY),
               observedPeakExact={f: {str(age): trainedAt(f, 'reapplied', age)['peakExact'] for age in PRIMARY} for f in faces})
P['P3'] = dict(holds=all(trainedAt('stripe', 'reapplied', age)['peakExact'] >= trainedAt(f, 'reapplied', age)['peakExact'] for f in faces for age in allAges),
               observedPeakExact={label: {str(age): trainedAt(label, 'reapplied', age)['peakExact'] for age in allAges} for label in targets})
stationaryAges = [3000, 5000, 10000, 20000]
returned = {label: {str(age): trainedAt(label, 'reapplied', age)['peakExact'] >= RETURNED for age in stationaryAges} for label in targets}
P['P4'] = dict(holds=all(len(set(v.values())) == 1 for v in returned.values()), returned=returned)
controlsReturned = {label: [dict(age=age, control=c, peakExact=m['peakExact']) for age in allAges for c, m in zip(results['targets'][label]['controls'], controlsAt(label, 'reapplied', age)) if m['peakExact'] >= RETURNED] for label in targets}
continuedReturned = {label: [age for age in allAges if trainedAt(label, 'continued', age)['peakExact'] >= RETURNED] for label in targets}
fromAge3000 = {label: [a for a in v if a >= 3000] for label, v in continuedReturned.items()}
P['P3']['agesWhereItFails'] = {f: [age for age in allAges if trainedAt('stripe', 'reapplied', age)['peakExact'] < trainedAt(f, 'reapplied', age)['peakExact']] for f in faces}
P['P5'] = dict(note='EXPLORATORY reading added after the results: at T = 1000 the unreapplied face runs return because the original run itself forms the face at its readout (iteration 2173 or 2143) inside tau 0-3299; the tissue is not yet in the ocean at T = 1000. Without that age the criterion holds.',
               holdsIfAgesFrom3000Only=not any(controlsReturned.values()) and not any(fromAge3000.values()), holds=not any(controlsReturned.values()) and not any(continuedReturned.values()), controlsThatReturned=controlsReturned, agesWhereTheContinuedRunReturned=continuedReturned)
zero = trainedAt('stripe', 'ageZero', 0)['firstExact']
first = {str(age): trainedAt('stripe', 'reapplied', age)['firstExact'] for age in PRIMARY}
P['P6'] = dict(holds=all(p1a.values()) and zero is not None and all(v is not None and abs(v - zero) <= TIMING_TOLERANCE for v in first.values()), firstExactAgeZero=zero, firstExactReapplied=first,
               applicable=all(p1a.values()))
results['summary'] = {k: v['holds'] for k, v in P.items()}
json.dump(results, open(args.outputPath, 'w'), indent=1)

# ------------------------------------------------------------------------------------------------------------ the printed table
print('V1:', {k: (v['codesCompared'], v['holds'], v['mismatches'][:3], 'dark sets identical:', v['darkSetsIdenticalInEveryFrame']) for k, v in results['validation'].items()})
print('\npeak exact IoU (tau 0-3299) of the trained code; returned = at least 0.9')
header = f'{"":14}' + ''.join(f'{"age " + str(a):>10}' for a in [0] + allAges)
print(header)
for label in targets:
    row = [trainedAt(label, 'ageZero', 0)['peakExact']] + [trainedAt(label, 'reapplied', a)['peakExact'] for a in allAges]
    print(f'{label:14}' + ''.join(f'{v:10.2f}' for v in row) + '   reapplied')
    print(f'{"":14}' + f'{"":10}' + ''.join(f'{trainedAt(label, "continued", a)["peakExact"]:10.2f}' for a in allAges) + '   continued, nothing applied')
print('\nfamily share (tau 301-2999) of the trained code; the largest of the 24 reapplied controls in brackets')
for label in targets:
    cells = [f'{trainedAt(label, "ageZero", 0)["familyShare"]:5.2f}      ']
    for a in allAges:
        cells.append(f'{trainedAt(label, "reapplied", a)["familyShare"]:5.2f} ({max(c["familyShare"] for c in controlsAt(label, "reapplied", a)):4.2f})')
    print(f'{label:14}' + ' '.join(f'{c:>12}' for c in cells))
print('\npredictions:', results['summary'])
print('wrote', args.outputPath)
