"""How far is each observed pattern from the stripe family and from the face family, and do the stripe set and the face set sit where one would expect? EXPLORATORY: nothing is registered; the measures were written first.

The expectation (the user's): the stripe set stays close to the stripe family and away from the face family, and the face set the other way round.

Families, as explicit sets of 9 x 9 interior patterns (so a distance to a family is a distance to a member):
  stripe family  a union of one, two or three vertical bars, each a solid run of 1 to 3 adjacent columns, runs separated by at least one empty column, every bar at least 3 cells tall; either (a) each bar's height is one of
                 3, 5, 7 or 9 rows centred on the middle row (the symmetric stripes the stripe code draws), or (b) all bars share one height of 3 to 9 rows at any vertical offset.
  face family    the deformed faces of analyzeCanalizationTalkFamilyVisits11x11.py: any union of at least two of the four face blocks (left eye, right eye, nose, mouth), each block moved independently by up to one cell.
Distance of a pattern to a family: 1 - (the largest IoU between the pattern's dark set and a member). 0 is a member, 1 shares nothing. "Closeness" is 1 - distance.

Data: the 75 runs of analyzeCanalizationTalkFamilyVisits11x11.py (64-bit, 20,000 iterations, a frame every 5). The stripe class is the stripe code, its 12 noisy copies and its 24 matched controls (tilt exactly 0, both mirror symmetries); the face class
likewise (orders 0-3, left-right symmetric). Frames from iteration 301 on, in the windows 301-3,000 (the window the codes were trained on), 3,000-10,000 and 10,000-20,000, and bins of 1,000.

Measures:
  calibrated  (added after the first run showed the raw comparison favours the larger family: the stripe family has 22,944 members and the face family 9,923, so even a random pattern is closer to the stripe family)
              every closeness is also expressed as its percentile among the closeness of all control frames (both classes, iterations 301 on) to the same family, so each axis has the same scale and a family's size drops out.
  own versus other  for each class and role (trained, copies, controls) and window: the mean closeness to the stripe family and to the face family, and the share of frames closer to the class's own family than to the other.
                    A stripe set is expected above the diagonal of "stripe closeness > face closeness", a face set below it.
  specificity       the same closeness for the near-trained runs against the matched controls of their own class (same symmetry), as a rank among the 24 controls and as a run-level bootstrap interval of the difference:
                    is the trained neighbourhood nearer its own family than generic codes of its symmetry are?
  clusters          from the two coordinates (stripe closeness, face closeness) alone, can a logistic regression assign a frame's run to the stripe class or the face class on runs it never saw (stratified group k-fold, 6
                    folds), pooled over each window, 100 frames per run, 200 label permutations for the null; and the same near-trained versus controls inside each class (balanced accuracy, 4 folds).
  time course       the mean closeness of each group to each family in bins of 1,000 iterations.

    python3 analyzeCanalizationTalkFamilyDistance11x11.py

Writes data/canalizationTalkFamilyDistance1888Hold301.json and data/canalizationTalkFamilyDistanceScores1888Hold301.npz (never overwriting).
"""
import argparse
import itertools
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import boundaryCodeUtilities as boundary
from canalizationTalkCommon import INTERIOR

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--trajectoryPath', type=str, default='data/canalizationTalkFamilyTrajectories1888Hold301.npz')
parser.add_argument('--outputPath', type=str, default='data/canalizationTalkFamilyDistance1888Hold301.json')
parser.add_argument('--scoresPath', type=str, default='data/canalizationTalkFamilyDistanceScores1888Hold301.npz')
parser.add_argument('--permutations', type=int, default=200)
parser.add_argument('--seed', type=int, default=20261004)
args = parser.parse_args()
for path in (args.outputPath, args.scoresPath):
    if os.path.exists(path):
        raise SystemExit(f'{path} exists; not overwriting')
rng = np.random.default_rng(args.seed)
interior = np.array(INTERIOR)

# ------------------------------------------------------------------------------------------------------------------ the families as explicit sets
columnSets = []
for mask in range(1, 512):
    columns = [(mask >> c) & 1 for c in range(9)]
    runs, length = [], 0
    for c in range(10):
        if c < 9 and columns[c]:
            length += 1
        elif length:
            runs.append((c - length, length))
            length = 0
    if len(runs) <= 3 and all(width <= 3 for _, width in runs):
        columnSets.append(runs)
stripeTemplates = set()
for runs in columnSets:
    for heights in itertools.product((3, 5, 7, 9), repeat=len(runs)):                              # (a) centred bars of independent height
        pattern = np.zeros((9, 9), bool)
        for (start, width), height in zip(runs, heights):
            first = (9 - height) // 2
            pattern[first:first + height, start:start + width] = True
        stripeTemplates.add(pattern.tobytes())
    for height in range(3, 10):                                                                    # (b) one shared height at any vertical offset
        for first in range(0, 10 - height):
            pattern = np.zeros((9, 9), bool)
            for start, width in runs:
                pattern[first:first + height, start:start + width] = True
            stripeTemplates.add(pattern.tobytes())
stripeTemplates = np.array([np.frombuffer(t, dtype=bool) for t in stripeTemplates]).astype(np.float32)

blocks = [np.isin(interior, boundary.featureParts[i]).reshape(9, 9) for i in range(4)]


def shifted(mask, dr, dc):
    out = np.zeros_like(mask)
    rows, columns = np.where(mask)
    rows, columns = rows + dr, columns + dc
    keep = (rows >= 0) & (rows < 9) & (columns >= 0) & (columns < 9)
    out[rows[keep], columns[keep]] = True
    return out


faceTemplates = []
for subset in itertools.chain.from_iterable(itertools.combinations(range(4), r) for r in (2, 3, 4)):
    for moves in itertools.product(itertools.product((-1, 0, 1), repeat=2), repeat=len(subset)):
        faceTemplates.append(np.logical_or.reduce([shifted(blocks[b], *move) for b, move in zip(subset, moves)]).reshape(-1))
faceTemplates = np.unique(np.array(faceTemplates), axis=0).astype(np.float32)
families = {'stripe': stripeTemplates, 'face': faceTemplates}
print('templates:', {k: len(v) for k, v in families.items()}, flush=True)

# ------------------------------------------------------------------------------------------------------------------ closeness of every frame
store = np.load(args.trajectoryPath, allow_pickle=True)
names, groups = list(store['names']), list(store['groups'])
stride = int(store['stride'])
dark = np.unpackbits(store['packed'], axis=2)[:, :, :81].astype(bool)
iteration = np.arange(dark.shape[1]) * stride
unique, inverse = np.unique(dark.reshape(-1, 81), axis=0, return_inverse=True)
inverse = inverse.reshape(dark.shape[:2])
print('frames', dark.shape[0] * dark.shape[1], 'unique patterns', len(unique), flush=True)
closeness = {}
for family, members in families.items():
    best = np.zeros(len(unique), np.float32)
    sizes = members.sum(1)
    for start in range(0, len(unique), 1500):
        part = unique[start:start + 1500].astype(np.float32)
        intersection = part @ members.T
        union = part.sum(1)[:, None] + sizes[None, :] - intersection
        best[start:start + 1500] = (intersection / np.maximum(union, 1)).max(1)
    closeness[family] = best[inverse]                                                                # runs x frames

# calibration: percentile of a frame's closeness among all control frames, per family
controlNames = [n for n in names if 'Control' in n]
selected = np.arange(dark.shape[1]) * stride >= 301
percentileReference = {f: np.sort(np.concatenate([closeness[f][names.index(n)][selected][::4] for n in controlNames])) for f in families}
percentile = {f: np.searchsorted(percentileReference[f], closeness[f], side='right') / len(percentileReference[f]) for f in families}

# reference: how close a random pattern of the typical density is to each family (the scale of the numbers)
reference = {}
for count in (8, 14, 20):
    patterns = np.zeros((2000, 81), np.float32)
    for row in patterns:
        row[rng.choice(81, count, replace=False)] = 1
    reference[str(count)] = {}
    for family, members in families.items():
        intersection = patterns @ members.T
        union = patterns.sum(1)[:, None] + members.sum(1)[None, :] - intersection
        reference[str(count)][family] = float(np.median((intersection / union).max(1)))

index = {n: k for k, n in enumerate(names)}
classOf = {k: (0 if g.startswith('stripe') else 1) for k, g in enumerate(groups) if g.startswith('stripe') or g.startswith('face')}
runs = np.array(sorted(classOf))
label = np.array([classOf[k] for k in runs])
role = np.array(['trained' if names[k].endswith('Trained') else ('copy' if 'Copy' in names[k] else 'control') for k in runs])
windows = {'inSample': (301, 3000), 'heldOutA': (3000, 10000), 'heldOutB': (10000, 20000)}
results = dict(note='EXPLORATORY; see the module docstring.', templates={k: int(len(v)) for k, v in families.items()}, uniquePatterns=int(len(unique)), randomPatternClosenessMedian=reference)

# ------------------------------------------------------------------------------------------------------------------ own versus other, per class, role and window
def windowMean(k, family, lo, hi):
    m = (iteration >= lo) & (iteration < hi)
    return float(closeness[family][k][m].mean())


ownOther = {}
for cls, key in ((0, 'stripe'), (1, 'face')):
    other = 'face' if key == 'stripe' else 'stripe'
    ownOther[key] = {}
    for window, (lo, hi) in windows.items():
        row = {}
        for roleName in ('trained', 'copy', 'control'):
            members = [runs[i] for i in range(len(runs)) if label[i] == cls and role[i] == roleName]
            own = np.array([windowMean(k, key, lo, hi) for k in members])
            oth = np.array([windowMean(k, other, lo, hi) for k in members])
            m = (iteration >= lo) & (iteration < hi)
            closerShare = float(np.mean([(closeness[key][k][m] > closeness[other][k][m]).mean() for k in members]))
            row[roleName] = dict(ownFamilyCloseness=float(own.mean()), otherFamilyCloseness=float(oth.mean()), shareOfFramesCloserToOwn=closerShare, runs=len(members))
        ownOther[key][window] = row
results['ownVersusOther'] = ownOther
calibrated = {}
for cls, key in ((0, 'stripe'), (1, 'face')):
    other = 'face' if key == 'stripe' else 'stripe'
    calibrated[key] = {}
    for window, (lo, hi) in windows.items():
        m = (iteration >= lo) & (iteration < hi)
        row = {}
        for roleName in ('trained', 'copy', 'control'):
            members = [runs[i] for i in range(len(runs)) if label[i] == cls and role[i] == roleName]
            row[roleName] = dict(ownPercentile=float(np.mean([percentile[key][k][m].mean() for k in members])), otherPercentile=float(np.mean([percentile[other][k][m].mean() for k in members])),
                                 ownAbove90=float(np.mean([(percentile[key][k][m] >= 0.9).mean() for k in members])), otherAbove90=float(np.mean([(percentile[other][k][m] >= 0.9).mean() for k in members])),
                                 ownMinusOtherPositive=float(np.mean([(percentile[key][k][m] > percentile[other][k][m]).mean() for k in members])))
        calibrated[key][window] = row
results['calibrated'] = calibrated

# ------------------------------------------------------------------------------------------------------------------ specificity: near-trained against matched controls of the same class
specificity = {}
for cls, key in ((0, 'stripe'), (1, 'face')):
    other = 'face' if key == 'stripe' else 'stripe'
    specificity[key] = {}
    controls = [runs[i] for i in range(len(runs)) if label[i] == cls and role[i] == 'control']
    near = [runs[i] for i in range(len(runs)) if label[i] == cls and role[i] != 'control']
    trained = [runs[i] for i in range(len(runs)) if label[i] == cls and role[i] == 'trained']
    for window, (lo, hi) in windows.items():
        row = {}
        for family in (key, other):
            controlValues = np.array([windowMean(k, family, lo, hi) for k in controls])
            nearValues = np.array([windowMean(k, family, lo, hi) for k in near])
            trainedValue = windowMean(trained[0], family, lo, hi)
            boot = [rng.choice(nearValues, len(nearValues)).mean() - rng.choice(controlValues, len(controlValues)).mean() for _ in range(2000)]
            row[f'{family}Family'] = dict(trained=trainedValue, nearTrainedMean=float(nearValues.mean()), controlsMean=float(controlValues.mean()), controlsAtLeastAsCloseAsTrained=int((controlValues >= trainedValue).sum()),
                                          controls=len(controlValues), differenceNearMinusControls=float(nearValues.mean() - controlValues.mean()), differenceInterval95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
        specificity[key][window] = row
results['specificity'] = specificity

# ------------------------------------------------------------------------------------------------------------------ clusters in the (stripe closeness, face closeness) plane
def balancedCV(features, y, groupsOfRows, folds, balanced):
    def run_(target, seed):
        guess = np.zeros(len(target))
        for train, test in StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(features, target, groupsOfRows):
            guess[test] = LogisticRegression(max_iter=2000, class_weight='balanced' if balanced else None).fit(features[train], target[train]).decision_function(features[test])
        predicted = (guess > 0).astype(int)
        return float(np.mean([(predicted[target == c] == c).mean() for c in (0, 1)]))
    return run_


clusters = {'classes': {}, 'within': {}}
for window, (lo, hi) in {'1000-4000': (1000, 4000), '4000-10000': (4000, 10000), '10000-20000': (10000, 20000)}.items():
    m = np.where((iteration >= lo) & (iteration < hi))[0]
    picked = [rng.choice(m, 100, replace=False) for _ in runs]
    coordinates = np.concatenate([np.stack([closeness['stripe'][k][p], closeness['face'][k][p]], 1) for k, p in zip(runs, picked)])
    scaled = StandardScaler().fit_transform(coordinates)
    y = np.concatenate([[label[i]] * len(p) for i, p in enumerate(picked)])
    rowRun = np.concatenate([[i] * len(p) for i, p in enumerate(picked)])
    score = balancedCV(scaled, y, rowRun, 6, False)
    accuracy = score(y, args.seed)
    null = np.array([score(rng.permutation(label)[rowRun], int(rng.integers(1e6))) for _ in range(args.permutations)])
    clusters['classes'][window] = dict(accuracy=accuracy, nullMean=float(null.mean()), null95=float(np.percentile(null, 95)), pValue=float((1 + (null >= accuracy).sum()) / (1 + len(null))))
    print(f'classes {window:12s} accuracy from the two closeness coordinates {accuracy:.3f}  null mean {null.mean():.3f}, 95th {np.percentile(null, 95):.3f}', flush=True)
    for cls, key in ((0, 'stripe'), (1, 'face')):
        subset = np.where(label == cls)[0]
        near = np.array([role[i] != 'control' for i in subset]).astype(int)
        coordinatesWithin = np.concatenate([np.stack([closeness['stripe'][runs[i]][picked[i]], closeness['face'][runs[i]][picked[i]]], 1) for i in subset])
        scaledWithin = StandardScaler().fit_transform(coordinatesWithin)
        yWithin = np.concatenate([[near[j]] * 100 for j in range(len(subset))])
        rowWithin = np.concatenate([[j] * 100 for j in range(len(subset))])
        score = balancedCV(scaledWithin, yWithin, rowWithin, 4, True)
        accuracy = score(yWithin, args.seed)
        null = np.array([score(rng.permutation(near)[rowWithin], int(rng.integers(1e6))) for _ in range(args.permutations)])
        clusters['within'].setdefault(key, {})[window] = dict(balancedAccuracy=accuracy, nullMean=float(null.mean()), null95=float(np.percentile(null, 95)), pValue=float((1 + (null >= accuracy).sum()) / (1 + len(null))))
        print(f'within  {key:6s} {window:12s} near-trained vs controls from the two coordinates: {accuracy:.3f}  null 95th {np.percentile(null, 95):.3f}', flush=True)
results['clusters'] = clusters

# ------------------------------------------------------------------------------------------------------------------ time course and plotting scores
edges = np.array([300] + list(range(1000, 20001, 1000)))
timeCourse = dict(edges=edges.tolist())
for cls, key in ((0, 'stripe'), (1, 'face')):
    timeCourse[key] = {}
    for roleName in ('trained', 'copy', 'control'):
        members = [runs[i] for i in range(len(runs)) if label[i] == cls and role[i] == roleName]
        timeCourse[key][roleName] = {family: [float(np.mean([closeness[family][k][(iteration >= lo) & (iteration < hi)].mean() for k in members])) for lo, hi in zip(edges[:-1], edges[1:])] for family in families}
        timeCourse[key][roleName].update({f'{family}Percentile': [float(np.mean([percentile[family][k][(iteration >= lo) & (iteration < hi)].mean() for k in members])) for lo, hi in zip(edges[:-1], edges[1:])] for family in families})
results['timeCourse'] = timeCourse
scores = {'label': label, 'role': role, 'runNames': np.array([names[k] for k in runs])}
for window, (lo, hi) in windows.items():
    m = np.where((iteration >= lo) & (iteration < hi))[0]
    picked = [rng.choice(m, 60, replace=False) for _ in runs]
    scores[f'{window}_stripe'] = np.concatenate([closeness['stripe'][k][p] for k, p in zip(runs, picked)]).astype(np.float32)
    scores[f'{window}_face'] = np.concatenate([closeness['face'][k][p] for k, p in zip(runs, picked)]).astype(np.float32)
    scores[f'{window}_stripePercentile'] = np.concatenate([percentile['stripe'][k][p] for k, p in zip(runs, picked)]).astype(np.float32)
    scores[f'{window}_facePercentile'] = np.concatenate([percentile['face'][k][p] for k, p in zip(runs, picked)]).astype(np.float32)
    scores[f'{window}_run'] = np.concatenate([[i] * 60 for i in range(len(runs))])
scores['trainedStripeCourse'] = np.stack([percentile['stripe'][index['stripeTrained']], percentile['face'][index['stripeTrained']]])
scores['trainedFaceCourse'] = np.stack([percentile['stripe'][index['faceTrained']], percentile['face'][index['faceTrained']]])
json.dump(results, open(args.outputPath, 'w'), indent=1)
np.savez_compressed(args.scoresPath, **scores)
print('wrote', args.outputPath)
