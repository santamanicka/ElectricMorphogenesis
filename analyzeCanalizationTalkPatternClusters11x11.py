"""Do the stripe-class and face-class pattern sets form different clusters, and do they merge over long horizons? EXPLORATORY: nothing is registered; the measures were written before the first run.

Data: the 75 runs of analyzeCanalizationTalkFamilyVisits11x11.py (64-bit, 20,000 iterations, interior dark set every 5 iterations). The stripe class is the stripe code, its 12 noisy copies and its 24 symmetry-matched controls;
the face class likewise (37 runs each). A frame is the 81-cell interior dark / light pattern. Frames from iteration 301 on are used.

Two feature sets, because the classes differ in symmetry: stripe-class codes keep both mirror symmetries, face-class codes only the left-right one, so the face class can leave the top-bottom-symmetric subspace and the stripe
class cannot, with no help from the patterns' content.
  raw          the 81 dark / light values.
  symmetrised  each pattern averaged with its top-bottom and left-right mirror images (a 9 x 9 map with values 0, 1/4, 1/2, 3/4, 1): the same content with the symmetry difference removed.

PCA: fitted once on every fourth frame of all 74 non-baseline runs, centred, on the chosen features; the first 10 components are kept. Reported: the variance they explain.

Separation, per time bin (1,000 iterations, the first from 301 to 1,000), 60 frames per run per bin (equal bins, so a wider bin cannot inflate the overlap):
  accuracy    a logistic regression on the 10 PCs separates the two classes, scored on runs it never saw (stratified group k-fold over runs, 6 folds): 0.5 is merged, 1 is fully separate. The null is the same procedure
              with the class labels permuted across runs (--permutations draws): its mean and 95th percentile say what "merged" looks like at this sample size.
  dPrime      the distance between the classes' mean out-of-fold decision scores over their pooled standard deviation.
  overlap     the histogram intersection of the two classes' densities in the plane of PC 1 and PC 2 (20 x 20 bins fixed over all time): 1 is merged, 0 is disjoint.
  centroid    the distance between the class centroids in the 10-PC space over the pooled within-class spread (root mean square distance to the class centroid).
  pooled      (added after the per-bin results were seen, so exploratory in a second sense) the same accuracy over whole windows (1,000-4,000; 4,000-10,000; 10,000-20,000), 100 frames per run, with a permutation
              p-value from 200 label permutations: the per-bin values are noisy because a run's frames are strongly correlated.
  within      (added after the pooled result: averaging a top-bottom-asymmetric pattern with its mirror images leaves fractional values, so the symmetrised features still carry a trace of the symmetry difference) the
              same pooled test inside each symmetry class: can the near-trained runs (the trained code and its 12 noisy copies) be told from the 24 matched controls on runs not seen, by balanced accuracy, on the raw and on the
              symmetrised features (the stripe controls have a tilt of exactly 0 and stay exactly top-bottom symmetric, the trained stripe code and its copies have a tilt of 0.0012 and slowly do not, so the raw
              stripe test can separate "exactly symmetric" from "slightly broken"), 200 label permutations. This asks whether the neighbourhood of a trained code is a realm of its own, with the symmetry held fixed.
  members     the share of the trained code's, the copies' and the controls' frames that the classifier (fitted on other runs) assigns to their own class.

    python3 analyzeCanalizationTalkPatternClusters11x11.py

Writes data/canalizationTalkPatternClusters1888Hold301.json and data/canalizationTalkPatternClusterScores1888Hold301.npz (PC scores for plotting; never overwriting).
"""
import argparse
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--trajectoryPath', type=str, default='data/canalizationTalkFamilyTrajectories1888Hold301.npz')
parser.add_argument('--outputPath', type=str, default='data/canalizationTalkPatternClusters1888Hold301.json')
parser.add_argument('--scoresPath', type=str, default='data/canalizationTalkPatternClusterScores1888Hold301.npz')
parser.add_argument('--components', type=int, default=10)
parser.add_argument('--framesPerRun', type=int, default=60)
parser.add_argument('--seed', type=int, default=20261004)
parser.add_argument('--permutations', type=int, default=20)
args = parser.parse_args()
for path in (args.outputPath, args.scoresPath):
    if os.path.exists(path):
        raise SystemExit(f'{path} exists; not overwriting')
rng = np.random.default_rng(args.seed)

store = np.load(args.trajectoryPath, allow_pickle=True)
names, groups = list(store['names']), list(store['groups'])
stride = int(store['stride'])
dark = np.unpackbits(store['packed'], axis=2)[:, :, :81].astype(np.float32)             # runs x frames x 81
iteration = np.arange(dark.shape[1]) * stride
classOf = {}
for k, (name, group) in enumerate(zip(names, groups)):
    if group.startswith('stripe'):
        classOf[k] = 0
    elif group.startswith('face'):
        classOf[k] = 1
runs = np.array(sorted(classOf))
label = np.array([classOf[k] for k in runs])
role = np.array(['trained' if names[k].endswith('Trained') else ('copy' if 'Copy' in names[k] else 'control') for k in runs])
edges = np.array([300] + list(range(1000, 20001, 1000)))


def symmetrise(patterns):
    square = patterns.reshape(-1, 9, 9)
    return ((square + square[:, ::-1, :] + square[:, :, ::-1] + square[:, ::-1, ::-1]) / 4).reshape(-1, 81)


features = {'raw': lambda x: x, 'symmetrised': symmetrise}
results = dict(note='EXPLORATORY; see the module docstring.', edges=edges.tolist(), classes={'stripe': int((label == 0).sum()), 'face': int((label == 1).sum())})
scores = {}
projectors = {}
for featureName, transform in features.items():
    fitFrames = np.concatenate([transform(dark[k][(iteration >= 301)][::4]) for k in runs])
    mean = fitFrames.mean(0)
    _, singular, vt = np.linalg.svd(fitFrames - mean, full_matrices=False)
    axes = vt[:args.components]
    explained = (singular ** 2 / (singular ** 2).sum())[:args.components]
    project = lambda frames: (transform(frames) - mean) @ axes.T
    block = dict(varianceExplained=explained.tolist(), cumulative=float(explained.sum()), bins=[])
    # fixed 2-D histogram grid for the overlap measure
    every = np.concatenate([project(dark[k][iteration >= 301][::8]) for k in runs])[:, :2]
    low, high = every.min(0), every.max(0)
    grid = [np.linspace(low[i], high[i], 21) for i in range(2)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = np.where((iteration >= lo) & (iteration < hi))[0]
        picked = [rng.choice(mask, min(args.framesPerRun, len(mask)), replace=False) for _ in runs]
        pcs = np.concatenate([project(dark[k][p]) for k, p in zip(runs, picked)])
        y = np.concatenate([[label[i]] * len(p) for i, p in enumerate(picked)])
        run = np.concatenate([[i] * len(p) for i, p in enumerate(picked)])
        scaled = StandardScaler().fit_transform(pcs)
        decision = np.zeros(len(y))
        for train, test in StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=args.seed).split(scaled, y, run):
            model = LogisticRegression(max_iter=2000, C=1.0).fit(scaled[train], y[train])
            decision[test] = model.decision_function(scaled[test])
        predicted = (decision > 0).astype(int)
        accuracy = float((predicted == y).mean())
        nullAccuracies = []
        for _ in range(args.permutations):                                                   # class labels permuted across runs
            shuffled = rng.permutation(label)[run]
            guess = np.zeros(len(y))
            for train, test in StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=int(rng.integers(1e6))).split(scaled, shuffled, run):
                guess[test] = LogisticRegression(max_iter=2000, C=1.0).fit(scaled[train], shuffled[train]).decision_function(scaled[test])
            nullAccuracies.append(float(((guess > 0).astype(int) == shuffled).mean()))
        d0, d1 = decision[y == 0], decision[y == 1]
        dPrime = float((d1.mean() - d0.mean()) / np.sqrt((d0.var() + d1.var()) / 2))
        histograms = [np.histogram2d(pcs[y == c][:, 0], pcs[y == c][:, 1], bins=grid)[0] for c in (0, 1)]
        p, q = [h / max(h.sum(), 1) for h in histograms]
        overlap = float(np.minimum(p, q).sum())
        centroids = [pcs[y == c].mean(0) for c in (0, 1)]
        spread = np.sqrt(np.mean([((pcs[y == c] - centroids[c]) ** 2).sum(1).mean() for c in (0, 1)]))
        members = {}
        for r in ('trained', 'copy', 'control'):
            own = [(predicted[run == i] == label[i]).mean() for i in range(len(runs)) if role[i] == r]
            members[r] = float(np.mean(own))
        block['bins'].append(dict(start=int(lo), end=int(hi), accuracy=accuracy, nullMean=float(np.mean(nullAccuracies)), null95=float(np.percentile(nullAccuracies, 95)), dPrime=dPrime, overlap=overlap, centroidOverSpread=float(np.linalg.norm(centroids[0] - centroids[1]) / spread), members=members))
    # scores for plotting: 40 frames per run per window
    for window, (lo, hi) in {'inSample': (301, 3000), 'heldOutA': (3000, 10000), 'heldOutB': (10000, 20000)}.items():
        mask = np.where((iteration >= lo) & (iteration < hi))[0]
        picked = [rng.choice(mask, 40, replace=False) for _ in runs]
        scores[f'{featureName}_{window}'] = np.concatenate([project(dark[k][p])[:, :3] for k, p in zip(runs, picked)]).astype(np.float32)
        scores[f'{featureName}_{window}_run'] = np.concatenate([[i] * 40 for i in range(len(runs))])
    results[featureName] = block
    projectors[featureName] = project
pooled = {}
for featureName in features:
    pooled[featureName] = {}
    for windowName, (lo, hi) in {'1000-4000': (1000, 4000), '4000-10000': (4000, 10000), '10000-20000': (10000, 20000)}.items():
        mask = np.where((iteration >= lo) & (iteration < hi))[0]
        picked = [rng.choice(mask, 100, replace=False) for _ in runs]
        pcs = np.concatenate([projectors[featureName](dark[k][p_]) for k, p_ in zip(runs, picked)])
        y = np.concatenate([[label[i]] * len(p_) for i, p_ in enumerate(picked)])
        run = np.concatenate([[i] * len(p_) for i, p_ in enumerate(picked)])
        scaled = StandardScaler().fit_transform(pcs)

        def crossValidated(target, seed):
            guess = np.zeros(len(target))
            for train, test in StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed).split(scaled, target, run):
                guess[test] = LogisticRegression(max_iter=2000).fit(scaled[train], target[train]).decision_function(scaled[test])
            return float(((guess > 0).astype(int) == target).mean())
        accuracy = crossValidated(y, args.seed)
        null = np.array([crossValidated(rng.permutation(label)[run], int(rng.integers(1e6))) for _ in range(200)])
        pooled[featureName][windowName] = dict(accuracy=accuracy, nullMean=float(null.mean()), null95=float(np.percentile(null, 95)), nullMax=float(null.max()), pValue=float((1 + (null >= accuracy).sum()) / (1 + len(null))))
        print(f'pooled {featureName:12s} {windowName:12s} accuracy {accuracy:.3f}  null mean {null.mean():.3f}, 95th {np.percentile(null, 95):.3f}, max {null.max():.3f}  p = {pooled[featureName][windowName]["pValue"]:.3f}', flush=True)
results['pooled'] = pooled
within = {}
for featureName in ('raw', 'symmetrised'):
    within[featureName] = {}
    for classNumber, key in ((0, 'stripe'), (1, 'face')):
        subset = np.where(label == classNumber)[0]
        near = np.array([role[i] != 'control' for i in subset])
        within[featureName][key] = {}
        for windowName, (lo, hi) in {'1000-4000': (1000, 4000), '4000-10000': (4000, 10000), '10000-20000': (10000, 20000)}.items():
            mask = np.where((iteration >= lo) & (iteration < hi))[0]
            picked = [rng.choice(mask, 100, replace=False) for _ in subset]
            pcs = np.concatenate([projectors[featureName](dark[runs[i]][p_]) for i, p_ in zip(subset, picked)])
            y = np.concatenate([[int(near[j])] * len(p_) for j, p_ in enumerate(picked)])
            run = np.concatenate([[j] * len(p_) for j, p_ in enumerate(picked)])
            scaled = StandardScaler().fit_transform(pcs)

            def balanced(target, seed):
                guess = np.zeros(len(target))
                for train, test in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(scaled, target, run):
                    guess[test] = LogisticRegression(max_iter=2000, class_weight='balanced').fit(scaled[train], target[train]).decision_function(scaled[test])
                predicted = (guess > 0).astype(int)
                return float(np.mean([(predicted[target == c] == c).mean() for c in (0, 1)]))
            accuracy = balanced(y, args.seed)
            null = np.array([balanced(rng.permutation(near.astype(int))[run], int(rng.integers(1e6))) for _ in range(200)])
            within[featureName][key][windowName] = dict(balancedAccuracy=accuracy, nullMean=float(null.mean()), null95=float(np.percentile(null, 95)), nullMax=float(null.max()), pValue=float((1 + (null >= accuracy).sum()) / (1 + len(null))))
            print(f'within {featureName:12s} {key:6s} {windowName:12s} near-trained vs controls: balanced accuracy {accuracy:.3f}  null mean {null.mean():.3f}, 95th {np.percentile(null, 95):.3f}  p = {within[featureName][key][windowName]["pValue"]:.3f}', flush=True)
results['within'] = within
scores['label'], scores['role'], scores['runNames'] = label, role, np.array([names[k] for k in runs])
json.dump(results, open(args.outputPath, 'w'), indent=1)
np.savez_compressed(args.scoresPath, **scores)
for featureName in features:
    print(f'== {featureName}: first {args.components} PCs explain {results[featureName]["cumulative"]:.2f} of the variance')
    for b in results[featureName]['bins']:
        print(f'  {b["start"]:5d}-{b["end"]:5d}  accuracy {b["accuracy"]:.2f} (null mean {b["nullMean"]:.2f}, 95th {b["null95"]:.2f})  dPrime {b["dPrime"]:.2f}  overlap {b["overlap"]:.2f}  centroid/spread {b["centroidOverSpread"]:.2f}  own-class share: trained {b["members"]["trained"]:.2f} copies {b["members"]["copy"]:.2f} controls {b["members"]["control"]:.2f}')
