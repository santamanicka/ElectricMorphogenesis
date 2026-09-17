"""Do the 384 trained band-hold boundary codes carry a code for face quality? (PolyPatterning_Sim.md, Section 12)

Reads the face scores from scoreBandHoldFaceMetrics11x11.py and the codes from the checkpoints. Groups are
mechanism x depth (96 seeds each: 4 training objectives x 2 hold durations x 12 seeds). A seed's code is the
set of 21 (depth 1) or 38 (depth 2) independent left-half G_pol values of its two-fold symmetric band; its
"shape" is that code with its own mean removed, and its dial is that mean. Each face metric is analysed with:
  1. extreme-split classification: can the code's shape tell the best 32 faces from the worst 32?
  2. the dial: Spearman correlation with face quality, pooled, configuration-controlled and per objective
  3. transfer: a discriminant fitted on the original cohort (correlation, globalsum) applied to the new one
  4. the discriminant direction itself, compared across metrics
plus code structure against a random-initialisation null, and checks for a training-objective confound.
All classifiers are L2 logistic regression (C = 0.1) on standardised shapes, 6-fold stratified CV.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--metricsPath', type=str, default='data/bandHoldFaceMetrics11x11_all384.csv')
parser.add_argument('--numPermutations', type=int, default=300)
parser.add_argument('--randomSeed', type=int, default=7)
args = parser.parse_args()

metrics = pd.read_csv(args.metricsPath).sort_values('fileNumber').reset_index(drop=True)
codes = boundary.loadBandHoldCodes(metrics.fileNumber.tolist())
randomGenerator = np.random.default_rng(args.randomSeed)
faceMetrics = [('featureRMS_mV', False), ('balancedRMS_mV', False), ('structuralIoU', True), ('partScore', True)]
groups = [('Gpol-only', 1), ('Gpol-only', 2), ('Gpol+Vmem', 1), ('Gpol+Vmem', 2)]


def members(mechanism, depth, lossMethodSubset=boundary.lossMethods):
    return metrics[(metrics.mechanism == mechanism) & (metrics.depth == depth) &
                   metrics.lossMethod.isin(lossMethodSubset)].reset_index(drop=True)


def foldedCodes(rows, channels=('Gpol',)):
    return np.stack([np.concatenate([codes[number]['folded'][channel] for channel in channels
                                     if channel in codes[number]['folded']]) for number in rows.fileNumber])


def shapes(codeMatrix):
    return codeMatrix - codeMatrix.mean(1, keepdims=True)


def goodFirst(values, higherIsBetter):
    """Values oriented so that lower means a better face."""
    return -np.asarray(values) if higherIsBetter else np.asarray(values)


def extremes(orientedValues, count):
    order = np.argsort(orientedValues)
    return np.concatenate([order[:count], order[-count:]]), np.concatenate([np.zeros(count, int), np.ones(count, int)])


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))


def crossValidatedAccuracy(features, labels, seed=0):
    return cross_val_score(classifier(), features, labels, scoring='accuracy',
                           cv=StratifiedKFold(6, shuffle=True, random_state=seed)).mean()


def accuracyWithPermutationTest(features, labels):
    observed = crossValidatedAccuracy(features, labels)
    null = np.array([crossValidatedAccuracy(features, randomGenerator.permutation(labels), seed=index)
                     for index in range(args.numPermutations)])
    return observed, (1 + (null >= observed).sum()) / (1 + len(null))


def discriminant(shapeMatrix, orientedValues, count):
    chosen, labels = extremes(orientedValues, count)
    scaler = StandardScaler().fit(shapeMatrix[chosen])
    return scaler, LogisticRegression(C=0.1, max_iter=5000).fit(scaler.transform(shapeMatrix[chosen]), labels)


print(f"{len(metrics)} checkpoints; permutation tests use {args.numPermutations} shuffles (seed {args.randomSeed})")

# ---------------------------------------------------------------------------- 1. extreme-split classification
print("\n1. EXTREME-SPLIT CLASSIFICATION: best 32 vs worst 32 of 96 (best 16 vs worst 16 within a cohort of 48)")
print(f"{'metric':16s} {'group':14s} {'all four':>17s} {'original cohort':>17s} {'new cohort':>17s}")
for metricName, higherIsBetter in faceMetrics:
    for mechanism, depth in groups:
        cells = []
        for subset, count in ((boundary.lossMethods, 32), (boundary.originalCohortLossMethods, 16), (boundary.newCohortLossMethods, 16)):
            rows = members(mechanism, depth, subset)
            chosen, labels = extremes(goodFirst(rows[metricName], higherIsBetter), count)
            accuracy, pValue = accuracyWithPermutationTest(shapes(foldedCodes(rows))[chosen], labels)
            cells.append(f"{accuracy:.3f} (p={pValue:.3f})")
        print(f"{metricName:16s} {mechanism + ' D' + str(depth):14s} " + " ".join(f"{cell:>17s}" for cell in cells))

# ---------------------------------------------------------------------------------------------- 2. the dial
print("\n2. THE DIAL (mean of the independent left-half G_pol values) vs face quality; positive = more dial, worse face")
for metricName, higherIsBetter in faceMetrics:
    for mechanism, depth in groups:
        rows = members(mechanism, depth)
        dial = foldedCodes(rows).mean(1)
        oriented = goodFirst(rows[metricName], higherIsBetter)
        pooled = spearmanr(dial, oriented)
        configurations = rows.holdIterations.astype(str) + '/' + rows.lossMethod
        controlled = spearmanr(boundary.rankWithinGroups(dial, configurations), boundary.rankWithinGroups(oriented, configurations))
        perObjective = []
        for lossMethod in boundary.lossMethods:
            inObjective = rows.lossMethod == lossMethod
            result = spearmanr(dial[inObjective], oriented[inObjective])
            perObjective.append(f"{result.statistic:+.3f}{'*' if result.pvalue < 0.05 else ' '}")
        print(f"{metricName:16s} {mechanism + ' D' + str(depth):14s} pooled {pooled.statistic:+.3f} (p={pooled.pvalue:.1g})  "
              f"controlled {controlled.statistic:+.3f} (p={controlled.pvalue:.1g})  by objective " + " ".join(perObjective))
print("   by objective = correlation, globalsum, facialFeatureOnly, facialFeatureBalanced (n = 24; * p < 0.05)")
rows = members('Gpol-only', 1)
for component in ('partCoverage', 'partSeparation', 'partSpurious'):
    result = spearmanr(foldedCodes(rows).mean(1), rows[component])
    print(f"   Gpol-only D1, dial vs {component}: rho={result.statistic:+.3f} p={result.pvalue:.4f}")

# ------------------------------------------------------------------------------ 3. transfer across cohorts
print("\n3. TRANSFER (Gpol-only D1): discriminant fitted on the original cohort's extremes, applied to the new cohort;"
      " positive rho = the projection ranks new faces the same way")
original, new = members('Gpol-only', 1, boundary.originalCohortLossMethods), members('Gpol-only', 1, boundary.newCohortLossMethods)
originalShapes, newShapes = shapes(foldedCodes(original)), shapes(foldedCodes(new))
for metricName, higherIsBetter in faceMetrics:
    scaler, model = discriminant(originalShapes, goodFirst(original[metricName], higherIsBetter), 16)
    projection = model.decision_function(scaler.transform(newShapes))
    newOriented = goodFirst(new[metricName], higherIsBetter)
    result = spearmanr(projection, newOriented)
    chosen, labels = extremes(newOriented, 16)
    withinObjectives = []
    for lossMethod in boundary.newCohortLossMethods:
        inObjective = (new.lossMethod == lossMethod).values
        within = spearmanr(projection[inObjective], newOriented[inObjective])
        withinObjectives.append(f"{lossMethod} {within.statistic:+.3f} (p={within.pvalue:.3f})")
    print(f"   {metricName:16s} rho={result.statistic:+.3f} p={result.pvalue:.4f}; new extremes classified "
          f"{((projection[chosen] > 0).astype(int) == labels).mean():.3f}; within " + ", ".join(withinObjectives))

# ---------------------------------------------------------------------------- 4. discriminant directions
print("\n4. DISCRIMINANT DIRECTIONS (Gpol-only D1, fitted on all 96 extremes); Spearman agreement between metrics")
allRows = members('Gpol-only', 1)
allShapes = shapes(foldedCodes(allRows))
weights = {metricName: discriminant(allShapes, goodFirst(allRows[metricName], higherIsBetter), 32)[1].coef_[0]
           for metricName, higherIsBetter in faceMetrics}
names = [name for name, _ in faceMetrics]
for first in range(len(names)):
    for second in range(first + 1, len(names)):
        print(f"   {names[first]:16s} vs {names[second]:16s}: rho={spearmanr(weights[names[first]], weights[names[second]]).statistic:+.3f}")
originalWeights = discriminant(originalShapes, goodFirst(original['featureRMS_mV'], False), 16)[1].coef_[0]
print(f"   featureRMS_mV fitted on the original 48 vs on all 96: rho={spearmanr(originalWeights, weights['featureRMS_mV']).statistic:+.3f}")
print("   metric agreement across the 96 seeds (oriented so positive = agree):")
for first in range(len(names)):
    for second in range(first + 1, len(names)):
        agreement = spearmanr(goodFirst(allRows[names[first]], faceMetrics[first][1]), goodFirst(allRows[names[second]], faceMetrics[second][1]))
        print(f"     {names[first]:16s} vs {names[second]:16s}: rho={agreement.statistic:+.3f}")

# ------------------------------------------------------------------ 5. code structure vs random-initialisation null
print("\n5. CODE STRUCTURE: held-out amplitude of the leading PCA modes vs iid-uniform data of the same shape (5-fold)")
nullGenerator = np.random.default_rng(202)


def heldOutModeAmplitudes(matrix):
    amplitudes = []
    for trainIndices, testIndices in KFold(5, shuffle=True, random_state=0).split(matrix):
        scaler = StandardScaler().fit(matrix[trainIndices])
        pca = PCA().fit(scaler.transform(matrix[trainIndices]))
        amplitudes.append(pca.transform(scaler.transform(matrix[testIndices])).std(0))
    shortest = min(len(amplitude) for amplitude in amplitudes)
    return np.mean([amplitude[:shortest] for amplitude in amplitudes], 0)


for mechanism, depth in groups:
    channels = ('Gpol', 'Vmem') if mechanism == 'Gpol+Vmem' else ('Gpol',)
    matrix = foldedCodes(members(mechanism, depth), channels)
    observed, null = heldOutModeAmplitudes(matrix), heldOutModeAmplitudes(nullGenerator.random(matrix.shape))
    print(f"   {mechanism + ' D' + str(depth):14s} {matrix.shape[1]:3d} values: mode 1 {observed[0]:.3f}, mode 2 {observed[1]:.3f}; "
          f"null mode 1 {null[0]:.3f}; ratio {observed[0] / null[0]:.2f}")

# ----------------------------------------------------------------------- 6. training-objective confound
print("\n6. CONFOUND CHECKS (Gpol-only D1)")
cohortLabels = allRows.lossMethod.isin(boundary.newCohortLossMethods).values.astype(int)
accuracy, pValue = accuracyWithPermutationTest(allShapes, cohortLabels)
print(f"   can the code's shape tell bulk-trained from feature-trained seeds? accuracy {accuracy:.3f} (p={pValue:.3f})")
for metricName, higherIsBetter in faceMetrics:
    cells = []
    for lossMethod in boundary.lossMethods:
        rows = members('Gpol-only', 1, (lossMethod,))
        chosen, labels = extremes(goodFirst(rows[metricName], higherIsBetter), 8)
        accuracy, pValue = accuracyWithPermutationTest(shapes(foldedCodes(rows))[chosen], labels)
        cells.append(f"{lossMethod} {accuracy:.3f} (p={pValue:.3f})")
    print(f"   {metricName:16s} best 8 vs worst 8 within each objective: " + "; ".join(cells))
