"""Analyse simulateBoundaryDialLandscape11x11.py: the dial across its physical range, and first-order gradients
within each single-cell regime (PolyPatterning_Sim.md, Section 12).

Dial sweep: for each dial level, which branch the ring and interior cells take (Vmem below or above the single-cell
saddle), where interior G_pol settles relative to the single-cell bistable window, pattern contrast, ongoing churn,
face scores, and where the late-window pattern jumps between neighbouring dial levels.

Regime pairs (once simulated): per regime, how strongly gradients act, whether they reach patterns no dial level
produces, how many dimensions they open, how much of their effect is fully symmetric, which modes are predictable from
the settings, what they do to face parts, and, on each rotation grid, how much of the response turns with the gradient.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--jumpThresholdMilliVolts', type=float, default=2.0)
parser.add_argument('--numPermutations', type=int, default=10)
parser.add_argument('--holdIterations', type=int, default=None,
                    help="hold length the runs used (default: the reference checkpoint's own); a different value reads and writes Hold<n> files")
parser.add_argument('--summaryPath', type=str, default=None,
                    help='JSON summary for plotBoundaryDial11x11.py (default data/boundaryDialLandscapeSummary<checkpoint>[Hold<n>].json)')
args = parser.parse_args()

reference = boundary.loadCheckpoint(args.referenceCheckpoint)
target = boundary.targetVmemMilliVolts(reference)
lowerThreshold, upperThreshold = boundary.singleCellBistableRange
ring, interior = boundary.boundaryRingCells, boundary.interiorCellIndices
referenceHoldIterations = int(reference['clampParameters']['clampEndIter']) + 1
holdIterations = args.holdIterations or referenceHoldIterations
holdTag = '' if holdIterations == referenceHoldIterations else f'Hold{holdIterations}'
sweep = np.load(f'data/boundaryDialSweep{args.referenceCheckpoint}{holdTag}.npz')
dials, patterns = sweep['dialLevel'], sweep['windowMeanVmem']


def rootMeanSquare(values):
    return np.sqrt((np.asarray(values) ** 2).mean(-1))


def regimeOf(dialLevel):
    return 'below' if dialLevel < lowerThreshold else ('inside' if dialLevel <= upperThreshold else 'above')


# ====================================================================================== dial sweep
print(f"=== DIAL SWEEP on checkpoint {args.referenceCheckpoint}, hold {holdIterations} iterations: {len(dials)} levels from {dials.min():.2f} to {dials.max():.2f}; "
      f"single-cell bistable window {lowerThreshold}-{upperThreshold} ===")
saddle = boundary.singleCellSaddleMilliVolts
print(f"\n{'dial':>5s} {'regime':>7s} {'ring hyp':>9s} {'int hyp':>8s} {'int hyp':>8s} {'int Gpol':>9s} {'in window':>10s} "
      f"{'contrast':>9s} {'churn':>6s} {'cover':>6s} {'separ':>6s} {'part':>6s} {'balRMS':>7s}")
print(f"{'':>5s} {'':>7s} {'(hold)':>9s} {'(hold)':>8s} {'(late)':>8s} {'(late)':>9s} {'(late)':>10s} {'(mV)':>9s} {'(mV)':>6s}")
for index, dialLevel in enumerate(dials):
    if not np.isclose((dialLevel * 10) % 1, 0, atol=1e-6) and not np.isclose((dialLevel * 10) % 1, 1, atol=1e-6):
        continue
    late, lateConductance = patterns[index], sweep['windowMeanGpol'][index]
    score, coverage, separation, _ = boundary.partSeparationScore(late)
    inWindow = np.mean((lateConductance[interior] >= lowerThreshold) & (lateConductance[interior] <= upperThreshold))
    print(f"{dialLevel:5.2f} {regimeOf(dialLevel):>7s} {np.mean(sweep['endOfHoldVmem'][index][ring] < saddle):9.2f} "
          f"{np.mean(sweep['endOfHoldVmem'][index][interior] < saddle):8.2f} {np.mean(late[interior] < saddle):8.2f} "
          f"{lateConductance[interior].mean():9.3f} {inWindow:10.2f} {late[interior].std():9.2f} "
          f"{sweep['windowStdVmem'][index][interior].mean():6.2f} {coverage:6.2f} {separation:6d} {score:6.2f} "
          f"{boundary.balancedRootMeanSquareError(late, target):7.2f}")
print("   ring/int hyp = fraction of ring / interior cells below the single-cell saddle (hyperpolarised branch); "
      "int Gpol = mean interior G_pol/G_ref; in window = fraction of interior cells inside the bistable window; "
      "contrast = spatial SD of interior Vmem; churn = mean per-cell SD over the late window")

steps = rootMeanSquare(patterns[1:] - patterns[:-1])
jumpIndices = np.where(steps > args.jumpThresholdMilliVolts)[0]
print(f"\nTipping points (late-window pattern changes > {args.jumpThresholdMilliVolts} mV RMS between neighbouring levels): {len(jumpIndices)}")
for index in jumpIndices:
    print(f"   {dials[index]:.2f} -> {dials[index + 1]:.2f}: {steps[index]:5.2f} mV  ({regimeOf(dials[index])} the bistable window)")
for name, low, high in (('below', 0, lowerThreshold), ('inside', lowerThreshold, upperThreshold), ('above', upperThreshold, 2.01)):
    members = (dials[:-1] >= low) & (dials[:-1] < high)
    print(f"   {name:6s} the window: {int((steps[members] > args.jumpThresholdMilliVolts).sum())} jumps across {members.sum()} steps; "
          f"median step {np.median(steps[members]):.2f} mV")
tippingDials = (dials[jumpIndices] + dials[jumpIndices + 1]) / 2
summaryPath = args.summaryPath or f'data/boundaryDialLandscapeSummary{args.referenceCheckpoint}{holdTag}.json'
summary = dict(tippingPoints=[dict(fromDial=float(dials[index]), toDial=float(dials[index + 1]), stepMilliVolts=float(steps[index])) for index in jumpIndices],
               regimes={}, rotationGrids=[])
dialOnlyVariance = PCA().fit(patterns - patterns.mean(0)).explained_variance_ratio_
print(f"   dial-only patterns span {boundary.participationRatio(dialOnlyVariance):.2f} effective dimensions; "
      f"end-to-end change {rootMeanSquare(patterns[-1] - patterns[0]):.2f} mV")


# ==================================================================================== regime pairs
def distanceToDialCurve(pattern):
    best = np.inf
    for start, end in zip(patterns[:-1], patterns[1:]):
        segment = end - start
        fraction = np.clip(((pattern - start) @ segment) / (segment @ segment + 1e-12), 0, 1)
        best = min(best, rootMeanSquare(pattern - (start + fraction * segment)))
    return best


pairsPath = f'data/boundaryRegimePairs{args.referenceCheckpoint}{holdTag}.npz'
if not os.path.exists(pairsPath):
    print(f"\n(regime pairs not analysed: {pairsPath} not found)")
    json.dump(summary, open(summaryPath, 'w'), indent=1)
    raise SystemExit
runs = np.load(pairsPath)
randomGenerator = np.random.default_rng(3)
print(f"\n=== REGIME PAIRS: {len(runs['pairDialLevel'])} gradient runs, each against its sweep twin ===")
for regime in ('monostableDepolarised', 'bistable', 'monostableHyperpolarised'):
    members = runs['pairRegime'] == regime
    dialLevel, strength, direction = runs['pairDialLevel'][members], runs['pairGradientStrength'][members], runs['pairGradientDirection'][members]
    gradientPatterns = runs['pairWindowMeanVmem'][members]
    twins = patterns[runs['pairSweepIndex'][members]]
    differentials = gradientPatterns - twins
    sizes = rootMeanSquare(differentials)
    print(f"\n-- {regime}: {members.sum()} pairs, dial {dialLevel.min():.2f}-{dialLevel.max():.2f}, G {strength.min():.3f}-{strength.max():.3f}")
    print(f"   size vs G rho={spearmanr(strength, sizes).statistic:+.3f}; size vs dial rho={spearmanr(dialLevel, sizes).statistic:+.3f}; "
          f"median size {np.median(sizes):.2f} mV (G >= 0.05: {np.median(sizes[strength >= 0.05]) if (strength >= 0.05).any() else float('nan'):.2f})")
    nearestTipping = np.array([np.min(np.abs(tippingDials - value)) if len(tippingDials) else np.nan for value in dialLevel])
    print(f"   size vs distance to the nearest tipping point: rho={spearmanr(nearestTipping, sizes).statistic:+.3f}")
    moved = sizes > 1.0
    regimeSummary = dict(numPairs=int(members.sum()), sizeVsStrengthRho=float(spearmanr(strength, sizes).statistic),
                         sizeVsTippingDistanceRho=float(spearmanr(nearestTipping, sizes).statistic), numMoved=int(moved.sum()))
    if moved.sum() >= 5:
        offCurve = np.array([distanceToDialCurve(pattern) for pattern in gradientPatterns[moved]])
        reproducible = 1 - offCurve ** 2 / sizes[moved] ** 2
        symmetric = np.array([boundary.symmetricShare(change) for change in differentials[moved]])
        print(f"   {moved.sum()} moved > 1 mV: reproducible by some dial level median {np.median(reproducible) * 100:.0f}%; "
              f"fully symmetric median {np.median(symmetric) * 100:.0f}%")
        regimeSummary.update(reproducibleByDialMedian=float(np.median(reproducible)), symmetricShareMedian=float(np.median(symmetric)))
    print(f"   differentials span {boundary.participationRatio(PCA().fit(differentials).explained_variance_ratio_):.2f} effective dimensions")
    features = np.column_stack([dialLevel, np.log(strength), strength * np.cos(direction), strength * np.sin(direction),
                                np.cos(direction), np.sin(direction), dialLevel * strength])
    modePca = PCA(8).fit(differentials)
    scores = modePca.transform(differentials)

    def crossValidatedR2(targets, seed=0):
        estimator = make_pipeline(StandardScaler(), GridSearchCV(KernelRidge(kernel='rbf'),
                                  {'alpha': [1e-3, 1e-2, 1e-1], 'gamma': [0.05, 0.2, 0.5]}, cv=3))
        predictions = np.column_stack([cross_val_predict(estimator, features, targets[:, column], cv=KFold(6, shuffle=True, random_state=seed))
                                       for column in range(targets.shape[1])])
        return 1 - ((targets - predictions) ** 2).sum(0) / ((targets - targets.mean(0)) ** 2).sum(0), predictions
    r2, predictions = crossValidatedR2(scores)
    nullRuns = [crossValidatedR2(scores[randomGenerator.permutation(len(scores))], seed=index)[0] for index in range(args.numPermutations)]
    nullBest = np.max(nullRuns, axis=0) if nullRuns else np.full(8, -np.inf)
    controlled = r2 > np.maximum(0.2, nullBest)
    wholeR2 = 1 - ((differentials - modePca.inverse_transform(predictions)) ** 2).sum() / ((differentials - differentials.mean(0)) ** 2).sum()
    print(f"   controllability: CV R2 by mode {np.round(r2, 2)}; controlled modes {list(np.where(controlled)[0] + 1)} "
          f"covering {modePca.explained_variance_ratio_[controlled].sum() * 100:.0f}% of variance; whole-map CV R2 {wholeR2:.3f}")
    tracking = [max(abs(spearmanr(scores[:, mode], np.cos(direction)).statistic), abs(spearmanr(scores[:, mode], np.sin(direction)).statistic))
                for mode in range(8)]
    print(f"   |rho| of each mode with gradient direction: {np.round(tracking, 2)}")
    regimeSummary.update(dimensions=boundary.participationRatio(PCA().fit(differentials).explained_variance_ratio_),
                         modeVariance=modePca.explained_variance_ratio_.tolist(), modeR2=r2.tolist(), modeNullBest=nullBest.tolist(),
                         modeControlled=controlled.tolist(), modeDirectionTracking=[float(value) for value in tracking],
                         modeSymmetricShare=[boundary.symmetricShare(component) for component in modePca.components_],
                         modeLoadings=[(component / np.abs(component).max()).round(3).tolist() for component in modePca.components_],
                         wholeMapR2=float(wholeR2))
    twinParts = np.array([boundary.partSeparationScore(pattern)[:3] for pattern in twins])
    gradientParts = np.array([boundary.partSeparationScore(pattern)[:3] for pattern in gradientPatterns])
    cells = []
    for column, name in ((1, 'coverage'), (2, 'separation'), (0, 'part score')):
        change = gradientParts[:, column] - twinParts[:, column]
        cells.append(f"{name} +{int((change > 1e-9).sum())}/-{int((change < -1e-9).sum())}")
    print(f"   face parts (improved/worsened): " + ", ".join(cells))
    regimeSummary['faceParts'] = {name: dict(improved=int(((gradientParts[:, column] - twinParts[:, column]) > 1e-9).sum()),
                                             worsened=int(((gradientParts[:, column] - twinParts[:, column]) < -1e-9).sum()))
                                  for column, name in ((1, 'coverage'), (2, 'separation'), (0, 'partScore'))}
    summary['regimes'][regime] = regimeSummary

print("\n=== ROTATION GRIDS: share of change direction-independent / turning with the gradient / 2nd-4th harmonics ===")
for gridDial in sorted(set(runs['gridDial'])):
    atDial = np.isclose(runs['gridDial'], gridDial)
    baseline = patterns[runs['gridSweepIndex'][atDial][0]]
    for strength in sorted(set(runs['gridGradientStrength'][atDial])):
        members = atDial & np.isclose(runs['gridGradientStrength'], strength)
        order = np.argsort(runs['gridGradientDirection'][members])
        directions = runs['gridGradientDirection'][members][order]
        changes = runs['gridWindowMeanVmem'][members][order] - baseline
        energy = (changes ** 2).sum()
        shares = [(changes.mean(0) ** 2).sum() * 8]
        for harmonic in (1, 2, 3):
            cosinePart = (changes * np.cos(harmonic * directions)[:, None]).sum(0) * 2 / 8
            sinePart = (changes * np.sin(harmonic * directions)[:, None]).sum(0) * 2 / 8
            shares.append(((cosinePart ** 2).sum() + (sinePart ** 2).sum()) * 4)
        shares.append((((changes * np.cos(4 * directions)[:, None]).sum(0) / 8) ** 2).sum() * 8)
        shares = np.array(shares) / max(energy, 1e-12)
        summary['rotationGrids'].append(dict(dial=float(gridDial), strength=float(strength), sizeMilliVolts=float(rootMeanSquare(changes).mean()),
                                             shares=shares.tolist(), steerableChanges=[((changes * np.cos(directions)[:, None]).sum(0) * 2 / 8).round(2).tolist(),
                                                                                       ((changes * np.sin(directions)[:, None]).sum(0) * 2 / 8).round(2).tolist()]))
        print(f"   dial {gridDial:.2f} ({regimeOf(gridDial)} window), G={strength:.2f}: size {rootMeanSquare(changes).mean():5.2f} mV; "
              + " / ".join(f"{share * 100:.0f}%" for share in shares))
json.dump(summary, open(summaryPath, 'w'), indent=1)
print(f"\nwrote {summaryPath}")
