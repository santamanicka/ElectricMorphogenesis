"""Analyse the synthetic boundary-code experiments from simulateBoundaryHarmonics11x11.py
(PolyPatterning_Sim.md, Section 12).

harmonicGrid: how far each single harmonic order moves the pattern away from the unpatterned dial, how that
scales with amplitude, which interior harmonic orders it drives, how deep it reaches, what rotating it does,
and how the reference model's own trained code compares.

firstOrderPairs: what a first-order gradient adds to the dial (differential = gradient run - its G = 0 twin):
its size and gating by the dial, whether it lands off the curve of dial-only patterns, how many dimensions it
opens, how much of it turns with the gradient, where it lands shell by shell, what it does to face parts, and
which of its modes are predictable from the settings.
"""
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices=['harmonicGrid', 'firstOrderPairs', 'both'], default='both')
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--numPermutations', type=int, default=10)
args = parser.parse_args()

reference = boundary.loadCheckpoint(args.referenceCheckpoint)
target = boundary.targetVmemMilliVolts(reference)


def rootMeanSquare(values):
    return np.sqrt((np.asarray(values) ** 2).mean(-1))


def shellHarmonicAmplitudes(pattern, maxOrder=8):
    return {shell: np.abs(boundary.circularHarmonicCoefficients(pattern[boundary.shellCells(shell)], maxOrder))
            for shell in range(5)}


def partScores(pattern):
    score, coverage, separation, spurious = boundary.partSeparationScore(pattern)
    return dict(coverage=coverage, separation=separation, partScore=score,
                balancedRMS=boundary.balancedRootMeanSquareError(pattern, target))


# ======================================================================================= harmonic grid
def analyzeHarmonicGrid():
    grid = np.load(f'data/boundaryHarmonicGrid{args.referenceCheckpoint}.npz')
    patterns, orders, amplitudes, phases, dialLevels = (grid[key] for key in ('patterns', 'order', 'amplitude', 'phase', 'dialLevel'))
    referenceDialLevel = float(grid['referenceDialLevel'])
    isUnpatterned = (amplitudes == 0) & np.isclose(dialLevels, referenceDialLevel)
    baseline = patterns[np.argmax(isUnpatterned)]
    print(f"=== HARMONIC GRID on checkpoint {args.referenceCheckpoint} (dial {referenceDialLevel:.3f}, {len(patterns)} runs) ===")

    def runIndex(order, amplitude, rotated=False):
        return int(np.argmax((orders == order) & np.isclose(amplitudes, amplitude) & ((phases > 0) == rotated)))

    print("\nFace scores of each run (late-window mean pattern)")
    print(f"{'run':>18s} {'coverage':>9s} {'separation':>11s} {'partScore':>10s} {'balancedRMS':>12s}")
    for index in range(len(patterns)):
        name = (f"dial {dialLevels[index]:.3f}" if amplitudes[index] == 0 else
                f"k{orders[index]} A{amplitudes[index]:.1f}{' rotated' if phases[index] > 0 else ''}")
        scores = partScores(patterns[index])
        print(f"{name:>18s} {scores['coverage']:9.3f} {scores['separation']:11d} {scores['partScore']:10.3f} {scores['balancedRMS']:12.2f}")

    distances = np.sqrt(((patterns[:, None, :] - patterns[None, :, :]) ** 2).mean(-1))
    representatives = []
    for index in range(len(patterns)):
        if all(distances[index, other] >= 1.0 for other in representatives):
            representatives.append(index)
    allPairs = distances[np.triu_indices(len(patterns), 1)]
    print(f"\nDistinct patterns (at least 1 mV RMS from every earlier one): {len(representatives)} of {len(patterns)}; "
          f"closest pair of all runs {allPairs.min():.2f} mV RMS")

    dialShift = rootMeanSquare(patterns[int(np.argmax((amplitudes == 0) & np.isclose(dialLevels, 0.55)))] - baseline)
    print(f"\nResponse size: RMS change from the unpatterned dial (moving the dial {referenceDialLevel:.3f} -> 0.55 gives {dialShift:.2f} mV)")
    print(f"{'k':>3s} {'A=0.1':>8s} {'A=0.2':>8s} {'A=0.4':>8s} {'A0.4/A0.1':>10s}")
    for order in (1, 2, 3, 4, 5, 6, 8, 10):
        sizes = [rootMeanSquare(patterns[runIndex(order, amplitude)] - baseline) for amplitude in (0.1, 0.2, 0.4)]
        print(f"{order:3d} " + " ".join(f"{size:8.2f}" for size in sizes) + f" {sizes[2] / sizes[0]:9.1f}x")

    baselineHarmonics = shellHarmonicAmplitudes(baseline)
    print("\nOutput harmonic orders at shell 2, A = 0.4 (change in amplitude, each row normalised to its maximum)")
    print(f"{'in k':>5s} " + " ".join(f"{'m=' + str(outputOrder):>6s}" for outputOrder in range(7)))
    for order in range(1, 7):
        change = np.abs(shellHarmonicAmplitudes(patterns[runIndex(order, 0.4)])[2] - baselineHarmonics[2])[:7]
        print(f"{order:5d} " + " ".join(f"{value:6.2f}" for value in change / change.max()))

    print("\nPenetration: summed harmonic-amplitude change by shell, A = 0.4 (shell 0 = boundary ring)")
    for order in (1, 2, 3, 4, 5, 6, 8, 10):
        harmonics = shellHarmonicAmplitudes(patterns[runIndex(order, 0.4)])
        print(f"   k={order:2d}: " + " ".join(f"{np.abs(harmonics[shell] - baselineHarmonics[shell]).sum():7.2f}" for shell in range(5)))

    print("\nRotation by half a period (A = 0.4)")
    for order in (2, 4):
        unrotated, rotated = patterns[runIndex(order, 0.4)], patterns[runIndex(order, 0.4, rotated=True)]
        print(f"   k={order}: rotation moves the pattern {rootMeanSquare(rotated - unrotated):.2f} mV "
              f"(the unrotated harmonic itself moves it {rootMeanSquare(unrotated - baseline):.2f} mV)")

    trained = grid['trainedCodePattern']
    print(f"\nReference model's own trained code, same readout: RMS {rootMeanSquare(trained - baseline):.2f} mV from the "
          f"unpatterned dial, {min(rootMeanSquare(trained - pattern) for pattern in patterns):.2f} mV from the nearest "
          f"synthetic run; balanced RMS {boundary.balancedRootMeanSquareError(trained, target):.2f} mV")


# ================================================================================== first-order pairs
def analyzeFirstOrderPairs():
    runs = np.load(f'data/boundaryFirstOrderPairs{args.referenceCheckpoint}.npz')
    dialLevel, gradientStrength, gradientDirection = runs['pairDialLevel'], runs['pairGradientStrength'], runs['pairGradientDirection']
    twins, gradients = runs['pairTwinPatterns'], runs['pairGradientPatterns']
    differentials = gradients - twins
    sizes = rootMeanSquare(differentials)
    print(f"\n=== FIRST-ORDER PAIRS on checkpoint {args.referenceCheckpoint} ({len(dialLevel)} pairs; "
          f"largest clipped fraction {runs['pairClippedFraction'].max():.2f}) ===")

    order = np.argsort(dialLevel)
    twinsByDial = twins[order]
    dialMode = twinsByDial[-15:].mean(0) - twinsByDial[:15].mean(0)
    print(f"\n1. SIZE. The dial's own range (15 highest-DC twins minus 15 lowest): {rootMeanSquare(dialMode):.2f} mV")
    for low, high in zip((0.01, 0.02, 0.05, 0.1, 0.2), (0.02, 0.05, 0.1, 0.2, 0.5)):
        members = (gradientStrength >= low) & (gradientStrength < high)
        print(f"   G {low:.2f}-{high:.2f}  n={members.sum():3d}  median {np.median(sizes[members]):5.2f} mV  (max {sizes[members].max():5.2f})")
    for name, values in (('G', gradientStrength), ('DC', dialLevel)):
        result = spearmanr(values, sizes)
        print(f"   size vs {name}: rho={result.statistic:+.3f} p={result.pvalue:.2g}")
    for low, high in ((-0.1, 0.15), (0.15, 0.4), (0.4, 0.65)):
        members = (dialLevel >= low) & (dialLevel < high) & (gradientStrength >= 0.05)
        print(f"   DC {low:+.2f}..{high:.2f}, G >= 0.05: median {np.median(sizes[members]):.2f} mV (n={members.sum()})")

    def distanceToDialCurve(pattern, curve):
        best = np.inf
        for start, end in zip(curve[:-1], curve[1:]):
            segment = end - start
            fraction = np.clip(((pattern - start) @ segment) / (segment @ segment + 1e-12), 0, 1)
            best = min(best, rootMeanSquare(pattern - (start + fraction * segment)))
        return best
    floor = np.array([distanceToDialCurve(twinsByDial[index], np.delete(twinsByDial, index, 0)) for index in range(len(twinsByDial))])
    offCurve = np.array([distanceToDialCurve(pattern, twinsByDial) for pattern in gradients])
    moved = sizes > 1.0
    explainable = 1 - offCurve[moved] ** 2 / sizes[moved] ** 2
    print(f"\n2. NEW OR MORE DIAL? leave-one-out floor median {np.median(floor):.2f} mV; gradient patterns' distance from the "
          f"dial curve median {np.median(offCurve[moved]):.2f} mV (their change {np.median(sizes[moved]):.2f})")
    print(f"   share of change reproducible by some dial setting: median {np.median(explainable) * 100:.0f}% "
          f"(IQR {np.percentile(explainable, 25) * 100:.0f}-{np.percentile(explainable, 75) * 100:.0f}%); "
          f"clearly off the curve: {int(((offCurve > 2 * np.median(floor)) & (offCurve > 1)).sum())} of {len(offCurve)}")

    movedPca = PCA().fit(differentials[moved])
    dialPca = PCA().fit(twins - twins.mean(0))
    dialModeCount = int(np.searchsorted(np.cumsum(dialPca.explained_variance_ratio_), 0.99) + 1)
    dialBasis = dialPca.components_[:dialModeCount]
    insideDial = ((differentials[moved] @ dialBasis.T) ** 2).sum() / (differentials[moved] ** 2).sum()
    print(f"\n3. DIMENSIONS. participation ratio: dial-only patterns {boundary.participationRatio(dialPca.explained_variance_ratio_):.2f}; "
          f"differentials {boundary.participationRatio(movedPca.explained_variance_ratio_):.2f} (the {moved.sum()} that moved > 1 mV), "
          f"{boundary.participationRatio(PCA().fit(differentials).explained_variance_ratio_):.2f} (all)")
    print(f"   share of differential energy inside the dial's own {dialModeCount}-mode subspace: {insideDial * 100:.0f}%")

    gridStrength, gridDirection = runs['gridGradientStrength'], runs['gridGradientDirection']
    gridChanges = runs['gridPatterns'] - runs['gridBaselinePattern']
    centredDialMode = dialMode - dialMode.mean()
    print("\n4. STEERING (rotation grid). Share of change: direction-independent / turns with gradient / 2nd / 3rd / 4th harmonic; "
          "rot90 equivariance; |r| of direction-independent part with the dial mode")
    for strength in sorted(set(gridStrength)):
        members = np.isclose(gridStrength, strength)
        directions = gridDirection[members]
        sortOrder = np.argsort(directions)
        changes, directions = gridChanges[members][sortOrder], directions[sortOrder]
        energy = (changes ** 2).sum()
        shares = [(changes.mean(0) ** 2).sum() * 8]
        for harmonic in (1, 2, 3):
            cosinePart = (changes * np.cos(harmonic * directions)[:, None]).sum(0) * 2 / 8
            sinePart = (changes * np.sin(harmonic * directions)[:, None]).sum(0) * 2 / 8
            shares.append(((cosinePart ** 2).sum() + (sinePart ** 2).sum()) * 8 / 2)
        shares.append((((changes * np.cos(4 * directions)[:, None]).sum(0) / 8) ** 2).sum() * 8)
        shares = np.array(shares) / energy
        grids = changes.reshape(-1, boundary.latticeRows, boundary.latticeCols)
        equivariance = np.mean([np.corrcoef(np.rot90(grids[index], -1).reshape(-1), changes[(index + 2) % 8])[0, 1] for index in range(8)])
        independentPart = changes.mean(0)
        dialAgreement = abs(np.corrcoef(independentPart - independentPart.mean(), centredDialMode)[0, 1])
        print(f"   G={strength:.2f}: " + " / ".join(f"{share * 100:.0f}%" for share in shares) +
              f"; equivariance {equivariance:+.2f}; |r| with dial mode {dialAgreement:.2f}")

    print("\n5. REACH. RMS by shell (0 = boundary ring ... 5 = centre)")
    print("   dial mode           : " + " ".join(f"{value:5.2f}" for value in boundary.shellRootMeanSquare(dialMode)))
    for low, high in ((0.01, 0.05), (0.05, 0.2), (0.2, 0.5)):
        members = (gradientStrength >= low) & (gradientStrength < high)
        profile = np.median([boundary.shellRootMeanSquare(change) for change in differentials[members]], 0)
        print(f"   gradient G {low:.2f}-{high:.2f}: " + " ".join(f"{value:5.2f}" for value in profile))

    twinParts = np.array([boundary.partSeparationScore(pattern)[:3] for pattern in twins])
    gradientParts = np.array([boundary.partSeparationScore(pattern)[:3] for pattern in gradients])
    print("\n6. FACE PARTS (paired change, gradient run minus twin)")
    for column, name in ((1, 'coverage'), (2, 'separation'), (0, 'partScore')):
        change = gradientParts[:, column] - twinParts[:, column]
        print(f"   {name:10s}: changed in {int((np.abs(change) > 1e-9).sum())}/{len(change)} pairs, improved "
              f"{int((change > 1e-9).sum())}, worsened {int((change < -1e-9).sum())}; mean change {change.mean():+.3f}")
    print(f"   |separation change| vs G: rho={spearmanr(gradientStrength, np.abs(gradientParts[:, 2] - twinParts[:, 2])).statistic:+.3f}")

    features = np.column_stack([dialLevel, np.log(gradientStrength), gradientStrength * np.cos(gradientDirection),
                                gradientStrength * np.sin(gradientDirection), np.cos(gradientDirection),
                                np.sin(gradientDirection), dialLevel * gradientStrength])
    modePca = PCA(8).fit(differentials)
    scores = modePca.transform(differentials)

    def crossValidatedR2(targets, seed=0):
        estimator = make_pipeline(StandardScaler(), GridSearchCV(KernelRidge(kernel='rbf'),
                                  {'alpha': [1e-3, 1e-2, 1e-1], 'gamma': [0.05, 0.2, 0.5]}, cv=3))
        predictions = np.column_stack([cross_val_predict(estimator, features, targets[:, column],
                                       cv=KFold(6, shuffle=True, random_state=seed)) for column in range(targets.shape[1])])
        return 1 - ((targets - predictions) ** 2).sum(0) / ((targets - targets.mean(0)) ** 2).sum(0), predictions
    r2, predictions = crossValidatedR2(scores)
    randomGenerator = np.random.default_rng(3)
    nullBest = np.max([crossValidatedR2(scores[randomGenerator.permutation(len(scores))], seed=index)[0]
                       for index in range(args.numPermutations)], axis=0)
    wholeR2 = 1 - ((differentials - modePca.inverse_transform(predictions)) ** 2).sum() / ((differentials - differentials.mean(0)) ** 2).sum()
    print(f"\n7. CONTROLLABILITY. 6-fold CV R^2 of kernel ridge from (DC, log G, G cos, G sin, cos, sin, DC*G); "
          f"null = best of {args.numPermutations} shuffles")
    print(f"{'mode':>5s} {'var%':>6s} {'CV R2':>7s} {'null':>7s} {'symmetric':>10s} {'rho cos':>8s} {'rho sin':>8s} {'rho DC':>7s} {'rho logG':>9s}")
    for mode in range(8):
        tracking = [spearmanr(scores[:, mode], values).statistic for values in
                    (np.cos(gradientDirection), np.sin(gradientDirection), dialLevel, np.log(gradientStrength))]
        controlled = r2[mode] > max(0.2, nullBest[mode])
        print(f"{mode + 1:5d} {modePca.explained_variance_ratio_[mode] * 100:6.1f} {r2[mode]:7.3f} {nullBest[mode]:7.3f} "
              f"{boundary.symmetricShare(modePca.components_[mode]) * 100:9.0f}% " + " ".join(f"{value:+8.2f}" for value in tracking) +
              ("  <- controlled" if controlled else ""))
    symmetricShares = np.array([boundary.symmetricShare(change) for change in differentials[moved]])
    print(f"   whole differential CV R^2 {wholeR2:.3f}; symmetric share of the {moved.sum()} differentials > 1 mV: median "
          f"{np.median(symmetricShares) * 100:.0f}% (IQR {np.percentile(symmetricShares, 25) * 100:.0f}-{np.percentile(symmetricShares, 75) * 100:.0f}%)")


if args.experiment in ('harmonicGrid', 'both'):
    analyzeHarmonicGrid()
if args.experiment in ('firstOrderPairs', 'both'):
    analyzeFirstOrderPairs()
