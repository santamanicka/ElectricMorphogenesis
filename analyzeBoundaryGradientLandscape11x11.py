"""What a single harmonic, order 1 or order 2, adds to the boundary dial below a ceiling of 1.3
(PolyPatterning_Sim.md, Section 12).

Reads the grid from simulateBoundaryGradientLandscape11x11.py (code = DC + G cos(k (theta - phi)), k = 1 by default or
2 with --codeTag Order2, every held value in [0, 1.3]) together with the dial sweep at the same hold, whose run at each
DC is the harmonic run's G = 0 twin, and maps the landscape the harmonic opens up:

  change          the late pattern minus its twin's, RMS over all 121 cells and over the 81 interior cells.
  jump crossing   whether the ring's own range [min, max of the held values] spans one of the dial sweep's jumps
                  (steps of more than --jumpMilliVolts between neighbouring dials 0.01 apart), and how well that
                  predicts a change of more than --changeMilliVolts.
  symmetry class  the change split into the three classes of the square's symmetry group: odd under a half turn (the
                  class an order-1 clamp drives directly, circular orders 1, 3, 5, ...), even under a half turn but odd
                  under a quarter turn (orders 2, 6, 10, ..., the class an order-2 clamp drives directly), and unchanged
                  by a quarter turn (orders 0, 4, 8, ..., the class a uniform dial moves).
  steps           RMS change between neighbouring grid points one grid step apart, along G at fixed DC and along DC
                  at fixed G; steps larger than --jumpMilliVolts are jumps.
  reach           how far each pattern sits from the nearest pattern the dial alone makes anywhere in [0, 1.3], the
                  dial whose pattern that is, and the size of the part of the pattern that is not fully symmetric.
  seams           whether each grid step carries any ring cell's held value across a dial jump, and how large the steps
                  that do and do not are; and how much pattern variance is explained by grouping runs on which side of
                  every jump each ring cell sits, against groupings of the same number formed by closeness in (DC, G).
  motifs          the interior binarised at the single-cell saddle (-29.3 mV): distinct motifs, and distinct motifs up
                  to the square's symmetries, for the dial alone and with the gradient, also at --motifThresholds; the
                  most common motifs with an example run each.
  pattern space   principal components of every pattern, the dial's own track among them, and the participation ratio.
  symmetry checks runs at phi = 90, 180, -22.5 and 67.5 degrees against the matching image of a grid run.

Writes data/boundaryGradientLandscapeSummary<checkpoint><hold>.json for plotBoundaryGradient11x11.py.
"""
import argparse
import json

import numpy as np
from sklearn.cluster import KMeans

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--holdTag', type=str, default='Hold301')
parser.add_argument('--codeTag', type=str, default='', help="'' for order 1, Order2 for order 2")
parser.add_argument('--jumpMilliVolts', type=float, default=2.0)
parser.add_argument('--changeMilliVolts', type=float, default=2.0)
parser.add_argument('--motifThresholds', type=str, default='-26.3,-29.3,-32.3,-34.6')
parser.add_argument('--galleryMotifs', type=int, default=16)
parser.add_argument('--proximityRepeats', type=int, default=5)
args = parser.parse_args()

size = boundary.latticeRows
interior = boundary.interiorCellIndices
grid = np.load(f'data/boundaryGradientLandscape{args.referenceCheckpoint}{args.holdTag}{args.codeTag}.npz')
harmonicOrder = int(grid['harmonicOrder']) if 'harmonicOrder' in grid.files else 1
drivenClass = 0 if harmonicOrder == 1 else 1   # the symmetry class the harmonic drives directly
sweep = np.load(f'data/boundaryDialSweep{args.referenceCheckpoint}{args.holdTag}.npz')
free = np.load(f'data/boundaryFreeRun{args.referenceCheckpoint}.npz')
dialLimit, gridStep = float(grid['dialLimit']), float(grid['gridStep'])
windowStart = int(grid['windowStart'])
freePattern = free['vmem'][windowStart:windowStart + 1000].astype(float).mean(0)

sweepDials = sweep['dialLevel']
inLimit = sweepDials <= dialLimit + 1e-9
dialPatterns, dialHeld, dialStd = sweep['windowMeanVmem'][inLimit], sweep['endOfHoldVmem'][inLimit], sweep['windowStdVmem'][inLimit]
dialLevels = sweepDials[inLimit]
dialSteps = np.sqrt(((dialPatterns[1:] - dialPatterns[:-1]) ** 2).mean(1))
jumpDials = ((dialLevels[1:] + dialLevels[:-1]) / 2)[dialSteps > args.jumpMilliVolts]


def rms(values, cells=None):
    values = values if cells is None else values[..., cells]
    return np.sqrt((values ** 2).mean(-1))


def image(pattern, transform):
    """A lattice map carried through one of the square's symmetries, back as a flat array."""
    square = np.asarray(pattern).reshape(size, size)
    return {'rotateClockwise': np.rot90(square, -1), 'halfTurn': np.rot90(square, 2), 'mirrorVertical': square[:, ::-1],
            'mirrorAntiDiagonal': np.rot90(square, 2).T, 'rotateCounterclockwise': np.rot90(square, 1)}[transform].reshape(-1)


def symmetryClasses(pattern):
    """Energy in the three classes: odd under a half turn; even under it but odd under a quarter turn; unchanged."""
    halfTurnOdd = (pattern - image(pattern, 'halfTurn')) / 2
    even = pattern - halfTurnOdd
    quarterTurnOdd = (even - image(even, 'rotateClockwise')) / 2
    quarterTurnEven = even - quarterTurnOdd
    return np.array([(part ** 2).sum() for part in (halfTurnOdd, quarterTurnOdd, quarterTurnEven)])


def fullySymmetricPart(pattern):
    square = np.asarray(pattern).reshape(size, size)
    images = [np.rot90(square, turns) for turns in range(4)]
    images += [each[:, ::-1] for each in images]
    return np.mean(images, axis=0).reshape(-1)


def motifKey(pattern, threshold=boundary.singleCellSaddleMilliVolts):
    return np.packbits(pattern[interior] < threshold).tobytes()


def canonicalMotifKey(pattern, threshold=boundary.singleCellSaddleMilliVolts):
    dark = (np.asarray(pattern).reshape(size, size) < threshold)[1:-1, 1:-1]
    images = [np.rot90(dark, turns) for turns in range(4)]
    images += [each[:, ::-1] for each in images]
    return min(np.packbits(each).tobytes() for each in images)


# ------------------------------------------------------------------------------------------ the grid
onGrid = ~grid['symmetryCheck']
directions = sorted(set(grid['gradientDirection'][onGrid].tolist()))
numSteps = int(round(dialLimit / gridStep))
dialIndexOfSweep = {int(round(level / gridStep)): index for index, level in enumerate(dialLevels)
                    if abs(level / gridStep - round(level / gridStep)) < 1e-6}
summary = dict(harmonicOrder=harmonicOrder, drivenClass=drivenClass, dialLimit=dialLimit, gridStep=gridStep,
               holdIterations=int(grid['holdIterations']), windowStart=windowStart,
               jumpMilliVolts=args.jumpMilliVolts, changeMilliVolts=args.changeMilliVolts, directions=directions,
               dialSweep=dict(dials=dialLevels.round(3).tolist(), steps=dialSteps.round(3).tolist(), jumpDials=jumpDials.round(3).tolist(),
                              windowStdMax=float(dialStd.max()), windowStdMaxDial=float(dialLevels[dialStd.max(1).argmax()])),
               free=dict(pattern=freePattern.round(2).tolist()), perDirection={})
print(f"dial sweep ({args.holdTag}), dials 0-{dialLimit}: jumps at " + ", ".join(f"{value:.3f}" for value in jumpDials))
print(f"   largest late-window Vmem SD on the dial alone: {dialStd.max():.1f} mV at dial {summary['dialSweep']['windowStdMaxDial']:.2f}")

allGradientPatterns, dialOnlyMotifs, dialOnlyCanonical = [], {motifKey(pattern) for pattern in dialPatterns}, {canonicalMotifKey(pattern) for pattern in dialPatterns}
motifCounts = {}
for direction in directions:
    members = np.where(onGrid & (grid['gradientDirection'] == direction))[0]
    dialIndex = np.rint(grid['dialLevel'][members] / gridStep).astype(int)
    gradientIndex = np.rint(grid['gradientStrength'][members] / gridStep).astype(int)
    patterns, held, windowStd = grid['windowMeanVmem'][members], grid['endOfHoldVmem'][members], grid['windowStdVmem'][members]
    ringValues = grid['ringValues'][members]
    twins = np.array([dialPatterns[dialIndexOfSweep[index]] for index in dialIndex])
    change = patterns - twins
    changeAll, changeInterior = rms(change), rms(change, interior)
    classEnergy = np.array([symmetryClasses(each) for each in change])
    classShare = classEnergy / np.maximum(classEnergy.sum(1, keepdims=True), 1e-12)
    ringLow, ringHigh = ringValues.min(1), ringValues.max(1)
    crossesJump = np.array([bool(((jumpDials >= low) & (jumpDials <= high)).any()) for low, high in zip(ringLow, ringHigh)])
    changed = changeAll > args.changeMilliVolts
    # reach: nearest dial-only pattern anywhere in [0, limit], and the part that is not fully symmetric
    distances = np.sqrt(((patterns[:, None, :] - dialPatterns[None, :, :]) ** 2).mean(-1))
    nearestDial = dialLevels[distances.argmin(1)]
    nearestDistance = distances.min(1)
    asymmetric = rms(patterns - np.array([fullySymmetricPart(each) for each in patterns]))
    symmetricDistances = np.sqrt(((np.array([fullySymmetricPart(each) for each in patterns])[:, None, :] - dialPatterns[None]) ** 2).mean(-1))
    equivalentDial = dialLevels[symmetricDistances.argmin(1)]
    allGradientPatterns.append(patterns)

    # neighbouring steps along G (fixed DC) and along DC (fixed G), G = 0 taken from the sweep; each step also counts the
    # ring cells whose held value it carries across a dial jump
    lookup = {(i, j): patterns[k] for k, (i, j) in enumerate(zip(dialIndex, gradientIndex))}
    for i, index in dialIndexOfSweep.items():
        lookup[(i, 0)] = dialPatterns[index]
    ringProfile = np.cos(harmonicOrder * (grid['ringAngles'] - np.deg2rad(direction)))

    def cellsCrossing(start, end):
        low = np.minimum(start, end)[:, None]
        high = np.maximum(start, end)[:, None]
        return int(((jumpDials[None, :] > low) & (jumpDials[None, :] <= high)).any(1).sum())
    alongGradient, alongDial = [], []
    for (i, j), pattern in lookup.items():
        for (di, dj), store in (((0, 1), alongGradient), ((1, 0), alongDial)):
            if (i + di, j + dj) in lookup:
                crossing = cellsCrossing((i + j * ringProfile) * gridStep, (i + di + (j + dj) * ringProfile) * gridStep)
                store.append((i, j, float(rms(lookup[(i + di, j + dj)] - pattern)), crossing))
    alongGradient, alongDial = np.array(alongGradient), np.array(alongDial)
    stepSummary = {}
    allSteps = np.concatenate([alongGradient, alongDial])
    for name, steps in (('alongGradient', alongGradient), ('alongDial', alongDial), ('alongDialAtZeroGradient', alongDial[alongDial[:, 1] == 0]),
                        ('alongDialWithGradient', alongDial[alongDial[:, 1] > 0]), ('crossing', allSteps[allSteps[:, 3] > 0]),
                        ('notCrossing', allSteps[allSteps[:, 3] == 0])):
        values = steps[:, 2]
        stepSummary[name] = dict(count=int(len(values)), jumps=int((values > args.jumpMilliVolts).sum()),
                                 jumpShare=float((values > args.jumpMilliVolts).mean()), median=float(np.median(values)),
                                 percentile95=float(np.percentile(values, 95)),
                                 movementInJumps=float(values[values > args.jumpMilliVolts].sum() / values.sum()),
                                 movementShare=float(values.sum() / allSteps[:, 2].sum()))
    crossingCounts = allSteps[:, 3].astype(int)
    stepSummary['byCellsCrossing'] = [dict(cells=int(cells), count=int((crossingCounts == cells).sum()),
                                           median=float(np.median(allSteps[crossingCounts == cells, 2])),
                                           quartiles=np.percentile(allSteps[crossingCounts == cells, 2], [25, 75]).round(3).tolist())
                                      for cells in np.unique(crossingCounts)]

    # which side of every jump each ring cell sits on, against equally many groups formed by closeness in (DC, G)
    def explainedShare(groups):
        within = sum(((patterns[groups == group] - patterns[groups == group].mean(0)) ** 2).sum() for group in np.unique(groups))
        return float(1 - within / ((patterns - patterns.mean(0)) ** 2).sum())
    sides = np.unique(np.searchsorted(jumpDials, ringValues), axis=0, return_inverse=True)[1].ravel()
    numGroups = len(np.unique(sides))
    coordinates = np.c_[dialIndex, gradientIndex].astype(float)
    proximity = [explainedShare(KMeans(numGroups, n_init=1, random_state=repeat).fit_predict(coordinates)) for repeat in range(args.proximityRepeats)]
    seams = dict(groups=numGroups, sideExplained=explainedShare(sides), proximityExplained=float(np.mean(proximity)),
                 proximitySpread=float(np.std(proximity)))

    motifs = {motifKey(pattern) for pattern in patterns}
    canonical = {canonicalMotifKey(pattern) for pattern in patterns}
    motifCounts[direction] = dict(motifs=len(motifs), newMotifs=len(motifs - dialOnlyMotifs), canonical=len(canonical),
                                  newCanonical=len(canonical - dialOnlyCanonical))
    entry = dict(
        dial=(dialIndex * gridStep).round(3).tolist(), gradient=(gradientIndex * gridStep).round(3).tolist(),
        pattern=patterns.round(1).tolist(), heldPattern=held.round(1).tolist(),
        changeAll=changeAll.round(3).tolist(), changeInterior=changeInterior.round(3).tolist(),
        heldChangeInterior=rms(held - np.array([dialHeld[dialIndexOfSweep[index]] for index in dialIndex]), interior).round(3).tolist(),
        classShare=classShare.round(4).tolist(), ringLow=ringLow.round(4).tolist(), ringHigh=ringHigh.round(4).tolist(),
        crossesJump=crossesJump.tolist(), churn=windowStd[:, interior].mean(1).round(3).tolist(),
        nearestDial=nearestDial.round(3).tolist(), nearestDistance=nearestDistance.round(3).tolist(),
        equivalentDial=equivalentDial.round(3).tolist(), asymmetric=asymmetric.round(3).tolist(),
        steps=dict(alongGradient=alongGradient.round(3).tolist(), alongDial=alongDial.round(3).tolist(), summary=stepSummary), seams=seams,
        crossing=dict(bothTrue=int((crossesJump & changed).sum()), crossOnly=int((crossesJump & ~changed).sum()),
                      changeOnly=int((~crossesJump & changed).sum()), neither=int((~crossesJump & ~changed).sum())),
        motifs=motifCounts[direction])
    oddShareSmall = classShare[~changed, 0]
    entry['classSummary'] = dict(
        smallChangeOddMedian=float(np.median(oddShareSmall)) if len(oddShareSmall) else None,
        largeChangeOddMedian=float(np.median(classShare[changed, 0])) if changed.any() else None,
        largeChangeDialLikeMedian=float(np.median(classShare[changed, 2])) if changed.any() else None,
        quietDrivenMedian=float(np.median(classShare[~crossesJump, drivenClass])) if (~crossesJump).any() else None,
        crossingDrivenMedian=float(np.median(classShare[crossesJump, drivenClass])) if crossesJump.any() else None,
        crossingDialLikeMedian=float(np.median(classShare[crossesJump, 2])) if crossesJump.any() else None)
    summary['perDirection'][f'{direction:g}'] = entry

    print(f"\n=== phi = {direction:g} degrees: {len(members)} runs ===")
    print(f"   change vs twin: median {np.median(changeAll):.2f} mV, {changed.sum()} runs above {args.changeMilliVolts} mV, largest {changeAll.max():.1f}")
    print(f"   ring range spans a dial jump: {crossesJump.sum()} runs; of those {(crossesJump & changed).sum()} changed; "
          f"of the {(~crossesJump).sum()} that do not, {(~crossesJump & changed).sum()} changed")
    print(f"   share in the class the harmonic drives: median {entry['classSummary']['quietDrivenMedian']:.2f} when the ring spans no jump, "
          f"{entry['classSummary']['crossingDrivenMedian']:.2f} when it spans one (dial-like {entry['classSummary']['crossingDialLikeMedian']:.2f})")
    for name, values in stepSummary.items():
        if name != 'byCellsCrossing':
            print(f"   steps {name}: {values['jumps']}/{values['count']} jumps, median {values['median']:.2f} mV (95th percentile {values['percentile95']:.2f}), "
                  f"{100 * values['movementInJumps']:.0f}% of their movement in jumps, {100 * values['movementShare']:.0f}% of all movement")
    print("   median step by ring cells carried across a jump: " + ", ".join(f"{entry['cells']}: {entry['median']:.2f} (n={entry['count']})" for entry in stepSummary['byCellsCrossing']))
    print(f"   sides of the jumps: {numGroups} groups explain {100 * seams['sideExplained']:.1f}% of pattern variance; as many groups by closeness in (DC, G) "
          f"{100 * seams['proximityExplained']:.1f}% (SD {100 * seams['proximitySpread']:.1f})")
    print(f"   reach: distance to the nearest dial-only pattern median {np.median(nearestDistance):.2f} mV (max {nearestDistance.max():.1f}); "
          f"not fully symmetric part median {np.median(asymmetric):.2f} mV (max {asymmetric.max():.1f})")
    print(f"   motifs: {motifCounts[direction]}  (dial alone: {len(dialOnlyMotifs)} motifs, {len(dialOnlyCanonical)} up to symmetry)")
    print(f"   churn (late-window Vmem SD, mean over interior cells): median {np.median(windowStd[:, interior].mean(1)):.2f} mV "
          f"(dial alone {np.median(dialStd[:, interior].mean(1)):.2f})")

allGradientPatterns = np.concatenate(allGradientPatterns)
runLabels = [(direction, dial, gradient) for direction in directions
             for dial, gradient in zip(summary['perDirection'][f'{direction:g}']['dial'], summary['perDirection'][f'{direction:g}']['gradient'])]
unionMotifs = {motifKey(pattern) for pattern in allGradientPatterns} | dialOnlyMotifs
unionCanonical = {canonicalMotifKey(pattern) for pattern in allGradientPatterns} | dialOnlyCanonical
byThreshold = []
for threshold in (float(value) for value in args.motifThresholds.split(',')):
    dialSet = {canonicalMotifKey(pattern, threshold) for pattern in dialPatterns}
    byThreshold.append(dict(threshold=threshold, dialOnly=len({motifKey(pattern, threshold) for pattern in dialPatterns}), dialOnlyCanonical=len(dialSet),
                            withGradient=len({motifKey(pattern, threshold) for pattern in allGradientPatterns} | {motifKey(pattern, threshold) for pattern in dialPatterns}),
                            withGradientCanonical=len({canonicalMotifKey(pattern, threshold) for pattern in allGradientPatterns} | dialSet)))
dialMotif = canonicalMotifKey(dialPatterns[0])
runMotifs = [canonicalMotifKey(pattern) for pattern in allGradientPatterns]
motifRuns = {}
for index, key in enumerate(runMotifs):
    motifRuns.setdefault(key, []).append(index)
gallery = []
for key, members in sorted(motifRuns.items(), key=lambda item: -len(item[1]))[:args.galleryMotifs]:
    centre = allGradientPatterns[members].mean(0)
    example = members[int(np.argmin(rms(allGradientPatterns[members] - centre)))]
    direction, dial, gradient = runLabels[example]
    darkCells = np.unpackbits(np.frombuffer(key, dtype=np.uint8))[:(size - 2) ** 2]
    gallery.append(dict(runs=len(members), isDialMotif=key == dialMotif, darkCells=int(darkCells.sum()),
                        perDirection={f'{each:g}': int(sum(runLabels[index][0] == each for index in members)) for each in directions},
                        example=dict(direction=direction, dial=dial, gradient=gradient, pattern=allGradientPatterns[example].round(1).tolist())))
summary['motifs'] = dict(dialOnly=len(dialOnlyMotifs), dialOnlyCanonical=len(dialOnlyCanonical), union=len(unionMotifs),
                         unionCanonical=len(unionCanonical), perDirection={f'{key:g}': value for key, value in motifCounts.items()},
                         byThreshold=byThreshold, keepDialMotif=float(np.mean([key == dialMotif for key in runMotifs])), gallery=gallery,
                         dialMotifDarkCells=int((dialPatterns[0][interior] < boundary.singleCellSaddleMilliVolts).sum()))
print(f"\nmotifs overall: dial alone {len(dialOnlyMotifs)} ({len(dialOnlyCanonical)} up to symmetry); with the gradient "
      f"{len(unionMotifs)} ({len(unionCanonical)} up to symmetry); {100 * summary['motifs']['keepDialMotif']:.0f}% of gradient runs keep the dial's motif")
for entry in byThreshold:
    print(f"   at {entry['threshold']} mV: dial alone {entry['dialOnly']} ({entry['dialOnlyCanonical']}), with the gradient {entry['withGradient']} ({entry['withGradientCanonical']})")
print("   most common: " + ", ".join(f"{entry['runs']} runs ({entry['darkCells']} dark{', the dial motif' if entry['isDialMotif'] else ''})" for entry in gallery[:8]))

# pattern space: principal components of every pattern, the dial's own track included
everything = np.concatenate([dialPatterns, allGradientPatterns])
mean = everything.mean(0)
_, singular, components = np.linalg.svd(everything - mean, full_matrices=False)
explained = singular ** 2 / (singular ** 2).sum()
scores = (everything - mean) @ components[:3].T
summary['patternSpace'] = dict(explained=explained[:6].round(4).tolist(), dialTrack=scores[:len(dialPatterns)].round(2).tolist(),
                               dialTrackDials=dialLevels.round(3).tolist(),
                               perDirection={f'{direction:g}': scores[len(dialPatterns):][[label[0] == direction for label in runLabels]].round(2).tolist()
                                             for direction in directions},
                               components=components[:3].round(4).tolist())
print("pattern space: leading components explain " + ", ".join(f"{100 * value:.1f}%" for value in explained[:4]))


# effective dimension of the pattern sets
def participation(patterns):
    centred = patterns - patterns.mean(0)
    return boundary.participationRatio(np.linalg.svd(centred, compute_uv=False) ** 2)


summary['dimension'] = dict(dialOnly=participation(dialPatterns),
                            **{f'withGradient{key:g}': participation(np.concatenate([dialPatterns, np.array(summary['perDirection'][f'{key:g}']['pattern'])]))
                               for key in directions},
                            withGradientAll=participation(np.concatenate([dialPatterns, allGradientPatterns])))
print("effective dimension (participation ratio): " + ", ".join(f"{key} {value:.2f}" for key, value in summary['dimension'].items()))

# symmetry checks: phi' runs against the matching image of a grid run
checks = []
expected = {90.0: (0.0, 'rotateClockwise'), 180.0: (0.0, 'halfTurn'), -22.5: (22.5, 'mirrorVertical'), 67.5: (22.5, 'mirrorAntiDiagonal')}
for index in np.where(grid['symmetryCheck'])[0]:
    direction = float(grid['gradientDirection'][index])
    source, transform = expected[direction]
    match = np.where(onGrid & np.isclose(grid['dialLevel'], grid['dialLevel'][index]) & np.isclose(grid['gradientStrength'], grid['gradientStrength'][index])
                     & (grid['gradientDirection'] == source))[0]
    if not len(match):
        continue
    difference = np.abs(grid['windowMeanVmem'][index] - image(grid['windowMeanVmem'][match[0]], transform)).max()
    checks.append(dict(dial=float(grid['dialLevel'][index]), gradient=float(grid['gradientStrength'][index]), direction=direction,
                       source=source, transform=transform, maxDifference=float(difference)))
summary['symmetryChecks'] = checks
print("symmetry checks, largest |difference| (mV): " + ", ".join(f"{check['dial']}:{check['gradient']}@{check['direction']:g} {check['maxDifference']:.1e}" for check in checks))
# mirror symmetry kept by the axial and diagonal gradients
for direction, transform in ((0.0, 'mirrorVertical'), (45.0, 'mirrorAntiDiagonal')):
    patterns = np.array(summary['perDirection'][f'{direction:g}']['pattern'])
    residual = max(np.abs(pattern - image(pattern, transform)).max() for pattern in patterns)
    summary[f'mirrorResidual{direction:g}'] = float(residual)
    print(f"phi = {direction:g}: largest departure from its mirror symmetry {residual:.2e} mV")

outputPath = f'data/boundaryGradientLandscapeSummary{args.referenceCheckpoint}{args.holdTag}{args.codeTag}.json'
json.dump(summary, open(outputPath, 'w'), separators=(',', ':'))
print(f"\nwrote {outputPath}")
