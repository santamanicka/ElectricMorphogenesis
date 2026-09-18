"""What orders 1 and 2 do together on the boundary dial below a ceiling of 1.3 (PolyPatterning_Sim.md, Section 12).

Reads the combined grid from simulateBoundaryGradientLandscape11x11.py --combinedOrders (code = DC + G1 cos(theta) +
G2 cos(2 theta), every held value in [0, 1.3], G1 >= 0, G2 signed, steps of 0.04), the dial sweep at the same hold, and
the single-order grids (order 1 and order 2, steps of 0.02) as references. Each code is classed by its shape around the
ring: order 1 alone (G2 = 0), order 2 alone (G1 = 0), both with |G2| > G1 / 4 (the profile has four turning points and
four inflection points, two more of each than the gradient alone), or both below that threshold (still two of each).

  change          the late pattern minus the dial alone's at the same DC, RMS over all 121 cells.
  additivity      for codes with both orders, the change against the sum of the two single-order changes at the same
                  DC, G1 and G2: the interaction share is |change - (change1 + change2)|^2 / |change|^2. Codes are split by
                  whether the ring spans a dial jump, and by whether the combination puts ring cells on a side of a jump
                  that neither single order does.
  seams           grid steps along DC, G1 and G2 (0.04), labelled by how many ring cells they carry across a dial jump,
                  and the grouping test of analyzeBoundaryGradientLandscape11x11.py.
  reach           each code's distance to the nearest pattern made by the dial alone or by either order alone (the finer
                  single-order grids, with order 2's negative amplitudes as quarter turns of its positive ones).
  inflection test whether crossing |G2| = G1 / 4 matters in itself: codes with the same |G2| on either side of the
                  threshold compared for reach beyond the single orders and for interaction, and steps along G1 that cross
                  the threshold compared with steps that do not, separately for steps that do and do not cross a dial jump.
  motifs          distinct interior motifs (binarised at the single-cell saddle, and at --motifThresholds) for the dial
                  alone, each order alone, both single orders together, and the combined codes.
  checks          codes run with G1 negated against the vertical mirror image, and the mirror symmetry every code keeps.

Writes data/boundaryOrderPairLandscapeSummary<checkpoint><hold>.json for plotBoundarySecondOrder11x11.py.
"""
import argparse
import json

import numpy as np
from sklearn.cluster import KMeans

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--holdTag', type=str, default='Hold301')
parser.add_argument('--jumpMilliVolts', type=float, default=2.0)
parser.add_argument('--changeMilliVolts', type=float, default=2.0)
parser.add_argument('--motifThresholds', type=str, default='-26.3,-29.3,-32.3,-34.6')
parser.add_argument('--galleryMotifs', type=int, default=16)
parser.add_argument('--proximityRepeats', type=int, default=5)
args = parser.parse_args()

size = boundary.latticeRows
interior = boundary.interiorCellIndices
prefix = f'data/boundaryGradientLandscape{args.referenceCheckpoint}{args.holdTag}'
combined = np.load(f'{prefix}Orders12.npz')
firstOrder, secondOrder = np.load(f'{prefix}.npz'), np.load(f'{prefix}Order2.npz')
sweep = np.load(f'data/boundaryDialSweep{args.referenceCheckpoint}{args.holdTag}.npz')
dialLimit, step = float(combined['dialLimit']), float(combined['gridStep'])
angles = combined['ringAngles']

inLimit = sweep['dialLevel'] <= dialLimit + 1e-9
dialLevels, dialPatterns = sweep['dialLevel'][inLimit], sweep['windowMeanVmem'][inLimit]
dialSteps = np.sqrt(((dialPatterns[1:] - dialPatterns[:-1]) ** 2).mean(1))
jumpDials = ((dialLevels[1:] + dialLevels[:-1]) / 2)[dialSteps > args.jumpMilliVolts]
dialAt = lambda level: dialPatterns[int(round(level * 100))]


def rms(values, cells=None):
    values = values if cells is None else values[..., cells]
    return np.sqrt((values ** 2).mean(-1))


def ringValues(dialLevel, firstStrength, secondStrength):
    return dialLevel + firstStrength * np.cos(angles) + secondStrength * np.cos(2 * angles)


def sides(values):
    return np.searchsorted(jumpDials, values)


def motifKey(pattern, threshold=boundary.singleCellSaddleMilliVolts):
    return np.packbits(pattern[interior] < threshold).tobytes()


def participation(patterns):
    return boundary.participationRatio(np.linalg.svd(patterns - patterns.mean(0), compute_uv=False) ** 2)


# ------------------------------------------------------------------------------------------ the combined grid
onGrid = ~combined['symmetryCheck']
dial, first, second = combined['dialLevel'][onGrid], combined['gradientStrength'][onGrid], combined['secondOrderStrength'][onGrid]
patterns, held, windowStd = combined['windowMeanVmem'][onGrid], combined['endOfHoldVmem'][onGrid], combined['windowStdVmem'][onGrid]
index = np.rint(np.c_[dial, first, second] / step).astype(int)
lookup = {tuple(key): row for row, key in enumerate(index)}
kind = np.where(first == 0, 'order2Alone', np.where(second == 0, 'order1Alone',
                np.where(np.abs(second) > first / 4 + 1e-9, 'addsInflections', 'belowThreshold')))
values = np.array([ringValues(*code) for code in zip(dial, first, second)])
twins = np.array([dialAt(level) for level in dial])
change = patterns - twins
changeAll = rms(change)
crossesJump = np.array([bool(((jumpDials >= row.min()) & (jumpDials <= row.max())).any()) for row in values])
print(f"combined grid: {len(dial)} codes -- " + ", ".join(f"{name} {int((kind == name).sum())}" for name in ('order1Alone', 'order2Alone', 'addsInflections', 'belowThreshold')))
print(f"   dial jumps: " + ", ".join(f"{value:.3f}" for value in jumpDials))

summary = dict(dialLimit=dialLimit, step=step, jumpDials=jumpDials.round(3).tolist(), jumpMilliVolts=args.jumpMilliVolts,
               codes=dict(dial=dial.round(3).tolist(), first=first.round(3).tolist(), second=second.round(3).tolist(), kind=kind.tolist(),
                          change=changeAll.round(3).tolist(), crosses=crossesJump.astype(int).tolist(), pattern=patterns.round(1).tolist(),
                          churn=windowStd[:, interior].mean(1).round(3).tolist(),
                          ringLow=values.min(1).round(4).tolist(), ringHigh=values.max(1).round(4).tolist()))
byKind = {}
for name in ('order1Alone', 'order2Alone', 'addsInflections', 'belowThreshold'):
    members = kind == name
    byKind[name] = dict(count=int(members.sum()), medianChange=float(np.median(changeAll[members])),
                        spanShare=float(crossesJump[members].mean()),
                        changedWhenSpanning=float((changeAll[members & crossesJump] > args.changeMilliVolts).mean()) if (members & crossesJump).any() else None,
                        changedWhenQuiet=float((changeAll[members & ~crossesJump] > args.changeMilliVolts).mean()) if (members & ~crossesJump).any() else None,
                        quietMedian=float(np.median(changeAll[members & ~crossesJump])) if (members & ~crossesJump).any() else None)
    print(f"   {name}: median change {byKind[name]['medianChange']:.2f} mV; {100 * byKind[name]['spanShare']:.0f}% span a jump; "
          f"changed > {args.changeMilliVolts} mV: {byKind[name]['changedWhenSpanning']} when spanning, {byKind[name]['changedWhenQuiet']} when not")
summary['byKind'] = byKind

# ------------------------------------------------------------------------------------------ additivity
print("\n=== ADDITIVITY: change with both orders against the sum of the single-order changes ===")
additivity = []
for row in np.where(np.isin(kind, ['addsInflections', 'belowThreshold']))[0]:
    i, j, k = index[row]
    alone1, alone2 = lookup.get((i, j, 0)), lookup.get((i, 0, k))
    if alone1 is None or alone2 is None:
        continue
    predicted = change[alone1] + change[alone2]
    energy = (change[row] ** 2).sum()
    interaction = float(((change[row] - predicted) ** 2).sum() / max(energy, 1e-12))
    base = sides(np.full(40, dial[row]))
    newSides = int(((sides(values[row]) != base) & (sides(values[alone1]) == base) & (sides(values[alone2]) == base)).sum())
    otherSides = int(((sides(values[row]) != sides(values[alone1])) & (sides(values[row]) != sides(values[alone2]))).sum())
    additivity.append((row, interaction, newSides, otherSides, float(rms(change[row])), float(rms(predicted - change[row]))))
additivity = np.array(additivity)
rows, share, newSides, otherSides = additivity[:, 0].astype(int), additivity[:, 1], additivity[:, 2], additivity[:, 3]
quiet = ~crossesJump[rows]
groups = {'quiet': quiet, 'spansNoNewSide': ~quiet & (otherSides == 0), 'spansNewSide': ~quiet & (otherSides > 0)}
summary['additivity'] = dict(count=int(len(rows)), rows=rows.tolist(), interactionShare=share.round(4).tolist(), otherSides=otherSides.astype(int).tolist(),
                             interactionMilliVolts=additivity[:, 5].round(3).tolist(),
                             groups={name: dict(count=int(members.sum()), medianShare=float(np.median(share[members])) if members.any() else None,
                                                medianMilliVolts=float(np.median(additivity[members, 5])) if members.any() else None,
                                                medianChange=float(np.median(additivity[members, 4])) if members.any() else None)
                                     for name, members in groups.items()},
                             byKind={name: dict(count=int((kind[rows] == name).sum()), medianShare=float(np.median(share[kind[rows] == name])))
                                     for name in ('addsInflections', 'belowThreshold')})
for name, entry in summary['additivity']['groups'].items():
    print(f"   {name}: {entry['count']} codes, median interaction share {entry['medianShare']}, median |interaction| {entry['medianMilliVolts']} mV, median change {entry['medianChange']}")
for name, entry in summary['additivity']['byKind'].items():
    print(f"   {name}: {entry['count']} codes, median interaction share {entry['medianShare']:.3f}")

# ------------------------------------------------------------------------------------------ seams
print("\n=== SEAMS: steps of one grid unit along DC, G1 and G2 ===")
steps = []
for key, row in lookup.items():
    for axis in range(3):
        neighbour = list(key)
        neighbour[axis] += 1
        other = lookup.get(tuple(neighbour))
        if other is None:
            continue
        crossing = int((sides(values[row]) != sides(values[other])).sum())
        steps.append((axis, crossing, float(rms(patterns[other] - patterns[row])), row, other))
for i in range(int(round(dialLimit / step))):       # the dial alone as the G1 = G2 = 0 line
    steps.append((0, 40 if ((jumpDials > i * step) & (jumpDials <= (i + 1) * step)).any() else 0,
                  float(rms(dialAt((i + 1) * step) - dialAt(i * step))), -1, -1))
steps = np.array(steps)
axisNames = ('dial', 'first', 'second')
stepSummary = {}
for name, members in (('crossing', steps[:, 1] > 0), ('notCrossing', steps[:, 1] == 0)):
    sizes = steps[members, 2]
    stepSummary[name] = dict(count=int(members.sum()), median=float(np.median(sizes)), jumpShare=float((sizes > args.jumpMilliVolts).mean()),
                             movementShare=float(sizes.sum() / steps[:, 2].sum()))
    print(f"   {name}: {stepSummary[name]['count']} steps, median {stepSummary[name]['median']:.2f} mV, "
          f"{100 * stepSummary[name]['jumpShare']:.1f}% above {args.jumpMilliVolts} mV, {100 * stepSummary[name]['movementShare']:.0f}% of movement")
for axis, name in enumerate(axisNames):
    members = steps[:, 0] == axis
    stepSummary[f'along_{name}'] = dict(count=int(members.sum()), median=float(np.median(steps[members, 2])),
                                        jumpShare=float((steps[members, 2] > args.jumpMilliVolts).mean()))
cells = np.minimum(steps[:, 1].astype(int), 12)
stepSummary['byCellsCrossing'] = [dict(cells=int(value), count=int((cells == value).sum()),
                                       quartiles=np.percentile(steps[cells == value, 2], [25, 50, 75]).round(3).tolist()) for value in np.unique(cells)]
sideGroups = np.unique(sides(values), axis=0, return_inverse=True)[1].ravel()


def explainedShare(groups):
    within = sum(((patterns[groups == group] - patterns[groups == group].mean(0)) ** 2).sum() for group in np.unique(groups))
    return float(1 - within / ((patterns - patterns.mean(0)) ** 2).sum())


numGroups = len(np.unique(sideGroups))
proximity = [explainedShare(KMeans(numGroups, n_init=1, random_state=repeat).fit_predict(index.astype(float))) for repeat in range(args.proximityRepeats)]
stepSummary['seams'] = dict(groups=numGroups, sideExplained=explainedShare(sideGroups), proximityExplained=float(np.mean(proximity)))
print(f"   sides of the jumps: {numGroups} groups explain {100 * stepSummary['seams']['sideExplained']:.1f}%; "
      f"as many groups by closeness {100 * stepSummary['seams']['proximityExplained']:.1f}%")
summary['steps'] = stepSummary

# ------------------------------------------------------------------------------------------ reach
print("\n=== REACH: distance to the nearest pattern of the dial alone or of either order alone ===")
reference = {'dial': dialPatterns}
for name, grid in (('order1', firstOrder), ('order2', secondOrder)):
    members = (~grid['symmetryCheck']) & (grid['gradientDirection'] == 0)
    reference[name] = grid['windowMeanVmem'][members]
reference['order2Negative'] = np.array([np.rot90(pattern.reshape(size, size), -1).reshape(-1) for pattern in reference['order2']])
singles = np.concatenate(list(reference.values()))
singleLabels = np.concatenate([[name] * len(value) for name, value in reference.items()])
distanceToSingles = np.array([rms(singles - pattern).min() for pattern in patterns])
distanceToDial = np.array([rms(dialPatterns - pattern).min() for pattern in patterns])
reach = {}
for name in ('order1Alone', 'order2Alone', 'addsInflections', 'belowThreshold'):
    members = kind == name
    reach[name] = dict(offDialMedian=float(np.median(distanceToDial[members])), offDialMax=float(distanceToDial[members].max()),
                       offSinglesMedian=float(np.median(distanceToSingles[members])), offSinglesMax=float(distanceToSingles[members].max()),
                       beyondDialSpread=int((distanceToSingles[members] > rms(dialPatterns[:, None] - dialPatterns[None]).max()).sum()))
    print(f"   {name}: off the dial median {reach[name]['offDialMedian']:.2f} (max {reach[name]['offDialMax']:.1f}); "
          f"off every single-order pattern median {reach[name]['offSinglesMedian']:.2f} (max {reach[name]['offSinglesMax']:.1f})")
summary['reach'] = reach
summary['codes']['offSingles'] = distanceToSingles.round(3).tolist()
summary['codes']['offDial'] = distanceToDial.round(3).tolist()
summary['dialSpread'] = float(rms(dialPatterns[:, None] - dialPatterns[None]).max())

# ------------------------------------------------------------------------------------------ inflection test
print("\n=== DOES CROSSING THE INFLECTION THRESHOLD MATTER IN ITSELF? ===")
interactionMilliVolts = np.full(len(kind), np.nan)
interactionMilliVolts[rows] = additivity[:, 5]
matched = []
for strength in np.unique(np.round(np.abs(second[second != 0]), 6)):
    same = np.isclose(np.abs(second), strength) & (first > 0)
    above, below = same & (kind == 'addsInflections'), same & (kind == 'belowThreshold')
    if above.sum() >= 10 and below.sum() >= 10:
        matched.append(dict(strength=float(strength), aboveCount=int(above.sum()), belowCount=int(below.sum()),
                            aboveOffSingles=float(np.median(distanceToSingles[above])), belowOffSingles=float(np.median(distanceToSingles[below])),
                            aboveInteraction=float(np.nanmedian(interactionMilliVolts[above])), belowInteraction=float(np.nanmedian(interactionMilliVolts[below]))))
        print(f"   |G2| {strength:.2f}: above {above.sum()} codes, off the singles {matched[-1]['aboveOffSingles']:.2f} mV, interaction {matched[-1]['aboveInteraction']:.2f} mV; "
              f"below {below.sum()}, {matched[-1]['belowOffSingles']:.2f} mV, {matched[-1]['belowInteraction']:.2f} mV")
thresholdSteps = []
for key, row in lookup.items():
    i, j, k = key
    other = lookup.get((i, j + 1, k))
    if k == 0 or other is None:
        continue
    crossesThreshold = (abs(k) > j / 4 + 1e-9) and not (abs(k) > (j + 1) / 4 + 1e-9)
    crossesAJump = bool((sides(values[row]) != sides(values[other])).any())
    thresholdSteps.append((crossesThreshold, crossesAJump, float(rms(patterns[other] - patterns[row]))))
thresholdSteps = np.array(thresholdSteps)
stepTest = {}
for jumpName, jumpFlag in (('noJump', 0), ('jump', 1)):
    for thresholdName, thresholdFlag in (('crossesThreshold', 1), ('otherSteps', 0)):
        members = (thresholdSteps[:, 0] == thresholdFlag) & (thresholdSteps[:, 1] == jumpFlag)
        stepTest[f'{jumpName}_{thresholdName}'] = dict(count=int(members.sum()), median=float(np.median(thresholdSteps[members, 2])),
                                                       jumpShare=float((thresholdSteps[members, 2] > args.jumpMilliVolts).mean()))
        print(f"   G1 steps, {jumpName}, {thresholdName}: {members.sum()} steps, median {stepTest[f'{jumpName}_{thresholdName}']['median']:.2f} mV")
summary['inflection'] = dict(matched=matched, steps=stepTest)

# ------------------------------------------------------------------------------------------ motifs and dimensions
print("\n=== MOTIFS AND DIMENSIONS ===")
sets = {'dial': dialPatterns, 'order1': reference['order1'], 'order2': np.concatenate([reference['order2'], reference['order2Negative']]),
        'bothSingles': singles, 'addsInflections': patterns[kind == 'addsInflections'], 'belowThreshold': patterns[kind == 'belowThreshold'],
        'combinedAll': patterns}
motifTable = []
for threshold in (float(value) for value in args.motifThresholds.split(',')):
    keys = {name: {motifKey(pattern, threshold) for pattern in value} for name, value in sets.items()}
    motifTable.append(dict(threshold=threshold, **{name: len(value) for name, value in keys.items()},
                           newInCombined=len(keys['combinedAll'] - keys['bothSingles']),
                           newAddsInflections=len(keys['addsInflections'] - keys['bothSingles'])))
    print(f"   {threshold} mV: " + ", ".join(f"{name} {len(value)}" for name, value in keys.items())
          + f"; new with both orders {motifTable[-1]['newInCombined']} (adds inflections {motifTable[-1]['newAddsInflections']})")
summary['motifs'] = motifTable
singleKeys = {motifKey(pattern) for pattern in singles}
combinedKeys = [motifKey(pattern) for pattern in patterns]
newRuns = {}
for row, key in enumerate(combinedKeys):
    if key not in singleKeys and kind[row] in ('addsInflections', 'belowThreshold'):
        newRuns.setdefault(key, []).append(row)
gallery = []
for key, members in sorted(newRuns.items(), key=lambda item: -len(item[1]))[:args.galleryMotifs]:
    centre = patterns[members].mean(0)
    example = members[int(np.argmin(rms(patterns[members] - centre)))]
    gallery.append(dict(runs=len(members), darkCells=int((patterns[example][interior] < boundary.singleCellSaddleMilliVolts).sum()),
                        addsInflections=int(sum(kind[row] == 'addsInflections' for row in members)),
                        example=dict(dial=float(dial[example]), first=float(first[example]), second=float(second[example]), pattern=patterns[example].round(1).tolist())))
summary['newMotifGallery'] = gallery
summary['newMotifCodes'] = int(sum(len(members) for members in newRuns.values()))
newRows = np.array(sorted(row for members in newRuns.values() for row in members))
bothOrders = np.isin(kind, ['addsInflections', 'belowThreshold'])
summary['newMotifStats'] = dict(
    horizontalOvalShare=float((second[newRows] < 0).mean()), horizontalOvalShareAll=float((second[bothOrders] < 0).mean()),
    ovalStrengthMedian=float(np.median(np.abs(second[newRows]))), ovalStrengthMedianAll=float(np.median(np.abs(second[bothOrders]))),
    tiltMedian=float(np.median(first[newRows])), tiltMedianAll=float(np.median(first[bothOrders])),
    dialQuartiles=np.percentile(dial[newRows], [25, 50, 75]).round(3).tolist(),
    slices={f'{level:g}': dict(bothOrders=int((bothOrders & np.isclose(dial, level)).sum()), newMotifs=int(np.isclose(dial[newRows], level).sum()),
                               ovalAloneMedian=float(np.median(changeAll[(kind == 'order2Alone') & np.isclose(dial, level)])),
                               tiltAloneMedian=float(np.median(changeAll[(kind == 'order1Alone') & np.isclose(dial, level)])),
                               bothMedian=float(np.median(changeAll[bothOrders & np.isclose(dial, level)])))
            for level in (0.48, 0.64, 0.80, 1.00)})
print("   new-motif codes: " + json.dumps(summary['newMotifStats']))
print(f"   codes with both orders whose motif no single order makes: {summary['newMotifCodes']}; most common: "
      + ", ".join(f"{entry['runs']} ({entry['darkCells']} dark)" for entry in gallery[:6]))
summary['dimension'] = {name: participation(value) for name, value in sets.items()}
print("   effective dimensions: " + ", ".join(f"{name} {value:.2f}" for name, value in summary['dimension'].items()))
everything = np.concatenate([dialPatterns, singles[len(dialPatterns):], patterns])
mean = everything.mean(0)
_, singular, components = np.linalg.svd(everything - mean, full_matrices=False)
summary['patternSpace'] = dict(explained=(singular[:4] ** 2 / (singular ** 2).sum()).round(4).tolist(),
                               dialTrack=((dialPatterns - mean) @ components[:2].T).round(2).tolist(),
                               singles=((singles[len(dialPatterns):] - mean) @ components[:2].T).round(2).tolist(),
                               singleLabels=singleLabels[len(dialPatterns):].tolist(),
                               combined=((patterns - mean) @ components[:2].T).round(2).tolist())

# ------------------------------------------------------------------------------------------ checks
checks = []
for row in np.where(combined['symmetryCheck'])[0]:
    match = np.where(onGrid & np.isclose(combined['dialLevel'], combined['dialLevel'][row]) & np.isclose(combined['gradientStrength'], -combined['gradientStrength'][row])
                     & np.isclose(combined['secondOrderStrength'], combined['secondOrderStrength'][row]))[0]
    if len(match):
        mirrored = combined['windowMeanVmem'][match[0]].reshape(size, size)[::-1, :].reshape(-1)
        checks.append(dict(dial=float(combined['dialLevel'][row]), first=float(-combined['gradientStrength'][row]),
                           second=float(combined['secondOrderStrength'][row]), maxDifference=float(np.abs(combined['windowMeanVmem'][row] - mirrored).max())))
summary['flipChecks'] = checks
mirrorResidual = np.abs(patterns.reshape(-1, size, size) - patterns.reshape(-1, size, size)[:, :, ::-1]).max((1, 2))
summary['mirrorResidual'] = dict(max=float(mirrorResidual.max()), median=float(np.median(mirrorResidual)))
print("\nvertical-flip checks, largest |difference| (mV): " + ", ".join(f"{check['maxDifference']:.1e}" for check in checks))
print(f"left-right mirror symmetry: largest departure {mirrorResidual.max():.2e} mV (median {np.median(mirrorResidual):.1e})")
consistency = []
for name, grid in (('order1', firstOrder), ('order2', secondOrder)):
    members = np.where((~grid['symmetryCheck']) & (grid['gradientDirection'] == 0))[0]
    for member in members:
        key = (int(round(grid['dialLevel'][member] / step)), int(round(grid['gradientStrength'][member] / step)) if name == 'order1' else 0,
               0 if name == 'order1' else int(round(grid['gradientStrength'][member] / step)))
        onStep = abs(grid['dialLevel'][member] / step - key[0]) < 1e-6 and abs(grid['gradientStrength'][member] / step - round(grid['gradientStrength'][member] / step)) < 1e-6
        if onStep and key in lookup:
            consistency.append(float(np.abs(grid['windowMeanVmem'][member] - patterns[lookup[key]]).max()))
summary['singleGridConsistency'] = dict(count=len(consistency), max=float(max(consistency)))
print(f"single-order codes run in both grids: {len(consistency)}, largest difference {max(consistency):.1e} mV")

outputPath = f'data/boundaryOrderPairLandscapeSummary{args.referenceCheckpoint}{args.holdTag}.json'
json.dump(summary, open(outputPath, 'w'), separators=(',', ':'))
print(f"\nwrote {outputPath}")
