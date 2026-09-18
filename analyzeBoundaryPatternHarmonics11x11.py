"""Circular harmonics of the boundary clamp and of the interior pattern, in one language (PolyPatterning_Sim.md,
Section 12).

Every lattice map is split into concentric square shells (shell 0 = the 40-cell boundary ring ... shell 4 = the 8 cells
around the centre, plus the centre cell), and each shell, read clockwise from its top-left corner, into circular
harmonic orders. The clamp code is the same kind of sequence on shell 0, so a clamp order and a pattern order can be
compared directly. On a square lattice with a symmetric model, a clamp of order k can only drive pattern orders
k, |k - 4|, k + 4, |k - 8|, ...: a uniform dial (order 0) gives orders 0, 4, 8, ...; a first-order gradient (order 1)
gives odd orders to first order in its strength.

  1. Dial sweeps (one per entry of --sweeps): the late patterns are fully symmetric, so each is 21 signed
     amplitudes (orders 0, 4, 8, ... per shell, plus the centre). The order-4 amplitude of a shell is negative when its
     corners (the diagonals) are more hyperpolarised than its edge midpoints (an X) and positive for a plus. Counts
     X <-> plus switches along the dial, how abrupt they are, how often dials a short distance apart show opposite
     motifs, and how much of each amplitude's movement happens at the pattern's jumps.
     The same sweeps in the lattice's own harmonics, the 2D cosine modes (p, q) that are the eigenmodes of the
     gap-junction Laplacian on the non-periodic grid: an axial-versus-diagonal index (energy in modes whose wavevector
     lies within 22.5 degrees of an axis, minus that within 22.5 degrees of a diagonal, over their sum), its switches,
     per-mode sign switches of the leading symmetric modes, and a basis-free check (correlation of patterns at nearby
     dials).
  2. First-order gradient pairs (hold 100): the share of each change's energy in odd orders (what an order-1 clamp
     drives directly), orders 2 mod 4 and orders 0 mod 4, by distance to the nearest jump and gradient strength.
  3. Trained seeds (Gpol-only D1): code energy and pattern energy in the same order classes, seed by seed.
  4. Overall progression along the dial, for every pair of holds, by region (all dials, below the bistable window,
     out to --extendedDialLimit, below the ring-flip zone, in it and above it): the 21 amplitudes as a trajectory, with the dial as
     its time axis. Per amplitude (scale-free): share of its movement in steps larger than a tenth of its range, total
     variation over range, largest step over range, and share of its variance at dial periods of at least 0.4; the
     longer hold counts as smoother on an amplitude when that measure is lower (higher for the last), compared across
     the 21 with a Wilcoxon signed-rank test. Whole trajectory, weighted by energy (equal to cell-level RMS) or with every
     amplitude scaled by its standard deviation over the pair: path length, straightness, share of the path in the
     largest tenth of the pair's steps, and share of variance at dial periods of at least 0.4. Each by dial region.
     Also, every sweep on one common footing (scale and large-step threshold pooled over all of them), and the PCA of
     the scaled amplitudes pooled over the sweeps with the fixed readout, for drawing the trajectories.

Writes data/boundaryPatternHarmonicsSummary<checkpoint>.json for plotBoundaryDial11x11.py.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.fft import dct, dctn
from scipy.stats import spearmanr, wilcoxon

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--motifThresholds', type=str, default='2,4', help='order-4 amplitudes (mV) that count as a clear X or plus')
parser.add_argument('--jumpThresholdMilliVolts', type=float, default=2.0)
parser.add_argument('--ringFlipZone', type=str, default='1.44,1.62', help='dial range around the ring flip, reported separately')
parser.add_argument('--nearbyDialDistance', type=float, default=0.2)
parser.add_argument('--extendedDialLimit', type=float, default=1.3,
                    help='upper edge of an extra low-dial region reported alongside the one below the bistable window')
parser.add_argument('--sweeps', type=str, default='100,301,500,301aligned,500aligned',
                    help="dial sweeps to include, by hold length (the reference checkpoint's own hold, then Hold<n> files); "
                         "<n>aligned reads the sweep whose readout was aligned to release (Hold<n>Aligned); missing sweeps are skipped")
args = parser.parse_args()

numShells = 5
centreCell = 5 * boundary.latticeCols + 5
motifThresholds = [float(value) for value in args.motifThresholds.split(',')]
flipZoneLow, flipZoneHigh = (float(value) for value in args.ringFlipZone.split(','))
windowLow = boundary.singleCellBistableRange[0]   # dials below the single-cell bistable window, reported as their own region
extendedLimit = args.extendedDialLimit            # a wider low-dial region, reported the same way
regionLimits = dict(belowWindow=windowLow, belowExtended=extendedLimit, belowFlipZone=flipZoneLow)


def symmetricAmplitudes(patterns):
    """Signed cosine amplitudes of orders 0, 4, 8, ... on shells 0-4, plus the centre cell, for fully symmetric maps."""
    amplitudes, names = [], []
    for shell in range(numShells):
        cells = boundary.shellCells(shell)
        count = len(cells)
        for order in range(0, count // 2 + 1, 4):
            coefficients = np.array([boundary.circularHarmonicCoefficients(pattern[cells], order)[order] for pattern in patterns])
            if np.abs(coefficients.imag).max() > 1e-6 * max(1.0, np.abs(coefficients).max()):
                raise ValueError(f"shell {shell} order {order} is not real: the maps are not fully symmetric")
            amplitudes.append(coefficients.real * (1 if order in (0, count // 2) else 2))
            names.append(f'shell{shell}order{order}')
    amplitudes.append(patterns[:, centreCell])
    names.append('centre')
    return np.array(amplitudes).T, names


def orderEnergy(fieldValues, maxOrder=20):
    """Energy of a lattice map in each circular order, summed over shells 0-4; the centre cell counts as order 0."""
    energy = np.zeros(maxOrder + 1)
    for shell in range(numShells):
        cells = boundary.shellCells(shell)
        count = len(cells)
        power = count * np.abs(np.fft.fft(fieldValues[cells]) / count) ** 2
        for order in range(count):
            energy[min(order, count - order)] += power[order]
    energy[0] += fieldValues[centreCell] ** 2
    return energy


def ringOrderEnergy(ringValues):
    count = len(ringValues)
    power = count * np.abs(np.fft.fft(ringValues) / count) ** 2
    energy = np.zeros(count // 2 + 1)
    for order in range(count):
        energy[min(order, count - order)] += power[order]
    return energy


orderClasses = {'odd': lambda order: order % 2 == 1, 'twoModFour': lambda order: order % 4 == 2,
                'zeroModFour': lambda order: order % 4 == 0}


def classShares(energy, skipOrderZero=False):
    orders = np.arange(len(energy))
    keep = orders > 0 if skipOrderZero else np.ones(len(energy), bool)
    total = energy[keep].sum() + 1e-12
    return {name: float(energy[keep & np.array([rule(order) for order in orders])].sum() / total) for name, rule in orderClasses.items()}


def latticeCosineMode(rowFrequency, columnFrequency):
    rows = np.cos(np.pi * rowFrequency * (np.arange(boundary.latticeRows) + 0.5) / boundary.latticeRows)
    columns = np.cos(np.pi * columnFrequency * (np.arange(boundary.latticeCols) + 0.5) / boundary.latticeCols)
    mode = np.outer(rows, columns).reshape(-1)
    return mode / np.linalg.norm(mode)


rowFrequencies, columnFrequencies = np.meshgrid(np.arange(boundary.latticeRows), np.arange(boundary.latticeCols), indexing='ij')
wavevectorAngle = np.degrees(np.arctan2(columnFrequencies, rowFrequencies))
nonConstant = (rowFrequencies + columnFrequencies) > 0
axialModes = nonConstant & ((wavevectorAngle < 22.5) | (wavevectorAngle > 67.5))
diagonalModes = nonConstant & (wavevectorAngle >= 22.5) & (wavevectorAngle <= 67.5)


def symmetricCosineMode(first, second):
    """The fully symmetric combination of cosine modes (first, second) and (second, first); both frequencies even."""
    mode = latticeCosineMode(first, second) + latticeCosineMode(second, first) if first != second else latticeCosineMode(first, second)
    return mode / np.linalg.norm(mode)


symmetricModeLabels = [(first, second) for first in range(0, boundary.latticeRows, 2) for second in range(first, boundary.latticeCols, 2)]
symmetricModes = np.array([symmetricCosineMode(first, second) for first, second in symmetricModeLabels])


def axialDiagonalIndex(patterns):
    power = np.array([dctn(pattern.reshape(boundary.latticeRows, boundary.latticeCols), norm='ortho') ** 2 for pattern in patterns])
    axial, diagonal = power[:, axialModes].sum(1), power[:, diagonalModes].sum(1)
    return (axial - diagonal) / (axial + diagonal)


def signSwitches(values, dials, threshold):
    """Sign changes along the dial, ignoring dials where the value is within +-threshold."""
    state = np.where(values < -threshold, -1, np.where(values > threshold, 1, 0))
    switches, lastState, lastIndex = [], 0, None
    for index, value in enumerate(state):
        if value == 0:
            continue
        if lastState and value != lastState:
            switches.append(dict(fromDial=float(dials[lastIndex]), toDial=float(dials[index]), toSign=int(value)))
        lastState, lastIndex = value, index
    return switches


def motifSwitches(amplitude, dials, threshold):
    """X <-> plus switches of one shell's order-4 amplitude (negative = X, positive = plus)."""
    return [dict(fromDial=switch['fromDial'], toDial=switch['toDial'], toMotif='plus' if switch['toSign'] > 0 else 'X')
            for switch in signSwitches(amplitude, dials, threshold)]


def region(dialLevel):
    return 'belowFlipZone' if dialLevel < flipZoneLow else ('flipZone' if dialLevel <= flipZoneHigh else 'aboveFlipZone')


summary = dict(motifThresholds=motifThresholds, ringFlipZone=[flipZoneLow, flipZoneHigh], sweeps={})
sweepAmplitudes, amplitudeNames = {}, None

# ============================================================================ 1. dial sweeps, both holds
print("=== 1. DIAL SWEEPS: order-4 motif (negative = X, positive = plus) by shell ===")
referenceHoldIterations = int(boundary.loadCheckpoint(args.referenceCheckpoint)['clampParameters']['clampEndIter']) + 1
sweepPaths = {}
for token in args.sweeps.split(','):
    hold, aligned = int(token.replace('aligned', '')), token.endswith('aligned')
    suffix = '' if hold == referenceHoldIterations else f"Hold{hold}{'Aligned' if aligned else ''}"
    path = f'data/boundaryDialSweep{args.referenceCheckpoint}{suffix}.npz'
    if os.path.exists(path):
        sweepPaths[f'hold{token}'] = path
    else:
        print(f"(skipping sweep {token}: {path} not found)")
for label, path in sweepPaths.items():
    sweep = np.load(path)
    dials, patterns = sweep['dialLevel'], sweep['windowMeanVmem']
    symmetricShares = [boundary.symmetricShare(pattern - pattern.mean()) for pattern in patterns]
    amplitudes, names = symmetricAmplitudes(patterns)
    steps = np.sqrt((np.diff(patterns, axis=0) ** 2).mean(1))
    jumps = steps > args.jumpThresholdMilliVolts
    print(f"\n-- {label}: smallest symmetric share {min(symmetricShares):.6f}; {len(names)} amplitudes; {int(jumps.sum())} jumps")
    orderFour = {shell: amplitudes[:, names.index(f'shell{shell}order4')] for shell in range(numShells)}
    sweepSummary = dict(dials=dials.round(3).tolist(), minimumSymmetricShare=float(min(symmetricShares)),
                        orderFour={str(shell): values.round(2).tolist() for shell, values in orderFour.items()},
                        centre=amplitudes[:, names.index('centre')].round(2).tolist(),
                        shellMeans={str(shell): amplitudes[:, names.index(f'shell{shell}order0')].round(2).tolist() for shell in range(numShells)},
                        switches={}, nearbyOpposite={})
    for threshold in motifThresholds:
        byShell = {str(shell): motifSwitches(orderFour[shell], dials, threshold) for shell in range(numShells)}
        allSwitches = [dict(shell=int(shell), **switch) for shell, switches in byShell.items() for switch in switches]
        counts = {}
        for name in ('belowFlipZone', 'flipZone', 'aboveFlipZone', 'belowWindow', 'belowExtended'):
            members = [switch for switch in allSwitches
                       if (switch['toDial'] <= regionLimits[name] if name in ('belowWindow', 'belowExtended')
                           else region(switch['toDial']) == name)]
            counts[name] = dict(total=len(members), abrupt=sum(1 for switch in members if switch['toDial'] - switch['fromDial'] <= 0.021))
        sweepSummary['switches'][str(threshold)] = dict(byShell=byShell, counts=counts)
        print(f"   threshold {threshold} mV: " + "; ".join(f"{name} {value['total']} switches ({value['abrupt']} between neighbouring dials)"
                                                      for name, value in counts.items()))
        for switch in allSwitches:
            print(f"      shell {switch['shell']}: {switch['fromDial']:.2f} -> {switch['toDial']:.2f} to {switch['toMotif']}")
        interiorFour = np.column_stack([orderFour[shell] for shell in range(1, numShells)])
        opposite = {name: [0, 0] for name in ('all', 'belowFlipZone', 'belowWindow', 'belowExtended')}
        pairs = int(round(args.nearbyDialDistance * 100))
        for first in range(len(dials)):
            for second in range(first + 1, min(len(dials), first + pairs + 1)):
                clash = bool(np.any((interiorFour[first] * interiorFour[second] < 0) & (np.abs(interiorFour[first]) > threshold)
                                    & (np.abs(interiorFour[second]) > threshold)))
                opposite['all'][0] += clash
                opposite['all'][1] += 1
                for name, limit in regionLimits.items():
                    if dials[second] <= limit:
                        opposite[name][0] += clash
                        opposite[name][1] += 1
        sweepSummary['nearbyOpposite'][str(threshold)] = {name: value[0] / value[1] for name, value in opposite.items()}
        print(f"      dial pairs at most {args.nearbyDialDistance} apart with opposite motifs on some interior shell: "
              f"{opposite['all'][0] / opposite['all'][1] * 100:.1f}% (both below {flipZoneLow}: {opposite['belowFlipZone'][0] / opposite['belowFlipZone'][1] * 100:.1f}%)")
    movement = np.abs(np.diff(amplitudes, axis=0))
    inJumps = movement[jumps].sum(0) / (movement.sum(0) + 1e-12)
    spread = amplitudes.std(0)
    leading = np.argsort(-spread)[:10]
    sweepSummary['leadingAmplitudes'] = [dict(name=names[index], standardDeviation=float(spread[index]), shareInJumps=float(inJumps[index]))
                                         for index in leading]
    print("   the 10 amplitudes that vary most: share of their movement at the jumps "
          + ", ".join(f"{names[index]} {inJumps[index] * 100:.0f}%" for index in leading))
    centre = amplitudes[:, names.index('centre')]
    for name, members in (('belowFlipZone', dials < flipZoneLow), ('belowWindow', dials <= windowLow),
                          ('belowExtended', dials <= extendedLimit), ('all', np.ones(len(dials), bool))):
        print(f"   centre cell ({name}): {centre[members].min():.1f} to {centre[members].max():.1f} mV")
        sweepSummary[f'centreRange_{name}'] = [float(centre[members].min()), float(centre[members].max())]
    # the same sweep in the lattice's own cosine modes
    orientation = axialDiagonalIndex(patterns)
    below = dials < flipZoneLow
    lattice = dict(axialDiagonalIndex=orientation.round(3).tolist(), switches={}, modeSignSwitches={})
    for threshold in (0.1, 0.2):
        switches = signSwitches(orientation, dials, threshold)
        lattice['switches'][str(threshold)] = switches
        belowSwitches = [switch for switch in switches if switch['toDial'] < flipZoneLow]
        print(f"   lattice axial-diagonal index, threshold {threshold}: {len(switches)} switches, {len(belowSwitches)} below {flipZoneLow} "
              f"({sum(1 for switch in belowSwitches if switch['toDial'] - switch['fromDial'] <= 0.021)} between neighbouring dials): "
              + ", ".join(f"{switch['fromDial']:.2f}->{switch['toDial']:.2f}" for switch in switches))
    lattice['indexRangeBelowFlipZone'] = [float(orientation[below].min()), float(orientation[below].max())]
    for name, limit in (('BelowWindow', windowLow), ('BelowExtended', extendedLimit)):
        lattice[f'indexRange{name}'] = [float(orientation[dials <= limit].min()), float(orientation[dials <= limit].max())]
    modeCoefficients = patterns @ symmetricModes.T / np.sqrt(len(patterns[0]))
    leadingModes = np.argsort(-modeCoefficients.std(0))[:8]
    for threshold in (1.0, 2.0):
        allSwitches = [switch for mode in leadingModes for switch in signSwitches(modeCoefficients[:, mode], dials, threshold)]
        belowSwitches = [switch for switch in allSwitches if switch['toDial'] < flipZoneLow]
        lattice['modeSignSwitches'][str(threshold)] = dict(total=len(allSwitches), belowFlipZone=len(belowSwitches),
                                                           abruptBelowFlipZone=sum(1 for switch in belowSwitches if switch['toDial'] - switch['fromDial'] <= 0.021))
    lattice['leadingModes'] = [list(symmetricModeLabels[mode]) for mode in leadingModes]
    lowFrequency = np.array([np.hypot(first, second) <= 4 for first, second in symmetricModeLabels])
    centredCoefficients = modeCoefficients - modeCoefficients.mean(0)
    lattice['lowFrequencyVarianceShare'] = float((centredCoefficients[:, lowFrequency] ** 2).sum() / (centredCoefficients ** 2).sum())
    print(f"   leading symmetric cosine modes {lattice['leadingModes']}; sign switches of those 8 (above 1 / 2 mV RMS): "
          f"{lattice['modeSignSwitches']['1.0']} / {lattice['modeSignSwitches']['2.0']}; variance at spatial frequency <= 4: {lattice['lowFrequencyVarianceShare'] * 100:.0f}%")
    unitPatterns = patterns - patterns.mean(1, keepdims=True)
    unitPatterns /= np.linalg.norm(unitPatterns, axis=1, keepdims=True)
    for name, limit in (('BelowFlipZone', flipZoneLow), ('BelowWindow', windowLow), ('BelowExtended', extendedLimit)):
        correlation = np.array([unitPatterns[first] @ unitPatterns[second] for first in range(len(dials))
                                for second in range(first + 1, min(len(dials), first + pairs + 1)) if dials[second] < limit])
        lattice[f'nearbyCorrelation{name}'] = dict(median=float(np.median(correlation)), fifthPercentile=float(np.percentile(correlation, 5)),
                                                   shareBelowHalf=float(np.mean(correlation < 0.5)))
    nearbyCorrelation = np.array([unitPatterns[first] @ unitPatterns[second] for first in range(len(dials))
                                  for second in range(first + 1, min(len(dials), first + pairs + 1)) if dials[second] < flipZoneLow])
    print(f"   basis-free: correlation of centred patterns at most {args.nearbyDialDistance} apart, both below {flipZoneLow}: median "
          f"{np.median(nearbyCorrelation):.2f}, 5th percentile {np.percentile(nearbyCorrelation, 5):.2f}, below 0.5 in {np.mean(nearbyCorrelation < 0.5) * 100:.1f}%")
    sweepSummary['lattice'] = lattice
    thumbnailIndices = [int(np.argmin(np.abs(dials - value))) for value in np.round(np.arange(0, 2.001, 0.1), 2)]
    sweepSummary['thumbnails'] = [dict(dial=float(dials[index]), pattern=patterns[index].round(1).tolist()) for index in thumbnailIndices]
    lowDialIndices = [int(np.argmin(np.abs(dials - value))) for value in np.round(np.arange(0, extendedLimit + 0.001, 0.05), 2)]
    sweepSummary['belowWindowThumbnails'] = [dict(dial=float(dials[index]), pattern=patterns[index].round(1).tolist()) for index in lowDialIndices]
    summary['sweeps'][label] = sweepSummary
    sweepAmplitudes[label], amplitudeNames, sweepDials = amplitudes, names, dials

# ============================================================== 4 (computed here). overall progression
slowPeriod = 0.4


def slowVarianceShare(values, dialStep=0.01):
    """Share of a series' variance (along the dial) in DCT components with period at least slowPeriod."""
    centred = values - values.mean(0)
    spectrum = dct(centred, axis=0, norm='ortho') ** 2
    spectrum = spectrum.sum(1) if spectrum.ndim > 1 else spectrum
    spectrum[0] = 0
    slowest = int(np.floor(2 * (len(values) - 1) * dialStep / slowPeriod))
    return float(spectrum[:slowest + 1].sum() / (spectrum.sum() + 1e-12))


def amplitudeMeasures(values):
    steps = np.abs(np.diff(values))
    valueRange, totalVariation = np.ptp(values) + 1e-9, steps.sum() + 1e-9
    return dict(abruptShare=steps[steps > 0.1 * valueRange].sum() / totalVariation, backAndForth=totalVariation / valueRange,
                largestStep=steps.max() / valueRange, slowShare=slowVarianceShare(values))


measureDirection = dict(abruptShare=-1, backAndForth=-1, largestStep=-1, slowShare=1)
orthonormalWeights = np.array([np.sqrt(len(boundary.shellCells(shell)) * (1 if order in (0, len(boundary.shellCells(shell)) // 2) else 0.5))
                               if shell < numShells else 1.0 for shell, order in
                               [(int(name[5]), int(name.split('order')[1])) if name != 'centre' else (numShells, 0) for name in amplitudeNames]])
progressionRegions = {'all': np.ones(len(sweepDials), bool), 'belowWindow': sweepDials <= windowLow,
                      'belowExtended': sweepDials <= extendedLimit, 'belowFlipZone': sweepDials < flipZoneLow,
                      'flipZone': (sweepDials >= flipZoneLow) & (sweepDials <= flipZoneHigh), 'aboveFlipZone': sweepDials > flipZoneHigh}


def compareProgression(shortLabel, longLabel):
    """Smoothness of the longer hold's trajectory against the shorter hold's, by dial region. The equal weights and the
    large-step threshold are pooled over these two sweeps only, so each pair's numbers do not depend on other holds."""
    shortHold, longHold = sweepAmplitudes[shortLabel], sweepAmplitudes[longLabel]
    pooledScale = np.concatenate([shortHold, longHold]).std(0) + 1e-9
    weightings = dict(energy=orthonormalWeights, equal=1 / pooledScale)
    largeStep = {name: np.percentile(np.concatenate([np.linalg.norm(np.diff(hold * weight, axis=0), axis=1) for hold in (shortHold, longHold)]), 90)
                 for name, weight in weightings.items()}
    comparison = dict(shortHold=shortLabel, longHold=longLabel, regions={})
    print(f"\n=== 4. OVERALL PROGRESSION: {longLabel} (long) vs {shortLabel} (short), all 21 amplitudes ===")
    for regionName, members in progressionRegions.items():
        span = sweepDials[members].max() - sweepDials[members].min()
        entry = dict(numDials=int(members.sum()), perAmplitude={}, trajectory={})
        perHold = {label: [amplitudeMeasures(hold[members][:, index]) for index in range(len(amplitudeNames))]
                   for label, hold in (('short', shortHold), ('long', longHold))}
        print(f"-- {regionName} ({members.sum()} dials)")
        for measure, direction in measureDirection.items():
            if measure == 'slowShare' and span < slowPeriod:
                continue
            short = np.array([values[measure] for values in perHold['short']])
            long = np.array([values[measure] for values in perHold['long']])
            test = wilcoxon(short, long) if np.any(np.abs(short - long) > 1e-9) else None
            entry['perAmplitude'][measure] = dict(shortMedian=float(np.median(short)), longMedian=float(np.median(long)),
                                                  longSmoother=int(np.sum(direction * (long - short) > 1e-9)),
                                                  shortSmoother=int(np.sum(direction * (short - long) > 1e-9)), p=float(test.pvalue) if test else 1.0)
            value = entry['perAmplitude'][measure]
            print(f"   per amplitude, {measure:13s}: median {value['shortMedian']:.2f} (short) vs {value['longMedian']:.2f} (long); long hold smoother on "
                  f"{value['longSmoother']}/21, short hold on {value['shortSmoother']}/21 (p={value['p']:.2g})")
        for weightName, weight in weightings.items():
            values = {}
            for label, hold in (('short', shortHold), ('long', longHold)):
                trajectory = hold[members] * weight
                steps = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
                values[label] = dict(path=float(steps.sum()), straightness=float(np.linalg.norm(trajectory[-1] - trajectory[0]) / steps.sum()),
                                     largeStepShare=float(steps[steps > largeStep[weightName]].sum() / steps.sum()),
                                     slowShare=slowVarianceShare(trajectory) if span >= slowPeriod else None)
            entry['trajectory'][weightName] = values
            print(f"   trajectory ({weightName} weights): " + "; ".join(
                f"{label}: path {value['path']:.1f}, straightness {value['straightness']:.2f}, in large steps {value['largeStepShare'] * 100:.0f}%"
                + (f", slow {value['slowShare'] * 100:.0f}%" if value['slowShare'] is not None else '') for label, value in values.items()))
        comparison['regions'][regionName] = entry
    return comparison


holdLabels = list(sweepAmplitudes)
progression = dict(slowPeriod=slowPeriod, comparisons={})
for shortIndex, shortLabel in enumerate(holdLabels):
    for longLabel in holdLabels[shortIndex + 1:]:
        progression['comparisons'][f'{longLabel}_vs_{shortLabel}'] = compareProgression(shortLabel, longLabel)
stacked = np.concatenate([sweepAmplitudes[label] for label in holdLabels])
fixedReadoutLabels = [label for label in holdLabels if not label.endswith('aligned')]
# every hold on one common footing (scale and large-step threshold pooled over all holds), for trends with hold length
commonWeights = dict(energy=orthonormalWeights, equal=1 / (stacked.std(0) + 1e-9))
commonLargeStep = {name: np.percentile(np.concatenate([np.linalg.norm(np.diff(sweepAmplitudes[label] * weight, axis=0), axis=1)
                                                       for label in holdLabels]), 90) for name, weight in commonWeights.items()}
progression['byHold'] = {}
print("\n   all holds on a common footing (path in mV RMS for energy weights):")
for regionName, members in progressionRegions.items():
    progression['byHold'][regionName] = {}
    for weightName, weight in commonWeights.items():
        progression['byHold'][regionName][weightName] = {}
        for label in holdLabels:
            trajectory = sweepAmplitudes[label][members] * weight
            steps = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            progression['byHold'][regionName][weightName][label] = dict(
                path=float(steps.sum() / (np.sqrt(boundary.numCells) if weightName == 'energy' else 1)),
                straightness=float(np.linalg.norm(trajectory[-1] - trajectory[0]) / steps.sum()),
                largeStepShare=float(steps[steps > commonLargeStep[weightName]].sum() / steps.sum()),
                largestStep=float(steps.max() / (np.sqrt(boundary.numCells) if weightName == 'energy' else 1)))
        print(f"   {regionName:14s} {weightName:6s}: " + "; ".join(f"{label} path {value['path']:.1f}, straightness {value['straightness']:.2f}, "
                                                             f"large steps {value['largeStepShare'] * 100:.0f}%"
                                                             for label, value in progression['byHold'][regionName][weightName].items()))
fixedStacked = np.concatenate([sweepAmplitudes[label] for label in fixedReadoutLabels])
scaled = fixedStacked / (fixedStacked.std(0) + 1e-9)
scaled -= scaled.mean(0)
_, singularValues, components = np.linalg.svd(scaled, full_matrices=False)
scores = (scaled @ components[:2].T).reshape(len(fixedReadoutLabels), len(sweepDials), 2)
progression['pca'] = dict(varianceShare=(singularValues[:2] ** 2 / (singularValues ** 2).sum()).round(4).tolist(),
                          **{label: scores[index].round(3).tolist() for index, label in enumerate(fixedReadoutLabels)})
print(f"\n   PCA of scaled amplitudes pooled over {', '.join(fixedReadoutLabels)}: PC1 {progression['pca']['varianceShare'][0] * 100:.0f}%, "
      f"PC2 {progression['pca']['varianceShare'][1] * 100:.0f}% of variance")
summary['progression'] = progression

# =================================================================== 2. first-order gradients (hold 100)
print("\n=== 2. FIRST-ORDER GRADIENT PAIRS (hold 100): which pattern orders the order-1 clamp reaches ===")
sweep = np.load(f'data/boundaryDialSweep{args.referenceCheckpoint}.npz')
landscape = json.load(open(f'data/boundaryDialLandscapeSummary{args.referenceCheckpoint}.json'))
runs = np.load(f'data/boundaryRegimePairs{args.referenceCheckpoint}.npz')
jumpMidpoints = np.array([(jump['fromDial'] + jump['toDial']) / 2 for jump in landscape['tippingPoints']])
changes = runs['pairWindowMeanVmem'] - sweep['windowMeanVmem'][runs['pairSweepIndex']]
sizes = np.sqrt((changes ** 2).mean(1))
strength, dialLevel = runs['pairGradientStrength'], runs['pairDialLevel']
distance = np.array([np.abs(jumpMidpoints - value).min() for value in dialLevel])
energies = np.array([orderEnergy(change) for change in changes])
shares = [classShares(energy) for energy in energies]
oddShare = np.array([share['odd'] for share in shares])
bins = []
for distanceLow, distanceHigh in ((0, 0.025), (0.025, 0.1), (0.1, 10)):
    for strengthLow, strengthHigh in ((0.01, 0.05), (0.05, 0.51)):
        members = (distance >= distanceLow) & (distance < distanceHigh) & (strength >= strengthLow) & (strength < strengthHigh)
        entry = dict(distance=[distanceLow, min(distanceHigh, 2.0)], strength=[strengthLow, min(strengthHigh, 0.5)], count=int(members.sum()),
                     medianSize=float(np.median(sizes[members])),
                     shares={name: float(np.median([share[name] for share, member in zip(shares, members) if member])) for name in orderClasses})
        bins.append(entry)
        print(f"   nearest jump {distanceLow}-{distanceHigh}, G {strengthLow}-{strengthHigh}: n={entry['count']:3d}, median change {entry['medianSize']:5.2f} mV; "
              + ", ".join(f"{name} {value * 100:.0f}%" for name, value in entry['shares'].items()))
clean = (distance >= 0.1) & (strength < 0.05)
oddEnergy = energies[clean][:, 1::2].sum(1)
withinOdd = {'1': energies[clean][:, 1], '3': energies[clean][:, 3], '5': energies[clean][:, 5], '7+': energies[clean][:, 7::2].sum(1)}
withinOdd = {order: float(np.median(values / (oddEnergy + 1e-12))) for order, values in withinOdd.items()}
print(f"   clean runs (nearest jump >= 0.1, G < 0.05, n={int(clean.sum())}): odd-order energy split " + ", ".join(f"order {order} {value * 100:.0f}%" for order, value in withinOdd.items()))
print(f"   odd share vs change size rho={spearmanr(oddShare, sizes).statistic:+.2f}; vs distance to jump {spearmanr(oddShare, distance).statistic:+.2f}; vs G {spearmanr(oddShare, strength).statistic:+.2f}")
summary['gradientTransfer'] = dict(bins=bins, cleanWithinOdd=withinOdd, oddShareVsSizeRho=float(spearmanr(oddShare, sizes).statistic),
                                   points=[dict(distance=round(float(distance[index]), 3), strength=round(float(strength[index]), 4),
                                                size=round(float(sizes[index]), 3), odd=round(float(oddShare[index]), 3)) for index in range(len(sizes))])

# ======================================================================================= 3. trained seeds
print("\n=== 3. TRAINED SEEDS (Gpol-only D1): code order classes vs pattern order classes ===")
metrics = pd.read_csv('data/bandHoldFaceMetrics11x11_all384.csv')
seeds = metrics[(metrics.mechanism == 'Gpol-only') & (metrics.depth == 1)].sort_values('fileNumber').reset_index(drop=True)
codes = boundary.loadBandHoldCodes(seeds.fileNumber.tolist())
patternFile = np.load('data/bandHoldPatterns11x11_all384.npz')
patternRow = {int(number): index for index, number in enumerate(patternFile['fileNumbers'])}
codeShares = [classShares(ringOrderEnergy(codes[number]['field'][boundary.boundaryRingCells]), skipOrderZero=True) for number in seeds.fileNumber]
patternShares = [classShares(orderEnergy(patternFile['windowMean'][patternRow[number]]), skipOrderZero=True) for number in seeds.fileNumber]
seedSummary = dict(codeMedian={}, patternMedian={}, rho={}, p={},
                   points=[dict(file=int(number), code={name: round(share[name], 3) for name in orderClasses},
                                pattern={name: round(pattern[name], 3) for name in orderClasses})
                           for number, share, pattern in zip(seeds.fileNumber, codeShares, patternShares)])
for name in orderClasses:
    codeValues, patternValues = [share[name] for share in codeShares], [share[name] for share in patternShares]
    test = spearmanr(codeValues, patternValues)
    seedSummary['codeMedian'][name], seedSummary['patternMedian'][name] = float(np.median(codeValues)), float(np.median(patternValues))
    seedSummary['rho'][name], seedSummary['p'][name] = float(test.statistic), float(test.pvalue)
    print(f"   {name:12s}: median share of non-dial code energy {np.median(codeValues) * 100:3.0f}%, of pattern energy outside order 0 "
          f"{np.median(patternValues) * 100:3.0f}%; seed-by-seed rho {test.statistic:+.2f} (p={test.pvalue:.2g})")
summary['trainedSeeds'] = seedSummary

outputPath = f'data/boundaryPatternHarmonicsSummary{args.referenceCheckpoint}.json'
json.dump(summary, open(outputPath, 'w'), separators=(',', ':'))
print(f"\nwrote {outputPath}")
