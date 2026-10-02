"""Which part of the ring writes the stripe? Hold only part of the ring and read what the stripe becomes.

The stripe counterpart of analyzeBoundaryHarmonicWallCounterfactual11x11.py (whose core loop it reuses), for a stripe code that
forms its stripe at its own best moment, which can come well before the face's. Every run is the pure re-implementation of
the model step (boundaryHarmonicStep.Step) with only part of the ring held to the trained code's values for the hold
(iterations 0-300, with the clamp's tissue-wide extra Vmem update in every run), against the same extraUpdateOnly baseline the
relay uses (the re-solve, but no ring written), taken to the trained code's best moment. Read there: the selectivity
difference (mean conductance of the 27 centre-stripe cells minus the mean of the 54 flank cells, relative to the baseline, in
G_ref), the number of centre-stripe cells that are dark (Vmem below -34.6 mV), the number of stray dark interior cells, and the
structural overlap with the stripe.

Held sets:
  fullRing, noRing                  the trained code, and nothing (the baseline)
  topRow, bottomRow, leftColumn, rightColumn
  topAndBottom, leftAndRight        the walls facing the stripe's ends, and the walls facing the flanks
  stripeFacing                      the six ring cells above and below the centre stripe (columns 4-6 of the top and bottom rows)
  topAndBottomWithoutStripeFacing   the walls facing the stripe's ends, less those six
  halfArcs                          every contiguous run of twenty ring cells (forty offsets) held alone, and the complement
  singleCells                       each ring cell alone
  randomSubsets                     --numRandomSubsets random subsets of each size of a structured set, held alone (the controls)

    python3 analyzeBoundaryHarmonicStripeRingSegments11x11.py --trainedRunPath data/boundaryHarmonicTraining...StripesInterior.../order2_restart02.npz

Writes data/boundaryHarmonicStripeRingSegments<checkpoint>Hold<hold><target><--tag>.json (never overwriting).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from boundaryHarmonicStep import Step

parser = argparse.ArgumentParser()
parser.add_argument('--trainedRunPath', type=str, required=True, help='the training restart (.npz) whose best code is analysed')
parser.add_argument('--numRandomSubsets', type=int, default=200)
parser.add_argument('--seed', type=int, default=20261001)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--outputPath', type=str, default=None, help='default: data/boundaryHarmonicStripeRingSegments<checkpoint>Hold<hold><target><tag>.json')
parser.add_argument('--predictionsPath', type=str, default=None, help='the registered criteria, copied into the output')
args = parser.parse_args()
torch.set_grad_enabled(False)
started = time.time()

run = dict(np.load(args.trainedRunPath))
orders = run['orders'] if 'orders' in run else np.arange(len(run['bestCoefficients']))
hold, checkpoint = int(run['holdIterations']), int(run['referenceCheckpoint'])
scoredIteration = int(run['bestIteration'])
ceiling = float(run['ceiling'])
ringValues = np.clip(np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), orders)) @ run['bestCoefficients'], 0, ceiling)
outputPath = args.outputPath or f"data/boundaryHarmonicStripeRingSegments{checkpoint}Hold{hold}{run['targetName']}{args.tag}.json"
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

step = Step(ringCode=ringValues)
n, Gref = step.numCells, step.Gref
T = scoredIteration + 1                                                   # state index of the scored moment
stripe = np.array(sorted(boundary.centreStripeCellIndices.tolist()))
interior = np.array(boundary.interiorCellIndices)
flank = np.array([c for c in interior if c not in set(stripe.tolist())])
fullMask = step.ringMask.clone()
ring = [int(c) for c in boundary.boundaryRingCells]
rows, columns = np.array(ring) // 11, np.array(ring) % 11
cellsOf = lambda keep: [c for c, r, k in zip(ring, rows, columns) if keep(r, k)]
stripeFacing = cellsOf(lambda r, c: r in (0, 10) and 4 <= c <= 6)
sets = dict(
    topRow=cellsOf(lambda r, c: r == 0), bottomRow=cellsOf(lambda r, c: r == 10),
    leftColumn=cellsOf(lambda r, c: c == 0 and 0 < r < 10), rightColumn=cellsOf(lambda r, c: c == 10 and 0 < r < 10),
    topAndBottom=cellsOf(lambda r, c: r in (0, 10)), leftAndRight=cellsOf(lambda r, c: c in (0, 10) and 0 < r < 10),
    stripeFacing=stripeFacing, topAndBottomWithoutStripeFacing=[c for c in cellsOf(lambda r, k: r in (0, 10)) if c not in set(stripeFacing)])


def simulate(held):
    """The state at the scored moment when only the cells in `held` are held; held=None is the baseline."""
    mask = torch.zeros(n, dtype=torch.double)
    if held:
        mask[list(held)] = 1.0
    step.ringMask = mask
    V, G = step.initialVmem.clone(), step.initialGpol.clone()
    for m in range(T):
        V, G = step(V, G, bool(held) and m < hold, m < hold)
    step.ringMask = fullMask
    return V.numpy() * 1000.0, G.numpy() / Gref


def readout(V, G):
    dark = V < boundary.hyperpolarizedThresholdMilliVolts
    return dict(selectivity=float(G[stripe].mean() - G[flank].mean()), stripeDark=int(dark[stripe].sum()),
                strayDark=int(dark[interior].sum() - dark[stripe].sum()), overlap=float(boundary.structuralIntersectionOverUnion(V, stripe)))


baselineV, baselineG = simulate(None)
baseline = readout(baselineV, baselineG)
print(f'[{time.time() - started:4.0f}s] baseline (no ring held): selectivity {baseline["selectivity"]:+.4f}, {baseline["stripeDark"]} stripe cells dark, '
      f'{baseline["strayDark"]} strays', flush=True)
outcomes = {}
patterns = dict(noRing=np.round(baselineV, 0).astype(int).tolist())          # the pattern at the scored moment of each named set, for the report


def hold_(name, held):
    V, G = simulate(held)
    if name in sets or name in ('fullRing', 'topAndBottomExcluded', 'leftAndRightExcluded'):
        patterns[name] = np.round(V, 0).astype(int).tolist()
    outcome = readout(V, G)
    outcome['selectivityDifference'] = outcome['selectivity'] - baseline['selectivity']
    outcome['numCells'] = len(held)
    outcomes[name] = outcome
    return outcome


full = hold_('fullRing', ring)
print(f'[{time.time() - started:4.0f}s] full ring: selectivity difference {full["selectivityDifference"]:+.4f}, {full["stripeDark"]}/27 stripe cells dark, '
      f'{full["strayDark"]} strays, overlap {full["overlap"]:.3f}', flush=True)
for name, held in sets.items():
    o = hold_(name, held)
    print(f'[{time.time() - started:4.0f}s] {name:34s} ({len(held):2d} cells): difference {o["selectivityDifference"]:+.4f} '
          f'({o["selectivityDifference"] / full["selectivityDifference"]:+.2f} of the full ring), {o["stripeDark"]}/27 dark, {o["strayDark"]} strays', flush=True)
for name in ('topAndBottom', 'leftAndRight'):                                  # and what is left when each is taken out
    hold_(name + 'Excluded', [c for c in ring if c not in set(sets[name])])

halfArcs = []
for offset in range(len(ring)):
    arc = [ring[(offset + k) % len(ring)] for k in range(20)]
    rest = [c for c in ring if c not in set(arc)]
    held, other = hold_(f'arc{offset}', arc), hold_(f'arc{offset}Complement', rest)
    halfArcs.append(dict(offset=offset, firstCell=arc[0], held=held['selectivityDifference'], complement=other['selectivityDifference'],
                         additivity=(held['selectivityDifference'] + other['selectivityDifference'] - full['selectivityDifference']) / full['selectivityDifference'],
                         heldStripeDark=held['stripeDark'], complementStripeDark=other['stripeDark']))
print(f'[{time.time() - started:4.0f}s] half arcs done', flush=True)
singleCells = {str(c): hold_(f'cell{c}', [c])['selectivityDifference'] for c in ring}
print(f'[{time.time() - started:4.0f}s] single cells done', flush=True)

generator = np.random.default_rng(args.seed)
controls = {}
for name in ('topAndBottom', 'leftAndRight', 'stripeFacing', 'topRow', 'leftColumn'):
    size = len(sets[name])
    values = []
    for _ in range(args.numRandomSubsets):
        subset = [int(c) for c in generator.choice(ring, size=size, replace=False)]
        V, G = simulate(subset)
        values.append(readout(V, G)['selectivity'] - baseline['selectivity'])
    controls[name] = dict(size=size, differences=values, percentile95=float(np.percentile(values, 95)), median=float(np.median(values)),
                          setDifference=outcomes[name]['selectivityDifference'],
                          setExceedsShare=float(np.mean(np.array(values) < outcomes[name]['selectivityDifference'])))
    print(f'[{time.time() - started:4.0f}s] random subsets of {size}: median {np.median(values):+.4f}, 95th percentile {np.percentile(values, 95):+.4f}; '
          f'{name} {outcomes[name]["selectivityDifference"]:+.4f} beats {100 * controls[name]["setExceedsShare"]:.0f}% of them', flush=True)

# ------------------------------------------------------------------ the registered criteria (boundaryHarmonicStripeMechanismPredictions...json)
def replayScore():
    """V2: the full ring replayed single-sample for the whole run, balanced RMS at every iteration from the release."""
    target = np.asarray(run['target']).ravel()
    isStripe = np.zeros(n, dtype=bool)
    isStripe[stripe] = True
    V, G = step.initialVmem.clone(), step.initialGpol.clone()
    best = (np.inf, 0)
    for m in range(int(run['numIterations'])):
        V, G = step(V, G, m < hold, m < hold)
        if m >= hold:
            values = V.numpy() * 1000.0
            score = 0.5 * np.sqrt(np.mean((values[isStripe] - target[isStripe]) ** 2)) + 0.5 * np.sqrt(np.mean((values[~isStripe] - target[~isStripe]) ** 2))
            best = min(best, (score, m))
    return best


replay = replayScore()
fullDifference = full['selectivityDifference']
verdicts = dict(
    V2=dict(storedScore=float(run['bestScore']), storedIteration=scoredIteration, replayScore=float(replay[0]), replayIteration=int(replay[1]),
            holds=bool(abs(replay[0] - float(run['bestScore'])) < 1e-6 and replay[1] == scoredIteration)),
    W1=dict(stripeFacingShareOfFull=outcomes['stripeFacing']['selectivityDifference'] / fullDifference,
            holds=bool(outcomes['stripeFacing']['selectivityDifference'] / fullDifference >= 0.5)),
    W2=dict(topAndBottomShare=outcomes['topAndBottom']['selectivityDifference'] / fullDifference,
            leftAndRightShare=outcomes['leftAndRight']['selectivityDifference'] / fullDifference,
            holds=bool(outcomes['topAndBottom']['selectivityDifference'] / fullDifference >= 0.75
                       and outcomes['leftAndRight']['selectivityDifference'] / fullDifference <= 0.25)),
    W3=dict(stripeFacing=dict(difference=outcomes['stripeFacing']['selectivityDifference'], percentile95=controls['stripeFacing']['percentile95']),
            topAndBottom=dict(difference=outcomes['topAndBottom']['selectivityDifference'], percentile95=controls['topAndBottom']['percentile95']),
            holds=bool(outcomes['stripeFacing']['selectivityDifference'] > controls['stripeFacing']['percentile95']
                       and outcomes['topAndBottom']['selectivityDifference'] > controls['topAndBottom']['percentile95'])),
    W4=dict(topRowOverlap=outcomes['topRow']['overlap'], bottomRowOverlap=outcomes['bottomRow']['overlap'], topAndBottomOverlap=outcomes['topAndBottom']['overlap'],
            holds=bool(outcomes['topRow']['overlap'] < 0.9 and outcomes['bottomRow']['overlap'] < 0.9 and outcomes['topAndBottom']['overlap'] >= 0.9)),
    W5=dict(sumOfHalves=outcomes['topAndBottom']['selectivityDifference'] + outcomes['leftAndRight']['selectivityDifference'], full=fullDifference,
            relativeError=(outcomes['topAndBottom']['selectivityDifference'] + outcomes['leftAndRight']['selectivityDifference'] - fullDifference) / fullDifference,
            holds=bool(abs(outcomes['topAndBottom']['selectivityDifference'] + outcomes['leftAndRight']['selectivityDifference'] - fullDifference) <= 0.25 * abs(fullDifference))))
for key, verdict in verdicts.items():
    print(f"{key}: {'holds' if verdict['holds'] else 'FAILS'}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in verdict.items() if k != 'holds'}, flush=True)

result = dict(note='EXPLORATORY unless a predictions file was given; criteria in it were registered before this run.', trainedRunPath=args.trainedRunPath, verdicts=verdicts,
              predictions=json.load(open(args.predictionsPath)) if args.predictionsPath else None, coefficients=run['bestCoefficients'].tolist(),
              orders=[int(o) for o in orders], ringValues=ringValues.tolist(), scoredIteration=scoredIteration, hold=hold, ceiling=ceiling,
              baseline=baseline, outcomes=outcomes, patterns=patterns, sets=sets, halfArcs=halfArcs, singleCells=singleCells, ring=ring, randomSubsetControls=controls)
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print('wrote', outputPath, flush=True)
