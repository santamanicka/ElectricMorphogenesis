"""Check the coarse-graining library on a synthetic system, where the answers are known by construction.

Random step operators with the fine relay's structure (three role blocks, a hold with a source) generate a difference
D that obeys the exact identity D(n+1) = Jbar(n) D(n) + src(n). Then:

  1. the fine flux sums to the readout of D at every state after the hold;
  2. blocking every cell on its own (the identity partition) reproduces the fine relay in the reduced relay;
  3. the aggregated relay sums to the same fine gap for any grouping, and its edges add up to the fine ones;
  4. the reduced relay's own flux sums to its own final gap after the hold, for any grouping;
  5. the cheap readout-only path agrees with the whole reduced relay;
  6. the stock and flow identity holds for the reduced relay: a block's share changes by what it takes in minus what it gives away;
  7. started from a later state (the block average of the true difference there), the identity partition reproduces the
     fine relay from that state on, and any grouping's flux still sums to its own final gap.

    python3 checkBoundaryHarmonicCoarseGrain.py
"""
import numpy as np

import boundaryHarmonicCoarseGrain as coarse

generator = np.random.default_rng(3)
n, steps, hold = coarse.NUM_CELLS, 40, 12
scale = 0.9 / np.sqrt(n)
jacobians = scale * generator.standard_normal((steps, 3, 2 * n, n)) * (generator.random((steps, 3, 2 * n, n)) < 0.15)
source = generator.standard_normal((hold, 2 * n)) * 0.1
readouts = generator.standard_normal((2, 2 * n)) * 0.05
difference = np.zeros((steps + 1, 2 * n))
for step in range(steps):
    total = np.concatenate([jacobians[step, 0] + jacobians[step, 1], jacobians[step, 2]], axis=1)
    difference[step + 1] = total @ difference[step] + (source[step] if step < hold else 0.0)
phases = dict(first=(0, 15), second=(15, steps))
relay = coarse.FineRelay(jacobians, difference, source, readouts, steps, phases, windowLength=10)
sweep = relay.sweep()
gap = readouts @ difference[steps]
failures = []


def check(name, error, tolerance):
    ok = error <= tolerance
    print(f'{"ok  " if ok else "FAIL"} {name}: {error:.2e} (limit {tolerance:.0e})')
    if not ok:
        failures.append(name)


# 1
worst = max(np.abs(sweep['cellFlux'][state].sum(1) - gap).max() for state in range(hold, steps + 1))
check('fine flux sums to the readout after the hold', worst / np.abs(gap).max(), 1e-10)

# 2
identity = np.arange(n)
reduced = relay.reducedRelay(identity)
check('identity partition: flux equals the fine flux', np.abs(reduced['flux'] - sweep['cellFlux']).max(), 1e-10)
check('identity partition: edges equal the fine edges',
      max(np.abs(reduced['edges'][name] - sweep['phaseEdges'][name]).max() for name in phases), 1e-10)
check('identity partition: injection equals the fine injection', np.abs(reduced['injection'] - sweep['injection']).max(), 1e-10)

# 3, 4, 5, 6 on a square tiling and on a scattered partition
for label, labels in (('2x2 squares', coarse.squareTilings(2)[7]['labels']),
                      ('scattered', coarse.randomPartition(np.array([30, 40, 20, 31]), np.random.default_rng(5)))):
    m = labels.max() + 1
    aggregated = relay.aggregate(sweep, labels)
    worst = max(np.abs(aggregated['flux'][state].sum(1) - gap).max() for state in range(hold, steps + 1))
    check(f'{label}: aggregated flux sums to the fine gap', worst / np.abs(gap).max(), 1e-10)
    fineTotal = sum(sweep['phaseEdges'][name].sum((2, 3)) for name in phases)
    blockTotal = sum(aggregated['edges'][name].sum((2, 3)) for name in phases)
    within = 0.0
    same = labels[:, None] == labels[None, :]
    for name in phases:
        within = within + (sweep['phaseEdges'][name] * same).sum((2, 3))
    check(f'{label}: aggregated edges + internal edges = fine edges', np.abs(fineTotal - blockTotal - within).max(), 1e-9)

    result = relay.reducedRelay(labels)
    finalGap = result['curve'][steps]
    worst = max(np.abs(result['flux'][state].sum(1) - finalGap).max() for state in range(hold, steps + 1))
    check(f'{label}: reduced flux sums to its own final gap', worst / np.abs(finalGap).max(), 1e-10)
    curve, _ = relay.reducedReadoutCurve(labels)
    check(f'{label}: readout-only path agrees with the whole relay', np.abs(curve - result['curve']).max(), 1e-10)

    # stock and flow: after the hold, a block's share changes by what it takes in minus what it gives away
    first, last = 20, 31
    spanRelay = coarse.FineRelay(jacobians, difference, source, readouts, steps, dict(span=(first, last)), windowLength=10)
    spanResult = spanRelay.reducedRelay(labels)
    both = spanResult['edges']['span'].sum(1)                      # (readout, into I, from J)
    netIn = both.sum(2) - both.sum(1)
    change = spanResult['flux'][last] - spanResult['flux'][first]
    check(f'{label}: share change = transfers in - transfers out', np.abs(change - netIn).max() / np.abs(change).max(), 1e-10)

# 7
startState = 20
fromLater = relay.reducedRelay(identity, start=startState)
check('identity partition from a later start: flux equals the fine flux from there',
      np.abs(fromLater['flux'][startState:] - sweep['cellFlux'][startState:]).max(), 1e-10)
laterGrouping = coarse.squareTilings(3)[2]['labels']
laterRelay = relay.reducedRelay(laterGrouping, start=startState)
laterGap = laterRelay['curve'][steps]
check('grouping from a later start: flux sums to its own final gap',
      max(np.abs(laterRelay['flux'][state].sum(1) - laterGap).max() for state in range(startState, steps + 1)) / np.abs(laterGap).max(), 1e-10)

print('\nALL PASS' if not failures else f'\nFAILED: {failures}')
raise SystemExit(1 if failures else 0)
