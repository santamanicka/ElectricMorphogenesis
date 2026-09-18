"""Map the boundary dial across its whole physical range, then test first-order gradients regime by regime
(PolyPatterning_Sim.md, Section 12).

The reference checkpoint's parameters, initial state, hold and run length are kept (the hold can be changed with
--holdIterations, and the run lengthened by the extra hold with --alignReadoutToRelease so that the late window keeps
its distance from release); only the G_pol / G_ref values held on the 40 outer-ring cells are replaced. All held values
lie within the physical range [0, 2]. theta is each ring cell's angle about the lattice centre, measured clockwise from
straight up. Each run records Vmem and G_pol / G_ref at the last held iteration, their per-cell means over the last
1000 iterations, and the per-cell Vmem standard deviation over that window.

  --experiment freeRun       no clamp at all: the reference run left alone for --freeRunIterations, with Vmem and
                             G_pol / G_ref kept at every iteration, as the no-hold baseline.
  --experiment dialSweep     the dial alone, code = DC on every ring cell, for DC = 0, step, 2 * step, ..., 2.
  --experiment regimePairs   first-order gradients code = DC + G cos(theta - phi), stratified by the single-cell
                             regime of the dial (boundaryCodeUtilities.singleCellBistableRange): --pairsPerRegime pairs
                             with DC below, inside and above the bistable window. DC is drawn from the sweep's own grid,
                             so each pair's G = 0 twin is the sweep run at that DC; G is log-uniform on
                             [0.01, min(0.5, DC, 2 - DC)] and phi uniform on [0, 2 pi). Also rotation grids at each of
                             --gridDials: G in {0.02, 0.05, 0.1, 0.2, 0.4}, capped to stay in range (duplicates dropped), x phi every 45 degrees.
"""
import argparse
import time

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices=['dialSweep', 'regimePairs', 'freeRun'], required=True)
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--dialStep', type=float, default=0.01)
parser.add_argument('--pairsPerRegime', type=int, default=150)
parser.add_argument('--gridDials', type=str, default='0.70,0.95,1.18,1.75')
parser.add_argument('--randomSeed', type=int, default=2)
parser.add_argument('--windowIterations', type=int, default=1000)
parser.add_argument('--holdIterations', type=int, default=None,
                    help="iterations the ring is held for (default: the reference checkpoint's own); a different value adds Hold<n> to the output names")
parser.add_argument('--freeRunIterations', type=int, default=3400, help='length of the clamp-free reference run')
parser.add_argument('--alignReadoutToRelease', action='store_true',
                    help="lengthen the run by the extra hold, so the late window starts as long after release as with the reference "
                         "checkpoint's own hold; adds Aligned to the output names")
args = parser.parse_args()

reference = boundary.loadCheckpoint(args.referenceCheckpoint)
referenceHoldIterations = int(reference['clampParameters']['clampEndIter']) + 1
holdIterations = args.holdIterations or referenceHoldIterations
extraHold = holdIterations - referenceHoldIterations
holdTag = '' if extraHold == 0 else f"Hold{holdIterations}{'Aligned' if args.alignReadoutToRelease else ''}"
numIterations = reference['simParameters']['numSimIters'] + (extraHold if args.alignReadoutToRelease else 0)
simulationParameters = dict(reference, simParameters=dict(reference['simParameters'], numSimIters=numIterations))
angles = boundary.ringAngles(boundary.boundaryRingCells)
sweepPath = f'data/boundaryDialSweep{args.referenceCheckpoint}{holdTag}.npz'
startTime = time.time()


def runRingCode(ringValues):
    """Replay with the ring held at ringValues; returns the recorded readouts."""
    if ringValues.min() < 0 or ringValues.max() > 2:
        raise ValueError(f"held values leave the physical range [0, 2]: {ringValues.min():.3f} to {ringValues.max():.3f}")
    readout = dict(windowMeanVmem=np.zeros(boundary.numCells), windowMeanGpol=np.zeros(boundary.numCells),
                   windowSquaredVmem=np.zeros(boundary.numCells))

    def onIteration(iteration, vmem, circuit):
        conductance = circuit.G_pol[0, :, 0].detach().numpy() / circuit.G_ref
        if iteration == holdIterations - 1:
            readout['endOfHoldVmem'], readout['endOfHoldGpol'] = vmem.copy(), conductance.copy()
        if iteration >= numIterations - args.windowIterations:
            readout['windowMeanVmem'] += vmem
            readout['windowSquaredVmem'] += vmem ** 2
            readout['windowMeanGpol'] += conductance
    boundary.replay(simulationParameters, boundary.ringClamp(reference, ringValues, holdIterations), onIteration, passCircuit=True)
    readout['windowMeanVmem'] /= args.windowIterations
    readout['windowMeanGpol'] /= args.windowIterations
    readout['windowStdVmem'] = np.sqrt(np.maximum(readout.pop('windowSquaredVmem') / args.windowIterations
                                                  - readout['windowMeanVmem'] ** 2, 0))
    return readout


readoutKeys = ('endOfHoldVmem', 'endOfHoldGpol', 'windowMeanVmem', 'windowMeanGpol', 'windowStdVmem')

if args.experiment == 'freeRun':
    # the reference with no boundary clamp at all, same parameters and initial state; the whole time course is kept so
    # the free pattern can be read over any window a sweep uses
    freeParameters = dict(reference, simParameters=dict(reference['simParameters'], numSimIters=args.freeRunIterations))
    vmemCourse, conductanceCourse = [], []

    def recordFree(iteration, vmem, circuit):
        vmemCourse.append(vmem.astype(np.float32))
        conductanceCourse.append((circuit.G_pol[0, :, 0].detach().numpy() / circuit.G_ref).astype(np.float32))
    boundary.replay(freeParameters, boundary.ringClamp(reference, np.zeros(len(boundary.boundaryRingCells)), 0), recordFree, passCircuit=True)
    freePath = f'data/boundaryFreeRun{args.referenceCheckpoint}.npz'
    np.savez_compressed(freePath, vmem=np.stack(vmemCourse), conductance=np.stack(conductanceCourse), referenceCheckpoint=args.referenceCheckpoint)
    print(f"wrote {freePath} ({len(vmemCourse)} iterations) in {time.time() - startTime:.0f}s")

elif args.experiment == 'dialSweep':
    dialLevels = np.round(np.arange(0, 2 + args.dialStep / 2, args.dialStep), 6)
    readouts = []
    for index, dialLevel in enumerate(dialLevels):
        readouts.append(runRingCode(np.full(len(boundary.boundaryRingCells), dialLevel)))
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(dialLevels)} dial levels  {time.time() - startTime:.0f}s", flush=True)
    np.savez_compressed(sweepPath, dialLevel=dialLevels, **{key: np.stack([readout[key] for readout in readouts]) for key in readoutKeys},
                        holdIterations=holdIterations, numIterations=numIterations, windowStart=numIterations - args.windowIterations,
                        referenceCheckpoint=args.referenceCheckpoint)
    print(f"wrote {sweepPath} in {time.time() - startTime:.0f}s")

else:
    sweep = np.load(sweepPath)
    sweepDials = sweep['dialLevel']
    lowerThreshold, upperThreshold = boundary.singleCellBistableRange
    regimes = {'monostableDepolarised': (sweepDials >= 0.01) & (sweepDials < lowerThreshold),
               'bistable': (sweepDials >= lowerThreshold) & (sweepDials <= upperThreshold),
               'monostableHyperpolarised': (sweepDials > upperThreshold) & (sweepDials <= 1.99)}
    randomGenerator = np.random.default_rng(args.randomSeed)
    pairs = {key: [] for key in ('regime', 'sweepIndex', 'dialLevel', 'gradientStrength', 'gradientDirection') + readoutKeys}
    for regime, eligible in regimes.items():
        candidates = np.where(eligible)[0]
        for _ in range(args.pairsPerRegime):
            sweepIndex = int(randomGenerator.choice(candidates))
            dialLevel = float(sweepDials[sweepIndex])
            ceiling = min(0.5, dialLevel, 2 - dialLevel)
            gradientStrength = float(np.exp(randomGenerator.uniform(np.log(0.01), np.log(ceiling))))
            gradientDirection = float(randomGenerator.uniform(0, 2 * np.pi))
            readout = runRingCode(dialLevel + gradientStrength * np.cos(angles - gradientDirection))
            for key, value in zip(('regime', 'sweepIndex', 'dialLevel', 'gradientStrength', 'gradientDirection'),
                                  (regime, sweepIndex, dialLevel, gradientStrength, gradientDirection)):
                pairs[key].append(value)
            for key in readoutKeys:
                pairs[key].append(readout[key])
        print(f"  {regime}: {args.pairsPerRegime} pairs done  {time.time() - startTime:.0f}s", flush=True)
    grids = {key: [] for key in ('gridDial', 'sweepIndex', 'gradientStrength', 'gradientDirection') + readoutKeys}
    for requestedDial in (float(value) for value in args.gridDials.split(',')):
        sweepIndex = int(np.argmin(np.abs(sweepDials - requestedDial)))
        dialLevel = float(sweepDials[sweepIndex])
        ceiling = np.floor(100 * min(0.5, dialLevel, 2 - dialLevel)) / 100
        for gradientStrength in sorted(set(min(requestedStrength, ceiling) for requestedStrength in (0.02, 0.05, 0.1, 0.2, 0.4))):
            for step in range(8):
                gradientDirection = step * np.pi / 4
                readout = runRingCode(dialLevel + gradientStrength * np.cos(angles - gradientDirection))
                for key, value in zip(('gridDial', 'sweepIndex', 'gradientStrength', 'gradientDirection'),
                                      (dialLevel, sweepIndex, gradientStrength, gradientDirection)):
                    grids[key].append(value)
                for key in readoutKeys:
                    grids[key].append(readout[key])
        print(f"  rotation grid at dial {dialLevel:.2f} done  {time.time() - startTime:.0f}s", flush=True)
    outputPath = f'data/boundaryRegimePairs{args.referenceCheckpoint}{holdTag}.npz'
    np.savez_compressed(outputPath, **{f'pair{key[0].upper()}{key[1:]}': np.array(value) for key, value in pairs.items()},
                        **{f'grid{key[0].upper()}{key[1:]}' if not key.startswith('grid') else key: np.array(value)
                           for key, value in grids.items()},
                        ringAngles=angles, randomSeed=args.randomSeed, referenceCheckpoint=args.referenceCheckpoint)
    print(f"wrote {outputPath} in {time.time() - startTime:.0f}s")
