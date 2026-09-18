"""How much each boundary hold changes the pattern, measured against the same tissue left unclamped
(PolyPatterning_Sim.md, Section 12).

The clamp-free reference (simulateBoundaryDialLandscape11x11.py --experiment freeRun) keeps the whole time course, so
it can be read over exactly the iterations a sweep uses. A sweep's late pattern at each dial is compared with the free
pattern averaged over the same late window (the same absolute iterations: 2000-2999 for the fixed readout, 2201-3200
or 2400-3399 for the 301- and 500-iteration holds with the readout aligned to release), and its state at the last held
iteration with the free state at that iteration. Differences are RMS over all 121 cells, over the 81 interior cells and
over the 40 ring cells. Also reported: how much the free pattern itself changes between those late windows, and the
free ring's own G_pol / G_ref over each hold, the value a clamp would have to match to leave the ring unchanged.

Writes data/boundaryHoldInfluence<checkpoint>.json for plotBoundaryDial11x11.py.
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--sweeps', type=str, default='100,301,500,301aligned,500aligned')
parser.add_argument('--windowIterations', type=int, default=1000)
parser.add_argument('--neutralMilliVolts', type=float, default=1.0, help='influence below which a dial counts as leaving no trace')
parser.add_argument('--regionLimits', type=str, default='0.802,1.3,1.44,1.62')
args = parser.parse_args()

ring, interior = boundary.boundaryRingCells, boundary.interiorCellIndices
windowLow, extendedLimit, flipZoneLow, flipZoneHigh = (float(value) for value in args.regionLimits.split(','))
free = np.load(f'data/boundaryFreeRun{args.referenceCheckpoint}.npz')
freeVmem, freeConductance = free['vmem'].astype(float), free['conductance'].astype(float)
referenceHoldIterations = int(boundary.loadCheckpoint(args.referenceCheckpoint)['clampParameters']['clampEndIter']) + 1


def rootMeanSquare(difference, cells=None):
    difference = difference if cells is None else difference[..., cells]
    return np.sqrt((difference ** 2).mean(-1))


def freeWindow(start):
    return freeVmem[start:start + args.windowIterations].mean(0)


summary = dict(windowIterations=args.windowIterations, neutralMilliVolts=args.neutralMilliVolts, sweeps={}, free={})
print("=== FREE RUN ===")
windowStarts = set()
for token in args.sweeps.split(','):
    hold = int(token.replace('aligned', ''))
    windowStarts.add(3000 - args.windowIterations + (hold - referenceHoldIterations if token.endswith('aligned') else 0))
windowStarts = sorted(windowStarts)
for start in windowStarts:
    pattern = freeWindow(start)
    summary['free'][f'window{start}'] = dict(pattern=pattern.round(2).tolist(), interiorDarkShare=float(np.mean(pattern[interior] < boundary.singleCellSaddleMilliVolts)),
                                            meanMilliVolts=float(pattern.mean()))
    print(f"   iterations {start}-{start + args.windowIterations - 1}: mean {pattern.mean():.1f} mV, "
          f"symmetric share {boundary.symmetricShare(pattern - pattern.mean()):.6f}")
summary['free']['windowChanges'] = {f'{first}-{second}': float(rootMeanSquare(freeWindow(first) - freeWindow(second)))
                                   for index, first in enumerate(windowStarts) for second in windowStarts[index + 1:]}
print("   change of the free pattern between those windows (mV RMS): " + ", ".join(f"{key} {value:.2f}" for key, value in summary['free']['windowChanges'].items()))
for hold in (100, 301, 500):
    summary['free'][f'atIteration{hold - 1}'] = dict(pattern=freeVmem[hold - 1].round(2).tolist(),
                                                   interiorDarkShare=float(np.mean(freeVmem[hold - 1][interior] < boundary.singleCellSaddleMilliVolts)),
                                                   ringConductanceDuringHold=float(freeConductance[:hold][:, ring].mean()),
                                                   ringConductanceAtEnd=float(freeConductance[hold - 1][ring].mean()))
    print(f"   iteration {hold - 1}: interior dark share {summary['free'][f'atIteration{hold - 1}']['interiorDarkShare']:.2f}; free ring G_pol/G_ref "
          f"{summary['free'][f'atIteration{hold - 1}']['ringConductanceDuringHold']:.3f} on average over the first {hold} iterations, "
          f"{summary['free'][f'atIteration{hold - 1}']['ringConductanceAtEnd']:.3f} at iteration {hold - 1}")

print("\n=== INFLUENCE OF EACH HOLD: sweep pattern minus free pattern over the same iterations ===")
for token in args.sweeps.split(','):
    hold, aligned = int(token.replace('aligned', '')), token.endswith('aligned')
    suffix = '' if hold == referenceHoldIterations else f"Hold{hold}{'Aligned' if aligned else ''}"
    path = f'data/boundaryDialSweep{args.referenceCheckpoint}{suffix}.npz'
    if not os.path.exists(path):
        print(f"(skipping {token}: {path} not found)")
        continue
    sweep = np.load(path)
    dials, late, held = sweep['dialLevel'], sweep['windowMeanVmem'], sweep['endOfHoldVmem']
    start = 3000 - args.windowIterations + (hold - referenceHoldIterations if aligned else 0)
    freeLate, freeHeld = freeWindow(start), freeVmem[hold - 1]
    influence = {name: rootMeanSquare(late - freeLate, cells) for name, cells in (('all', None), ('interior', interior), ('ring', ring))}
    heldInfluence = {name: rootMeanSquare(held - freeHeld, cells) for name, cells in (('all', None), ('interior', interior), ('ring', ring))}
    regions = {'all': np.ones(len(dials), bool), 'belowWindow': dials <= windowLow, 'belowExtended': dials <= extendedLimit,
               'belowFlipZone': dials < flipZoneLow, 'flipZone': (dials >= flipZoneLow) & (dials <= flipZoneHigh), 'aboveFlipZone': dials > flipZoneHigh}
    neutral = influence['all'] < args.neutralMilliVolts
    entry = dict(hold=hold, aligned=aligned, windowStart=start, dials=dials.round(3).tolist(),
                 influence={name: values.round(3).tolist() for name, values in influence.items()},
                 heldInfluence={name: values.round(3).tolist() for name, values in heldInfluence.items()},
                 regionMedians={name: float(np.median(influence['all'][members])) for name, members in regions.items()},
                 heldRegionMedians={name: float(np.median(heldInfluence['interior'][members])) for name, members in regions.items()},
                 minimum=dict(dial=float(dials[influence['all'].argmin()]), milliVolts=float(influence['all'].min())),
                 maximum=dict(dial=float(dials[influence['all'].argmax()]), milliVolts=float(influence['all'].max())),
                 neutralDials=[float(value) for value in dials[neutral]])
    summary['sweeps'][f'hold{token}'] = entry
    print(f"\n-- hold {token} (late window {start}-{start + args.windowIterations - 1}): median {np.median(influence['all']):.2f} mV "
          f"(interior {np.median(influence['interior']):.2f}, ring {np.median(influence['ring']):.2f}); smallest {influence['all'].min():.2f} at "
          f"{dials[influence['all'].argmin()]:.2f}, largest {influence['all'].max():.1f} at {dials[influence['all'].argmax()]:.2f}")
    print("   medians by region: " + ", ".join(f"{name} {value:.2f}" for name, value in entry['regionMedians'].items()))
    print(f"   dials within {args.neutralMilliVolts} mV of the free pattern: {neutral.sum()}"
          + (f" ({dials[neutral].min():.2f}-{dials[neutral].max():.2f})" if neutral.any() else ''))
    print(f"   at the last held iteration ({hold - 1}): interior median {np.median(heldInfluence['interior']):.2f} mV, ring median "
          f"{np.median(heldInfluence['ring']):.2f} mV; interior below dial {extendedLimit}: {np.median(heldInfluence['interior'][dials <= extendedLimit]):.2f}")

outputPath = f'data/boundaryHoldInfluence{args.referenceCheckpoint}.json'
json.dump(summary, open(outputPath, 'w'), separators=(',', ':'))
print(f"\nwrote {outputPath}")
