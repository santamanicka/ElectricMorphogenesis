"""Clamp with the trained signal for its stored duration, then release, fixed weight=700, across
fieldScreenSize, following the full trajectory.

Same protocol as analyzeReleasedByWeight.py, with fieldScreenSize swept instead of weight, weight
held fixed at 700 throughout.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat', default='./data/bestModelParameters_fieldVector_30x30_616.dat')
parser.add_argument('--weight', type=float, default=700)
parser.add_argument('--screens', type=int, nargs='+', default=[4, 6, 8, 10])
parser.add_argument('--numSimIters', type=int, default=2500)
parser.add_argument('--snapshots', type=int, nargs='+',
                     default=[0, 100, 200, 500, 1000, 1750, 2499])
parser.add_argument('--output', default='./figures/releasedByScreen_weight700.png')
args = parser.parse_args()

torch.set_grad_enabled(False)

reference = torch.load(args.sourceDat, weights_only=False)
numRows, numCols = reference['latticeDims']
numCells = numRows * numCols
boundary = np.zeros((numRows, numCols), bool)
boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = True
interiorMask = ~boundary.reshape(-1)
print(f"  weight fixed at {args.weight:.0f}, source {args.sourceDat.split('/')[-1]}, "
      f"clampEndIter {int(reference['clampParameters']['clampEndIter'])}")


def run(screen):
    parameters = torch.load(args.sourceDat, weights_only=False)
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['fieldParameters'] = dict(parameters['fieldParameters'])
    parameters['fieldParameters']['fieldTransductionWeight'] = torch.DoubleTensor([args.weight])
    parameters['fieldParameters']['fieldScreenSize'] = screen
    values = parameters['simParameters']['initialValues']
    if 'ligandConc' not in values:
        values['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
    instance = model(parameters, 1)
    instance.simulate(numSimIters=args.numSimIters, fieldModulation=True, perturbation=None,
                       clampParameters=parameters['clampParameters'], storeVariables=['Vmem'])
    return np.stack([v[0, :, 0].detach().numpy() for v in instance.timeseriesVmem])


results = {}
print(f"\n  {'screen':>7s} {'final spatial std (mV)':>23s} {'final rate (mV/iter)':>21s} {'peak rate':>10s}")
for screen in args.screens:
    series = run(screen)
    results[screen] = series
    rate = np.linalg.norm(np.diff(series[:, interiorMask], axis=0), axis=1) * 1000
    print(f"  {screen:7d} {series[-1, interiorMask].std()*1000:23.4f} {rate[-1]:21.3e} {rate.max():10.3e}")

figure = plt.figure(figsize=(2.0 * len(args.snapshots), 2.35 * len(args.screens) + 3.5))
grid = gridspec.GridSpec(len(args.screens) + 1, len(args.snapshots), figure=figure,
                          height_ratios=[1] * len(args.screens) + [1.6], hspace=0.5, wspace=0.08)
for rowIndex, screen in enumerate(args.screens):
    series = results[screen]
    interiorValues = series[:, interiorMask] * 1000
    span = max(np.abs(interiorValues - np.median(interiorValues)).max(), 1e-9)
    centre = np.median(interiorValues)
    for columnIndex, iteration in enumerate(args.snapshots):
        axis = figure.add_subplot(grid[rowIndex, columnIndex])
        axis.imshow(series[min(iteration, len(series) - 1)].reshape(numRows, numCols) * 1000,
                    cmap='RdBu_r', vmin=centre - span, vmax=centre + span)
        axis.set_xticks([]); axis.set_yticks([])
        if rowIndex == 0:
            axis.set_title(f'iter {iteration}', fontsize=9)
        if columnIndex == 0:
            axis.set_ylabel(f'screen {screen}', fontsize=9)

axisRate = figure.add_subplot(grid[len(args.screens), :])
for screen, series in results.items():
    rate = np.linalg.norm(np.diff(series[:, interiorMask], axis=0), axis=1) * 1000
    axisRate.plot(rate, label=f'screen {screen}')
axisRate.axvline(int(reference['clampParameters']['clampEndIter']), color='0.6', linestyle=':', lw=1)
axisRate.set_xlabel('iteration'); axisRate.set_ylabel('interior rate of change (mV/iter)')
axisRate.set_yscale('log'); axisRate.legend(fontsize=8)
figure.suptitle(f'30x30, weight={args.weight:.0f}, same trained clamp released after iteration 100: '
                 f'effect of screen size', fontsize=11)
figure.savefig(args.output, dpi=130, bbox_inches='tight')
print(f"\n  wrote {args.output}")
