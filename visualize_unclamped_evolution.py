"""
Show what the tissue does on its own, with and without a boundary code.

The seed file starts every cell at the same voltage with uniform conductances and zero field,
so a run with no clamp and no noise stays uniform forever -- there is nothing to break the
symmetry. "Unclamped" therefore has two distinct meanings, and they answer different questions:

  autonomous  -- seed with faint noise and never clamp. Asks whether the uniform resting state
                 is unstable, i.e. whether the tissue patterns of its own accord.
  released    -- clamp briefly, then let go. Asks what becomes of a written pattern once the
                 code stops, which is what the ensemble protocol actually measures.

Comparing action ranges on both is what distinguishes a tissue that decodes a boundary signal
from one that merely uses it to trigger dynamics it was going to run regardless. A tissue whose
autonomous run stays flat has no competing dynamics of its own; whatever structure appears in
its released run must have come from the code.
"""

import argparse, ast, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utilities
from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat',        type=str,   default='data/StigmergicModelParameters_30x30.dat')
parser.add_argument('--fieldScreenSizes', type=str,   default='[2,5]')
parser.add_argument('--numSimIters',      type=int,   default=2500)
parser.add_argument('--clampIters',       type=int,   default=100)
parser.add_argument('--noiseStd',         type=float, default=1e-4, help='volts of symmetry-breaking noise')
parser.add_argument('--snapshots',        type=str,   default='[0,25,50,100,200,500,1000,1750,2499]')
parser.add_argument('--seed',             type=int,   default=11)
parser.add_argument('--outputPrefix',     type=str,   default='data/unclampedEvolution')
args = parser.parse_args()

screenSizes = ast.literal_eval(args.fieldScreenSizes)
snapshots   = ast.literal_eval(args.snapshots)
torch.set_grad_enabled(False)

reference = torch.load(args.sourceDat, weights_only=False)
numRows, numCols = reference['latticeDims']
numCells = numRows * numCols
reference['ATPParameters'] = None
reference['latticePeriodicBoundaryGJ'] = False
initial = reference['simParameters']['initialValues']
if 'ligandConc' not in initial:
    initial['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)

utils = utilities.utilities()
referenceModel = model(reference, 1)
circuit = referenceModel.electricNetwork
domeIndices = utils.computeDomeIndices(circuit, mode='tissue')
boundaryMask = np.zeros(numCells, dtype=bool); boundaryMask[domeIndices] = True
interiorMask = ~boundaryMask
leftHalf = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
mirrored = utils.computeSymmetricalIndices(circuit, leftHalf, mode='field', symmetry='twofold')
allIndices = np.concatenate((leftHalf, mirrored))
_, uniqueIdx = np.unique(allIndices, return_index=True)
clampPointIndices = allIndices[uniqueIdx]
clampIndices = (np.zeros(len(clampPointIndices), dtype=int), clampPointIndices)
numHalf = len(leftHalf)
timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
del referenceModel, circuit; gc.collect()

generator = torch.Generator().manual_seed(args.seed)
frequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * 900 + 100
phases      = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
amplitudes  = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 - 1

def run(screenSize, condition):
    parameters = torch.load(args.sourceDat, weights_only=False)
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['fieldParameters']['fieldScreenSize'] = screenSize
    values = parameters['simParameters']['initialValues']
    if 'ligandConc' not in values:
        values['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
    clampParameters = None
    if condition == 'autonomous':
        noise = torch.tensor(np.random.default_rng(args.seed).standard_normal((1, numCells, 1)) * args.noiseStd)
        values['Vmem'] = values['Vmem'] + noise
    else:
        clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                       * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
        clampParameters = {'clampMode': 'fieldDomeTwoFoldSymmetry', 'clampIndices': clampIndices,
                           'clampValues': clampValues, 'clampStartIter': 0,
                           'clampEndIter': args.clampIters}
    instance = model(parameters, 1)
    instance.simulate(numSimIters=args.numSimIters, fieldModulation=True, perturbation=None,
                      clampParameters=clampParameters, storeVariables=['Vmem'])
    series = np.stack([v[0, :, 0].detach().numpy() for v in instance.timeseriesVmem])
    del instance; gc.collect()
    return series

conditions = ['autonomous', 'released']
results = {}
for screenSize in screenSizes:
    for condition in conditions:
        series = run(screenSize, condition)
        results[(screenSize, condition)] = series
        rate = np.linalg.norm(np.diff(series[:, interiorMask], axis=0), axis=1)
        print(f"  range {screenSize:2d} {condition:>11}: final spatial std "
              f"{series[-1, interiorMask].std()*1000:7.3f} mV, "
              f"final rate of change {rate[-1]*1000:.3e} mV/iter, "
              f"peak {rate.max()*1000:.3e}")

# ── Figure ───────────────────────────────────────────────────────────────────
numRowsFigure = len(screenSizes) * len(conditions)
figure = plt.figure(figsize=(2.0 * len(snapshots), 2.35 * numRowsFigure + 5.0))
grid = gridspec.GridSpec(numRowsFigure + 2, len(snapshots), figure=figure,
                         height_ratios=[1] * numRowsFigure + [1.5, 1.5], hspace=0.55, wspace=0.08)
rowIndex = 0
for screenSize in screenSizes:
    for condition in conditions:
        series = results[(screenSize, condition)]
        interiorValues = series[:, interiorMask] * 1000
        span = max(np.abs(interiorValues - np.median(interiorValues)).max(), 1e-9)
        centre = np.median(interiorValues)
        for columnIndex, iteration in enumerate(snapshots):
            axis = figure.add_subplot(grid[rowIndex, columnIndex])
            axis.imshow(series[min(iteration, len(series) - 1)].reshape(numRows, numCols) * 1000,
                        cmap='RdBu_r', vmin=centre - span, vmax=centre + span)
            axis.set_xticks([]); axis.set_yticks([])
            if rowIndex == 0:
                axis.set_title(f'iteration {iteration}', fontsize=9)
            if columnIndex == 0:
                axis.set_ylabel(f'range {screenSize}\n{condition}', fontsize=9)
        rowIndex += 1

colours = {2: 'steelblue', 5: 'crimson', 11: 'darkorange', 3: 'seagreen'}
styles  = {'autonomous': '--', 'released': '-'}
axisRate = figure.add_subplot(grid[numRowsFigure, :])
axisStd  = figure.add_subplot(grid[numRowsFigure + 1, :])
for (screenSize, condition), series in results.items():
    rate = np.linalg.norm(np.diff(series[:, interiorMask], axis=0), axis=1) * 1000
    spatialStd = series[:, interiorMask].std(axis=1) * 1000
    label = f'range {screenSize}, {condition}'
    axisRate.semilogy(np.maximum(rate, 1e-16), styles[condition], color=colours.get(screenSize, 'k'),
                      linewidth=1.5, label=label)
    axisStd.plot(spatialStd, styles[condition], color=colours.get(screenSize, 'k'),
                 linewidth=1.5, label=label)
axisRate.axvline(args.clampIters, color='0.5', linestyle=':', linewidth=1.2)
axisRate.annotate('clamp released', xy=(args.clampIters, axisRate.get_ylim()[1]),
                  xytext=(args.clampIters + 40, axisRate.get_ylim()[1] * 0.02), fontsize=9, color='0.4')
axisRate.set_ylabel('rate of change\n(mV per iteration)', fontsize=10)
axisRate.set_title('A tissue that settles goes flat on this axis; one with dynamics of its own does not',
                   fontsize=10)
axisRate.legend(fontsize=8, ncol=2); axisRate.set_xlabel('iteration', fontsize=10)
axisStd.axvline(args.clampIters, color='0.5', linestyle=':', linewidth=1.2)
axisStd.set_ylabel('spatial variation\nacross interior (mV)', fontsize=10)
axisStd.set_xlabel('iteration', fontsize=10); axisStd.legend(fontsize=8, ncol=2)
plt.savefig(f'{args.outputPrefix}.png', dpi=150, bbox_inches='tight')
np.savez_compressed(f'{args.outputPrefix}.npz',
                    **{f'screen{s}_{c}': results[(s, c)] for s in screenSizes for c in conditions})
print(f"Saved {args.outputPrefix}.png")
