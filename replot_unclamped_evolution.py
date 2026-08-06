"""
Replot the unclamped runs as spatial pattern, with the mean removed.

Plotting raw voltage lets a drifting spatial mean read as pattern: a tissue whose cells all move
together looks vivid while being perfectly uniform. Subtracting each frame's own spatial mean
shows only what varies across the tissue, which is the quantity every claim here is about.
Boundary cells are masked -- they are clamped or adjacent to the clamp, and their excursions
otherwise dominate the colour scale that the interior has to share.
"""
import numpy as np, torch, gc, utilities
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from embryo import model
torch.set_grad_enabled(False)

parameters = torch.load('data/StigmergicModelParameters_30x30.dat', weights_only=False)
numRows, numCols = parameters['latticeDims']; numCells = numRows * numCols
parameters['ATPParameters'] = None; parameters['latticePeriodicBoundaryGJ'] = False
values = parameters['simParameters']['initialValues']
if 'ligandConc' not in values:
    values['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
utils = utilities.utilities(); instance = model(parameters, 1)
domeIndices = utils.computeDomeIndices(instance.electricNetwork, mode='tissue')
boundaryMask = np.zeros(numCells, dtype=bool); boundaryMask[domeIndices] = True
interiorMask = ~boundaryMask; del instance; gc.collect()

series = {}
for path in ['data/unclampedEvolution.npz', 'data/unclampedEvolutionMid.npz']:
    archive = np.load(path)
    for key in archive.files:
        screenSize = int(key.split('_')[0].replace('screen', ''))
        series[(screenSize, key.split('_')[1])] = archive[key]
screenSizes = sorted({s for s, _ in series})
conditions = ['autonomous', 'released']
snapshots = [0, 25, 50, 100, 200, 500, 1000, 1750, 2499]

# Shared colour scale: the 99th percentile of interior deviation across every run, so that a row
# with no pattern renders blank rather than being stretched to fill the map.
allDeviations = []
for value in series.values():
    centred = value - value[:, interiorMask].mean(axis=1, keepdims=True)
    allDeviations.append(np.abs(centred[:, interiorMask]))
span = float(np.percentile(np.concatenate([d.ravel() for d in allDeviations]), 99) * 1000)

figure = plt.figure(figsize=(1.85 * len(snapshots), 1.95 * len(screenSizes) * 2 + 4.4))
grid = gridspec.GridSpec(len(screenSizes) * 2 + 2, len(snapshots), figure=figure,
                         height_ratios=[1] * (len(screenSizes) * 2) + [1.5, 1.5],
                         hspace=0.35, wspace=0.06)
rowIndex = 0
for screenSize in screenSizes:
    for condition in conditions:
        value = series[(screenSize, condition)]
        for columnIndex, iteration in enumerate(snapshots):
            frame = value[min(iteration, len(value) - 1)].copy()
            frame = (frame - frame[interiorMask].mean()) * 1000
            display = np.where(boundaryMask, np.nan, frame).reshape(numRows, numCols)
            axis = figure.add_subplot(grid[rowIndex, columnIndex])
            colourmap = plt.get_cmap('RdBu_r').copy(); colourmap.set_bad('0.85')
            image = axis.imshow(display, cmap=colourmap, vmin=-span, vmax=span)
            axis.set_xticks([]); axis.set_yticks([])
            if rowIndex == 0: axis.set_title(f'iteration {iteration}', fontsize=9)
            if columnIndex == 0:
                axis.set_ylabel(f'range {screenSize}\n{condition}', fontsize=8.5)
            if columnIndex == len(snapshots) - 1:
                axis.text(1.06, 0.5, f'{value[min(iteration,len(value)-1)][interiorMask].std()*1000:.1f} mV',
                          transform=axis.transAxes, fontsize=8, va='center', color='0.35')
        rowIndex += 1

colours = {2: 'steelblue', 3: 'seagreen', 4: 'darkorange', 5: 'crimson'}
axisRate = figure.add_subplot(grid[len(screenSizes) * 2, :])
axisStd = figure.add_subplot(grid[len(screenSizes) * 2 + 1, :])
for (screenSize, condition), value in sorted(series.items()):
    style = '--' if condition == 'autonomous' else '-'
    rate = np.linalg.norm(np.diff(value[:, interiorMask], axis=0), axis=1) * 1000
    axisRate.semilogy(np.maximum(rate, 1e-16), style, color=colours[screenSize], linewidth=1.4,
                      label=f'range {screenSize}, {condition}')
    axisStd.plot(value[:, interiorMask].std(axis=1) * 1000, style, color=colours[screenSize],
                 linewidth=1.4, label=f'range {screenSize}, {condition}')
for axis in (axisRate, axisStd):
    axis.axvline(100, color='0.5', linestyle=':', linewidth=1.2)
    axis.set_xlabel('iteration', fontsize=10); axis.legend(fontsize=7.5, ncol=4)
axisRate.set_ylabel('rate of change\n(mV per iteration)', fontsize=10)
axisRate.set_title('Only range 2 settles. Every other range is still moving at iteration 2500, '
                   'clamped or not.', fontsize=10)
axisStd.set_ylabel('spatial variation\nacross interior (mV)', fontsize=10)
axisStd.set_title('Range 2 alone shows a gap between autonomous and released: '
                  'the pattern is the code’s doing', fontsize=10)
figure.colorbar(image, ax=[axisRate, axisStd], location='right', fraction=0.012,
                label='deviation from spatial mean (mV)')
plt.savefig('data/unclampedEvolutionCombined.png', dpi=150, bbox_inches='tight')
print("Saved data/unclampedEvolutionCombined.png")
