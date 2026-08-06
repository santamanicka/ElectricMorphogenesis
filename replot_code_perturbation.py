"""
Replot the code-perturbation runs with per-row scaling and honest column labels.

Two flaws in the first rendering. The columns were labelled by the perturbation parameter, which
rounds to "0.0% change" for four of the six levels and in any case names the input to a saturating
map rather than the quantity of interest; they are now labelled by the code displacement actually
achieved, matching the axis of the plot below. And a single colour range shared across ranges
saturated range 2 into a solid block, since its pattern amplitude is roughly twice range 11's --
each row now takes its own scale, printed alongside, and each frame has its spatial mean removed
so that structure rather than offset is what shows.

The amplification panel is the sharpest of the three views: below 1 the tissue contracts code
differences, which is what makes small code noise survivable; above 1 it magnifies them, which is
what makes similar codes land on unrelated patterns.
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
instance = model(parameters, 1)
dome = utilities.utilities().computeDomeIndices(instance.electricNetwork, mode='tissue')
boundaryMask = np.zeros(numCells, bool); boundaryMask[dome] = True
interiorMask = ~boundaryMask; del instance; gc.collect()

archive = np.load('data/codePerturbation30x30.npz', allow_pickle=True)
results = {int(k.replace('screen', '')): archive[k].item() for k in archive.files}
screenSizes = sorted(results)
numColumns = len(next(iter(results.values()))['thumbnails']) + 1

figure = plt.figure(figsize=(2.0 * numColumns, 2.5 * len(screenSizes) + 7.5))
grid = gridspec.GridSpec(len(screenSizes) + 2, numColumns, figure=figure,
                         height_ratios=[1] * len(screenSizes) + [2.0, 2.0], hspace=0.5, wspace=0.08)
for rowIndex, screenSize in enumerate(screenSizes):
    entry = results[screenSize]
    frames = [entry['baseVmem']] + list(entry['thumbnails'])
    labels = ['unperturbed'] + [f'{c / entry["codeScale"]:.1%} of\nunrelated code' for c in entry['code']]
    centred = [(f - f[interiorMask].mean()) * 1000 for f in frames]
    span = max(np.percentile(np.abs(np.concatenate([c[interiorMask] for c in centred])), 99), 1e-9)
    for columnIndex, (frame, label) in enumerate(zip(centred, labels)):
        axis = figure.add_subplot(grid[rowIndex, columnIndex])
        colourmap = plt.get_cmap('RdBu_r').copy(); colourmap.set_bad('0.85')
        axis.imshow(np.where(boundaryMask, np.nan, frame).reshape(numRows, numCols),
                    cmap=colourmap, vmin=-span, vmax=span)
        axis.set_xticks([]); axis.set_yticks([]); axis.set_title(label, fontsize=8)
        if columnIndex == 0:
            axis.set_ylabel(f'action range {screenSize}\n(scale +-{span:.0f} mV)', fontsize=9)

colours = {2: 'steelblue', 5: 'crimson', 11: 'darkorange'}
axisCurve = figure.add_subplot(grid[len(screenSizes), :])
axisGain  = figure.add_subplot(grid[len(screenSizes) + 1, :])
for screenSize in screenSizes:
    entry = results[screenSize]
    codeFraction = entry['code'] / entry['codeScale']
    patternFraction = entry['pattern'] / entry['patternScale']
    axisCurve.plot(codeFraction, patternFraction, 'o-', color=colours.get(screenSize, 'k'),
                   linewidth=1.8, markersize=6, label=f'action range {screenSize}')
    axisGain.plot(codeFraction, patternFraction / codeFraction, 'o-', color=colours.get(screenSize, 'k'),
                  linewidth=1.8, markersize=6, label=f'action range {screenSize}')
axisCurve.axhline(1.0, color='0.35', linestyle='--', linewidth=1.2)
axisCurve.text(0.012, 1.03, 'unrelated pattern — all relation to the original destroyed',
               fontsize=9, color='0.3')
axisCurve.set_xscale('log'); axisCurve.set_ylim(0, 1.25)
axisCurve.set_xlabel('code displacement, relative to two unrelated codes', fontsize=10)
axisCurve.set_ylabel('pattern displacement,\nrelative to two unrelated patterns', fontsize=10)
axisCurve.legend(fontsize=9, loc='upper left'); axisCurve.set_title(
    'Range 5 has already moved a tenth of the way to an unrelated pattern before the code has '
    'meaningfully changed', fontsize=10)
axisGain.axhline(1.0, color='0.35', linestyle='--', linewidth=1.2)
axisGain.set_xscale('log'); axisGain.set_yscale('log')
axisGain.set_xlabel('code displacement, relative to two unrelated codes', fontsize=10)
axisGain.set_ylabel('amplification\n(pattern change / code change)', fontsize=10)
axisGain.legend(fontsize=9); axisGain.set_title(
    'Below the dashed line the tissue absorbs code differences; above it, magnifies them. '
    'Only range 2 starts below.', fontsize=10)
plt.savefig('data/codePerturbation30x30.png', dpi=150, bbox_inches='tight')
print("Saved data/codePerturbation30x30.png")
