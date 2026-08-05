"""
Vmem patterns over the tail of a 30x30 run at field action range 5.

Shows the last frames of free evolution as a spatial small-multiple series, plus the
per-step rate of change over the whole run. The pair matters together: the patterns
alone cannot tell you whether what you are looking at is settled, and the Manicka &
Levin (2025) "patternable" property says it will not be -- these are patterning states
that keep evolving, not fixed points.

  python visualize_30x30_tail.py --numSimIters 2000 --numFrames 12
"""

import argparse
import copy
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--latticeDim',      type=int,   default=30)
parser.add_argument('--fieldScreenSize', type=int,   default=5)
parser.add_argument('--numSimIters',     type=int,   default=2000)
parser.add_argument('--clampIters',      type=int,   default=100,
                    help='absolute clamp duration, held fixed across lattice sizes so only '
                         'the free-evolution window changes with grid size')
parser.add_argument('--numFrames',       type=int,   default=12, help='tail frames to display')
parser.add_argument('--frameStride',     type=int,   default=20, help='iterations between frames')
parser.add_argument('--frameStart',      type=int,   default=None,
                    help='first displayed iteration; frames then run to the end of the simulation '
                         'at frameStride. Default (None) shows the last numFrames frames instead.')
parser.add_argument('--seed',            type=int,   default=7)
parser.add_argument('--vLimit',          type=float, default=None,
                    help='half-width of the colour scale in mV about V_th. Pin it to the same '
                         'value across figures that are meant to be compared; the default '
                         'auto-scales each figure to its own extremes.')
parser.add_argument('--noClamp',         action='store_true',
                    help='run with no clamp at all, to separate what the boundary writes from '
                         'what the tissue does on its own')
parser.add_argument('--sourceDat',       type=str,   default='data/StigmergicModelParameters.dat')
parser.add_argument('--outputPrefix',    type=str,   default='data/vmem_30x30_tail')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
utils = utilities.utilities()

latticeDim = args.latticeDim
numCells = latticeDim * latticeDim
preIdx = args.clampIters + 1


def buildParameters(numFieldGridPoints=None):
    parameters = copy.deepcopy(torch.load(args.sourceDat, weights_only=False))
    parameters['latticeDims'] = (latticeDim, latticeDim)
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['fieldParameters']['fieldScreenSize'] = args.fieldScreenSize
    if numFieldGridPoints is not None:
        initialValues = parameters['simParameters']['initialValues']
        initialValues['Vmem'] = torch.full((1, numCells, 1), -9.2e-3, dtype=torch.float64)
        initialValues['eV'] = torch.zeros((1, numFieldGridPoints, 1), dtype=torch.float64)
        initialValues['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
        initialValues['G_pol'] = {'cells': [[list(range(numCells))]],
                                  'values': [[torch.ones(numCells, dtype=torch.float64)]]}
        initialValues['G_dep'] = {'cells': [], 'values': torch.DoubleTensor([])}
    return parameters


# Two passes: the first only to learn the field grid size, which sets the eV init shape.
modelInstance = model(buildParameters(), 1)
parameters = buildParameters(modelInstance.electricNetwork.numFieldGridPoints)
modelInstance = model(parameters, 1)
modelInstance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))

# Random two-fold symmetric boundary field clamp (the fieldDomeTwoFoldSymmetry convention:
# the left half is parameterised independently, then mirrored to the right).
circuit = modelInstance.electricNetwork
leftHalfIndices = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
mirroredIndices = utils.computeSymmetricalIndices(circuit, leftHalfIndices, mode='field',
                                                  symmetry='twofold')
allIndices = np.concatenate((leftHalfIndices, mirroredIndices))
_, uniqueIdx = np.unique(allIndices, return_index=True)
clampPointIndices = allIndices[uniqueIdx]

numHalf = len(leftHalfIndices)
timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
frequencies = torch.rand(numHalf, dtype=torch.double) * 900.0 + 100.0
phases = torch.rand(numHalf, dtype=torch.double) * 2 * torch.pi
amplitudes = torch.rand(numHalf, dtype=torch.double) * 2.0 - 1.0
clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
               * torch.tile(amplitudes, (2,)))[:, uniqueIdx]

clampParameters = None
if not args.noClamp:
    clampParameters = {'clampMode': 'fieldDomeTwoFoldSymmetry',
                       'clampIndices': (np.zeros(len(clampPointIndices), dtype=int), clampPointIndices),
                       'clampValues': clampValues,
                       'clampStartIter': 0,
                       'clampEndIter': args.clampIters}

print(f"{latticeDim}x{latticeDim} lattice, field action range {args.fieldScreenSize}, "
      f"{'NO CLAMP' if args.noClamp else f'clamp 0-{args.clampIters}'}, "
      f"{args.numSimIters} iterations")
tStart = time.time()
modelInstance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                       clampParameters=clampParameters, perturbation=None,
                       numSimIters=args.numSimIters, storeVariables=('Vmem', 'Gpol'))
print(f"  ran in {time.time() - tStart:.0f}s")

Vmem = modelInstance.timeseriesVmem[:, 0, :, 0].detach().numpy() * 1000.0   # mV
gpolPre = modelInstance.timeseriesGpol[preIdx][0, :, 0].detach().numpy()
print(f"  G_pol pre-pattern (t={preIdx}): std/G_ref = {gpolPre.std() / 1e-9:.3f}")

# Per-step rate of change, mean over cells. This is what says whether the tail is settled.
changeRate = np.abs(np.diff(Vmem, axis=0)).mean(axis=1)

if args.frameStart is None:
    frameIters = [args.numSimIters - 1 - k * args.frameStride for k in range(args.numFrames)][::-1]
    frameIters = [i for i in frameIters if i >= 0]
else:
    frameIters = list(range(args.frameStart, args.numSimIters, args.frameStride))
    if frameIters[-1] != args.numSimIters - 1:
        frameIters.append(args.numSimIters - 1)
frames = Vmem[frameIters]

# One shared diverging scale across every panel. Per-frame normalisation would rescale each
# panel to its own extremes and hide exactly the drift the figure exists to show.
# The neutral midpoint is the model's threshold voltage V_th, so blue/red read as
# hyperpolarised/depolarised relative to it rather than relative to an arbitrary mean.
vCentre = modelInstance.electricNetwork.V_th * 1000.0
vLimit = args.vLimit if args.vLimit is not None else np.abs(frames - vCentre).max()
vMin, vMax = vCentre - vLimit, vCentre + vLimit
print(f"  colour scale: [{vMin:.1f}, {vMax:.1f}] mV (vLimit={vLimit:.1f} about V_th)")

numCols = 6
numRows = int(np.ceil(len(frames) / numCols))
fig = plt.figure(figsize=(2.1 * numCols + 1.2, 2.1 * numRows + 3.0))
gs = fig.add_gridspec(numRows + 1, numCols, height_ratios=[2.1] * numRows + [1.5], hspace=0.35)

for k, (iteration, frame) in enumerate(zip(frameIters, frames)):
    ax = fig.add_subplot(gs[k // numCols, k % numCols])
    im = ax.imshow(frame.reshape(latticeDim, latticeDim), cmap='RdBu_r', vmin=vMin, vmax=vMax, interpolation='nearest')
    ax.set_title(f't = {iteration}', fontsize=9, color='0.25')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('0.85')

colorbarAx = fig.add_axes([0.92, 0.42, 0.014, 0.44])
colorbar = fig.colorbar(im, cax=colorbarAx)
colorbar.set_label(f'Vmem (mV)   —   white = V_th = {vCentre:.0f} mV', fontsize=9, color='0.25')
colorbar.ax.tick_params(labelsize=8, colors='0.35')
colorbar.outline.set_edgecolor('0.85')

axRate = fig.add_subplot(gs[numRows, :])
axRate.plot(np.arange(1, args.numSimIters), changeRate, linewidth=1.5, color='#3b6ea5')
axRate.axvspan(frameIters[0], frameIters[-1], color='#3b6ea5', alpha=0.10, linewidth=0)
if not args.noClamp:
    axRate.axvline(args.clampIters, color='0.55', linewidth=1.0, linestyle='--')
    axRate.annotate('clamp ends', xy=(args.clampIters, axRate.get_ylim()[1]),
                    xytext=(args.clampIters + args.numSimIters * 0.012, axRate.get_ylim()[1] * 0.75),
                    fontsize=8, color='0.35')
axRate.annotate('frames above', xy=(frameIters[0], 0), fontsize=8, color='#3b6ea5',
                xytext=(frameIters[0], axRate.get_ylim()[1] * 0.45))
axRate.set_yscale('log')
axRate.set_xlabel('iteration', fontsize=9, color='0.25')
axRate.set_ylabel('mean |ΔVmem| per step (mV)', fontsize=9, color='0.25')
axRate.tick_params(labelsize=8, colors='0.35')
axRate.grid(True, alpha=0.18, linewidth=0.6)
for spine in ['top', 'right']:
    axRate.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    axRate.spines[spine].set_edgecolor('0.8')

if args.frameStart is None:
    windowLabel = f'over the last {(len(frames) - 1) * args.frameStride} steps'
else:
    startLabel = ('' if args.noClamp else ' (first free-evolution frame after the clamp)')
    windowLabel = (f'every {args.frameStride} steps from t = {args.frameStart}{startLabel} '
                   f'to t = {args.numSimIters - 1}')
conditionLabel = 'UNCLAMPED' if args.noClamp else f'clamp seed {args.seed}'
fig.suptitle(f'Vmem {windowLabel}  '
             f'({latticeDim}x{latticeDim}, field action range {args.fieldScreenSize}, {conditionLabel})',
             fontsize=11, color='0.2', y=0.97)
plt.savefig(f'{args.outputPrefix}.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  mean |dVmem|/step over displayed window: {changeRate[frameIters[0]:].mean():.3e} mV")
print(f"  Vmem range over displayed window: [{frames.min():.1f}, {frames.max():.1f}] mV")
print(f"  first-to-last displayed frame RMS difference: "
      f"{np.sqrt(((frames[-1] - frames[0]) ** 2).mean()):.2f} mV")
print(f"Saved {args.outputPrefix}.png")
