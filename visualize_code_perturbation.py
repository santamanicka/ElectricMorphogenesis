"""
Show what a small change in the boundary code does to the interior pattern.

Addressability, measured as a rank statistic, says whether similar codes give similar patterns
but gives no sense of scale: an index of +0.153 is reliably above chance without indicating
whether the tissue is usefully instructable. This makes it concrete. One clamp is perturbed by
increasing amounts, and the resulting pattern displacement is measured against the distance
between two unrelated patterns from the same ensemble.

That normalisation is what makes the plot readable. A displacement of 1.0 means the perturbed
pattern is as different from the original as a pattern grown from a completely different clamp
-- the perturbation has destroyed all relation to the original. A decoder should rise gradually
and stay well below 1.0 for small perturbations; a tissue that merely amplifies should jump to
1.0 immediately, since there the code only selects which arbitrary state is reached.
"""

import argparse, ast, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist
import utilities
from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat',        type=str,   default='data/StigmergicModelParameters_30x30.dat')
parser.add_argument('--fieldScreenSizes', type=str,   default='[5,11]')
parser.add_argument('--perturbations',    type=str,   default='[0.005,0.01,0.02,0.05,0.1,0.3]',
                    help='fraction of each clamp parameter range by which to perturb')
parser.add_argument('--numReplicates',    type=int,   default=3)
parser.add_argument('--numSimIters',      type=int,   default=2500)
parser.add_argument('--clampIters',       type=int,   default=100)
parser.add_argument('--readoutIters',     type=int,   default=200)
parser.add_argument('--freqRange',        type=str,   default='(100.0,1000.0)')
parser.add_argument('--ampRange',         type=str,   default='(-1.0,1.0)')
parser.add_argument('--seed',             type=int,   default=11)
parser.add_argument('--ensembleDir',      type=str,   default='data/fieldRangeSweep',
                    help='merged ensembles supplying the unrelated-pattern reference scale')
parser.add_argument('--outputPrefix',     type=str,   default='data/codePerturbation')
args = parser.parse_args()

screenSizes  = ast.literal_eval(args.fieldScreenSizes)
perturbations = ast.literal_eval(args.perturbations)
minFreq, maxFreq = ast.literal_eval(args.freqRange)
minAmp,  maxAmp  = ast.literal_eval(args.ampRange)
torch.set_grad_enabled(False)

baseParameters = torch.load(args.sourceDat, weights_only=False)
numRows, numCols = baseParameters['latticeDims']
numCells = numRows * numCols
baseParameters['ATPParameters'] = None
baseParameters['latticePeriodicBoundaryGJ'] = False
initialValues = baseParameters['simParameters']['initialValues']
if 'ligandConc' not in initialValues:
    initialValues['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)

utils = utilities.utilities()
referenceModel = model(baseParameters, 1)
referenceCircuit = referenceModel.electricNetwork
domeIndices = utils.computeDomeIndices(referenceCircuit, mode='tissue')
boundaryMask = np.zeros(numCells, dtype=bool); boundaryMask[domeIndices] = True
interiorMask = ~boundaryMask

leftHalf = utils.computeDomeIndices(referenceCircuit, mode='field', region='leftHalf')
mirrored = utils.computeSymmetricalIndices(referenceCircuit, leftHalf, mode='field', symmetry='twofold')
allIndices = np.concatenate((leftHalf, mirrored))
_, uniqueIdx = np.unique(allIndices, return_index=True)
clampPointIndices = allIndices[uniqueIdx]
clampIndices = (np.zeros(len(clampPointIndices), dtype=int), clampPointIndices)
numHalf = len(leftHalf)
clampEndIter = args.clampIters
timeIndices = torch.linspace(0, 0.5, clampEndIter + 1).view(-1, 1)
del referenceModel, referenceCircuit; gc.collect()

generator = torch.Generator().manual_seed(args.seed)
baseFrequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * (maxFreq - minFreq) + minFreq
basePhases      = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
baseAmplitudes  = torch.rand(numHalf, generator=generator, dtype=torch.double) * (maxAmp - minAmp) + minAmp
rng = np.random.default_rng(args.seed)

def runClamp(frequencies, phases, amplitudes, screenSize):
    parameters = torch.load(args.sourceDat, weights_only=False)
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['fieldParameters']['fieldScreenSize'] = screenSize
    values = parameters['simParameters']['initialValues']
    if 'ligandConc' not in values:
        values['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
    clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                   * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    instance = model(parameters, 1)
    instance.simulate(numSimIters=args.numSimIters, fieldModulation=True,
                      clampParameters={'clampMode': 'fieldDomeTwoFoldSymmetry',
                                       'clampIndices': clampIndices, 'clampValues': clampValues,
                                       'clampStartIter': 0, 'clampEndIter': clampEndIter},
                      perturbation=None, storeVariables=['Vmem', 'Gpol'])
    gpol = instance.timeseriesGpol[clampEndIter + 1][0, :, 0].detach().numpy().copy()
    vmem = instance.timeseriesVmem[-args.readoutIters:, 0, :, 0].detach().numpy().mean(axis=0).copy()
    del instance; gc.collect()
    return gpol, vmem

results = {}
for screenSize in screenSizes:
    baseGpol, baseVmem = runClamp(baseFrequencies, basePhases, baseAmplitudes, screenSize)
    codeDistances, patternDistances, thumbnails = [], [], []
    for delta in perturbations:
        perturbedCode, perturbedPattern, firstVmem = [], [], None
        for _ in range(args.numReplicates):
            frequencies = baseFrequencies + delta * (maxFreq - minFreq) * torch.tensor(rng.standard_normal(numHalf))
            phases      = basePhases      + delta * 2 * np.pi          * torch.tensor(rng.standard_normal(numHalf))
            amplitudes  = baseAmplitudes  + delta * (maxAmp - minAmp)  * torch.tensor(rng.standard_normal(numHalf))
            gpol, vmem = runClamp(frequencies, phases, amplitudes, screenSize)
            perturbedCode.append(np.linalg.norm(gpol[boundaryMask] - baseGpol[boundaryMask]))
            perturbedPattern.append(np.linalg.norm(vmem[interiorMask] - baseVmem[interiorMask]))
            if firstVmem is None: firstVmem = vmem
        codeDistances.append(np.mean(perturbedCode))
        patternDistances.append(np.mean(perturbedPattern))
        thumbnails.append(firstVmem)
        print(f"  screen {screenSize}, delta {delta:.3f}: code {codeDistances[-1]:.3e}, "
              f"pattern {patternDistances[-1]*1000:.1f} mV")
    ensembleGpol = np.load(f'{args.ensembleDir}/screen{screenSize:02d}_gpol_prepatterns.npy')
    ensembleVmem = np.load(f'{args.ensembleDir}/screen{screenSize:02d}_vmem_final.npy')
    results[screenSize] = dict(baseVmem=baseVmem, thumbnails=thumbnails,
                               code=np.array(codeDistances), pattern=np.array(patternDistances),
                               codeScale=float(np.mean(pdist(ensembleGpol[:, boundaryMask]))),
                               patternScale=float(np.mean(pdist(ensembleVmem[:, interiorMask]))))
    print(f"  screen {screenSize}: unrelated-pattern reference distance "
          f"{results[screenSize]['patternScale']*1000:.1f} mV")

# ── Figure ───────────────────────────────────────────────────────────────────
numColumns = len(perturbations) + 1
figure = plt.figure(figsize=(2.0 * numColumns, 3.0 * len(screenSizes) + 4.2))
grid = gridspec.GridSpec(len(screenSizes) + 1, numColumns, figure=figure,
                         height_ratios=[1] * len(screenSizes) + [1.9], hspace=0.45, wspace=0.12)
centre = np.median([results[s]['baseVmem'] for s in screenSizes]) * 1000
for rowIndex, screenSize in enumerate(screenSizes):
    entry = results[screenSize]
    panels = [entry['baseVmem']] + entry['thumbnails']
    labels = ['unperturbed'] + [f'{d:.1%} change' for d in perturbations]
    for columnIndex, (pattern, label) in enumerate(zip(panels, labels)):
        axis = figure.add_subplot(grid[rowIndex, columnIndex])
        axis.imshow(pattern.reshape(numRows, numCols) * 1000, cmap='RdBu_r', vmin=centre - 25, vmax=centre + 25)
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_title(label, fontsize=8)
        if columnIndex == 0:
            axis.set_ylabel(f'action range {screenSize}', fontsize=10)

axis = figure.add_subplot(grid[len(screenSizes), :])
for screenSize, colour in zip(screenSizes, ['steelblue', 'crimson', 'darkorange', 'seagreen']):
    entry = results[screenSize]
    axis.plot(entry['code'] / entry['codeScale'], entry['pattern'] / entry['patternScale'],
              'o-', color=colour, linewidth=1.8, markersize=6, label=f'action range {screenSize}')
axis.axhline(1.0, color='0.35', linestyle='--', linewidth=1.2)
axis.annotate('unrelated pattern — all relation to the original destroyed',
              xy=(0.015, 1.0), xytext=(0.015, 1.06), fontsize=9, color='0.3')
axis.set_xscale('log')
axis.set_xlabel('code displacement, relative to two unrelated codes', fontsize=10)
axis.set_ylabel('pattern displacement,\nrelative to two unrelated patterns', fontsize=10)
axis.set_ylim(0, 1.25)
axis.legend(fontsize=9, loc='lower right')
axis.set_title('A decoder rises gradually and stays low for small code changes; '
               'an amplifier jumps to the dashed line at once', fontsize=10)
plt.savefig(f'{args.outputPrefix}.png', dpi=150, bbox_inches='tight')
np.savez(f'{args.outputPrefix}.npz', **{f'screen{s}': results[s] for s in screenSizes})
print(f"Saved {args.outputPrefix}.png")
