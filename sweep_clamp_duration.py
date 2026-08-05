"""
Is a 100-iteration clamp long enough to write the 30x30 boundary?

The 100-step figure comes from the 11x11 protocol, where the boundary is 40 cells and 44
field-dome points. At 30x30 it is 116 cells and 120 dome points -- nearly three times the
write surface driven for the same duration. If 100 steps only partially drives those points,
the ensemble would sample a fraction of the available address space, which is exactly what
the dimensionality result is meant to measure.

Sweeps clamp duration at two clamp seeds and reports, per duration:
  - boundary-only std(G_pol)/G_ref, and the fraction of boundary cells driven to saturation
    (the quantity that says how much was written)
  - interior decorrelation between the two clamps (the quantity that says how much of what
    was written is distinguishable downstream)
  - when the post-clamp transient ends (so the readout window can be sized across clamps)

Free-evolution length is held fixed at --freeIters rather than total length, so duration is
the only variable. Saturation at or before 100 means the current protocol is sufficient; a
still-climbing curve means a longer clamp buys address space for free.

  python sweep_clamp_duration.py
"""

import argparse
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--latticeDim',      type=int, default=30)
parser.add_argument('--fieldScreenSize', type=int, default=5)
parser.add_argument('--clampIters',      type=str, default='[25,50,100,200,400]')
parser.add_argument('--clampSeeds',      type=str, default='[7,99]')
parser.add_argument('--freeIters',       type=int, default=2400,
                    help='free-evolution steps after the clamp, held fixed across durations')
parser.add_argument('--readoutIters',    type=int, default=200, help='steps averaged for the readout')
parser.add_argument('--sourceDat',       type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--outputPrefix',    type=str, default='data/clampDurationSweep')
args = parser.parse_args()

clampIterList = eval(args.clampIters)
clampSeeds = eval(args.clampSeeds)
latticeDim = args.latticeDim
numCells = latticeDim * latticeDim
utils = utilities.utilities()


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


referenceModel = model(buildParameters(), 1)
numFieldGridPoints = referenceModel.electricNetwork.numFieldGridPoints
boundaryMask = np.zeros(numCells, dtype=bool)
boundaryMask[utils.computeDomeIndices(referenceModel.electricNetwork, mode='tissue')] = True
interiorMask = ~boundaryMask
G_ref = referenceModel.electricNetwork.G_ref
del referenceModel


def run(clampIters, clampSeed):
    parameters = buildParameters(numFieldGridPoints)
    modelInstance = model(parameters, 1)
    modelInstance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    circuit = modelInstance.electricNetwork

    clampParameters = None
    if clampSeed is not None:
        torch.manual_seed(clampSeed)
        leftHalf = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
        mirrored = utils.computeSymmetricalIndices(circuit, leftHalf, mode='field', symmetry='twofold')
        allIndices = np.concatenate((leftHalf, mirrored))
        _, uniqueIdx = np.unique(allIndices, return_index=True)
        points = allIndices[uniqueIdx]
        # The oscillation is defined over a fixed phase interval, so a longer clamp samples the
        # same waveform more finely rather than extending it into new phase territory.
        timeIndices = torch.linspace(0, 0.5, clampIters + 1).view(-1, 1)
        frequencies = torch.rand(len(leftHalf), dtype=torch.double) * 900.0 + 100.0
        phases = torch.rand(len(leftHalf), dtype=torch.double) * 2 * torch.pi
        amplitudes = torch.rand(len(leftHalf), dtype=torch.double) * 2.0 - 1.0
        values = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                  * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
        clampParameters = {'clampMode': 'fieldDomeTwoFoldSymmetry',
                           'clampIndices': (np.zeros(len(points), dtype=int), points),
                           'clampValues': values, 'clampStartIter': 0, 'clampEndIter': clampIters}

    numSimIters = clampIters + args.freeIters
    modelInstance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                           clampParameters=clampParameters, perturbation=None,
                           numSimIters=numSimIters, storeVariables=('Vmem', 'Gpol'))

    gpol = modelInstance.timeseriesGpol[clampIters + 1][0, :, 0].detach().numpy() / G_ref
    Vmem = modelInstance.timeseriesVmem[:, 0, :, 0].detach().numpy() * 1000.0
    readout = Vmem[-args.readoutIters:].mean(axis=0)
    changeRate = np.abs(np.diff(Vmem, axis=0)).mean(axis=1)

    # Transient end: last time the rate exceeds twice the plateau median. The plateau is taken
    # from the final 500 steps, which the 5000-iteration run showed is fluctuation, not decay.
    plateauMedian = np.median(changeRate[-500:])
    aboveThreshold = np.where(changeRate > 2 * plateauMedian)[0]
    transientEnd = int(aboveThreshold[-1]) if len(aboveThreshold) else clampIters
    peakIter = int(np.argmax(changeRate[clampIters:])) + clampIters

    # A boundary cell counts as driven if G_pol left the neighbourhood of its uniform start (1.0)
    # and approached either absorbing end of [0, 2].
    boundaryGpol = gpol[boundaryMask]
    saturated = float(np.mean((boundaryGpol < 0.1) | (boundaryGpol > 1.9)))
    return {'boundaryStd': float(boundaryGpol.std()), 'fullStd': float(gpol.std()),
            'saturatedFraction': saturated, 'readout': readout,
            'transientEnd': transientEnd, 'peakIter': peakIter}


print(f"{latticeDim}x{latticeDim} fieldScreenSize={args.fieldScreenSize} | clamp durations {clampIterList} | seeds {clampSeeds}")
print(f"free evolution fixed at {args.freeIters} steps, readout = mean of last {args.readoutIters}\n")

results = {}
for clampIters in clampIterList:
    for clampSeed in clampSeeds:
        results[(clampIters, clampSeed)] = run(clampIters, clampSeed)
        r = results[(clampIters, clampSeed)]
        print(f"  clamp={clampIters:>3} seed={clampSeed:<3} | boundary std(G_pol)/G_ref={r['boundaryStd']:.4f} "
              f"| saturated={r['saturatedFraction']*100:>5.1f}% | rate peak t={r['peakIter']:<5} "
              f"transient ends t={r['transientEnd']}")
unclamped = run(100, None)
print(f"  {'unclamped':<14} | boundary std(G_pol)/G_ref={unclamped['boundaryStd']:.4f} "
      f"| saturated={unclamped['saturatedFraction']*100:>5.1f}%")

print(f"\n{'clamp':>6} {'boundary std':>14} {'saturated %':>13} {'interior corr':>15} {'transient end':>15}")
print('-' * 68)
summary = {'boundaryStd': [], 'saturatedFraction': [], 'interiorCorr': [], 'transientEnd': []}
for clampIters in clampIterList:
    runs = [results[(clampIters, s)] for s in clampSeeds]
    boundaryStd = np.mean([r['boundaryStd'] for r in runs])
    saturated = np.mean([r['saturatedFraction'] for r in runs])
    transientEnd = max(r['transientEnd'] for r in runs)
    interiorCorr = np.corrcoef(runs[0]['readout'][interiorMask], runs[1]['readout'][interiorMask])[0, 1]
    for key, value in [('boundaryStd', boundaryStd), ('saturatedFraction', saturated),
                       ('interiorCorr', interiorCorr), ('transientEnd', transientEnd)]:
        summary[key].append(value)
    print(f"{clampIters:>6} {boundaryStd:>14.4f} {saturated*100:>12.1f}% {interiorCorr:>15.3f} {transientEnd:>15}")

print(f"\nunclamped boundary std = {unclamped['boundaryStd']:.4f} (the floor to beat)")
print(f"latest transient end across all runs = {max(summary['transientEnd'])} "
      f"-> numSimIters must exceed this plus the {args.readoutIters}-step readout window")

torch.save({'results': {f'{k[0]}_{k[1]}': {kk: vv for kk, vv in v.items() if kk != 'readout'}
                        for k, v in results.items()},
            'summary': summary, 'clampIters': clampIterList, 'args': vars(args)},
           f'{args.outputPrefix}.dat')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, key, label, colour in [
        (axes[0], 'boundaryStd', 'boundary std(G_pol) / G_ref', '#3b6ea5'),
        (axes[1], 'saturatedFraction', 'fraction of boundary cells saturated', '#3b6ea5'),
        (axes[2], 'interiorCorr', 'interior corr between the two clamps', '#b5484d')]:
    values = np.array(summary[key])
    if key == 'saturatedFraction':
        values = values * 100
    ax.plot(clampIterList, values, marker='o', linewidth=2, markersize=8, color=colour)
    ax.axvline(100, color='0.6', linestyle='--', linewidth=1)
    ax.annotate('current protocol', xy=(100, ax.get_ylim()[0]), fontsize=8, color='0.4',
                rotation=90, va='bottom', xytext=(105, ax.get_ylim()[0]))
    ax.set_xscale('log')
    ax.set_xticks(clampIterList)
    ax.set_xticklabels([str(c) for c in clampIterList])
    ax.set_xlabel('clamp iterations', fontsize=9, color='0.25')
    ax.set_ylabel(label, fontsize=9, color='0.25')
    ax.tick_params(labelsize=8, colors='0.35')
    ax.grid(True, alpha=0.18, linewidth=0.6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
axes[0].axhline(unclamped['boundaryStd'], color='0.55', linestyle=':', linewidth=1.5)
axes[0].annotate('unclamped floor', xy=(clampIterList[0], unclamped['boundaryStd']),
                 fontsize=8, color='0.4', va='bottom')
axes[2].set_ylim(-0.1, 1.0)
fig.suptitle(f'Clamp duration sweep ({latticeDim}x{latticeDim}, field action range {args.fieldScreenSize})',
             fontsize=11, color='0.2')
plt.tight_layout()
plt.savefig(f'{args.outputPrefix}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {args.outputPrefix}.dat, {args.outputPrefix}.png")
