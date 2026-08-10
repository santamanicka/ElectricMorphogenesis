"""Compare how the pattern develops at two field screen sizes under the correlation objective.

The screen sets how wide a patch of extracellular field each cell reads, so it controls how far
coordination can reach. A target defined over the whole tissue needs coupling at that range, which is
the reason to expect the wider screen to do better, and the runs so far do: every screen 10 restart
scores better than all but one screen 4 restart, and none of them is anti-correlated with the target
where half the screen 4 runs still are.

Frames share one grayscale across both rows so the two are directly comparable, and each carries its
correlation with the target rather than a distance in millivolts, since correlation is what these
runs were trained on. Both models are still training, so this is a snapshot rather than a result.
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+',
                    default=['data/bestModelParameters_fieldVector_30x30_904.dat',
                             'data/bestModelParameters_fieldVector_30x30_954.dat'])
parser.add_argument('--frames', type=int, nargs='+', default=[0, 500, 1000, 1500, 2000, 2499])
parser.add_argument('--output', default='./figures/screenComparison.png')
args = parser.parse_args()

def correlation(observed, target):
    a = observed - observed.mean()
    b = target - target.mean()
    denominator = np.sqrt((a**2).sum()) * np.sqrt((b**2).sum())
    return float((a*b).sum()/denominator) if denominator > 0 else 0.0

rows = []
for path in args.models:
    parameters = torch.load(path, weights_only=False)
    latticeDims = parameters['latticeDims']
    numCells = latticeDims[0]*latticeDims[1]
    numSamples = parameters['simParameters']['numSamples']
    initialValues = parameters['simParameters']['initialValues']
    if 'ligandConc' not in initialValues:
        initialValues['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['ATPParameters'] = None
    numSimIters = parameters['simParameters']['numSimIters']
    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValues, numSamples))
    instance.simulate(clampParameters=parameters['clampParameters'], fieldModulation=True,
                      numSimIters=numSimIters, storeVariables=['Vmem'], storeStride=100,
                      storeIters=[numSimIters-1])
    trajectory = torch.stack(list(instance.timeseriesVmem)).reshape(-1, numCells).numpy()*1000
    target = parameters['trainParameters']['targetVmem'].detach().numpy().reshape(-1)*1000
    rows.append(dict(screen=parameters['fieldParameters']['fieldScreenSize'],
                     loss=float(parameters['trainParameters']['bestLoss']),
                     dims=latticeDims, trajectory=trajectory, target=target,
                     path=path.split('/')[-1]))
    print(f"  {rows[-1]['path']}: screen {rows[-1]['screen']}, 1-r {rows[-1]['loss']:.4f}")

allValues = np.concatenate([r['trajectory'].ravel() for r in rows] + [rows[0]['target']])
vmin, vmax = allValues.min(), allValues.max()

ncol = 1 + len(args.frames)
fig, axes = plt.subplots(len(rows), ncol, figsize=(2.35*ncol, 2.75*len(rows)))
axes = np.atleast_2d(axes)
for r, row in enumerate(rows):
    n = row['dims'][0]
    axes[r][0].imshow(row['target'].reshape(n, n), cmap='gray', vmin=vmin, vmax=vmax)
    axes[r][0].set_title(f"target\nscreen {row['screen']}  (1-r {row['loss']:.3f})", fontsize=9)
    for c, iteration in enumerate(args.frames):
        index = min(iteration//100, len(row['trajectory'])-1)
        image = row['trajectory'][index].reshape(n, n)
        ax = axes[r][c+1]
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"iter {iteration}\nr = {correlation(image.ravel(), row['target']):+.3f}", fontsize=9)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Correlation objective: a wider field screen changes what the tissue makes', fontsize=12)
fig.tight_layout()
fig.savefig(args.output, dpi=150)
print(f"  wrote {args.output}")
