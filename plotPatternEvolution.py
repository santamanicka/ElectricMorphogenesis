"""Follow one trained clamp from the pre-pattern state to the end of its horizon.

A single scored number says where the tissue ended up but not how it got there, and the horizon work
turned on exactly that: whether the pattern is still forming when the loss reads it, and whether the
structure it passes through is closer to the target than the structure it lands on. Frames are drawn
on one shared grayscale so brightness differences between them are real rather than per-frame
rescaling, and each carries its distance to the target so the picture and the number can be read
together.

The clamp is released at iteration 100, so everything after the second frame is the tissue on its own.
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
parser.add_argument('--parameterfile', default='./data/bestModelParameters_fieldVector_30x30_511.dat')
parser.add_argument('--every', type=int, default=200)
parser.add_argument('--output', default='./figures/patternEvolution.png')
parser.add_argument('--numSimIters', type=int, default=None,
                    help='run past the trained horizon; the clamp still ends where it was trained to')
args = parser.parse_args()

p = torch.load(args.parameterfile, weights_only=False)
dims = p['latticeDims']; rows, cols = dims[0], dims[1]; numCells = rows*cols
numSamples = p['simParameters']['numSamples']
initialValues = p['simParameters']['initialValues']
if 'ligandConc' not in initialValues:
    initialValues['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)
numSimIters = args.numSimIters or p['simParameters']['numSimIters']
p['latticePeriodicBoundaryGJ'] = False
p['ATPParameters'] = None
target = p['trainParameters']['targetVmem'].detach().numpy().reshape(rows, cols)*1000
loss = float(p['trainParameters']['bestLoss'])

wanted = list(range(0, numSimIters, args.every)) + [numSimIters-1]
instance = model(p, numSamples)
instance.setExperimentalConditions((initialValues, numSamples))
instance.simulate(clampParameters=p['clampParameters'], fieldModulation=True,
                  numSimIters=numSimIters, storeVariables=['Vmem'],
                  storeStride=args.every, storeIters=[numSimIters-1])
V = torch.stack(list(instance.timeseriesVmem)).reshape(-1, numCells).numpy()*1000
print(f"  {dims} lattice, horizon {numSimIters}, loss {loss:.4g}, {len(V)} frames stored")

frames = [(it, V[min(i, len(V)-1)].reshape(rows, cols)) for i, it in enumerate(wanted)]
vmin = min(target.min(), min(f.min() for _, f in frames))
vmax = max(target.max(), max(f.max() for _, f in frames))

ncol = 5
nrow = int(np.ceil((len(frames)+1)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.5*ncol, 2.75*nrow))
axes = np.atleast_2d(axes)
axes.ravel()[0].imshow(target, cmap='gray', vmin=vmin, vmax=vmax)
axes.ravel()[0].set_title('target', fontsize=9)
for k, (it, img) in enumerate(frames):
    ax = axes.ravel()[k+1]
    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    rms = np.sqrt(((img-target)**2).mean())
    note = ' (pre-pattern)' if it == 0 else (' (clamp off)' if it == args.every else '')
    ax.set_title(f'iter {it}{note}\n{rms:.1f} mV to target', fontsize=9)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.ravel()[len(frames)+1:]:
    ax.axis('off')
fig.suptitle(f'{rows}x{cols} best at horizon {numSimIters} (loss {loss:.4g}): pattern every {args.every} iterations',
             fontsize=12)
fig.tight_layout()
fig.savefig(args.output, dpi=150)
print(f"  wrote {args.output}")
