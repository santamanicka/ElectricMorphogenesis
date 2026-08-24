"""Render the G_pol pre-pattern state (t = clampEndIter+1) for the same trained clamp, across
fieldTransductionWeight 1000 / 800 / 700.

This is the 2D counterpart to the depth-resolved g-curves from the weight sweep: those numbers
average each depth shell into a single value, which hides the actual spatial arrangement within a
shell. Plotting the full map shows whether "committed" vs "still open" territory forms a clean ring
around the boundary (as the depth-shell averages implied) or something less regular.

One colour scale (centred at G_pol/G_ref = 1, the uniform starting value) is shared across all three
panels so darker/lighter directly reflects how far a cell has moved, comparably at every weight.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat', default='./data/bestModelParameters_fieldVector_30x30_616.dat')
parser.add_argument('--weights', type=float, nargs='+', default=[1000, 800, 700])
parser.add_argument('--output', default='./figures/prepatternByWeight.png')
args = parser.parse_args()

torch.set_grad_enabled(False)


def prepattern(weight):
    parameters = torch.load(args.sourceDat, weights_only=False)
    parameters['fieldParameters'] = dict(parameters['fieldParameters'])
    parameters['fieldParameters']['fieldTransductionWeight'] = torch.DoubleTensor([weight])
    rows, cols = parameters['latticeDims']
    numCells = rows * cols
    numSamples = parameters['simParameters']['numSamples']
    initialValues = parameters['simParameters']['initialValues']
    if 'ligandConc' not in initialValues:
        initialValues['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['ATPParameters'] = None
    clampParameters = dict(parameters['clampParameters'])
    clampEnd = int(clampParameters['clampEndIter'])

    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValues, numSamples))
    circuit = instance.electricNetwork
    instance.simulate(clampParameters=clampParameters, fieldModulation=True,
                       numSimIters=clampEnd + 1, storeVariables=['Vmem'])
    gpol = (circuit.G_pol.detach().clone().reshape(-1) / circuit.G_ref).numpy()
    vmem = circuit.Vmem.detach().clone().reshape(-1).numpy() * 1000
    return gpol, vmem, rows, cols


results = {}
for weight in args.weights:
    gpol, vmem, rows, cols = prepattern(weight)
    results[weight] = (gpol, vmem)
    print(f"  weight {weight:5.0f}: G_pol/G_ref range [{gpol.min():.3f}, {gpol.max():.3f}], "
          f"Vmem range [{vmem.min():.2f}, {vmem.max():.2f}] mV")

fig, axes = plt.subplots(2, len(args.weights), figsize=(3.6 * len(args.weights), 7.4))
for column, weight in enumerate(args.weights):
    gpol, vmem = results[weight]
    axTop = axes[0, column]
    im = axTop.imshow(gpol.reshape(rows, cols), cmap='RdBu_r', vmin=0, vmax=2)
    axTop.set_title(f'weight {weight:.0f}\nG_pol / G_ref', fontsize=10)
    axTop.set_xticks([]); axTop.set_yticks([])
    axBottom = axes[1, column]
    imV = axBottom.imshow(vmem.reshape(rows, cols), cmap='RdBu_r', vmin=vmem.min(), vmax=vmem.max())
    axBottom.set_title('Vmem (mV)', fontsize=10)
    axBottom.set_xticks([]); axBottom.set_yticks([])
    fig.colorbar(imV, ax=axBottom, fraction=0.046, pad=0.04)
fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04)
fig.suptitle('Pre-pattern state (t = clampEndIter+1), same trained clamp, across transduction weight',
             fontsize=12)
fig.tight_layout()
fig.savefig(args.output, dpi=140, bbox_inches='tight')
print(f"\n  wrote {args.output}")
