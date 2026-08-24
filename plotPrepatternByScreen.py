"""Render the G_pol pre-pattern state for the same trained clamp and fixed weight=700, across
fieldScreenSize.

Weight 700 was chosen because its pre-pattern depth profile (at screen 4, the clamp's native screen)
most closely matched what the working 11x11 model natively produces. This holds that weight fixed
and asks how the field's action range changes the picture -- the source clamp is unchanged
throughout, so any difference is attributable to screen alone.
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
parser.add_argument('--weight', type=float, default=700)
parser.add_argument('--screens', type=int, nargs='+', default=[4, 6, 8, 10])
parser.add_argument('--output', default='./figures/prepatternByScreen_weight700.png')
args = parser.parse_args()

torch.set_grad_enabled(False)


def prepattern(screen):
    parameters = torch.load(args.sourceDat, weights_only=False)
    parameters['fieldParameters'] = dict(parameters['fieldParameters'])
    parameters['fieldParameters']['fieldTransductionWeight'] = torch.DoubleTensor([args.weight])
    parameters['fieldParameters']['fieldScreenSize'] = screen
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
print(f"  weight fixed at {args.weight:.0f}, source {args.sourceDat.split('/')[-1]}")
for screen in args.screens:
    gpol, vmem, rows, cols = prepattern(screen)
    results[screen] = (gpol, vmem)
    print(f"  screen {screen:3d}: G_pol/G_ref range [{gpol.min():.3f}, {gpol.max():.3f}], "
          f"Vmem range [{vmem.min():.2f}, {vmem.max():.2f}] mV")

fig, axes = plt.subplots(2, len(args.screens), figsize=(3.6 * len(args.screens), 7.4))
for column, screen in enumerate(args.screens):
    gpol, vmem = results[screen]
    axTop = axes[0, column]
    im = axTop.imshow(gpol.reshape(rows, cols), cmap='RdBu_r', vmin=0, vmax=2)
    axTop.set_title(f'screen {screen}\nG_pol / G_ref', fontsize=10)
    axTop.set_xticks([]); axTop.set_yticks([])
    axBottom = axes[1, column]
    imV = axBottom.imshow(vmem.reshape(rows, cols), cmap='RdBu_r', vmin=vmem.min(), vmax=vmem.max())
    axBottom.set_title('Vmem (mV)', fontsize=10)
    axBottom.set_xticks([]); axBottom.set_yticks([])
    fig.colorbar(imV, ax=axBottom, fraction=0.046, pad=0.04)
fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04)
fig.suptitle(f'Pre-pattern state, same trained clamp, weight={args.weight:.0f}, across screen size',
             fontsize=12)
fig.tight_layout()
fig.savefig(args.output, dpi=140, bbox_inches='tight')
print(f"\n  wrote {args.output}")
