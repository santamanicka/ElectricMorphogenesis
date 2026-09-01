"""Build a 30x30 initial condition from the 11x11 trained face model's clamp-release state.

The 11x11 Stigmergic face model clamps eV on its boundary ring for iterations 0-100 inclusive
(clampEndIter=100), then runs unclamped through iteration 1000. "The prepattern left by the clamp"
is the tissue state -- Vmem and the field-driven G_pol imprint -- at the instant the clamp lifts,
i.e. after exactly clampEndIter+1 = 101 iterations (iteration 100 is still clamped; 101 is the first
free step). That state is bilinearly upsampled (align_corners=True, so the boundary ring maps onto
the new boundary ring rather than bleeding inward) onto a 30x30 grid, to be released unclamped there.

Native field-transduction parameters (screen=4, weight=1000, gain=-1, bias=0.0005, timeConstant=10)
are carried over unchanged for the later release run -- same mechanism, bigger board -- matching what
analyzeCrossLatticeGTrend.py already assumes when transplanting 11x11 params onto 30x30.

This script only builds and visualizes the scaled prepattern; it does not run the release itself.
"""
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

torch.set_grad_enabled(False)

SOURCE = './data/StigmergicModelParameters.dat'
TARGET_ROWS, TARGET_COLS = 30, 30
OUTPUT = './data/scaledFacePrepattern_11to30.dat'
FIGURE = './figures/scaledFacePrepatternPreview.png'

parameters = torch.load(SOURCE, weights_only=False)
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
instance.simulate(clampParameters=clampParameters, fieldModulation=True,
                   numSimIters=clampEnd + 1, storeVariables=['Vmem'])

circuit = instance.electricNetwork
prepatternVmem = circuit.Vmem.detach().clone().reshape(rows, cols)                      # volts
prepatternGpolRatio = (circuit.G_pol.detach().clone() / circuit.G_ref).reshape(rows, cols)  # ratio to G_ref


def upsample(grid2d, targetRows, targetCols):
    t = grid2d.reshape(1, 1, *grid2d.shape).double()
    up = F.interpolate(t, size=(targetRows, targetCols), mode='bilinear', align_corners=True)
    return up.reshape(targetRows, targetCols)


scaledVmem = upsample(prepatternVmem, TARGET_ROWS, TARGET_COLS)
scaledGpolRatio = upsample(prepatternGpolRatio, TARGET_ROWS, TARGET_COLS)

# The 11x11 model's own iter-1000 result (the comparison target for the eventual 30x30 release),
# scaled up the same way purely so it can be looked at on the 30x30 grid.
targetVmem1000 = parameters['trainParameters']['actualVmem'].detach().clone().reshape(rows, cols)
scaledTarget1000 = upsample(targetVmem1000, TARGET_ROWS, TARGET_COLS)

torch.save({'scaledVmem': scaledVmem, 'scaledGpolRatio': scaledGpolRatio,
            'scaledTarget1000': scaledTarget1000,
            'sourceFile': SOURCE, 'clampEndIter': clampEnd,
            'sourceRows': rows, 'sourceCols': cols,
            'targetRows': TARGET_ROWS, 'targetCols': TARGET_COLS,
            'fieldParameters': parameters['fieldParameters']},
           OUTPUT)
print(f'wrote {OUTPUT}')

# ── figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4))
panels = [(prepatternVmem.numpy() * 1000, f'11x11 prepattern Vmem (mV)\niter {clampEnd + 1} (clamp release)', 'RdBu_r'),
          (prepatternGpolRatio.numpy(), f'11x11 prepattern G_pol/G_ref\niter {clampEnd + 1} (clamp release)', 'PuOr_r'),
          (targetVmem1000.numpy() * 1000, '11x11 reference face\niter 1000 (actualVmem)', 'gray'),
          (scaledVmem.numpy() * 1000, '30x30 scaled Vmem\n(bilinear upsample)', 'RdBu_r'),
          (scaledGpolRatio.numpy(), '30x30 scaled G_pol/G_ref\n(bilinear upsample)', 'PuOr_r'),
          (scaledTarget1000.numpy() * 1000, '30x30 scaled reference\n(for later comparison)', 'gray')]
for ax, (img, title, cmap) in zip(axes.flat, panels):
    im = ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle('Scaling the 11x11 clamp-release prepattern onto a 30x30 lattice (bilinear, align_corners=True)',
             fontsize=11)
fig.tight_layout()
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'wrote {FIGURE}')
