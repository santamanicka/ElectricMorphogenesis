"""Visualize the actual Vmem pattern evolution (not just the rate/std summary) for each of the four
random band-clamp configurations checked in measureQuiescence11x11.py: both mechanisms
(tissueBandGpolTwoFoldSymmetry, tissueBandGpolVmemTwoFoldSymmetry) crossed with both depths (1, 2),
sampled every 50 iterations from t=1 to t=500 -- the window spanning the clamp (0-100) and the
quiescence trough identified in that script (~150-300) into the start of the following active phase.

Same StigmergicModelParameters.dat base (correct fieldTransductionBias=0.0005) and same seed=0
construction as measureQuiescence11x11.py's default, so this is a direct visual companion to those
rate/std numbers, not a separate run.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model
import utilities

torch.manual_seed(0)
np.random.seed(0)
torch.set_grad_enabled(False)

snapIters = list(range(1, 501, 50))  # 1, 51, 101, ..., 451
numSimIters = 501

configs = [
    ('Gpol-only, depth 1', 'tissueBandGpolTwoFoldSymmetry', 1),
    ('Gpol-only, depth 2', 'tissueBandGpolTwoFoldSymmetry', 2),
    ('Gpol+Vmem, depth 1', 'tissueBandGpolVmemTwoFoldSymmetry', 1),
    ('Gpol+Vmem, depth 2', 'tissueBandGpolVmemTwoFoldSymmetry', 2),
]

fig, axes = plt.subplots(len(configs), len(snapIters), figsize=(2.0 * len(snapIters), 2.3 * len(configs)))

for row, (label, clampMode, depth) in enumerate(configs):
    torch.manual_seed(0)
    np.random.seed(0)
    p = torch.load('data/StigmergicModelParameters.dat', map_location='cpu', weights_only=False)
    p['latticePeriodicBoundaryGJ'] = False
    p['ATPParameters'] = None
    system = model(p, p['simParameters']['numSamples'])
    system.setExperimentalConditions((p['simParameters']['initialValues'], p['simParameters']['numSamples']))
    circuit = system.electricNetwork
    utils = utilities.utilities()

    bandLeftHalfIndices = utils.computeBandIndices(circuit, mode='tissue', region='leftHalf', depth=depth)
    clampPointIndices = np.array(bandLeftHalfIndices)
    verticalReflectedIndices = utils.computeSymmetricalIndices(circuit, clampPointIndices, mode='tissue', symmetry='twofold')
    clampPointIndices = np.concatenate((clampPointIndices, verticalReflectedIndices))
    _, uniqueClampPointIndices = np.unique(clampPointIndices, return_index=True)
    clampPointIndices = clampPointIndices[uniqueClampPointIndices]
    numClampPoints = len(clampPointIndices)
    sampleIndices = np.repeat(range(p['simParameters']['numSamples']), numClampPoints)
    clampIndices = (sampleIndices, clampPointIndices)

    clampIters = 100
    clampValuesStatic = torch.rand(len(bandLeftHalfIndices), dtype=torch.double)
    clampValuesStaticActual = torch.tile(clampValuesStatic, (2,))
    clampValues = clampValuesStaticActual.repeat((clampIters + 1, 1))[:, uniqueClampPointIndices]

    clampParameters = {
        'clampMode': clampMode,
        'clampIndices': clampIndices,
        'clampValues': clampValues,
        'clampStartIter': 0,
        'clampEndIter': clampIters,
    }
    if clampMode == 'tissueBandGpolVmemTwoFoldSymmetry':
        minClampValueVmem, maxClampValueVmem = -0.060, -0.0092
        clampValuesStaticVmem = torch.rand(len(bandLeftHalfIndices), dtype=torch.double) * (maxClampValueVmem - minClampValueVmem) + minClampValueVmem
        clampValuesStaticVmemActual = torch.tile(clampValuesStaticVmem, (2,))
        clampParameters['clampValuesVmem'] = clampValuesStaticVmemActual.repeat((clampIters + 1, 1))[:, uniqueClampPointIndices]

    system.simulate(clampParameters=clampParameters, numSimIters=numSimIters, storeVariables=['Vmem'])
    vmem = system.timeseriesVmem[:, 0, :, 0].numpy() * 1000  # (numSimIters, numCells), mV

    for col, it in enumerate(snapIters):
        ax = axes[row, col]
        frame = vmem[it].reshape(11, 11)
        ax.imshow(frame, cmap='gray')
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f'iter {it}', fontsize=9)
        if col == 0:
            ax.set_ylabel(label, fontsize=10)

fig.suptitle('11x11 pattern evolution, t=1 to 500 (random band clamp, StigmergicModelParameters.dat)', fontsize=12)
fig.tight_layout()
outPath = 'figures/quiescence11x11_patternEvolution.png'
fig.savefig(outPath, dpi=140, bbox_inches='tight')
print(f'wrote {outPath}')
