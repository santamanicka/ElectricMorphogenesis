"""Generate a genuinely random 2-fold-symmetric prepattern the way the real one was made: run a random
boundary field clamp through the actual clamp+field-transduction dynamics, natively on the 30x30
lattice -- not hand-crafted i.i.d. noise. That gives it the same qualitative character as a real
clamp-derived prepattern (content concentrated on the boundary, smooth field-propagated structure in
the bulk) without carrying any of the learned 11x11 face's specific content.

Clamp-generation mechanics mirror map_basin_structure.py's prePattern(): random per-boundary-point
cosine time series (frequency/phase/amplitude), mirrored across the vertical axis for exact 2-fold
symmetry, applied via the same 'fieldDomeTwoFoldSymmetry' clamp mode and clampIters+1 horizon the real
model uses, amplitude matched to the real clampValues' scale (std~0.31, range ~[-0.84,0.84]).

The resulting prepattern is then released (unclamped) at the same field parameters used for the
scaled-11x11-prepattern releases, so its t~680 behavior is directly comparable to both the baseline
and the small-noise perturbations already tested.
"""
import argparse
import gc

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utilities
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--fieldScreenSize', type=float, default=10.909090909090908)
parser.add_argument('--fieldStrength', type=float, default=0.25)
parser.add_argument('--clampIters', type=int, default=100)
parser.add_argument('--numSimIters', type=int, default=1000)
parser.add_argument('--numTrials', type=int, default=8)
parser.add_argument('--readIter', type=int, default=680)
parser.add_argument('--seed0', type=int, default=2000)
args = parser.parse_args()

PREPATTERN_FILE = './data/scaledFacePrepattern_11to30.dat'
BASE_PARAMETER_FILE = './data/StigmergicModelParameters.dat'
STORE_STRIDE = 10
tag = f'strength{args.fieldStrength:g}'
FIGURE = f'./figures/randomClampDerivedPrepattern_{tag}.png'
OUTPUT_DATA = f'./data/randomClampDerivedPrepattern_{tag}.dat'
prePatternIter = args.clampIters + 1

reference30x30 = torch.load(PREPATTERN_FILE, weights_only=False)
rows, cols = reference30x30['targetRows'], reference30x30['targetCols']
numCells = rows * cols
baseVmem = reference30x30['scaledVmem'].reshape(rows, cols).clone()
baseGpolRatio = reference30x30['scaledGpolRatio'].reshape(rows, cols).clone()
scaledTarget1000 = reference30x30['scaledTarget1000'].reshape(rows, cols).numpy()

base = torch.load(BASE_PARAMETER_FILE, weights_only=False)
fieldParameters = dict(reference30x30['fieldParameters'])
fieldParameters['fieldScreenSize'] = args.fieldScreenSize
fieldParameters['fieldStrength'] = args.fieldStrength
parameters = {
    'latticeDims': (rows, cols),
    'GJParameters': dict(base['GJParameters']),
    'fieldParameters': fieldParameters,
    'ligandParameters': dict(base['ligandParameters']),
    'GRNParameters': dict(base['GRNParameters']),
    'latticePeriodicBoundaryGJ': False,
    'ATPParameters': None,
}
numSamples = 1
initialValuesTemplate = {
    'Vmem': torch.zeros((numSamples, numCells, 1), dtype=torch.float64),
    'eV': torch.zeros((numSamples, (rows + 1) * (cols + 1), 1), dtype=torch.float64),
    'ligandConc': torch.zeros((numSamples, numCells, 1), dtype=torch.float64),
    'G_pol': {'cells': [[[0]]] * numSamples, 'values': [torch.DoubleTensor([1.0])] * numSamples},
    'G_dep': {'cells': [], 'values': torch.DoubleTensor([])},
}

# ── boundary field-clamp index set, 2-fold symmetric (same convention as the real training clamp
# and as map_basin_structure.py) ───────────────────────────────────────────
utils = utilities.utilities()
probe = model(parameters, numSamples)
probe.setExperimentalConditions((initialValuesTemplate, numSamples))
circuit = probe.electricNetwork
leftHalf = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
mirrored = utils.computeSymmetricalIndices(circuit, leftHalf, mode='field', symmetry='twofold')
allIndices = np.concatenate((leftHalf, mirrored))
_, uniqueIdx = np.unique(allIndices, return_index=True)
clampIndices = (np.zeros(len(allIndices[uniqueIdx]), dtype=int), allIndices[uniqueIdx])
numHalf = len(leftHalf)
timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
del probe
gc.collect()
print(f"  {len(allIndices[uniqueIdx])} boundary field-clamp points (2-fold symmetric), "
      f"screen={args.fieldScreenSize:g} strength={args.fieldStrength:g}")


def randomClampValues(seed):
    generator = torch.Generator().manual_seed(seed)
    frequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * 900 + 100
    phases = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
    amplitudes = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 - 1
    values = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
              * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    return values


def buildRandomPrepattern(seed):
    # numSimIters = clampEndIter+1 executed iterations (0..clampEndIter inclusive) so that, after the
    # loop, the *live* circuit state reflects the update from the last clamped iteration -- reading a
    # stored timeseries row instead would be off by one (it snapshots state *before* that iteration's
    # update runs), which is exactly the bug caught earlier in this session for the 11x11 prepattern.
    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValuesTemplate, numSamples))
    instance.simulate(numSimIters=prePatternIter, fieldModulation=True, perturbation=None,
                       clampParameters={'clampMode': 'fieldDomeTwoFoldSymmetry', 'clampIndices': clampIndices,
                                         'clampValues': randomClampValues(seed), 'clampStartIter': 0,
                                         'clampEndIter': args.clampIters},
                       storeVariables=[])
    circuit = instance.electricNetwork
    vmem = circuit.Vmem[0, :, 0].detach().clone()
    gpolRatio = (circuit.G_pol[0, :, 0].detach().clone() / circuit.G_ref)
    del instance
    gc.collect()
    return vmem.reshape(rows, cols), gpolRatio.reshape(rows, cols)


def release(vmem, gpolRatio):
    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValuesTemplate, numSamples))
    circuit = instance.electricNetwork
    circuit.Vmem = vmem.reshape(1, numCells, 1).clone()
    circuit.G_pol = (gpolRatio.reshape(1, numCells, 1).clone() * circuit.G_ref)
    instance.simulate(clampParameters=None, fieldModulation=True, numSimIters=args.numSimIters,
                       storeVariables=['Vmem'], storeStride=STORE_STRIDE)
    storedIters = np.array(instance.storedIters)
    V = np.stack([v[0, :, 0].detach().numpy() for v in instance.timeseriesVmem]) * 1000
    targetFlat = scaledTarget1000.reshape(-1) * 1000
    correlation = np.array([np.corrcoef(frame, targetFlat)[0, 1] for frame in V])
    del instance
    gc.collect()
    return storedIters, V, correlation


trials = [('baseline', baseVmem, baseGpolRatio)]
for i in range(args.numTrials):
    seed = args.seed0 + i
    vmem, gpolRatio = buildRandomPrepattern(seed)
    trials.append((f'seed{seed}', vmem, gpolRatio))

results = []
for label, vmem, gpolRatio in trials:
    storedIters, V, correlation = release(vmem, gpolRatio)
    readRow = min(int(np.searchsorted(storedIters, args.readIter)), len(storedIters) - 1)
    print(f"  {label:10s}  r@iter{args.readIter}={correlation[readRow]:+.3f}  "
          f"peak r={correlation.max():+.3f} at iter {storedIters[np.argmax(correlation)]}")
    results.append((label, storedIters, V, correlation, readRow))

torch.save({'results': [(l, si, V, c, rr) for l, si, V, c, rr in results],
            'scaledTarget1000_mV': scaledTarget1000 * 1000, 'rows': rows, 'cols': cols,
            'readIter': args.readIter}, OUTPUT_DATA)
print(f'  wrote {OUTPUT_DATA}')

# ── figure ───────────────────────────────────────────────────────────────
n = len(results)
fig = plt.figure(figsize=(2.1 * (n + 1), 5.6))
grid = gridspec.GridSpec(2, n + 1, height_ratios=[1, 1.2], hspace=0.5, wspace=0.12)

axTarget = fig.add_subplot(grid[0, 0])
axTarget.imshow(scaledTarget1000, cmap='gray')
axTarget.set_title('target', fontsize=8)
axTarget.set_xticks([]); axTarget.set_yticks([])

for col, (label, storedIters, V, correlation, readRow) in enumerate(results, start=1):
    frame = V[readRow].reshape(rows, cols)
    axis = fig.add_subplot(grid[0, col])
    axis.imshow(frame, cmap='gray', vmin=frame.min(), vmax=frame.max())
    axis.set_title(f'{label}\niter {storedIters[readRow]}, r={correlation[readRow]:.2f}', fontsize=7)
    axis.set_xticks([]); axis.set_yticks([])

axCorr = fig.add_subplot(grid[1, :])
for label, storedIters, V, correlation, readRow in results:
    style = dict(linewidth=2.2, color='black') if label == 'baseline' else dict(linewidth=1.0, alpha=0.8)
    axCorr.plot(storedIters, correlation, label=label, **style)
axCorr.axvline(args.readIter, color='gray', linewidth=0.7, linestyle='--')
axCorr.axhline(0, color='gray', linewidth=0.5)
axCorr.set_xlabel('iteration'); axCorr.set_ylabel('correlation to\n11x11 iter-1000 target')
axCorr.legend(fontsize=6, ncol=5, loc='upper right')
axCorr.set_title('Prepatterns built from a random 2-fold-symmetric boundary clamp, run natively on '
                  '30x30 (same clamp mechanism, random content)', fontsize=9)
fig.suptitle(f'Does the t~{args.readIter} echo need the learned boundary code, or does any '
             f'clamp-derived prepattern produce it?', fontsize=11)
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'  wrote {FIGURE}')
