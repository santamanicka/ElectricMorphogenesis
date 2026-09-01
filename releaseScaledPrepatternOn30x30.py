"""Release the 11x11-scaled prepattern on a 30x30 tissue and watch it run unclamped for 5000 iters.

Loads the Vmem/G_pol state built by scaleFacePrepatternTo30x30.py (the 11x11 face model's
clamp-release state, bilinearly upsampled to 30x30) as the initial condition for a fresh 30x30
tissue, applies no clamp at all, and lets field-modulated dynamics run freely. The question: does
the tissue hold or regrow a face-like pattern resembling the 11x11 model's own iter-1000 result
(also upsampled, for comparison), or does it decay/drift into something else?

Field-transduction weight/gain/bias/timeConstant are always the 11x11 model's native values.
fieldScreenSize defaults to that same native value (--fieldScreenSize 4, i.e. the "same mechanism,
bigger board" test); pass e.g. --fieldScreenSize 10.909 to instead preserve the field's reach
*relative to tissue size* (screenSize/latticeSize held equal to the 11x11 case: 4/11 = X/30).
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--fieldScreenSize', type=float, default=None,
                     help='defaults to the native value stored with the prepattern (4)')
parser.add_argument('--fieldStrength', type=float, default=None,
                     help='defaults to the native value stored with the prepattern (1.0); scales the '
                          'Coulomb-law constant used to build eV from Vmem (see updateExtracellularVoltage)')
parser.add_argument('--numSimIters', type=int, default=5000)
parser.add_argument('--outputTag', default=None,
                     help='defaults to "screenNATIVE" or "screen<value>"')
args = parser.parse_args()

PREPATTERN_FILE = './data/scaledFacePrepattern_11to30.dat'
BASE_PARAMETER_FILE = './data/StigmergicModelParameters.dat'
NUM_SIM_ITERS = args.numSimIters
STORE_STRIDE = 10
SNAPSHOT_ITERS = [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]

prepattern = torch.load(PREPATTERN_FILE, weights_only=False)
rows, cols = prepattern['targetRows'], prepattern['targetCols']
numCells = rows * cols
scaledVmem = prepattern['scaledVmem'].reshape(1, numCells, 1).clone()
scaledTarget1000 = prepattern['scaledTarget1000'].reshape(rows, cols).numpy()

base = torch.load(BASE_PARAMETER_FILE, weights_only=False)
fieldParameters = dict(prepattern['fieldParameters'])
nativeScreenSize = fieldParameters['fieldScreenSize']
nativeFieldStrength = fieldParameters['fieldStrength']
if args.fieldScreenSize is not None:
    fieldParameters['fieldScreenSize'] = args.fieldScreenSize
if args.fieldStrength is not None:
    fieldParameters['fieldStrength'] = args.fieldStrength
screenTag = f'screenNative{nativeScreenSize}' if args.fieldScreenSize is None else f'screen{args.fieldScreenSize:g}'
strengthTag = '' if args.fieldStrength is None else f'_strength{args.fieldStrength:g}'
tag = args.outputTag or (screenTag + strengthTag)
OUTPUT_DATA = f'./data/releasedScaledFace_30x30_{tag}.dat'
FIGURE = f'./figures/releasedScaledFace_30x30_{tag}.png'
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
initialValues = {
    'Vmem': torch.zeros((numSamples, numCells, 1), dtype=torch.float64),
    'eV': torch.zeros((numSamples, (rows + 1) * (cols + 1), 1), dtype=torch.float64),
    'ligandConc': torch.zeros((numSamples, numCells, 1), dtype=torch.float64),
    'G_pol': {'cells': [[[0]]] * numSamples, 'values': [torch.DoubleTensor([1.0])] * numSamples},
    'G_dep': {'cells': [], 'values': torch.DoubleTensor([])},
}

instance = model(parameters, numSamples)
instance.setExperimentalConditions((initialValues, numSamples))

circuit = instance.electricNetwork
circuit.Vmem = scaledVmem.clone()
circuit.G_pol = (prepattern['scaledGpolRatio'].reshape(1, numCells, 1).clone() * circuit.G_ref)

print(f"  released on {rows}x{cols} ({numCells} cells), no clamp, field weight "
      f"{float(parameters['fieldParameters']['fieldTransductionWeight'])}, "
      f"screen {parameters['fieldParameters']['fieldScreenSize']}, "
      f"fieldStrength {parameters['fieldParameters']['fieldStrength']}")

instance.simulate(clampParameters=None, fieldModulation=True, numSimIters=NUM_SIM_ITERS,
                   storeVariables=['Vmem'], storeStride=STORE_STRIDE)

storedIters = np.array(instance.storedIters)
V = np.stack([v[0, :, 0].detach().numpy() for v in instance.timeseriesVmem]) * 1000  # (numStored, numCells) mV

# correlation against the (scaled) 11x11 iter-1000 reference, at every stored iteration
targetFlat = scaledTarget1000.reshape(-1) * 1000
correlation = np.array([np.corrcoef(frame, targetFlat)[0, 1] for frame in V])

torch.save({'storedIters': storedIters, 'Vmem_mV': V, 'correlation': correlation,
            'scaledTarget1000_mV': scaledTarget1000 * 1000, 'rows': rows, 'cols': cols,
            'numSimIters': NUM_SIM_ITERS}, OUTPUT_DATA)
print(f'  wrote {OUTPUT_DATA}')

bestIdx = int(np.nanargmax(correlation))
print(f"  best correlation to iter-1000 target: {correlation[bestIdx]:.3f} at iteration {storedIters[bestIdx]}")
print(f"  final correlation (iteration {storedIters[-1]}): {correlation[-1]:.3f}")

# ── figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(2.05 * (len(SNAPSHOT_ITERS) + 1), 5.6))
grid = gridspec.GridSpec(2, len(SNAPSHOT_ITERS) + 1, height_ratios=[1, 1.2], hspace=0.55, wspace=0.08)

axTarget = fig.add_subplot(grid[0, 0])
axTarget.imshow(scaledTarget1000 * 1000, cmap='gray')
axTarget.set_title('target\n(11x11 iter1000,\nscaled)', fontsize=8)
axTarget.set_xticks([]); axTarget.set_yticks([])

for col, iteration in enumerate(SNAPSHOT_ITERS, start=1):
    row = int(np.searchsorted(storedIters, iteration))
    row = min(row, len(storedIters) - 1)
    frame = V[row].reshape(rows, cols)
    axis = fig.add_subplot(grid[0, col])
    axis.imshow(frame, cmap='gray', vmin=frame.min(), vmax=frame.max())
    axis.set_title(f'iter {storedIters[row]}\nr={correlation[row]:.2f}\n[{frame.min():.0f},{frame.max():.0f}]mV', fontsize=7)
    axis.set_xticks([]); axis.set_yticks([])

axCorr = fig.add_subplot(grid[1, :])
axCorr.plot(storedIters, correlation, color='steelblue')
axCorr.axhline(0, color='gray', linewidth=0.7)
axCorr.set_xlabel('iteration'); axCorr.set_ylabel('correlation to\n11x11 iter-1000 target')
axCorr.set_title('Pattern similarity over the unclamped 30x30 release', fontsize=10)

fig.suptitle(f'11x11 clamp-release prepattern, scaled to 30x30, released unclamped for {NUM_SIM_ITERS} iters '
             f'(fieldScreenSize={fieldParameters["fieldScreenSize"]:g}, fieldStrength={fieldParameters["fieldStrength"]:g}, '
             f'per-panel colour scale)', fontsize=11)
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'  wrote {FIGURE}')
