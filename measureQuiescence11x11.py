"""Find the quiescence-to-propagation transition for the 11x11 tissue, at StigmergicModelParameters.dat's
own field/GJ parameters (fieldScreenSize=4, fieldTransductionWeight=1000, fieldStrength=1.0,
fieldTransductionBias=0.0005, GJStrength=0.05) -- the analog of PolyPatterning_Sim.md Sec 8.5's staging
measurement, which was done only at 30x30 and is explicitly flagged there as not expected to transfer to
a different lattice size. Uses this file specifically, not bestModelParameters_fieldVector_0.dat's very
similar-looking but critically different bias (0.0021) -- that difference alone flips the unclamped
tissue's fate between a stable fixed point and a genuinely never-settling oscillation (confirmed
directly, see measureQuiescence11x11_unclamped.py), so getting the base file right matters here too.

Method: apply a short (100-iteration) random band-limited static hold to break symmetry (with either
clampMode -- tissueBandGpolTwoFoldSymmetry, Gpol only, or tissueBandGpolVmemTwoFoldSymmetry, Gpol+Vmem --
matching the mechanisms the real experiment will use), then track mean per-step |delta Vmem| over free
evolution to find where the rate stops falling (quiescence) and starts climbing again
(propagation/nucleation), the same reasoning Sec 8.5 used at 30x30.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model
import utilities

parser = argparse.ArgumentParser()
parser.add_argument('--clampMode', type=str, default='tissueBandGpolTwoFoldSymmetry',
                     choices=['tissueBandGpolTwoFoldSymmetry', 'tissueBandGpolVmemTwoFoldSymmetry'])
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--clampIters', type=int, default=100)
parser.add_argument('--numSimIters', type=int, default=5000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--paramsFile', type=str, default='data/StigmergicModelParameters.dat')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_grad_enabled(False)

p = torch.load(args.paramsFile, map_location='cpu', weights_only=False)
rows, cols = p['latticeDims']
assert (rows, cols) == (11, 11), f"expected an 11x11 checkpoint, got {(rows, cols)}"
p['latticePeriodicBoundaryGJ'] = False
p['ATPParameters'] = None
system = model(p, p['simParameters']['numSamples'])
system.setExperimentalConditions((p['simParameters']['initialValues'], p['simParameters']['numSamples']))
circuit = system.electricNetwork
utils = utilities.utilities()

# Build a random band clamp exactly the way learnCellularFieldNetwork.py does for clampType='staticRandom'
# -- left-half band indices, mirrored, one random G_pol/G_ref ratio per clamped point (clampValue=1.0
# convention). The Gpol+Vmem variant adds a second, independently-random held-Vmem value per point, over
# the same default range learnCellularFieldNetwork.py's --clampVmemRange uses.
bandLeftHalfIndices = utils.computeBandIndices(circuit, mode='tissue', region='leftHalf', depth=args.depth)
clampPointIndices = np.array(bandLeftHalfIndices)
verticalReflectedIndices = utils.computeSymmetricalIndices(circuit, clampPointIndices, mode='tissue', symmetry='twofold')
clampPointIndices = np.concatenate((clampPointIndices, verticalReflectedIndices))
_, uniqueClampPointIndices = np.unique(clampPointIndices, return_index=True)
clampPointIndices = clampPointIndices[uniqueClampPointIndices]
numClampPoints = len(clampPointIndices)
sampleIndices = np.repeat(range(p['simParameters']['numSamples']), numClampPoints)
clampIndices = (sampleIndices, clampPointIndices)
print(f"clampMode={args.clampMode}, depth={args.depth}: {numClampPoints} clamped cells "
      f"({numClampPoints/circuit.numCells*100:.1f}% of {circuit.numCells})")

clampValuesStatic = torch.rand(len(bandLeftHalfIndices), dtype=torch.double)
clampValuesStaticActual = torch.tile(clampValuesStatic, (2,))
clampValues = clampValuesStaticActual.repeat((args.clampIters + 1, 1))[:, uniqueClampPointIndices]

clampParameters = {
    'clampMode': args.clampMode,
    'clampIndices': clampIndices,
    'clampValues': clampValues,
    'clampStartIter': 0,
    'clampEndIter': args.clampIters,
}
if args.clampMode == 'tissueBandGpolVmemTwoFoldSymmetry':
    minClampValueVmem, maxClampValueVmem = -0.060, -0.0092
    clampValuesStaticVmem = torch.rand(len(bandLeftHalfIndices), dtype=torch.double) * (maxClampValueVmem - minClampValueVmem) + minClampValueVmem
    clampValuesStaticVmemActual = torch.tile(clampValuesStaticVmem, (2,))
    clampParameters['clampValuesVmem'] = clampValuesStaticVmemActual.repeat((args.clampIters + 1, 1))[:, uniqueClampPointIndices]

system.simulate(clampParameters=clampParameters, numSimIters=args.numSimIters, storeVariables=['Vmem'])
vmem = system.timeseriesVmem[:, 0, :, 0].numpy() * 1000  # (numSimIters, numCells), mV

perStepChange = np.abs(np.diff(vmem, axis=0)).mean(axis=1)  # mean |delta Vmem| across cells, per step
windowSize = 20
windowedRate = np.convolve(perStepChange, np.ones(windowSize) / windowSize, mode='valid')

fig, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[1, 1.2])
axes[0].plot(perStepChange, alpha=0.3, label='raw per-step |dVmem|')
axes[0].plot(np.arange(windowSize - 1, len(perStepChange)), windowedRate, label=f'{windowSize}-step rolling mean')
axes[0].axvline(args.clampIters, color='gray', linestyle='--', label=f'clamp releases ({args.clampIters})')
axes[0].set_ylabel('mean |ΔVmem| per step (mV)')
axes[0].legend(fontsize=8)
axes[0].set_title(f'11x11 ({args.clampMode}, depth={args.depth}, {args.clampIters}-iter hold): settling rate')

axes[1].plot(vmem.std(axis=1), label='Vmem std across cells (mV)')
axes[1].axvline(args.clampIters, color='gray', linestyle='--')
axes[1].set_xlabel('iteration')
axes[1].set_ylabel('spatial std (mV)')
axes[1].legend(fontsize=8)

fig.tight_layout()
suffix = 'GpolVmem' if args.clampMode == 'tissueBandGpolVmemTwoFoldSymmetry' else 'Gpol'
outPath = f'figures/quiescence11x11_{suffix}_depth{args.depth}_hold{args.clampIters}.png'
fig.savefig(outPath, dpi=140, bbox_inches='tight')
print(f"wrote {outPath}")

print(f"\nwindowed rate (mV/step, {windowSize}-step rolling mean) at checkpoints:")
for it in range(0, args.numSimIters - windowSize, 200):
    print(f"  iter {it:5d}: rate={windowedRate[it]:.5f}  std={vmem[it].std():.4f} mV")
