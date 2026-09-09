"""Find the quiescence-to-propagation transition for the 11x11 tissue evolving completely freely from
its uniform initial condition -- no clamp at all, not even a brief symmetry-breaking one. Native field
parameters (fieldScreenSize=4, fieldTransductionWeight=1000, fieldStrength=1.0). The analog of
PolyPatterning_Sim.md Sec 8.5's staging measurement (done only at 30x30, under an actual clamp, and
flagged there as not expected to transfer to a different lattice size).

The initial condition is perfectly uniform (every cell at the same Vmem, no noise anywhere in this
codebase's deterministic pipeline -- stochasticIonChannels is hardcoded False throughout), so a
translation-symmetric system would stay uniform forever. It does not have to: the lattice itself is
not translation-symmetric -- a finite, non-periodic boundary means edge cells have fewer field
neighbors than centre cells, which is a real geometric asymmetry independent of any perturbation. This
script tests directly whether that alone is enough to break the tissue out of uniformity.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--numSimIters', type=int, default=3000)
parser.add_argument('--paramsFile', type=str, default='data/StigmergicModelParameters.dat')
args = parser.parse_args()

torch.set_grad_enabled(False)

p = torch.load(args.paramsFile, map_location='cpu', weights_only=False)
rows, cols = p['latticeDims']
assert (rows, cols) == (11, 11), f"expected the 11x11 native checkpoint, got {(rows, cols)}"
p['latticePeriodicBoundaryGJ'] = False
p['ATPParameters'] = None
system = model(p, p['simParameters']['numSamples'])
system.setExperimentalConditions((p['simParameters']['initialValues'], p['simParameters']['numSamples']))
circuit = system.electricNetwork

initialVmem = system.electricNetwork.Vmem[0, :, 0].detach().numpy() * 1000
print(f"initial Vmem: min={initialVmem.min():.6f} max={initialVmem.max():.6f} mV "
      f"(should be identical everywhere -- confirming the uniform IC before evolving)")

system.simulate(clampParameters=None, numSimIters=args.numSimIters, storeVariables=['Vmem'])
vmem = system.timeseriesVmem[:, 0, :, 0].numpy() * 1000  # (numSimIters, numCells), mV

perStepChange = np.abs(np.diff(vmem, axis=0)).mean(axis=1)  # mean |delta Vmem| across cells, per step
windowSize = 20
windowedRate = np.convolve(perStepChange, np.ones(windowSize) / windowSize, mode='valid')

fig, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[1, 1.2])
axes[0].plot(perStepChange, alpha=0.3, label='raw per-step |dVmem|')
axes[0].plot(np.arange(windowSize - 1, len(perStepChange)), windowedRate, label=f'{windowSize}-step rolling mean')
axes[0].set_ylabel('mean |ΔVmem| per step (mV)')
axes[0].legend(fontsize=8)
axes[0].set_title('11x11 native (screen4/weight1000), fully unclamped, uniform initial condition')

axes[1].plot(vmem.std(axis=1), label='Vmem std across cells (mV)')
axes[1].set_xlabel('iteration')
axes[1].set_ylabel('spatial std (mV)')
axes[1].legend(fontsize=8)

fig.tight_layout()
outPath = 'figures/quiescence11x11_unclamped.png'
fig.savefig(outPath, dpi=140, bbox_inches='tight')
print(f"wrote {outPath}")

print(f"\nwindowed rate (mV/step, {windowSize}-step rolling mean) at checkpoints:")
for it in range(0, args.numSimIters - windowSize, 100):
    print(f"  iter {it:5d}: {windowedRate[it]:.5f}")
