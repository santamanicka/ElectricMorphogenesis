"""Find the quiescence-to-complexity timeline for the 11x11 tissue under its own genuine, already-
trained boundary-dome clamp (StigmergicModelParameters.dat's stored fieldDomeTwoFoldSymmetry clamp --
not a random substitute, not no clamp at all). Earlier attempts used an unclamped run and a random
tissueBandGpolTwoFoldSymmetry clamp, both of which collapsed to a trivial fixed point; neither is what
Sec 3/4's complexity measurements (nonzero discreteTSE/gaussianTSE/participation ratio at every screen
size, matching this file's own field/GJ parameters) actually characterized. A genuinely trained signal,
not a random one, is what the field-feedback loop sustains into persistent complexity -- this replays
that real signal to find where the tissue's own quiescence dip sits, informing T for the band-hold
experiment even though the eventual mechanism (tissueBandGpol...) differs from this reference clamp.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

torch.set_grad_enabled(False)

p = torch.load('data/StigmergicModelParameters.dat', map_location='cpu', weights_only=False)
p['latticePeriodicBoundaryGJ'] = False
p['ATPParameters'] = None
system = model(p, p['simParameters']['numSamples'])
system.setExperimentalConditions((p['simParameters']['initialValues'], p['simParameters']['numSamples']))
circuit = system.electricNetwork
clampParameters = dict(p['clampParameters'])
clampEndIter = int(clampParameters['clampEndIter'])
numSimIters = 5000

system.simulate(clampParameters=clampParameters, numSimIters=numSimIters, storeVariables=['Vmem'])
vmem = system.timeseriesVmem[:, 0, :, 0].numpy() * 1000

perStepChange = np.abs(np.diff(vmem, axis=0)).mean(axis=1)
windowSize = 20
windowedRate = np.convolve(perStepChange, np.ones(windowSize) / windowSize, mode='valid')
spatialStd = vmem.std(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[1, 1.2])
axes[0].plot(perStepChange, alpha=0.3, label='raw per-step |dVmem|')
axes[0].plot(np.arange(windowSize - 1, len(perStepChange)), windowedRate, label=f'{windowSize}-step rolling mean')
axes[0].axvline(clampEndIter, color='gray', linestyle='--', label=f'clamp releases ({clampEndIter})')
axes[0].set_ylabel('mean |ΔVmem| per step (mV)')
axes[0].legend(fontsize=8)
axes[0].set_title('11x11, StigmergicModelParameters.dat\'s own trained boundary-dome clamp: settling rate')

axes[1].plot(spatialStd, label='Vmem std across cells (mV)')
axes[1].axvline(clampEndIter, color='gray', linestyle='--')
axes[1].set_xlabel('iteration')
axes[1].set_ylabel('spatial std (mV)')
axes[1].legend(fontsize=8)

fig.tight_layout()
outPath = 'figures/quiescence11x11_trainedClamp.png'
fig.savefig(outPath, dpi=140, bbox_inches='tight')
print(f"wrote {outPath}")

print(f"\nwindowed rate (mV/step) and spatial std at checkpoints:")
for it in range(0, numSimIters - windowSize, 100):
    print(f"  iter {it:5d}: rate={windowedRate[it]:.5f}  std={spatialStd[it]:.4f} mV")
