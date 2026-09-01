"""Statistically compare the rescaled-transplant prepattern (data/scaledFacePrepattern_11to30.dat)
against a family of genuinely random boundary-clamp-derived prepatterns, both built at the same
field parameters (fieldScreenSize=10.909 (~11), fieldStrength=0.25 by default -- the setting
identified in Sim §11.3/11.6). Clamp-generation mechanics mirror randomClampDerivedPrepattern.py's
buildRandomPrepattern(): a random 2-fold-symmetric boundary field clamp run through the real
clampEndIter+1 clamp dynamics, natively on the 30x30 lattice.

Question asked: does the transplant etch a larger portion of the tissue (reach further from the
boundary, commit more cells) than a same-parameter random clamp does, or are the two prepattern
families statistically indistinguishable except for which specific cells end up committed?
"""
import argparse
import gc

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utilities
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--fieldScreenSize', type=float, default=10.909090909090908)
parser.add_argument('--fieldStrength', type=float, default=0.25)
parser.add_argument('--clampIters', type=int, default=100)
parser.add_argument('--numTrials', type=int, default=8)
parser.add_argument('--seed0', type=int, default=3000)
parser.add_argument('--committedLow', type=float, default=0.1)
parser.add_argument('--committedHigh', type=float, default=1.9)
args = parser.parse_args()

PREPATTERN_FILE = './data/scaledFacePrepattern_11to30.dat'
BASE_PARAMETER_FILE = './data/StigmergicModelParameters.dat'
prePatternIter = args.clampIters + 1

reference30x30 = torch.load(PREPATTERN_FILE, weights_only=False)
rows, cols = reference30x30['targetRows'], reference30x30['targetCols']
numCells = rows * cols
transplantVmem = reference30x30['scaledVmem'].reshape(rows, cols).clone()
transplantGpolRatio = reference30x30['scaledGpolRatio'].reshape(rows, cols).clone()

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


def randomClampValues(seed):
    generator = torch.Generator().manual_seed(seed)
    frequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * 900 + 100
    phases = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
    amplitudes = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 - 1
    values = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
              * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    return values


def buildRandomPrepattern(seed):
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


# Depth shell = Chebyshev distance from the nearest tissue edge (0 = boundary ring).
rowIdx, colIdx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
depthShell = np.minimum.reduce([rowIdx, rows - 1 - rowIdx, colIdx, cols - 1 - colIdx])
numShells = depthShell.max() + 1


def committedFraction(gpolRatio):
    g = gpolRatio.numpy()
    committed = (g < args.committedLow) | (g > args.committedHigh)
    return committed


def depthProfile(gpolRatio):
    committed = committedFraction(gpolRatio)
    return np.array([committed[depthShell == s].mean() for s in range(numShells)])


randomVmems, randomGpols = [], []
for i in range(args.numTrials):
    seed = args.seed0 + i
    vmem, gpolRatio = buildRandomPrepattern(seed)
    randomVmems.append(vmem)
    randomGpols.append(gpolRatio)
    print(f"  seed {seed}: Vmem [{vmem.min()*1000:.2f},{vmem.max()*1000:.2f}] mV, "
          f"G_pol ratio [{gpolRatio.min():.3f},{gpolRatio.max():.3f}], "
          f"committed {committedFraction(gpolRatio).mean()*100:.1f}%")

transplantCommitted = committedFraction(transplantGpolRatio)
transplantProfile = depthProfile(transplantGpolRatio)
randomProfiles = np.stack([depthProfile(g) for g in randomGpols])

print()
print(f"transplant: Vmem [{transplantVmem.min()*1000:.2f},{transplantVmem.max()*1000:.2f}] mV "
      f"(mean {transplantVmem.mean()*1000:.2f}, std {transplantVmem.std()*1000:.2f}), "
      f"G_pol ratio [{transplantGpolRatio.min():.3f},{transplantGpolRatio.max():.3f}] "
      f"(mean {transplantGpolRatio.mean():.3f}, std {transplantGpolRatio.std():.3f}), "
      f"committed {transplantCommitted.mean()*100:.1f}%")
randomVmemStack = torch.stack(randomVmems)
randomGpolStack = torch.stack(randomGpols)
print(f"random ({args.numTrials} trials): Vmem mean {randomVmemStack.mean()*1000:.2f} +/- "
      f"{randomVmemStack.mean(dim=(1,2)).std()*1000:.2f} mV (within-trial std {randomVmemStack.std(dim=(1,2)).mean()*1000:.2f}), "
      f"G_pol ratio mean {randomGpolStack.mean():.3f} +/- {randomGpolStack.mean(dim=(1,2)).std():.3f} "
      f"(within-trial std {randomGpolStack.std(dim=(1,2)).mean():.3f}), "
      f"committed {np.array([committedFraction(g).mean() for g in randomGpols]).mean()*100:.1f}% "
      f"+/- {np.array([committedFraction(g).mean() for g in randomGpols]).std()*100:.1f}%")

print()
print("depth-shell committed fraction (shell 0 = boundary ring):")
print(f"{'shell':>6} {'transplant':>11} {'random mean':>12} {'random std':>11}")
for s in range(numShells):
    print(f"{s:>6} {transplantProfile[s]*100:>10.1f}% {randomProfiles[:,s].mean()*100:>11.1f}% "
          f"{randomProfiles[:,s].std()*100:>10.1f}%")

firstUncommittedTransplant = next((s for s in range(numShells) if transplantProfile[s] < 0.5), numShells)
firstUncommittedRandomMean = next((s for s in range(numShells) if randomProfiles[:, s].mean() < 0.5), numShells)
print()
print(f"first shell < 50% committed -- transplant: {firstUncommittedTransplant}, "
      f"random (mean): {firstUncommittedRandomMean}")

# Figure: prepattern maps + depth profile
fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(2, 4)

axTVmem = fig.add_subplot(gs[0, 0])
axTVmem.imshow(transplantVmem.numpy()*1000, cmap='RdBu_r')
axTVmem.set_title('transplant Vmem (mV)', fontsize=10); axTVmem.set_xticks([]); axTVmem.set_yticks([])

axTGpol = fig.add_subplot(gs[0, 1])
axTGpol.imshow(transplantGpolRatio.numpy(), cmap='RdBu_r', vmin=0, vmax=2)
axTGpol.set_title('transplant G_pol/G_ref', fontsize=10); axTGpol.set_xticks([]); axTGpol.set_yticks([])

axRVmem = fig.add_subplot(gs[0, 2])
axRVmem.imshow(randomVmems[0].numpy()*1000, cmap='RdBu_r')
axRVmem.set_title('random-clamp Vmem (seed0)', fontsize=10); axRVmem.set_xticks([]); axRVmem.set_yticks([])

axRGpol = fig.add_subplot(gs[0, 3])
axRGpol.imshow(randomGpols[0].numpy(), cmap='RdBu_r', vmin=0, vmax=2)
axRGpol.set_title('random-clamp G_pol/G_ref (seed0)', fontsize=10); axRGpol.set_xticks([]); axRGpol.set_yticks([])

axCommitTransplant = fig.add_subplot(gs[1, 0])
axCommitTransplant.imshow(transplantCommitted, cmap='gray_r')
axCommitTransplant.set_title('transplant: committed cells', fontsize=10); axCommitTransplant.set_xticks([]); axCommitTransplant.set_yticks([])

axCommitRandom = fig.add_subplot(gs[1, 1])
axCommitRandom.imshow(committedFraction(randomGpols[0]), cmap='gray_r')
axCommitRandom.set_title('random-clamp: committed cells (seed0)', fontsize=10); axCommitRandom.set_xticks([]); axCommitRandom.set_yticks([])

axProfile = fig.add_subplot(gs[1, 2:4])
axProfile.plot(range(numShells), transplantProfile*100, marker='o', label='transplant')
axProfile.errorbar(range(numShells), randomProfiles.mean(axis=0)*100, yerr=randomProfiles.std(axis=0)*100,
                    marker='s', label=f'random ({args.numTrials} trials, mean+/-std)')
axProfile.set_xlabel('depth shell (0 = boundary)')
axProfile.set_ylabel('% committed cells')
axProfile.legend(fontsize=9)
fig.suptitle(f'Transplant vs. random-clamp prepattern (screen={args.fieldScreenSize:g}, strength={args.fieldStrength:g})', fontsize=12)
fig.tight_layout()
outputPath = f'figures/transplantVsRandomClampPrepattern_strength{args.fieldStrength:g}.png'
fig.savefig(outputPath, dpi=140, bbox_inches='tight')
print(f"\nwrote {outputPath}")
