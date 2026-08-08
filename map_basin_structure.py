"""
Is the boundary-to-interior map a gain or a switch?

CT is fitted as a linear regression, which presumes the interior responds proportionally to the
boundary code. Three measurements say it does not. No linear response operator exists at the
readout horizon: opposite perturbations sometimes give identical responses. A dose-response sweep
along a real code direction stays linear only to about 0.1% of the separation between two clamps,
compresses through the next two decades, and then leaps -- at half the natural separation the
pattern moves 105 mV, comparable to its entire amplitude. And the underlying dynamics amplify a
single-cell perturbation through ten to thirteen doublings over one run.

Together those imply a different architecture: the code does not push the pattern, it selects which
attractor the pattern falls into. This maps that structure directly, by walking the boundary code
from one clamp to another in small steps and watching where the resulting pattern moves. A gain
gives even steps throughout. A switch gives plateaus -- stretches where the code changes and the
pattern does not -- separated by jumps at basin boundaries.

What this measures is the quantity CT cannot express. For a switching system the useful notion of
control is how many distinct outcomes the boundary can select and how reliably, not how many
millivolts of response it produces per unit of code. That is a count of basins, and it is what the
capacity-in-bits form of CT was implicitly reaching for while the variance form assumed a gain that
is not there.
"""

import argparse, gc
import numpy as np
import torch
import utilities
from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat',   type=str,   default='data/StigmergicModelParameters_30x30.dat')
parser.add_argument('--fieldScreenSize', type=float, default=4)
parser.add_argument('--numSteps',    type=int,   default=40)
parser.add_argument('--numSimIters', type=int,   default=2500)
parser.add_argument('--clampIters',  type=int,   default=100)
parser.add_argument('--readoutIters',type=int,   default=200)
parser.add_argument('--clampSeeds',  type=str,   default='(7,99)')
parser.add_argument('--jumpFactor',  type=float, default=5.0,
                    help='a step this many times the median counts as a basin transition')
parser.add_argument('--outputPrefix',type=str,   default='data/basinStructure')
args = parser.parse_args()
seedA, seedB = eval(args.clampSeeds)
torch.set_grad_enabled(False)
prePatternIter = args.clampIters + 1


def buildParameters():
    parameters = torch.load(args.sourceDat, weights_only=False)
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['fieldParameters']['fieldScreenSize'] = args.fieldScreenSize
    cells = parameters['latticeDims'][0] * parameters['latticeDims'][1]
    values = parameters['simParameters']['initialValues']
    if 'ligandConc' not in values:
        values['ligandConc'] = torch.zeros((1, cells, 1), dtype=torch.float64)
    return parameters


utils = utilities.utilities()
reference = model(buildParameters(), 1)
circuit = reference.electricNetwork
numCells, G_ref = circuit.numCells, circuit.G_ref
boundaryIndices = np.array(utils.computeDomeIndices(circuit, mode='tissue'))
boundaryMask = np.zeros(numCells, bool); boundaryMask[boundaryIndices] = True
interiorMask = ~boundaryMask
leftHalf = utils.computeDomeIndices(circuit, mode='field', region='leftHalf')
mirrored = utils.computeSymmetricalIndices(circuit, leftHalf, mode='field', symmetry='twofold')
allIndices = np.concatenate((leftHalf, mirrored))
_, uniqueIdx = np.unique(allIndices, return_index=True)
clampIndices = (np.zeros(len(allIndices[uniqueIdx]), dtype=int), allIndices[uniqueIdx])
numHalf = len(leftHalf)
timeIndices = torch.linspace(0, 0.5, args.clampIters + 1).view(-1, 1)
del reference; gc.collect()


def prePattern(seed):
    generator = torch.Generator().manual_seed(seed)
    frequencies = torch.rand(numHalf, generator=generator, dtype=torch.double) * 900 + 100
    phases = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 * torch.pi
    amplitudes = torch.rand(numHalf, generator=generator, dtype=torch.double) * 2 - 1
    clampValues = (torch.cos(timeIndices * torch.tile(frequencies, (2,)) + torch.tile(phases, (2,)))
                   * torch.tile(amplitudes, (2,)))[:, uniqueIdx]
    instance = model(buildParameters(), 1)
    instance.simulate(numSimIters=prePatternIter + 1, fieldModulation=True, perturbation=None,
                      clampParameters={'clampMode': 'fieldDomeTwoFoldSymmetry',
                                       'clampIndices': clampIndices, 'clampValues': clampValues,
                                       'clampStartIter': 0, 'clampEndIter': args.clampIters},
                      storeVariables=('Vmem', 'Gpol'))
    vmem = instance.timeseriesVmem[prePatternIter].clone()
    gpol = instance.timeseriesGpol[prePatternIter].clone()
    del instance; gc.collect()
    return vmem, gpol


def evolve(vmemPre, gpolPre):
    parameters = buildParameters()
    instance = model(parameters, 1)
    instance.setExperimentalConditions((parameters['simParameters']['initialValues'], 1))
    net = instance.electricNetwork
    initialValues = dict(parameters['simParameters']['initialValues'])
    initialValues['Vmem'] = vmemPre.clone().double()
    initialValues['eV'] = torch.zeros_like(net.eV)
    net.initVariables(initialValues)
    net.G_pol = gpolPre.clone().double()
    instance.simulate(externalInputs=parameters['simParameters']['externalInputs'],
                      clampParameters=None, perturbation=None,
                      numSimIters=args.numSimIters - prePatternIter, storeVariables=('Vmem',))
    preceding = instance.timeseriesVmem[-(args.readoutIters - 1):, 0, :, 0].detach().numpy()
    final = net.Vmem[0, :, 0].detach().numpy()
    del instance; gc.collect()
    values = ((preceding.sum(axis=0) + final) / args.readoutIters * 1000.0)[interiorMask]
    return values - values.mean()


vmemA, gpolA = prePattern(seedA)
_, gpolB = prePattern(seedB)
difference = gpolB - gpolA
naturalRMS = (difference[0, boundaryIndices, 0] / G_ref).pow(2).mean().sqrt().item()
print(f"{numCells} cells | fieldScreenSize {args.fieldScreenSize} | clamps {seedA} to {seedB} | "
      f"natural code separation {naturalRMS:.4f} G_ref per cell | {args.numSteps} steps")

patterns = [evolve(vmemA, gpolA)]
for fraction in np.linspace(0.0, 1.0, args.numSteps + 1)[1:]:
    gpol = gpolA.clone()
    gpol[0, boundaryIndices, 0] += difference[0, boundaryIndices, 0] * fraction
    patterns.append(evolve(vmemA, torch.clip(gpol, 0.0, 2.0 * G_ref)))
patterns = np.array(patterns)

consecutive = np.linalg.norm(np.diff(patterns, axis=0), axis=1)
median = float(np.median(consecutive))
transitions = consecutive > args.jumpFactor * median
# Basins are the runs between transitions: the code moves and the pattern does not.
basinCount = int(transitions.sum()) + 1
plateauSteps = consecutive[~transitions]
print(f"\n  median step {median:.2f} mV | largest {consecutive.max():.2f} mV "
      f"({consecutive.max()/max(median,1e-9):.0f}x median)")
print(f"  transitions above {args.jumpFactor:g}x median: {int(transitions.sum())} of {len(consecutive)}")
print(f"  implied basins along this path: {basinCount}")
print(f"  within a basin the pattern moves {plateauSteps.mean():.2f} mV per step; "
      f"across a transition {consecutive[transitions].mean() if transitions.any() else float('nan'):.2f} mV "
      f"({(consecutive[transitions].mean()/max(plateauSteps.mean(),1e-9)) if transitions.any() else float('nan'):.0f}x)")
print(f"  total path length {consecutive.sum():.1f} mV, of which "
      f"{100*consecutive[transitions].sum()/max(consecutive.sum(),1e-9):.0f}% is taken in transitions "
      f"occupying {100*transitions.mean():.0f}% of the code range")
np.savez(f'{args.outputPrefix}_screen{args.fieldScreenSize}.npz', patterns=patterns,
         consecutive=consecutive, transitions=transitions, naturalRMS=naturalRMS,
         basinCount=basinCount)
print(f"Saved {args.outputPrefix}_screen{args.fieldScreenSize}.npz")
