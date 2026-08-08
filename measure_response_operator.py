"""
Measure the boundary-to-interior response operator directly, with no observational fitting.

The fitted-direction test in validate_ct_causally.py inherits the weakness it was meant to check:
its intervention directions come from the same regression whose validity is in question, so a
poor result cannot distinguish a tissue that resists control from a regression that guessed wrong.
This measures the operator instead of fitting it.

Perturb one boundary cell's pre-pattern G_pol, restart, and record how the interior moves. Repeat
for every boundary cell, in both directions. The result is the response matrix R, whose column i
is the interior displacement per unit conductance change at boundary cell i -- a directly measured
causal object. Its singular value spectrum is then the empirical analogue of a controllability
Gramian: the singular vectors are the boundary patterns ordered by how much interior response they
produce, the singular values are the gains, and the number of singular values standing above the
noise floor is a *measured* count of independently controllable directions, to be compared against
the count CT infers by summing per-mode R^2.

Two properties are checked rather than assumed, because a linear operator only exists if they hold.
Opposite perturbations of equal size should produce opposite responses; if they do not, the tissue
is being pushed across a basin boundary and no linear response is defined there. And doubling the
step should double the response. Both are reported per cell, and the amplitude is chosen small
enough that they hold.
"""

import argparse, gc
import numpy as np
import torch
import utilities
from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--sourceDat',    type=str,   default='data/StigmergicModelParameters_30x30.dat')
parser.add_argument('--fieldScreenSize', type=float, default=4)
parser.add_argument('--numSimIters',  type=int,   default=2500)
parser.add_argument('--clampIters',   type=int,   default=100)
parser.add_argument('--readoutIters', type=int,   default=200)
parser.add_argument('--clampSeed',    type=int,   default=7)
parser.add_argument('--stepFraction', type=float, default=0.05,
                    help='per-cell G_pol step, as a fraction of G_ref')
parser.add_argument('--cellSubset',   type=str,   default='all',
                    help='"all", or "i:j" to measure a slice of boundary cells (for array jobs)')
parser.add_argument('--linearityCells', type=int, default=4,
                    help='cells on which to also run a doubled step, to check linearity')
parser.add_argument('--outputPrefix', type=str,   default='data/responseOperator')
args = parser.parse_args()
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


def clampedToPrePattern(seed):
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


def freeEvolveFrom(vmemPre, gpolPre):
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
    return (preceding.sum(axis=0) + final) / args.readoutIters * 1000.0


def interiorOf(pattern):
    """Interior readout, centred on its own spatial mean to match the CX/CT convention."""
    values = pattern[interiorMask]
    return values - values.mean()


vmemBase, gpolBase = clampedToPrePattern(args.clampSeed)
baseInterior = interiorOf(freeEvolveFrom(vmemBase, gpolBase))
step = args.stepFraction * G_ref

if args.cellSubset == 'all':
    subset = np.arange(len(boundaryIndices))
else:
    lo, hi = (int(x) for x in args.cellSubset.split(':'))
    subset = np.arange(lo, min(hi, len(boundaryIndices)))

print(f"{numCells} cells | fieldScreenSize {args.fieldScreenSize} | "
      f"{len(boundaryIndices)} boundary cells, measuring {len(subset)} | "
      f"step {args.stepFraction} G_ref")

def displacementFor(cell, magnitude):
    perturbed = gpolBase.clone()
    perturbed[0, boundaryIndices[cell], 0] += magnitude
    perturbed = torch.clip(perturbed, 0.0, 2.0 * G_ref)
    delivered = (perturbed - gpolBase)[0, boundaryIndices[cell], 0].item()
    return delivered, interiorOf(freeEvolveFrom(vmemBase, perturbed)) - baseInterior

columns, deliveredPlus, symmetry, linearity = {}, {}, {}, {}
for position, cell in enumerate(subset):
    plusDelivered, plusResponse = displacementFor(cell, +step)
    minusDelivered, minusResponse = displacementFor(cell, -step)
    # A linear operator requires opposite steps to give opposite responses. Where they do not, the
    # perturbation has crossed a basin boundary and no local linear response exists.
    denominator = np.linalg.norm(plusResponse) + np.linalg.norm(minusResponse)
    symmetry[cell] = float(np.linalg.norm(plusResponse + minusResponse) / denominator) \
        if denominator > 0 else np.nan
    scale = plusDelivered - minusDelivered
    columns[cell] = (plusResponse - minusResponse) / scale if scale != 0 else plusResponse * 0
    deliveredPlus[cell] = plusDelivered / G_ref
    if position < args.linearityCells:
        _, doubleResponse = displacementFor(cell, +2 * step)
        expected = 2 * np.linalg.norm(plusResponse)
        linearity[cell] = float(np.linalg.norm(doubleResponse) / expected) if expected > 0 else np.nan
    print(f"  cell {cell:>3} | delivered {plusDelivered/G_ref:>+7.4f} G_ref | "
          f"response {np.linalg.norm(plusResponse):>8.3f} mV | asymmetry {symmetry[cell]:>6.3f}"
          + (f" | doubling gives {linearity[cell]:>5.2f}x of 2x" if cell in linearity else ""))

np.savez(f'{args.outputPrefix}_screen{args.fieldScreenSize}_{args.cellSubset.replace(":","-")}.npz',
         columns=np.array([columns[c] for c in subset]), cells=subset,
         symmetry=np.array([symmetry[c] for c in subset]),
         delivered=np.array([deliveredPlus[c] for c in subset]),
         linearity={int(k): v for k, v in linearity.items()},
         baseInterior=baseInterior, stepFraction=args.stepFraction)
print(f"Saved. mean asymmetry {np.nanmean(list(symmetry.values())):.3f}, "
      f"mean response {np.mean([np.linalg.norm(columns[c]) for c in subset]) * step:.3f} mV")
