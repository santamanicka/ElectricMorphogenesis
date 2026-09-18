"""How far the clamp moves interior G_pol, against the scale at which a cell commits.

The interior receives a near-scalar echo of the boundary code -- participation ratio 1.6, most of its
variance in a single global mode -- and at 11x11 that is enough: the tissue elaborates it into a
twelve dimensional Vmem pattern, and restoring it reproduces the outcome exactly. So the question for
a larger tissue is not whether the bulk is seeded sharply but whether it is seeded at all, in the
sense of crossing the bifurcation that lets the tissue's own dynamics take hold.

G_pol starts uniform at G_ref and is clipped to [0, 2*G_ref], so a cell that fully commits moves by
one G_ref unit in either direction. That is the scale every displacement below is measured against.
A bulk sitting orders of magnitude under it is being nudged rather than instructed, and no amount of
elaboration downstream can act on an instruction that never arrived.

Boundary and interior are separated because the clamp writes to the rim by construction; the interior
figure is the one that carries the argument.
"""
import argparse

import numpy as np
import torch

from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+',
                    default=['./data/StigmergicModelParameters.dat',
                             './data/bestModelParameters_fieldVector_30x30_616.dat'])
parser.add_argument('--labels', nargs='+', default=['11x11 reference', '30x30 clamp-only'])
parser.add_argument('--amplitude', type=float, default=1.0,
                    help='scale applied to clampValues, to sweep how hard the clamp writes')
args = parser.parse_args()


def interiorMask(rows, cols):
    mask = np.ones((rows, cols), bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    return mask.reshape(-1)


def seeding(parameterfile, amplitude):
    parameters = torch.load(parameterfile, weights_only=False)
    rows, cols = parameters['latticeDims']
    numCells = rows*cols
    numSamples = parameters['simParameters']['numSamples']
    initialValues = parameters['simParameters']['initialValues']
    if 'ligandConc' not in initialValues:
        initialValues['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['ATPParameters'] = None
    clampParameters = dict(parameters['clampParameters'])
    clampParameters['clampValues'] = clampParameters['clampValues']*amplitude
    clampEnd = int(clampParameters['clampEndIter'])

    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValues, numSamples))
    circuit = instance.electricNetwork
    reference = circuit.G_pol.detach().clone().reshape(-1)

    # the pre-pattern step is the first free-evolution iteration, clampEndIter + 1
    instance.simulate(clampParameters=clampParameters, fieldModulation=True,
                      numSimIters=clampEnd+1, storeVariables=['Vmem'])
    prepattern = circuit.G_pol.detach().clone().reshape(-1)

    displacement = ((prepattern - reference)/circuit.G_ref).numpy()
    return dict(rows=rows, cols=cols, screen=parameters['fieldParameters']['fieldScreenSize'],
                clampEnd=clampEnd, displacement=displacement,
                interior=interiorMask(rows, cols))


print(f"  clamp amplitude scale: {args.amplitude:g}")
print(f"  displacements in G_ref units; a cell that fully commits moves 1.0\n")
print(f"  {'model':22s} {'screen':>7s} {'boundary mean':>14s} {'interior mean':>14s} "
      f"{'interior 95th':>14s} {'interior >0.1':>14s}")
for parameterfile, label in zip(args.models, args.labels):
    result = seeding(parameterfile, args.amplitude)
    magnitude = np.abs(result['displacement'])
    interior = magnitude[result['interior']]
    boundary = magnitude[~result['interior']]
    print(f"  {label:22s} {result['screen']:7d} {boundary.mean():14.5f} {interior.mean():14.5f} "
          f"{np.percentile(interior, 95):14.5f} {100*(interior > 0.1).mean():13.1f}%")
