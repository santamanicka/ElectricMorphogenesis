"""How much does the readout move when the clamp is perturbed, as a function of action range?

Backward inversion says the face becomes far more reachable as the screen widens, and forward
training says the opposite: at screen 24 both objectives barely move off their starting loss, while
screen 10 improves fifteen to twenty five percent. Reachability and trainability point in opposite
directions, which needs an explanation rather than a story.

The candidate is that a wide action range flattens the landscape. If every cell writes across the
whole field then changing one clamp point smears over the whole tissue, and the readout stops
depending on the particular structure of the clamp even though a wide range of patterns remains
attainable. That predicts the response to a clamp perturbation falls as the screen grows, which is
what this measures: perturb the clamp, run the full horizon, and see how far the final pattern moves.
"""
import argparse

import numpy as np
import torch

from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--parameterfile', default='./data/bestModelParameters_fieldVector_30x30_616.dat')
parser.add_argument('--screens', type=int, nargs='+', default=[4, 10, 24])
parser.add_argument('--numDirections', type=int, default=3)
parser.add_argument('--perturbation', type=float, default=1e-3,
                    help='clamp perturbation as a fraction of the clamp value spread')
args = parser.parse_args()

base = torch.load(args.parameterfile, weights_only=False)
numCells = base['latticeDims'][0]*base['latticeDims'][1]

def finalPattern(screen, clampValues):
    parameters = {k: v for k, v in base.items()}
    parameters['fieldParameters'] = dict(base['fieldParameters'])
    parameters['fieldParameters']['fieldScreenSize'] = screen
    parameters['clampParameters'] = dict(base['clampParameters'])
    parameters['clampParameters']['clampValues'] = clampValues
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['ATPParameters'] = None
    numSamples = parameters['simParameters']['numSamples']
    initialValues = parameters['simParameters']['initialValues']
    if 'ligandConc' not in initialValues:
        initialValues['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)
    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValues, numSamples))
    instance.simulate(clampParameters=parameters['clampParameters'], fieldModulation=True,
                      numSimIters=parameters['simParameters']['numSimIters'],
                      storeVariables=['Vmem'])
    return instance.electricNetwork.Vmem.detach().clone().reshape(-1)*1000.0

clamp = base['clampParameters']['clampValues']
spread = float(clamp.std())
generator = torch.Generator().manual_seed(0)
print(f"  clamp shape {tuple(clamp.shape)}, spread {spread:.4g}; "
      f"perturbing by {args.perturbation:g} of that\n")
print(f"  {'screen':>7s} {'response (mV)':>15s} {'per unit clamp change':>23s}")
for screen in args.screens:
    reference = finalPattern(screen, clamp)
    responses = []
    for _ in range(args.numDirections):
        direction = torch.randn(clamp.shape, generator=generator, dtype=clamp.dtype)
        direction = direction/direction.std()
        perturbed = clamp + args.perturbation*spread*direction
        responses.append(float(((finalPattern(screen, perturbed) - reference)**2).mean().sqrt()))
    response = float(np.mean(responses))
    print(f"  {screen:7d} {response:15.5f} {response/(args.perturbation*spread):23.3f}")
