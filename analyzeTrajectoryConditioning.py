"""Measure whether staged (multiple-shooting) inversion is viable on this model.

Two quantities decide it. How fast a perturbation grows with horizon sets the longest stage whose
gradients still carry signal: backpropagating through the full trajectory fails because the map
hashes, and staging only helps if some shorter horizon is better conditioned. Where the trajectory
actually moves sets where the stages should go, since a segment that has already equilibrated
carries no information about its own predecessor and a stage spent there is wasted.

Perturbations are applied to the initial Vmem and tracked against an unperturbed reference, so the
growth curve is measured on the model's own trajectory rather than on a linearisation of it.
"""
import argparse

import numpy as np
import torch
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--parameterfile', default='./data/bestModelParameters_fieldVector_24.dat')
parser.add_argument('--output', default='./data/trajectoryConditioning.npz',
                    help='written only if it does not already exist, unless --overwrite is given')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--numSimIters', type=int, default=None,
                    help='override the stored simulation length, to see past the trained horizon')
parser.add_argument('--numPerturbationDirections', type=int, default=4)
args = parser.parse_args()

parameterfilename = args.parameterfile
numPerturbationDirections = args.numPerturbationDirections
perturbationSize = 1e-6          # volts, small enough to stay in the linear regime

parameters = torch.load(parameterfilename, weights_only=False)
latticeDims = parameters['latticeDims']
numCells = latticeDims[0] * latticeDims[1]
numSamples = parameters['simParameters']['numSamples']
initialValues = parameters['simParameters']['initialValues']
if 'ligandConc' not in initialValues:
    initialValues['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)
clampParameters = parameters['clampParameters']
numSimIters = args.numSimIters or parameters['simParameters']['numSimIters']
parameters['latticePeriodicBoundaryGJ'] = False
parameters['ATPParameters'] = None
print(f"  {latticeDims} lattice, {numSamples} sample(s), {numSimIters} iterations\n")

def runTrajectory(vmemOffset=None):
    values = {k: v for k, v in initialValues.items()}
    if vmemOffset is not None:
        values['Vmem'] = initialValues['Vmem'] + vmemOffset
    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((values, numSamples))
    instance.simulate(clampParameters=clampParameters, fieldModulation=True,
                      numSimIters=numSimIters, storeVariables=['Vmem'])
    return torch.stack(list(instance.timeseriesVmem)).reshape(numSimIters, -1).numpy()

reference = runTrajectory()
print(f"  reference trajectory: {reference.shape[0]} steps x {reference.shape[1]} cells")

step = np.abs(np.diff(reference, axis=0)).mean(axis=1) * 1000
print(f"\n  where the trajectory moves -- mean |dVmem| per step, mV")
for t in (0, 10, 50, 100, 200, 400, 600, 800, numSimIters - 2):
    if t < len(step):
        print(f"    iter {t:5d}: {step[t]:.3e}")
settled = np.argmax(step < step[0] * 0.01) if (step < step[0]*0.01).any() else -1
print(f"    falls below 1% of its initial rate at iter {settled}"
      if settled > 0 else "    never falls below 1% of its initial rate")

growth = []
generator = np.random.default_rng(0)
for d in range(numPerturbationDirections):
    direction = generator.normal(size=(numSamples, numCells, 1))
    direction /= np.sqrt((direction ** 2).mean())
    offset = torch.tensor(direction * perturbationSize, dtype=torch.float64)
    diff = runTrajectory(offset) - reference
    growth.append(np.sqrt((diff ** 2).mean(axis=1)) / perturbationSize)
growth = np.array(growth).mean(axis=0)

print(f"\n  perturbation growth vs horizon (mean of {numPerturbationDirections} directions)")
print(f"    {'horizon':>9s} {'amplification':>14s} {'per step':>10s}")
for n in (1, 2, 5, 10, 20, 50, 100, 200, 500, numSimIters - 1):
    if n < len(growth):
        print(f"    {n:9d} {growth[n]:14.4g} {growth[n]**(1.0/n):10.4f}")
import os
if os.path.exists(args.output) and not args.overwrite:
    print(f"\n  {args.output} exists; not overwriting (pass --overwrite to replace it)")
else:
    np.savez(args.output, step=step, growth=growth)
    print(f"\n  wrote {args.output}")
