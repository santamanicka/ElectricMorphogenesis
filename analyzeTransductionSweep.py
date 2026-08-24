"""Does lowering fieldTransductionWeight let the 30x30 bulk carry graded, depth-modulated -- or even
bipolar -- information, instead of instantly saturating to a single pole?

The G_pol update is dp = 10*(-G_pol + (2*sigmoid(gain*eV+bias)-1)*weight)/timeConstant, and G_pol is
O(G_ref) while weight=1000 -- the "-G_pol" restoring term is nine orders of magnitude smaller than
the term it is meant to balance, so this behaves as an unbalanced push rather than a true relaxation.
At the stored weight/timeConstant, a single iteration with a moderately non-saturated sigmoid moves
G_pol several times its own valid range before clipping stops it, which is why depth was only ever
observed to change how MANY iterations until arrival, never which pole is reached. Lowering weight
(holding timeConstant fixed, since the two enter as a ratio) should let cells farther from the
boundary end the 100-iteration window still in transit rather than already clipped, and -- if the
sign of the driving signal genuinely varies with depth or position rather than being fixed by bias
everywhere -- should let some of them commit to the other pole.
"""
import numpy as np
import torch

from embryo import model

torch.set_grad_enabled(False)


def depthShell(rows, cols):
    r, c = np.indices((rows, cols))
    return np.minimum(np.minimum(r, rows - 1 - r), np.minimum(c, cols - 1 - c)).reshape(-1)


def prepattern(parameterfile, weight):
    parameters = torch.load(parameterfile, weights_only=False)
    parameters['fieldParameters'] = dict(parameters['fieldParameters'])
    parameters['fieldParameters']['fieldTransductionWeight'] = torch.DoubleTensor([weight])
    rows, cols = parameters['latticeDims']
    numCells = rows * cols
    numSamples = parameters['simParameters']['numSamples']
    initialValues = parameters['simParameters']['initialValues']
    if 'ligandConc' not in initialValues:
        initialValues['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['ATPParameters'] = None
    clampParameters = dict(parameters['clampParameters'])
    clampEnd = int(clampParameters['clampEndIter'])

    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValues, numSamples))
    circuit = instance.electricNetwork
    instance.simulate(clampParameters=clampParameters, fieldModulation=True,
                       numSimIters=clampEnd + 1, storeVariables=['Vmem'])
    gpol = (circuit.G_pol.detach().clone().reshape(-1) / circuit.G_ref).numpy()
    return gpol, rows, cols


path = './data/bestModelParameters_fieldVector_30x30_616.dat'
weights = [1000, 300, 100, 30, 10, 3, 1]

print(f"  {'weight':>7s} {'interior near-0':>16s} {'interior near-2':>16s} {'uncommitted':>12s} "
      f"{'open shells':>12s} {'first open':>11s}", flush=True)
for weight in weights:
    gpol, rows, cols = prepattern(path, weight)
    depth = depthShell(rows, cols)
    maxDepth = depth.max()
    interior = depth > 0
    gi = gpol[interior]
    near0 = 100 * (gi < 0.2).mean()
    near2 = 100 * (gi > 1.8).mean()
    uncommitted = 100 - near0 - near2
    openShells = sum(1 for d in range(1, maxDepth + 1)
                      if (((gpol[depth == d] >= 0.2) & (gpol[depth == d] <= 1.8)).mean() > 0.5))
    firstOpen = next((d for d in range(1, maxDepth + 1)
                       if ((gpol[depth == d] >= 0.2) & (gpol[depth == d] <= 1.8)).mean() > 0.5), None)
    print(f"  {weight:7.0f} {near0:15.1f}% {near2:15.1f}% {uncommitted:11.1f}% "
          f"{openShells:9d}/{maxDepth} {str(firstOpen):>11s}", flush=True)
