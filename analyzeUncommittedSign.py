"""Does the uncommitted (mid-range) interior population have a consistent sign of displacement, or
is "uncommitted" hiding a genuine split that the near-0/near-2 thresholds are too coarse to see?

The earlier weight sweep only reported near-0/near-2/uncommitted percentages from fixed thresholds
(<0.2, >1.8) and never printed the mean of the uncommitted population itself. "100% uncommitted"
is consistent with three different situations that this script distinguishes: cells sitting at
their 1.0 starting value doing nothing, cells uniformly drifting toward 0, or a real mix of cells
drifting toward 0 and toward 2 that never individually crosses the threshold. Only the signed mean
and, critically, the SPREAD of individual cell values (not just their average) can tell those apart
-- a mix with cancelling signs and a population that hasn't moved at all can share the same mean.
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

print("  interior displacement from the 1.0 starting value: g = G_pol/G_ref - 1", flush=True)
print("  g < 0 means drifting toward the depolarising pole (0), g > 0 toward the hyperpolarising pole (2)")
print(f"\n  {'weight':>7s} {'mean g':>9s} {'g<-0.01':>9s} {'g>+0.01':>9s} {'|g|<0.01':>9s} "
      f"{'min g':>9s} {'max g':>9s}", flush=True)
for weight in weights:
    gpol, rows, cols = prepattern(path, weight)
    depth = depthShell(rows, cols)
    interior = depth > 0
    g = gpol[interior] - 1.0
    negative = 100 * (g < -0.01).mean()
    positive = 100 * (g > 0.01).mean()
    unmoved = 100 - negative - positive
    print(f"  {weight:7.0f} {g.mean():9.5f} {negative:8.1f}% {positive:8.1f}% {unmoved:8.1f}% "
          f"{g.min():9.5f} {g.max():9.5f}", flush=True)

print("\n  same breakdown by depth shell, at weight=10 (well inside the 'fully uncommitted' regime)")
gpol, rows, cols = prepattern(path, 10)
depth = depthShell(rows, cols)
print(f"  {'depth':>6s} {'cells':>6s} {'mean g':>9s} {'g<-0.01':>9s} {'g>+0.01':>9s} {'min g':>9s} {'max g':>9s}",
      flush=True)
for d in range(1, depth.max() + 1):
    g = gpol[depth == d] - 1.0
    negative = 100 * (g < -0.01).mean()
    positive = 100 * (g > 0.01).mean()
    print(f"  {d:6d} {int((depth==d).sum()):6d} {g.mean():9.5f} {negative:8.1f}% {positive:8.1f}% "
          f"{g.min():9.5f} {g.max():9.5f}", flush=True)
