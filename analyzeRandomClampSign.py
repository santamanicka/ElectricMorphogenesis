"""Does the interior's uniformly negative displacement hold for random clamps too, or is it specific
to this one trained clamp?

Every signed-displacement check so far used the single trained clamp in bestModelParameters_
fieldVector_30x30_616.dat. A trained clamp could have converged on a systematic bias the loss
happens to reward; an untrained, structurally unrelated clamp is the control that would show
whether the one-directional pull is a property of the tissue and field geometry, or an artefact of
this particular optimised signal.

Five independent random clamps are drawn (matching clampValues' shape, drawn i.i.d. standard normal
-- not matched to the trained clamp's own spread, since the point is a structurally different signal,
not a rescaled one), each run at the trained weight (1000, where the earlier depth table showed a
clean gradient) and at weight 10 (where saturation is stripped away and the gradient is easiest to
see).
"""
import numpy as np
import torch

from embryo import model

torch.set_grad_enabled(False)


def depthShell(rows, cols):
    r, c = np.indices((rows, cols))
    return np.minimum(np.minimum(r, rows - 1 - r), np.minimum(c, cols - 1 - c)).reshape(-1)


def prepattern(parameterfile, weight, seed):
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
    generator = torch.Generator().manual_seed(seed)
    clampParameters['clampValues'] = torch.randn(clampParameters['clampValues'].shape,
                                                  generator=generator,
                                                  dtype=clampParameters['clampValues'].dtype)
    clampEnd = int(clampParameters['clampEndIter'])

    instance = model(parameters, numSamples)
    instance.setExperimentalConditions((initialValues, numSamples))
    circuit = instance.electricNetwork
    instance.simulate(clampParameters=clampParameters, fieldModulation=True,
                       numSimIters=clampEnd + 1, storeVariables=['Vmem'])
    gpol = (circuit.G_pol.detach().clone().reshape(-1) / circuit.G_ref).numpy()
    return gpol, rows, cols


path = './data/bestModelParameters_fieldVector_30x30_616.dat'
seeds = [1, 2, 3, 4, 5]

for weight in (1000, 10):
    print(f"\n  ===== weight = {weight} =====", flush=True)
    print(f"  {'seed':>5s} {'mean g':>9s} {'g<-0.01':>9s} {'g>+0.01':>9s} {'min g':>9s} {'max g':>9s}",
          flush=True)
    for seed in seeds:
        gpol, rows, cols = prepattern(path, weight, seed)
        depth = depthShell(rows, cols)
        interior = depth > 0
        g = gpol[interior] - 1.0
        negative = 100 * (g < -0.01).mean()
        positive = 100 * (g > 0.01).mean()
        print(f"  {seed:5d} {g.mean():9.5f} {negative:8.1f}% {positive:8.1f}% "
              f"{g.min():9.5f} {g.max():9.5f}", flush=True)

print("\n  depth-resolved, seed 1, weight 10 (clean-gradient regime)", flush=True)
gpol, rows, cols = prepattern(path, 10, 1)
depth = depthShell(rows, cols)
print(f"  {'depth':>6s} {'cells':>6s} {'mean g':>9s} {'min g':>9s} {'max g':>9s}", flush=True)
for d in range(1, depth.max() + 1):
    g = gpol[depth == d] - 1.0
    print(f"  {d:6d} {int((depth==d).sum()):6d} {g.mean():9.5f} {g.min():9.5f} {g.max():9.5f}",
          flush=True)
