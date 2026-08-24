"""Does clamp amplitude change how much of the 30x30 tissue's depth stays uncommitted?

The aggregate amplitude sweep showed interior near-0 jumping from 73.5% at 1x to 100% by 3x, which
would mean the uncommitted fraction COLLAPSES rather than grows with amplitude -- the opposite of
what would help. This resolves that by depth shell: for each amplitude, what fraction of shells (and
of total interior cells) remain majority uncommitted (mid-range) at the pre-pattern step.
"""
import numpy as np
import torch

from embryo import model

torch.set_grad_enabled(False)


def depthShell(rows, cols):
    r, c = np.indices((rows, cols))
    return np.minimum(np.minimum(r, rows-1-r), np.minimum(c, cols-1-c)).reshape(-1)


def prepattern(parameterfile, amplitude):
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
    instance.simulate(clampParameters=clampParameters, fieldModulation=True,
                      numSimIters=clampEnd+1, storeVariables=['Vmem'])
    gpol = (circuit.G_pol.detach().clone().reshape(-1)/circuit.G_ref).numpy()
    return gpol, rows, cols


path = './data/bestModelParameters_fieldVector_30x30_616.dat'
amplitudes = [0.3, 1, 2, 3, 5, 10]

for amplitude in amplitudes:
    gpol, rows, cols = prepattern(path, amplitude)
    depth = depthShell(rows, cols)
    maxDepth = depth.max()
    interior = depth > 0  # excludes the boundary ring, matching the earlier "interior" definition
    gi = gpol[interior]
    uncommittedCells = ((gi >= 0.2) & (gi <= 1.8)).mean()
    # a shell counts as "still open" if the majority of its cells are mid-range
    openShells = sum(1 for d in range(1, maxDepth+1)
                     if (((gpol[depth == d] >= 0.2) & (gpol[depth == d] <= 1.8)).mean() > 0.5))
    firstOpenDepth = next((d for d in range(1, maxDepth+1)
                          if ((gpol[depth == d] >= 0.2) & (gpol[depth == d] <= 1.8)).mean() > 0.5), None)
    print(f"  amplitude {amplitude:6.2f}: interior uncommitted {100*uncommittedCells:5.1f}%   "
          f"open shells {openShells}/{maxDepth} (of interior)   first open at depth {firstOpenDepth}")
