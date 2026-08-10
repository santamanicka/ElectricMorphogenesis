"""Walk backwards from the target one iteration at a time and record how well each step can be solved.

Every attempt so far reports a single number at the end -- the target was not reached -- and says
nothing about where the obstruction lies. Solving backwards localises it. Given a desired state, the
question at each step is whether some earlier state maps onto it under one forward iteration; the
residual left over is how far the best available predecessor falls short. Residuals near zero mean
the target is reachable at least that far back. A residual that jumps means the chain has left the
set of states the dynamics can produce, and the step at which it jumps is the measurement.

The forward map contracts, at roughly 0.81 per step, so its image shrinks and not every state has a
predecessor. That is the reason to expect a jump rather than a clean chain, and the reason the
residual is worth recording per step rather than only at the end.

The state is Vmem, the extracellular field, the polarising conductance and the ligand concentration.
Only Vmem is specified by the target, so the first step leaves the other three free and later steps
match the full state that the previous step settled on. Components are compared relative to their own
scale, since a millivolt of Vmem and a unit of conductance are not the same size of error.
"""
import argparse

import numpy as np
import torch

from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--parameterfile', default='./data/bestModelParameters_fieldVector_30x30_616.dat')
parser.add_argument('--numBackSteps', type=int, default=100)
parser.add_argument('--innerIters', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
# Rprop is what the training loop uses, and it suits this problem for the same reason: it follows the
# sign of the gradient with a step size adapted per parameter, so components differing by eight orders
# of magnitude need no rescaling by hand. Adam is kept available since the two disagreeing would say
# the residual is the solver's limit rather than the problem's answer.
parser.add_argument('--optimiser', choices=('adam', 'rprop'), default='adam')
parser.add_argument('--output', default='./data/backwardReachability.npz')
args = parser.parse_args()

parameters = torch.load(args.parameterfile, weights_only=False)
latticeDims = parameters['latticeDims']
numCells = latticeDims[0]*latticeDims[1]
numSamples = parameters['simParameters']['numSamples']
initialValues = parameters['simParameters']['initialValues']
if 'ligandConc' not in initialValues:
    initialValues['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)
parameters['latticePeriodicBoundaryGJ'] = False
parameters['ATPParameters'] = None
numSimIters = parameters['simParameters']['numSimIters']
# the stored target is float32 while the model runs in double, so it is cast rather than
# left to fail inside the field update several frames deep
target = parameters['trainParameters']['targetVmem'].detach().to(torch.float64).clone()

# a real trajectory supplies physically plausible values for the state components the target leaves
# unspecified, so the walk starts from a state the dynamics could actually be in apart from its Vmem
instance = model(parameters, numSamples)
instance.setExperimentalConditions((initialValues, numSamples))
instance.simulate(clampParameters=parameters['clampParameters'], fieldModulation=True,
                  numSimIters=numSimIters, storeVariables=['Vmem'])
circuit = instance.electricNetwork
print(f"  {latticeDims} lattice, screen {parameters['fieldParameters']['fieldScreenSize']}, "
      f"reference trajectory run to iteration {numSimIters}")

# ligandConc is left out: ligand is disabled in these runs, so it holds zero throughout and
# normalising by its own spread divides by nothing
STATE = ('Vmem', 'eV', 'G_pol')
desired = {name: getattr(circuit, name).detach().clone() for name in STATE}
desired['Vmem'] = target.clone()                      # the face, in place of whatever the run produced

def scalesOf(state):
    """Each component's own spread. Vmem, the field and the conductance differ by eight orders of
    magnitude, so both the mismatch and the search step have to be expressed relative to these or a
    single learning rate is meaninglessly large for one component and negligible for another."""
    return {name: max(float(value.std()), 1e-30) for name, value in state.items()}

def detachStaleIntermediates(keep):
    """Drop autograd history the circuit is still holding from the previous inner iteration.

    A step leaves graph-connected tensors on the circuit beyond the four state variables -- the
    conductance increment, the screened field average and so on -- and the next iteration would try
    to backward through a graph that has already been freed.
    """
    for name, value in list(circuit.__dict__.items()):
        if torch.is_tensor(value) and value.grad_fn is not None and name not in keep:
            setattr(circuit, name, value.detach())

def stepOnce(state):
    detachStaleIntermediates(set(state))
    for name, value in state.items():
        setattr(circuit, name, value)
    circuit.simulate(numSimIters=1, fieldModulation=True, saveData=False)
    return {name: getattr(circuit, name) for name in STATE}

def mismatch(produced, wanted, scale, components):
    """Mismatch over the components that are actually specified.

    The target fixes Vmem and nothing else, so at the first step back the field and the conductance
    are free: any values the dynamics would produce alongside the right Vmem are acceptable. Holding
    them to a reference trajectory's values instead asks for the face to be reached with one
    particular field and conductance, a different and much harder question than whether the face is
    reached at all. Every later step matches the full state, because there the thing being reproduced
    is a specific predecessor the previous step settled on.
    """
    return sum(((produced[n] - wanted[n])**2).mean()/(scale[n]**2) for n in components)

residuals, vmemResiduals = [], []
for backStep in range(args.numBackSteps):
    constrained = ('Vmem',) if backStep == 0 else STATE
    scale = scalesOf(desired)
    offsets = {n: torch.zeros_like(desired[n], requires_grad=True) for n in STATE}
    optimiser = (torch.optim.Rprop(list(offsets.values()), lr=args.lr) if args.optimiser == 'rprop'
                 else torch.optim.Adam(list(offsets.values()), lr=args.lr))
    for _ in range(args.innerIters):
        optimiser.zero_grad()
        guess = {n: desired[n] + scale[n]*offsets[n] for n in STATE}
        loss = mismatch(stepOnce(guess), desired, scale, constrained)
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        guess = {n: (desired[n] + scale[n]*offsets[n]).detach() for n in STATE}
        produced = stepOnce(guess)
        residuals.append(float(mismatch(produced, desired, scale, constrained).sqrt()))
        vmemResiduals.append(float(((produced['Vmem'] - desired['Vmem'])**2).mean().sqrt())*1000)
        desired = {n: guess[n].detach().clone() for n in STATE}
    if backStep < 20 or (backStep+1) % 10 == 0:
        print(f"    back step {backStep+1:4d}: residual {residuals[-1]:.3e}, "
              f"Vmem residual {vmemResiduals[-1]:.4f} mV")

np.savez(args.output, residual=np.array(residuals), vmemResidual=np.array(vmemResiduals))
print(f"\n  first residual {residuals[0]:.3e}, last {residuals[-1]:.3e}, "
      f"max {max(residuals):.3e} at step {int(np.argmax(residuals))+1}")
print(f"  wrote {args.output}")
