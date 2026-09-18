"""Run the state the backward solve inferred forward, and look at how it becomes the target.

The joint solve reports a residual and nothing else, so what it found stays invisible. This takes the
saved state, runs it forward for exactly the horizon it was solved over, and draws the trajectory.
That state is not produced by a clamp: it is a full specification of Vmem, the field and the
conductance at some moment, which is why it answers whether the target is reachable at all rather
than whether a boundary signal can induce it.

The final frame should match the target to within the residual the solve reported. The frames before
it show what a pre-pattern for this target looks like, which is the part worth seeing.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--stateFile', default='./data/inferredSeed_screen24.npz')
parser.add_argument('--parameterfile', default='./data/bestModelParameters_fieldVector_30x30_616.dat')
parser.add_argument('--output', default='./figures/inferredSeedEvolution.png')
args = parser.parse_args()

saved = np.load(args.stateFile)
jointSteps = int(saved['jointSteps'])
screen = int(saved['fieldScreenSize'])
residual = float(saved['vmemResidual'])
target = saved['target'].reshape(-1)*1000.0

parameters = torch.load(args.parameterfile, weights_only=False)
parameters['fieldParameters'] = dict(parameters['fieldParameters'])
parameters['fieldParameters']['fieldScreenSize'] = screen
parameters['latticePeriodicBoundaryGJ'] = False
parameters['ATPParameters'] = None
rows, cols = parameters['latticeDims']
numCells = rows*cols
numSamples = parameters['simParameters']['numSamples']
initialValues = parameters['simParameters']['initialValues']
if 'ligandConc' not in initialValues:
    initialValues['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)

instance = model(parameters, numSamples)
instance.setExperimentalConditions((initialValues, numSamples))
circuit = instance.electricNetwork
for name in ('Vmem', 'eV', 'G_pol'):
    setattr(circuit, name, torch.tensor(saved[f'solved_{name}'], dtype=torch.float64))

# no clamp: the inferred state is the whole instruction, which is the point of the exercise
frames = [circuit.Vmem.detach().clone().reshape(-1).numpy()*1000.0]
for _ in range(jointSteps):
    circuit.simulate(numSimIters=1, fieldModulation=True, saveData=False)
    frames.append(circuit.Vmem.detach().clone().reshape(-1).numpy()*1000.0)
print(f"  screen {screen}, {jointSteps} steps, solve residual {residual:.4f} mV")
print(f"  final frame vs target: {np.sqrt(((frames[-1]-target)**2).mean()):.4f} mV")

show = [0] + [round(jointSteps*k/6) for k in range(1, 7)]
allValues = np.concatenate(frames + [target])
vmin, vmax = np.percentile(allValues, 0.5), np.percentile(allValues, 99.5)
fig, axes = plt.subplots(1, len(show)+1, figsize=(2.2*(len(show)+1), 2.7))
axes[0].imshow(target.reshape(rows, cols), cmap='gray', vmin=vmin, vmax=vmax)
axes[0].set_title('target', fontsize=9)
for k, step in enumerate(show):
    image = frames[min(step, len(frames)-1)].reshape(rows, cols)
    axes[k+1].imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    rms = np.sqrt(((image.ravel()-target)**2).mean())
    label = 'inferred seed' if step == 0 else f'+{step} steps'
    axes[k+1].set_title(f'{label}\n{rms:.2f} mV to target', fontsize=9)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f'State inferred {jointSteps} steps before the target at screen {screen}, run forward '
             f'(no clamp)', fontsize=11)
fig.tight_layout()
fig.savefig(args.output, dpi=150)
print(f"  wrote {args.output}")
