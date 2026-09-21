"""Whole Vmem trajectories for the free-or-new check (PolyPatterning_Sim.md, Section 12).

After release, a clamped tissue follows the free model's equations; the clamp only sets its state at the end of the
hold. So a pattern that looks new in a fixed readout window could be the free tissue's own pattern at another time.
This script records every iteration of:

  free            the reference checkpoint with no clamp, from its own initial state, for --freeIterations.
  freeShifted     the same, with every cell's initial Vmem raised by --shiftMilliVolts: a control that should rejoin
                  the free run's cycle at a different phase.
  freeRandom<k>   the same with a random initial state that has all 8 symmetries of the square, as every clamp here
                  does: Vmem drawn uniformly from --randomVmemRange (mV) and G_pol / G_ref from --randomGpolRange, one
                  value per class of cells the symmetries carry onto each other (21 classes on the 11 x 11 lattice).
  dial<value>     the ring held at one dial value for --holdIterations, then released.

With --runSet tilted, the runs behind the Tilted Dial report instead, written to data/boundaryRecurrence<checkpoint>
Hold<hold>Tilted/:

  tilt<phi>_<DC>_<G>  the ring held at DC + G cos(theta - phi) (--tiltCodes, phi in degrees) for --holdIterations.
  dial<value>         the dial alone at each tilt code's DC, its twin.
  free<Symmetry><k>   free runs from random starting states with only the symmetry a tilt code keeps: Mirror (left-right,
                      as a tilt at 0 degrees), Diagonal (about the diagonal through the top-right corner, as at 45 degrees)
                      or Asymmetric (none, as at 22.5 degrees); --numRandomStarts of each.

Every run but the first lasts --runIterations. Each run is one --taskIndex, written to
data/boundaryRecurrence<checkpoint>Hold<hold>/<name>.npz with Vmem (mV) at every iteration and G_pol / G_ref every
--conductanceStride iterations.
"""
import argparse
import os
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--holdIterations', type=int, default=301)
parser.add_argument('--freeIterations', type=int, default=50000)
parser.add_argument('--runIterations', type=int, default=20000)
parser.add_argument('--dialLevels', type=str, default='0,0.6,1.3')
parser.add_argument('--numRandomStarts', type=int, default=20)
parser.add_argument('--randomVmemRange', type=str, default='-55,-5')
parser.add_argument('--randomGpolRange', type=str, default='0,1.5')
parser.add_argument('--shiftMilliVolts', type=float, default=1.0)
parser.add_argument('--conductanceStride', type=int, default=10)
parser.add_argument('--runSet', type=str, default='dial', choices=['dial', 'tilted'])
parser.add_argument('--tiltCodes', type=str, default='0:0.52:0.2,0:0.6:0.18,45:0.84:0.32,22.5:0.74:0.46,45:0.82:0.16,45:0.76:0.14,'
                    '0:0.74:0.5,0:0.82:0.48,45:0.74:0.48,22.5:0.74:0.54,22.5:0.82:0.26,45:0.66:0.58,45:0.8:0.32,45:0.76:0.12,'
                    '0:0.76:0.5,45:0.78:0.3', help='phi:DC:G per tilt code; default the typical codes of the Tilted Dial gallery')
parser.add_argument('--taskIndex', type=int, default=None, help='run only this entry of the run list; default runs all')
args = parser.parse_args()

size = boundary.latticeRows
reference = boundary.loadCheckpoint(args.referenceCheckpoint)
outputDirectory = f"data/boundaryRecurrence{args.referenceCheckpoint}Hold{args.holdIterations}{'Tilted' if args.runSet == 'tilted' else ''}"
os.makedirs(outputDirectory, exist_ok=True)


def symmetryClasses(symmetry='Square'):
    """Label each cell by its class under a symmetry group: the smallest index among its images. Square: all 8
    symmetries of the square; Mirror: the left-right mirror; Diagonal: the mirror about the diagonal through the top-right
    corner; Asymmetric: none."""
    index = np.arange(size * size).reshape(size, size)
    images = {'Square': [np.rot90(index, turns) for turns in range(4)] + [np.rot90(index, turns)[:, ::-1] for turns in range(4)],
              'Mirror': [index, index[:, ::-1]], 'Diagonal': [index, np.rot90(index, 2).T], 'Asymmetric': [index]}[symmetry]
    return np.min(np.stack(images), axis=0).reshape(-1)


def runList():
    if args.runSet == 'tilted':
        codes = [tuple(float(value) for value in code.split(':')) for code in args.tiltCodes.split(',')]
        runs = [(f'tilt{phi:g}_{dial:g}_{gradient:g}', dict(dial=dial, gradient=gradient, direction=phi)) for phi, dial, gradient in codes]
        runs += [(f'dial{dial:g}', dict(dial=dial)) for dial in sorted({code[1] for code in codes})]
        runs += [(f'free{symmetry}{seed}', dict(seed=seed, symmetry=symmetry)) for symmetry in ('Mirror', 'Diagonal', 'Asymmetric')
                 for seed in range(args.numRandomStarts)]
        return runs
    runs = [('free', dict(iterations=args.freeIterations)), ('freeShifted', dict(shift=args.shiftMilliVolts))]
    runs += [(f'freeRandom{seed}', dict(seed=seed)) for seed in range(args.numRandomStarts)]
    runs += [(f'dial{float(value):g}', dict(dial=float(value))) for value in args.dialLevels.split(',')]
    return runs


def run(name, spec):
    numIterations = spec.get('iterations', args.runIterations)
    initialValues = dict(reference['simParameters']['initialValues'])
    vmem = initialValues['Vmem'].clone()
    if 'shift' in spec:
        vmem = vmem + spec['shift'] / 1000.0
    startGpol = None
    if 'seed' in spec:
        generator = np.random.default_rng(spec['seed'])
        classes = symmetryClasses(spec.get('symmetry', 'Square'))
        classIds = np.unique(classes)
        vmemLow, vmemHigh = (float(value) for value in args.randomVmemRange.split(','))
        gpolLow, gpolHigh = (float(value) for value in args.randomGpolRange.split(','))
        classVmem = dict(zip(classIds, generator.uniform(vmemLow, vmemHigh, len(classIds))))
        classGpol = dict(zip(classIds, generator.uniform(gpolLow, gpolHigh, len(classIds))))
        vmem = torch.tensor([classVmem[label] / 1000.0 for label in classes], dtype=vmem.dtype).view(vmem.shape)
        startGpol = np.array([classGpol[label] for label in classes])
        initialValues['G_pol'] = dict(cells=[[list(range(size * size))]], values=[[torch.tensor(startGpol, dtype=torch.double)]])
    initialValues['Vmem'] = vmem
    parameters = dict(reference, simParameters=dict(reference['simParameters'], numSimIters=numIterations, initialValues=initialValues))
    holdIterations = args.holdIterations if 'dial' in spec else 0
    ringValues = np.full(len(boundary.boundaryRingCells), spec.get('dial', 0.0))
    if 'gradient' in spec:
        ringValues = np.maximum(spec['dial'] + spec['gradient'] * np.cos(boundary.ringAngles(boundary.boundaryRingCells) - np.deg2rad(spec['direction'])), 0)
    vmemCourse = np.zeros((numIterations, size * size), dtype=np.float32)
    conductanceCourse = []

    def record(iteration, vmemNow, circuit):
        vmemCourse[iteration] = vmemNow
        if iteration % args.conductanceStride == 0:
            conductanceCourse.append((circuit.G_pol[0, :, 0].detach().numpy() / circuit.G_ref).astype(np.float32))
    startTime = time.time()
    boundary.replay(parameters, boundary.ringClamp(reference, ringValues, holdIterations), record, passCircuit=True)
    path = f'{outputDirectory}/{name}.npz'
    np.savez_compressed(path, vmem=vmemCourse, conductance=np.stack(conductanceCourse), conductanceStride=args.conductanceStride,
                        startVmem=vmem.numpy().reshape(-1) * 1000.0, startGpol=startGpol if startGpol is not None else np.ones(size * size),
                        holdIterations=holdIterations, dialLevel=spec.get('dial', np.nan), gradientStrength=spec.get('gradient', 0.0),
                        gradientDirection=spec.get('direction', 0.0), symmetry=spec.get('symmetry', 'Square'), referenceCheckpoint=args.referenceCheckpoint)
    print(f"{name}: {numIterations} iterations in {time.time() - startTime:.0f}s -> {path}", flush=True)


runs = runList()
for taskIndex, (name, spec) in enumerate(runs):
    if args.taskIndex is None or args.taskIndex == taskIndex:
        run(name, spec)
