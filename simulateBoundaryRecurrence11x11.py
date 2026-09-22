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

With --runSet tiltedStarts, free runs from starting states that carry a tilt, written to data/boundaryRecurrence<checkpoint>
Hold<hold>TiltedStarts/:

  freeTilt<phi>_<k>   G_pol / G_ref = DC + G p + noise and Vmem = V + A p + noise over the whole tissue, where p is each
                      cell's offset from the centre along direction phi, in units of the half-width (p = cos(theta - phi)
                      at the middle of each edge). DC and G are drawn from the ranges of the tilt codes (--tiltStartDial,
                      --tiltStartGradient), V and A from --tiltStartVmem and --tiltStartVmemTilt (A of either sign); the noise
                      (--tiltStartNoise) has the symmetry a tilt at phi keeps. --numRandomStarts at each of 0, 45 and 22.5 degrees.
  release<Ring>_<code> each tilt code's state at release (Vmem and G_pol after its last held iteration) with the ring's
                      clamp imprint removed: RingGpol puts the ring's G_pol back to the free run's at the same iteration;
                      RingAll puts back the ring's Vmem as well. The interior keeps everything the hold left in it.
                      Nudge, a control, keeps the whole state but raises every cell's Vmem by --shiftMilliVolts.

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
parser.add_argument('--runSet', type=str, default='dial', choices=['dial', 'tilted', 'tiltedStarts'])
parser.add_argument('--tiltStartDial', type=str, default='0.52,0.84')
parser.add_argument('--tiltStartGradient', type=str, default='0.12,0.58')
parser.add_argument('--tiltStartVmem', type=str, default='-45,-15')
parser.add_argument('--tiltStartVmemTilt', type=str, default='0,20', help='range of |A| in mV')
parser.add_argument('--tiltStartNoise', type=str, default='0.1,5', help='G_pol / G_ref and Vmem (mV) noise half-widths')
parser.add_argument('--tiltCodes', type=str, default='0:0.52:0.2,0:0.6:0.18,45:0.84:0.32,22.5:0.74:0.46,45:0.82:0.16,45:0.76:0.14,'
                    '0:0.74:0.5,0:0.82:0.48,45:0.74:0.48,22.5:0.74:0.54,22.5:0.82:0.26,45:0.66:0.58,45:0.8:0.32,45:0.76:0.12,'
                    '0:0.76:0.5,45:0.78:0.3', help='phi:DC:G per tilt code; default the typical codes of the Tilted Dial gallery')
parser.add_argument('--taskIndex', type=int, default=None, help='run only this entry of the run list; default runs all')
args = parser.parse_args()

size = boundary.latticeRows
reference = boundary.loadCheckpoint(args.referenceCheckpoint)
outputDirectory = f"data/boundaryRecurrence{args.referenceCheckpoint}Hold{args.holdIterations}{dict(dial='', tilted='Tilted', tiltedStarts='TiltedStarts')[args.runSet]}"
os.makedirs(outputDirectory, exist_ok=True)


def symmetryClasses(symmetry='Square'):
    """Label each cell by its class under a symmetry group: the smallest index among its images. Square: all 8
    symmetries of the square; Mirror: the left-right mirror; Diagonal: the mirror about the diagonal through the top-right
    corner; Asymmetric: none."""
    index = np.arange(size * size).reshape(size, size)
    images = {'Square': [np.rot90(index, turns) for turns in range(4)] + [np.rot90(index, turns)[:, ::-1] for turns in range(4)],
              'Mirror': [index, index[:, ::-1]], 'Diagonal': [index, np.rot90(index, 2).T], 'Asymmetric': [index]}[symmetry]
    return np.min(np.stack(images), axis=0).reshape(-1)


symmetryOfDirection = {0.0: 'Mirror', 45.0: 'Diagonal', 22.5: 'Asymmetric'}
ranges = lambda text: tuple(float(value) for value in text.split(','))


def runList():
    codes = [tuple(float(value) for value in code.split(':')) for code in args.tiltCodes.split(',')]
    if args.runSet == 'tiltedStarts':
        runs = [(f'freeTilt{phi:g}_{seed}', dict(seed=seed, direction=phi, tiltedStart=True)) for phi in symmetryOfDirection for seed in range(args.numRandomStarts)]
        runs += [(f'release{ring}_{phi:g}_{dial:g}_{gradient:g}', dict(release=ring, dial=dial, gradient=gradient, direction=phi))
                 for ring in ('RingGpol', 'RingAll', 'Nudge') for phi, dial, gradient in codes]
        return runs
    if args.runSet == 'tilted':
        runs = [(f'tilt{phi:g}_{dial:g}_{gradient:g}', dict(dial=dial, gradient=gradient, direction=phi)) for phi, dial, gradient in codes]
        runs += [(f'dial{dial:g}', dict(dial=dial)) for dial in sorted({code[1] for code in codes})]
        runs += [(f'free{symmetry}{seed}', dict(seed=seed, symmetry=symmetry)) for symmetry in ('Mirror', 'Diagonal', 'Asymmetric')
                 for seed in range(args.numRandomStarts)]
        return runs
    runs = [('free', dict(iterations=args.freeIterations)), ('freeShifted', dict(shift=args.shiftMilliVolts))]
    runs += [(f'freeRandom{seed}', dict(seed=seed)) for seed in range(args.numRandomStarts)]
    runs += [(f'dial{float(value):g}', dict(dial=float(value))) for value in args.dialLevels.split(',')]
    return runs


def tiltedRingValues(spec):
    return np.maximum(spec['dial'] + spec['gradient'] * np.cos(boundary.ringAngles(boundary.boundaryRingCells) - np.deg2rad(spec['direction'])), 0)


def stateAfter(ringValues, holdIterations):
    """Vmem (V) and G_pol / G_ref of every cell after iteration holdIterations - 1, the ring held at ringValues until then
    (holdIterations - 1 = 0 hold for the free run's state)."""
    parameters = dict(reference, simParameters=dict(reference['simParameters'], numSimIters=args.holdIterations))
    state = {}

    def capture(iteration, vmemNow, circuit):
        if iteration == args.holdIterations - 1:
            state['vmem'], state['gpol'] = circuit.Vmem.clone(), (circuit.G_pol[0, :, 0] / circuit.G_ref).numpy().copy()
    boundary.replay(parameters, boundary.ringClamp(reference, ringValues, holdIterations), capture, passCircuit=True)
    return state['vmem'], state['gpol']


def run(name, spec):
    numIterations = spec.get('iterations', args.runIterations)
    initialValues = dict(reference['simParameters']['initialValues'])
    vmem = initialValues['Vmem'].clone()
    if 'shift' in spec:
        vmem = vmem + spec['shift'] / 1000.0
    startGpol = None
    if 'release' in spec:
        vmem, startGpol = stateAfter(tiltedRingValues(spec), args.holdIterations)
        freeVmem, freeGpol = stateAfter(np.zeros(len(boundary.boundaryRingCells)), 0)
        ring = boundary.boundaryRingCells
        if spec['release'] == 'Nudge':
            vmem = vmem + args.shiftMilliVolts / 1000.0
        else:
            startGpol[ring] = freeGpol[ring]
        if spec['release'] == 'RingAll':
            vmem[0, ring, 0] = freeVmem[0, ring, 0]
        initialValues['G_pol'] = dict(cells=[[list(range(size * size))]], values=[[torch.tensor(startGpol, dtype=torch.double)]])
    elif spec.get('tiltedStart'):
        generator = np.random.default_rng(spec['seed'])
        classes = symmetryClasses(symmetryOfDirection[spec['direction']])
        classIds = np.unique(classes)
        rows, columns = np.divmod(np.arange(size * size), size)
        angle = np.deg2rad(spec['direction'])
        offset = ((columns - boundary.latticeCentre) * np.sin(angle) + (boundary.latticeCentre - rows) * np.cos(angle)) / boundary.latticeCentre
        dial, gradient = generator.uniform(*ranges(args.tiltStartDial)), generator.uniform(*ranges(args.tiltStartGradient))
        vmemLevel, vmemTilt = generator.uniform(*ranges(args.tiltStartVmem)), generator.choice([-1, 1]) * generator.uniform(*ranges(args.tiltStartVmemTilt))
        gpolNoise, vmemNoise = ranges(args.tiltStartNoise)
        classGpolNoise = dict(zip(classIds, generator.uniform(-gpolNoise, gpolNoise, len(classIds))))
        classVmemNoise = dict(zip(classIds, generator.uniform(-vmemNoise, vmemNoise, len(classIds))))
        gpolLow, gpolHigh = ranges(args.randomGpolRange)
        vmemLow, vmemHigh = ranges(args.randomVmemRange)
        startGpol = np.clip(dial + gradient * offset + np.array([classGpolNoise[label] for label in classes]), gpolLow, gpolHigh)
        startVmem = np.clip(vmemLevel + vmemTilt * offset + np.array([classVmemNoise[label] for label in classes]), vmemLow, vmemHigh)
        vmem = torch.tensor(startVmem / 1000.0, dtype=vmem.dtype).view(vmem.shape)
        initialValues['G_pol'] = dict(cells=[[list(range(size * size))]], values=[[torch.tensor(startGpol, dtype=torch.double)]])
        spec = dict(spec, dial=dial, gradient=gradient, vmemLevel=vmemLevel, vmemTilt=vmemTilt)
    elif 'seed' in spec:
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
    free = 'release' in spec or spec.get('tiltedStart', False)
    holdIterations = args.holdIterations if 'dial' in spec and not free else 0
    ringValues = np.full(len(boundary.boundaryRingCells), spec.get('dial', 0.0))
    if 'gradient' in spec and not free:
        ringValues = tiltedRingValues(spec)
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
                        gradientDirection=spec.get('direction', 0.0), symmetry=spec.get('symmetry', symmetryOfDirection.get(spec.get('direction'), 'Square')),
                        release=spec.get('release', ''), vmemLevel=spec.get('vmemLevel', np.nan), vmemTilt=spec.get('vmemTilt', np.nan),
                        referenceCheckpoint=args.referenceCheckpoint)
    print(f"{name}: {numIterations} iterations in {time.time() - startTime:.0f}s -> {path}", flush=True)


runs = runList()
for taskIndex, (name, spec) in enumerate(runs):
    if args.taskIndex is None or args.taskIndex == taskIndex:
        run(name, spec)
