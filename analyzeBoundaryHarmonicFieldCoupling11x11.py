"""Which way does one cell's state push its neighbours, through the field and through contact?

EXPLORATORY. Written after the aggregate-nucleation result, to settle a sign that two earlier readings of the
intervention runs got wrong in opposite directions. Nothing here was registered in advance.

The intervention runs cannot answer this cleanly. Blocking eleven cells at once changes the whole trajectory of
a chaotic system, and averaging over "cells near a blocked cell" mixes each cell's distance to eleven different
sources. So the field coupling is measured as a derivative of the model's own equations instead: take a real
tissue state, move one cell from light to dark, recompute the extracellular field exactly, and read off what
that does to every other cell. No dynamics, no divergence, no averaging over sources.

Two channels are then comparable:

  field     eV is the magnitude of the net field vector, always positive, so a cell's charge raises the field
            its neighbours read. Transduction has gain -1, and because the relaxation term -G_pol carries units
            of siemens against a target of order one, what the field sets is the RATE:
                d(G_pol/G_ref)/dt  ~  500 * (fieldTransductionBias - fieldRead)
            A dark cell therefore pushes its neighbours' conductance DOWN, which depolarises them.

  contact   a dark neighbour lowers both switching thresholds through the gap junctions, by 0.0396 on G_up.
            That helps a neighbour cross.

The two oppose. Since the field acts on the rate and contact on the level, their relative size depends on how
long the neighbour waits: this reports the crossing time at which the field's suppression overtakes contact's
help.
"""
import argparse
import collections
import json

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--recordPath', type=str, required=True, help='npz from recordBoundaryHarmonicRuns11x11.py --mode trained')
parser.add_argument('--switchRulePath', type=str,
                    default='data/boundaryHarmonicSwitchRule1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str,
                    default='data/boundaryHarmonicFieldCoupling1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--moments', type=int, nargs='+', default=(1300, 1400, 1500, 1600, 1700))
parser.add_argument('--probeStride', type=int, default=4, help='every nth interior cell is used as a probe')
parser.add_argument('--darkVoltage', type=float, default=-40.0, help='mV a probe is moved to, a typical dark seat')
args = parser.parse_args()

switchRule = json.load(open(args.switchRulePath))
upperCoefficients = switchRule['upperCoefficients']
contactShift = abs(upperCoefficients[2])          # G_up falls by this much per dark neighbour

torch.set_grad_enabled(False)
reference = boundary.loadCheckpoint(1888)
parameters = dict(reference)
parameters['latticePeriodicBoundaryGJ'] = False
parameters['ATPParameters'] = None
initial = reference['simParameters']['initialValues']
batchInitial = {name: initial[name] for name in ('Vmem', 'eV', 'ligandConc')}
batchInitial['G_pol'] = dict(cells=[initial['G_pol']['cells'][0]], values=[initial['G_pol']['values'][0]])
batchInitial['G_dep'] = initial['G_dep']
system = model(parameters, 1)
system.setExperimentalConditions((batchInitial, 1))
circuit = system.electricNetwork
screen = circuit.fieldScreenMatrixIn[0].numpy().astype(float)
numFieldNeighbours = float(circuit.numFieldNeighbors)

gain = float(circuit.fieldTransductionGain)
weight = float(circuit.fieldTransductionWeight)
bias = float(circuit.fieldTransductionBias)
# d(G/G_ref)/dt = 10/tau * weight * tanh((bias - field)/2) ~ -rateGain * (field - bias) for the small arguments
# the runs actually visit; rateGain is the slope in units of G_ref per unit field per unit time.
rateGain = 10.0 / float(circuit.fieldTransductionTimeConstant) * weight * 0.5
iterationsPerUnitTime = 1.0 / float(circuit.timestep)


def fieldReadBy(voltageVolts):
    circuit.Vmem = torch.tensor(voltageVolts.reshape(1, -1, 1), dtype=torch.double)
    circuit.updateExtracellularVoltage(source='Vmem')
    return (circuit.eV[0, :, 0].numpy() @ screen) / numFieldNeighbours


record = np.load(args.recordPath)
orderIndex = list(record['orders']).index(3)
interior = [int(c) for c in boundary.interiorCellIndices]


def latticeStep(a, b):
    return abs(a // boundary.latticeCols - b // boundary.latticeCols) + abs(a % boundary.latticeCols - b % boundary.latticeCols)


rates = collections.defaultdict(list)
fields = collections.defaultdict(list)
probes = interior[::args.probeStride]
for moment in args.moments:
    voltage = record['vmem'][orderIndex][moment].astype(float) / 1000.0
    baseField = fieldReadBy(voltage)
    for probe in probes:
        if voltage[probe] * 1000.0 < boundary.hyperpolarizedThresholdMilliVolts:
            continue                                        # already dark, nothing to flip
        moved = voltage.copy()
        moved[probe] = args.darkVoltage / 1000.0
        deltaField = fieldReadBy(moved) - baseField
        for cell in interior:
            if cell == probe:
                continue
            step = latticeStep(cell, probe)
            if step <= 5:
                fields[step].append(float(deltaField[cell]))
                rates[step].append(float(-rateGain * deltaField[cell]))

profile = []
for step in sorted(rates):
    values = np.array(rates[step])
    profile.append(dict(steps=step, samples=len(values),
                        meanFieldChange=float(np.mean(fields[step])),
                        meanRateChange=round(float(values.mean()), 6),
                        share_negative=round(float((values < 0).mean()), 3)))

nearest = next(row for row in profile if row['steps'] == 1)
crossingIterations = (abs(contactShift / nearest['meanRateChange']) * iterationsPerUnitTime
                      if nearest['meanRateChange'] else None)

result = dict(
    exploratory=True,
    note='a derivative of the model equations at real tissue states, not a dynamical intervention',
    moments=list(args.moments), probes=len(probes), darkVoltage=args.darkVoltage,
    transduction=dict(gain=gain, weight=weight, bias=bias,
                      rateGain=rateGain, iterationsPerUnitTime=iterationsPerUnitTime,
                      law='d(G/G_ref)/dt = -rateGain * (fieldRead - bias)'),
    fieldChannel=dict(profile=profile,
                      direction='a cell going dark pushes its neighbours conductance DOWN, i.e. depolarises them',
                      screenSize=int(reference['fieldParameters']['fieldScreenSize'])),
    contactChannel=dict(thresholdShiftPerDarkNeighbour=contactShift,
                        direction='a dark neighbour lowers G_up, i.e. helps a neighbour cross'),
    comparison=dict(
        contactHelp=contactShift,
        fieldSuppressionPerHundredIterations=round(abs(nearest['meanRateChange']), 4),
        crossoverIterations=round(crossingIterations) if crossingIterations else None,
        reading=('contact wins for a neighbour that crosses soon after; the field wins for one that waits longer, '
                 'because the field acts on the rate and so accumulates')))
json.dump(result, open(args.outputPath, 'w'), indent=1)

print(f'transduction: d(G/G_ref)/dt = -{rateGain:.0f} * (field - {bias})', flush=True)
for row in profile:
    print(f"  {row['steps']} step(s): mean change in dG/dt {row['meanRateChange']:+.4f} "
          f"({100 * row['share_negative']:.0f}% negative, n={row['samples']})", flush=True)
print(f'contact help {contactShift:.4f} on G_up against field suppression '
      f"{abs(nearest['meanRateChange']):.4f} per 100 iterations "
      f'-> they cross at about {round(crossingIterations)} iterations', flush=True)
print('wrote', args.outputPath, flush=True)
