"""Block the nucleators as a set, not one at a time, and see whether the recruits still darken.

The single-cell test showed that no individual nucleator is needed for the neighbour that followed it. That
cannot settle the question, because the drive every cell reads is the tissue's own field: the nucleators could
still produce the recruits collectively, at range. This runs the aggregate version.

Conditions and criteria are registered in
data/boundaryHarmonicAggregateNucleationPredictions1888Hold301FaceMinus60Minus5.json, committed first.

    python3 runBoundaryHarmonicAggregateNucleation11x11.py <outPath.npz> <trainedRuns.npz>
"""
import sys
import json

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from embryo import model

outPath = sys.argv[1]
recorded = np.load(sys.argv[2])
summary = json.load(open('data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json'))
switchRule = json.load(open('data/boundaryHarmonicSwitchRule1888Hold301FaceMinus60Minus5.json'))
reference = boundary.loadCheckpoint(1888)
hold = int(summary['hold'])
best = 2173

winner = summary['orders']['3']['best']
coefficients = np.asarray(np.load(
    f"{summary['trainingDirs'][winner['round']]}/order3_restart{winner['restart']:02d}.npz")['bestCoefficients'], float)
angles = boundary.ringAngles(boundary.boundaryRingCells)
ringValues = np.cos(np.outer(angles, np.arange(len(coefficients)))) @ coefficients

orderIndex = list(recorded['orders']).index(3)
baselineGpol = recorded['gpol'][orderIndex].astype(float)

LATE_AFTER = 1200
WINDOW = (1250, best)
BLOCK_VALUE = 1.25          # the value the single-cell test used
RANDOM_DRAWS = 20
SEED = 101

# ------------------------------------------------------------------ who is a nucleator, who a recruit
late = [e for e in switchRule['runs']['3']['events'] if e['toDark'] and e['iteration'] > LATE_AFTER]
nucleatorOrder, nucleators = [], set()
for event in late:
    if event['darkNeighbours'] == 0 and event['cell'] not in nucleators:
        nucleators.add(event['cell'])
        nucleatorOrder.append(event['cell'])          # events are in time order
recruits = sorted({e['cell'] for e in late if e['darkNeighbours'] > 0} - nucleators)
everDarkLate = sorted({e['cell'] for e in late})
# The registered control drew from "interior cells dark at some point after 1200 but not nucleators", which is
# exactly the recruit set -- so it blocks the very cells it then scores, and cannot test what it was meant to.
# It is kept, run and reported as void, alongside an amended pool that excludes the scored recruits.
pool = [c for c in everDarkLate if c not in nucleators]
darkEver = sorted({e['cell'] for e in switchRule['runs']['3']['events'] if e['toDark']})
amendedPool = [c for c in darkEver if c not in nucleators and c not in set(recruits)]
print(f'{len(nucleatorOrder)} nucleators {nucleatorOrder}', flush=True)
print(f'{len(recruits)} recruits {recruits}', flush=True)

generator = np.random.default_rng(SEED)
runs = [dict(kind='baseline', cells=[])]
runs.append(dict(kind='allNucleators', cells=list(nucleatorOrder)))
for k in (2, 4, 6, 8):
    runs.append(dict(kind=f'ladder{k}', cells=list(nucleatorOrder[:k])))
for draw in range(RANDOM_DRAWS):
    pick = generator.choice(len(pool), size=min(len(nucleatorOrder), len(pool)), replace=False)
    runs.append(dict(kind=f'random{draw:02d}', cells=[int(pool[i]) for i in pick]))
for draw in range(RANDOM_DRAWS):
    pick = generator.choice(len(amendedPool), size=min(len(nucleatorOrder), len(amendedPool)), replace=False)
    runs.append(dict(kind=f'amended{draw:02d}', cells=[int(amendedPool[i]) for i in pick]))
runs.append(dict(kind='noop', cells=list(nucleatorOrder)))
numRuns = len(runs)
print(f'{len(pool)} in the registered control pool, {len(amendedPool)} in the amended pool', flush=True)
print(f'{numRuns} runs: baseline, all-nucleators, 4 ladder steps, {RANDOM_DRAWS} registered + {RANDOM_DRAWS} amended draws, no-op', flush=True)

torch.set_grad_enabled(False)
parameters = dict(reference)
parameters['latticePeriodicBoundaryGJ'] = False
parameters['ATPParameters'] = None
initial = reference['simParameters']['initialValues']
batchInitial = {n: initial[n].repeat(numRuns, 1, 1) for n in ('Vmem', 'eV', 'ligandConc')}
batchInitial['G_pol'] = dict(cells=[initial['G_pol']['cells'][0]] * numRuns,
                             values=[initial['G_pol']['values'][0]] * numRuns)
batchInitial['G_dep'] = initial['G_dep']
system = model(parameters, numRuns)
system.setExperimentalConditions((batchInitial, numRuns))
circuit = system.electricNetwork
clamp = dict(reference['clampParameters'])
clamp['clampIndices'] = (np.repeat(np.arange(numRuns), len(boundary.boundaryRingCells)),
                         np.tile(boundary.boundaryRingCells, numRuns))
clamp['clampValues'] = torch.tensor(np.tile(np.tile(ringValues, numRuns).reshape(1, -1), (hold, 1)), dtype=torch.double)
clamp['clampStartIter'], clamp['clampEndIter'] = 0, hold - 1

vmem = np.zeros((numRuns, best + 1, boundary.numCells), dtype=np.float32)
gpol = np.zeros((numRuns, best + 1, boundary.numCells), dtype=np.float32)
for iteration in range(best + 1):
    system.simulate(clampParameters=clamp if iteration < hold else None, numSimIters=1,
                    outerIter=iteration, fieldModulation=False)
    if WINDOW[0] <= iteration <= WINDOW[1]:
        for index, spec in enumerate(runs):
            if not spec['cells']:
                continue
            if spec['kind'] == 'noop':
                for cell in spec['cells']:
                    circuit.G_pol[index, cell, 0] = baselineGpol[iteration, cell] * circuit.G_ref
            else:
                for cell in spec['cells']:
                    circuit.G_pol[index, cell, 0] = BLOCK_VALUE * circuit.G_ref
    vmem[:, iteration] = circuit.Vmem[:, :, 0].numpy() * 1000.0
    gpol[:, iteration] = (circuit.G_pol[:, :, 0] / circuit.G_ref).numpy()
    if iteration % 500 == 0:
        print(f'  iteration {iteration}', flush=True)

np.savez_compressed(
    outPath, vmem=vmem, gpol=gpol,
    kinds=np.array([r['kind'] for r in runs]),
    cells=np.array([','.join(str(c) for c in r['cells']) for r in runs]),
    nucleators=np.array(nucleatorOrder), recruits=np.array(recruits),
    window=np.array(WINDOW), blockValue=BLOCK_VALUE, best=best, seed=SEED)
print('done', vmem.shape, flush=True)
