"""Replays the trained face code of the chosen order (BOUNDARY_FACE_ORDER, default 3) on the reference tissue and stores what the order-4 report
draws first: the ring's 40 held values, the tissue's voltage (mV) at the landmark iterations, and the interior mean polarising conductance through
the first 3000 iterations. EXPLORATORY: nothing here was predicted or registered; it only shows the committed codes.

Writes data/reportPatterns1888Hold301FaceOrder<N>.json (never overwriting).

    BOUNDARY_FACE_ORDER=4 python3 replayReportPatterns11x11.py
"""
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary
import canalizationTalkCommon as common

order = boundary.faceOrder
outputPath = f'data/reportPatterns1888Hold301FaceOrder{order}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

target = common.TARGETS['face']
coefficients, scored = target['coefficients'], target['readIteration']
landmarks = boundary.faceLandmarkStates                                           # states: release, trough, write peak, scored (recorded iteration = state - 1)
NUM_ITERATIONS = 3000
ringValues = common.ringValuesOf(coefficients, target['ceiling'])
replayer = common.Replayer()
vmem, conductance = replayer.run(ringValues, NUM_ITERATIONS)                       # row k = state after iteration k + 1 steps (recorded iteration k)

shown = sorted({100, 300, 301, landmarks['trough'] - 1, landmarks['peak'] - 1, scored, 2999})
interiorMean = conductance[:, common.INTERIOR].mean(1)
faceMean = conductance[:, common.FACE_CELLS].mean(1)
backgroundCells = np.array([c for c in common.INTERIOR if c not in set(common.FACE_CELLS.tolist())])
backgroundMean = conductance[:, backgroundCells].mean(1)
dark = vmem[scored] < common.THRESHOLD
overlap = float((dark[common.FACE_CELLS]).sum() / (dark[common.INTERIOR].sum() + len(common.FACE_CELLS) - dark[common.FACE_CELLS].sum()))
result = dict(order=order, coefficients=[round(float(c), 6) for c in coefficients], ringValues=[round(float(v), 4) for v in ringValues],
              ringAnglesRadians=[round(float(a), 4) for a in common.RING_ANGLES], ringCells=[int(c) for c in common.RING_CELLS],
              faceCells=[int(c) for c in common.FACE_CELLS], interiorCells=[int(c) for c in common.INTERIOR], darkThresholdMillivolts=common.THRESHOLD,
              hold=common.HOLD, scoredIteration=int(scored), landmarkStates={k: int(v) for k, v in landmarks.items()}, iterationsShown=shown,
              frames={str(i): np.round(vmem[i], 1).tolist() for i in shown},
              conductanceEvery5=dict(iterations=list(range(0, NUM_ITERATIONS, 5)), interiorMean=np.round(interiorMean[::5], 4).tolist(),
                                     faceMean=np.round(faceMean[::5], 4).tolist(), backgroundMean=np.round(backgroundMean[::5], 4).tolist()),
              darkInteriorAtScored=int(dark[common.INTERIOR].sum()), faceCellsDarkAtScored=int(dark[common.FACE_CELLS].sum()), faceCellCount=len(common.FACE_CELLS),
              faceIntersectionOverUnion=overlap)
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath}: order {order}, scored iteration {scored}, {result["faceCellsDarkAtScored"]} of {result["faceCellCount"]} face cells dark, '
      f'{result["darkInteriorAtScored"]} dark interior cells, IoU {overlap:.3f}')
