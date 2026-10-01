"""One ring code's whole causal net, read from its raw relay file (computeBoundaryHarmonicRelay11x11.py's .npz): the net
transfer between every pair of canonical blocks in each phase, field and contact channel, plus the face the code ends with
and whether its conductances stayed inside the model's range. Shared by extractRelayLoopFullNets11x11.py (the Relay Loop
page's codes) and extractRelayLoopSweepRecord11x11.py (the larger sweep), so both read a net the same way.

Net transfer is the custom layout's block net (weight.T @ (edges - edges.T) @ weight over the relay's selectivity readout),
summed over the windows of each phase (flood 0-5, clear 6-11, write 12-35, as the page does) and signed along the pair's
canonical direction (the earlier block in NODE_ORDER to the later one). Only the left / centre blocks are kept; the
right-hand blocks are mirror images to ~1e-8 over these windows.
"""
import itertools

import numpy as np

import boundaryCodeUtilities as boundary
import boundaryHarmonicCoarseGrain as coarse
from boundaryHarmonicStep import Step

NODE_ORDER = ['ringTop', 'ringBottom', 'ringLeft', 'ringRight', 'eyes', 'nose', 'mouth', 'bgTL', 'bgTR', 'bgBL', 'bgBR']
NAME = dict(ringTop='ring top', ringBottom='ring bottom', ringLeft='ring left', ringRight='ring right', eyes='eyes', nose='nose',
            mouth='mouth', bgTL='background top-left', bgTR='background top-right', bgBL='background bottom-left',
            bgBR='background bottom-right')
CANONICAL = [n for n in NODE_ORDER if n not in ('ringRight', 'bgTR', 'bgBR')]
PAIRS = list(itertools.combinations(CANONICAL, 2))                      # in node order, so a -> b runs earlier -> later
PHASES = dict(flood=range(0, 6), clear=range(6, 12), write=range(12, 36))
WRITE_PEAK_STATE, FACE_STATE = 1766, 2174                               # recorded iterations 1765 and 2173

_ring = np.array(boundary.boundaryRingCells)
_weight, _, _regionNames = coarse.namedRegionLabels(_ring, boundary.featureParts)
_index = {n: _regionNames.index(NAME[n]) for n in NODE_ORDER}
_step = Step(ringCode=np.zeros(len(_ring)))
GREF = float(_step.Gref)
FEATURE_CELLS = sorted(set(boundary.featureCellIndices.tolist()))
BACKGROUND_CELLS = [int(c) for c in boundary.interiorCellIndices if c not in set(FEATURE_CELLS)]


def readNet(rawPath):
    """{'field', 'contact': [phase][pair] net transfers, 'gap': feature minus background conductance at the write peak (G_ref),
    'faceOverlap': structural overlap of the dark cells with the face at recorded iteration 2173, 'gpolMin', 'gpolMax': the
    smallest and largest conductance of any cell at any time, in G_ref}."""
    raw = np.load(rawPath)
    edges = raw['edges'][0]                                               # selectivity readout: (channel, window, 121, 121)
    state = raw['trainedState']
    nets = {}
    for channel, name in ((0, 'field'), (1, 'contact')):
        perWindow = np.array([_weight.T @ (edges[channel, w] - edges[channel, w].T) @ _weight for w in range(36)])   # [to, from]
        nets[name] = [[round(float(sum(perWindow[w][_index[b], _index[a]] for w in windows)), 6) for a, b in PAIRS]
                      for windows in PHASES.values()]
    conductance = state[:, coarse.NUM_CELLS:] / GREF
    atPeak = conductance[WRITE_PEAK_STATE]
    return dict(gap=round(float(atPeak[FEATURE_CELLS].mean() - atPeak[BACKGROUND_CELLS].mean()), 4),
                faceOverlap=round(float(boundary.structuralIntersectionOverUnion(state[FACE_STATE, :coarse.NUM_CELLS].astype(float) * 1000.0)), 4),
                gpolMin=round(float(conductance.min()), 4), gpolMax=round(float(conductance.max()), 4), **nets)
