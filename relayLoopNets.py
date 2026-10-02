"""One ring code's whole causal net, read from its raw relay file (computeBoundaryHarmonicRelay11x11.py's .npz): the net
transfer between every pair of canonical blocks in each phase, field and contact channel, plus the target the code ends with
and whether its conductances stayed inside the model's range. Shared by extractRelayLoopFullNets11x11.py (the Relay Loop
page's codes) and extractRelayLoopSweepRecord11x11.py (the larger sweep), so both read a net the same way.

Net transfer is the custom layout's block net (weight.T @ (edges - edges.T) @ weight over the relay's selectivity readout),
summed over the windows of each phase (for the face flood 0-5, clear 6-11, write 12-35, as the page does) and signed along
the pair's canonical direction (the earlier block in the layout's node order to the later one). Only the left / centre blocks
are kept; the right-hand blocks are mirror images to ~1e-8 over these windows.

Three targets, two block layouts. 'face' (the module-level NODE_ORDER, NAME, CANONICAL, PAIRS, PHASES and the constants below,
unchanged) is the 11 named blocks of the Relay Loop report. 'stripesInterior' is the ten-block layout of the interior stripes
(boundaryHarmonicCoarseGrain.stripeRegionLabels: ring top, bottom, left and right; stripe upper and lower; each flank upper
and lower). 'doubleStripesInterior', the inverse pattern (the two flanks dark, the centre stripe light), uses the same ten
blocks; only the cells the gap and the overlap are read over differ (the flanks instead of the centre stripe). layoutFor(target)
returns the layout; readNet picks it from the raw file's own `targetName` (a raw file written before that key existed is the
face's), and the stripe targets' phase windows from the landmarks the file records.
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

# ------------------------------------------------------------------------------------------ the layouts
STRIPE_NODE_ORDER = ['ringTop', 'ringBottom', 'ringLeft', 'ringRight', 'stripeUpper', 'stripeLower', 'flankLeftUpper',
                     'flankRightUpper', 'flankLeftLower', 'flankRightLower']
STRIPE_NAME = dict(ringTop='ring top', ringBottom='ring bottom', ringLeft='ring left', ringRight='ring right',
                   stripeUpper='stripe upper', stripeLower='stripe lower', flankLeftUpper='left flank upper',
                   flankRightUpper='right flank upper', flankLeftLower='left flank lower', flankRightLower='right flank lower')
_STRIPE_MIRRORED = ('ringRight', 'flankRightUpper', 'flankRightLower')
STRIPE_TARGET_CELLS = dict(stripesInterior=boundary.centreStripeCellIndices, doubleStripesInterior=boundary.flankCellIndices)


class Layout:
    """A target's block layout: the node order and names, the canonical (left / centre) nodes and their pairs, the cell-to-block
    weight matrix, and the feature and background cells the gap is read over."""

    def __init__(self, target, nodeOrder, names, mirrored, weight, regionNames, featureCells):
        self.target, self.NODE_ORDER, self.NAME = target, list(nodeOrder), dict(names)
        self.CANONICAL = [n for n in self.NODE_ORDER if n not in mirrored]
        self.PAIRS = list(itertools.combinations(self.CANONICAL, 2))
        self.weight = weight
        self.index = {n: regionNames.index(self.NAME[n]) for n in self.NODE_ORDER}
        self.featureCells = sorted(set(int(c) for c in featureCells))
        self.backgroundCells = [int(c) for c in boundary.interiorCellIndices if c not in set(self.featureCells)]


_LAYOUTS = {}


def layoutFor(target):
    """The Layout of 'face', 'stripesInterior' or 'doubleStripesInterior' (built once)."""
    if target not in _LAYOUTS:
        if target == 'face':
            _LAYOUTS[target] = Layout(target, NODE_ORDER, NAME, ('ringRight', 'bgTR', 'bgBR'), _weight, _regionNames, FEATURE_CELLS)
        elif target in STRIPE_TARGET_CELLS:
            weight, _, regionNames = coarse.stripeRegionLabels(_ring, boundary.stripeParts)
            _LAYOUTS[target] = Layout(target, STRIPE_NODE_ORDER, STRIPE_NAME, _STRIPE_MIRRORED, weight, regionNames,
                                      STRIPE_TARGET_CELLS[target])
        else:
            raise ValueError(f'unknown target {target!r}')
    return _LAYOUTS[target]


def phaseWindows(raw, target):
    """{phase: range of 50-iteration edge windows} of a raw relay file. The face's are the page's (flood 0-5, clear 6-11,
    write 12-35). For a target whose raw file records its landmarks, the windows are read from them the way those were
    read from the face's: flood is the windows wholly inside the hold, clear runs to the window holding the trough, write to
    the window holding the write peak."""
    if target == 'face' or 'troughIteration' not in raw.files:
        return PHASES
    window, hold = int(raw['edgeWindow']), int(raw['hold'])
    floodEnd = hold // window
    writeEnd = -(-(int(raw['primaryIteration']) + 1) // window)
    # a code read before its conductance has turned (the trough is not inside the run) has no write phase: clear runs to the readout
    clearEnd = (-(-(int(raw['troughIteration']) + 1) // window)) if int(raw['troughIteration']) < int(raw['primaryIteration']) else writeEnd
    return dict(flood=range(0, floodEnd), clear=range(floodEnd, clearEnd), write=range(clearEnd, writeEnd))


def readNet(rawPath):
    """{'field', 'contact': [phase][pair] net transfers, 'gap': feature minus background conductance at the write peak (G_ref),
    'faceOverlap': structural overlap of the dark cells with the target (the face, the centre stripe or the two flanks) at the scored moment
    (recorded iteration 2173 for the face), 'gpolMin', 'gpolMax': the smallest and largest conductance of any cell at any
    time, in G_ref}. For a stripe raw file the pairs and phases are the stripe layout's (layoutFor('stripesInterior'))."""
    raw = np.load(rawPath)
    target = str(raw['targetName']) if 'targetName' in raw.files else 'face'
    layout = layoutFor(target)
    phases = phaseWindows(raw, target)
    peakState = int(raw['primaryIteration']) + 1 if 'primaryIteration' in raw.files else WRITE_PEAK_STATE
    scoredState = int(raw['scoredIteration']) + 1 if 'scoredIteration' in raw.files else FACE_STATE
    edges = raw['edges'][0]                                               # selectivity readout: (channel, window, 121, 121)
    state = raw['trainedState']
    lastWindow = max(max(windows) for windows in phases.values() if len(windows)) + 1
    nets = {}
    for channel, name in ((0, 'field'), (1, 'contact')):
        perWindow = np.array([layout.weight.T @ (edges[channel, w] - edges[channel, w].T) @ layout.weight
                              for w in range(lastWindow)])                # [to, from]
        nets[name] = [[round(float(sum(perWindow[w][layout.index[b], layout.index[a]] for w in windows)), 6)
                       for a, b in layout.PAIRS] for windows in phases.values()]
    conductance = state[:, coarse.NUM_CELLS:] / GREF
    atPeak = conductance[peakState]
    targetCells = None if target == 'face' else STRIPE_TARGET_CELLS[target]     # the centre stripe, or the two flanks; layoutFor has refused any other target
    extra = {}
    if target != 'face':        # the pattern at the scored moment and how much of the target's upper (rows 1-4) and lower (rows 6-9) halves is dark
        vmem = state[scoredState, :coarse.NUM_CELLS].astype(float) * 1000.0
        stripeRows = targetCells // boundary.latticeCols
        stripeDark = vmem[targetCells] < boundary.hyperpolarizedThresholdMilliVolts
        extra = dict(scoredVmem=[int(round(v)) for v in vmem], upperDark=int(stripeDark[stripeRows <= 4].sum()), lowerDark=int(stripeDark[stripeRows >= 6].sum()))
    return dict(**extra, gap=round(float(atPeak[layout.featureCells].mean() - atPeak[layout.backgroundCells].mean()), 4),
                faceOverlap=round(float(boundary.structuralIntersectionOverUnion(
                    state[scoredState, :coarse.NUM_CELLS].astype(float) * 1000.0, targetCells)), 4),
                gpolMin=round(float(conductance.min()), 4), gpolMax=round(float(conductance.max()), 4), **nets)


def readTrajectory(rawPath, stride=5):
    """What the Relay Loop page draws besides the net, read from a raw relay file's `trainedState` (the whole (Vmem, G_pol) trajectory of the run it
    decomposed, so nothing is re-simulated): the tissue's Vmem (mV, one decimal, 121 cells row by row) at the end of the flood (the last held state)
    and at the readout (`best`, the moment the net is read at), and the mean, lowest and highest G_pol / G_ref over time of the feature cells and of
    the rest of the interior, every `stride` states with the phase-end states always included. A state k is recorded iteration k - 1."""
    raw = np.load(rawPath)
    target = str(raw['targetName']) if 'targetName' in raw.files else 'face'
    layout = layoutFor(target)
    phases = phaseWindows(raw, target)
    peakState = int(raw['primaryIteration']) + 1 if 'primaryIteration' in raw.files else WRITE_PEAK_STATE
    state = raw['trainedState']
    holdState = int(raw['hold']) if 'hold' in raw.files else 301
    wanted = sorted(set(range(0, peakState + 1, stride)) | {holdState, peakState})
    conductance = state[:, coarse.NUM_CELLS:] / GREF
    curves = {}
    for name, cells in (('feature', layout.featureCells), ('background', layout.backgroundCells)):
        block = conductance[wanted][:, cells]
        curves.update({f'{name}Mean': block.mean(1), f'{name}Min': block.min(1), f'{name}Max': block.max(1)})
    millivolts = lambda k: [round(float(v), 1) for v in state[k, :coarse.NUM_CELLS] * 1000.0]
    return dict(states=wanted, **{k: [round(float(v), 4) for v in values] for k, values in curves.items()},
                vmem=dict(flood=millivolts(holdState), clear=millivolts(peakState), best=millivolts(peakState)))
