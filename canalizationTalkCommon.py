"""Shared pieces of the talk materials ("Can spatial bulk pattern development be canalized from the boundary?"): the two trained codes,
the replay of a ring code on the reference tissue, the drawing helpers and the palette of the reports. Used by
buildCanalizationTalkFigures11x11.py and buildCanalizationTalkMovies11x11.py. EXPLORATORY: nothing here was predicted or registered.
"""
import json

import matplotlib
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

import boundaryCodeUtilities as boundary
from boundaryHarmonicStep import Step

# ------------------------------------------------------------------ the two trained codes and where each is read
STRIPE_RUN_PATH = 'data/boundaryHarmonicTraining1888Hold301StripesInteriorMinus60Minus5Ceiling2/order2_restart06.npz'
FACE_RUN_PATH = 'data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz'
HOLD = 301
LATTICE = 11
THRESHOLD = -34.6
STRIPE_CELLS = np.array(sorted(boundary.centreStripeCellIndices.tolist()))
FACE_CELLS = np.array(sorted(boundary.featureCellIndices.tolist()))
INTERIOR = np.array(boundary.interiorCellIndices)
RING_CELLS = np.array(boundary.boundaryRingCells)
RING_ANGLES = boundary.ringAngles(RING_CELLS)                                   # clockwise from straight up, radians
TARGETS = {
    'stripe': dict(label='Stripe', runPath=STRIPE_RUN_PATH, cells=STRIPE_CELLS, readIteration=504, ceiling=2.0, colour='#0F7B9C'),
    'face': dict(label='Face', runPath=FACE_RUN_PATH, cells=FACE_CELLS, readIteration=2173, ceiling=1.3, colour='#C2622D'),
}
for key, target in TARGETS.items():
    run = np.load(target['runPath'], allow_pickle=True)
    target['coefficients'] = np.asarray(run['bestCoefficients'], float)
    target['score'] = float(run['bestScore'])
    assert int(run['bestIteration']) == target['readIteration']

# ------------------------------------------------------------------ palette of the reports
INK, INK_2, INK_3 = '#16202B', '#42525F', '#6B7A87'
TEAL, OCHRE, DIAL, SEQUENTIAL, GREEN, ROSE = '#0F7B9C', '#C2622D', '#7C4FA0', '#1B2A38', '#4F8A4B', '#B8456B'
VMEM_MAP = LinearSegmentedColormap.from_list('vmem', [SEQUENTIAL, '#FFFFFF'])           # drawn from -60 to -5 mV: dark = hyperpolarised, white = depolarised
CODE_MAP = LinearSegmentedColormap.from_list('code', ['#F4EEF8', DIAL])
VMEM_LOW, VMEM_HIGH = -60.0, -5.0

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.edgecolor': INK_3, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def ringValuesOf(coefficients, ceiling=2.0):
    """The 40 held values of a code a_0 + a_1 cos(theta) + ... , clipped to [0, ceiling] as in training."""
    return np.clip(np.cos(np.outer(RING_ANGLES, np.arange(len(coefficients)))) @ np.asarray(coefficients, float), 0.0, ceiling)


# ------------------------------------------------------------------ replay on the reference tissue
class Replayer:
    """One pure-step tissue (checkpoint 1888, the registered hold of 301 iterations), reused for every replay."""

    def __init__(self):
        torch.set_grad_enabled(False)
        self.step = Step(ringCode=np.zeros(len(RING_CELLS)))
        self.numCells = self.step.numCells

    def run(self, ringValues, numIterations, heldCells=None, keepEvery=1, holdIterations=HOLD):
        """Vmem (mV) of the whole tissue after every `keepEvery` iterations: row k is the state after iteration k + 1 steps, so row 504 is
        'iteration 504', the stripe code's best moment. `heldCells` holds only those ring cells (default every ring cell)."""
        step = self.step
        step.ringMask = torch.zeros(self.numCells, dtype=torch.double)
        step.ringMask[torch.as_tensor(RING_CELLS if heldCells is None else np.asarray(heldCells), dtype=torch.long)] = 1.0
        code = torch.zeros(self.numCells, dtype=torch.double)
        code[step.ring] = torch.as_tensor(ringValues, dtype=torch.double) * step.Gref
        step.ringCode = code
        vmem, gpol = step.initialVmem.clone(), step.initialGpol.clone()
        frames, conductance = [], []
        for iteration in range(numIterations):
            vmem, gpol = step(vmem, gpol, iteration < holdIterations, iteration < holdIterations)
            if iteration % keepEvery == 0 or iteration == numIterations - 1:
                frames.append(vmem.numpy() * 1000.0)
                conductance.append(gpol.numpy() / step.Gref)
        return np.array(frames), np.array(conductance)

    def readout(self, ringValues, readIteration, heldCells=None):
        return self.run(ringValues, readIteration + 1, heldCells=heldCells, keepEvery=readIteration + 1)[0][-1]


def darkCounts(vmem, cells):
    """(target cells dark, strays): dark interior cells inside and outside the target."""
    dark = np.asarray(vmem) < THRESHOLD
    inTarget = int(dark[cells].sum())
    strays = int(dark[[c for c in INTERIOR if c not in set(cells.tolist())]].sum())
    return inTarget, strays


# ------------------------------------------------------------------ drawing helpers
def drawTissue(axis, vmem, outlineCells=None, outlineColour=OCHRE, frameColour=None, ringGrey=True, lineWidth=2.0):
    """The 11 x 11 tissue, white = depolarised, dark = hyperpolarised (the reports' scale); the ring is set off by a thin frame."""
    image = np.asarray(vmem, float).reshape(LATTICE, LATTICE)
    axis.imshow(image, cmap=VMEM_MAP, vmin=VMEM_LOW, vmax=VMEM_HIGH, interpolation='nearest')
    for edge in np.arange(-0.5, LATTICE, 1.0):
        axis.axhline(edge, color='white', lw=0.8)
        axis.axvline(edge, color='white', lw=0.8)
    axis.add_patch(Rectangle((0.5, 0.5), LATTICE - 2, LATTICE - 2, fill=False, ec=INK_3, lw=0.9))
    if outlineCells is not None:
        outlineCellSet(axis, outlineCells, outlineColour, lineWidth)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(frameColour is not None)
        if frameColour is not None:
            spine.set_edgecolor(frameColour)
            spine.set_linewidth(3.0)


def outlineCellSet(axis, cells, colour, lineWidth=2.0, dashed=True):
    """Dashed outline around the union of the cells (the edges between a listed cell and an unlisted one)."""
    cellSet = set(int(c) for c in cells)
    for cell in cellSet:
        row, column = divmod(cell, LATTICE)
        for (dr, dc, xs, ys) in ((-1, 0, (-0.5, 0.5), (-0.5, -0.5)), (1, 0, (-0.5, 0.5), (0.5, 0.5)), (0, -1, (-0.5, -0.5), (-0.5, 0.5)), (0, 1, (0.5, 0.5), (-0.5, 0.5))):
            neighbourRow, neighbourColumn = row + dr, column + dc
            inside = 0 <= neighbourRow < LATTICE and 0 <= neighbourColumn < LATTICE
            if not inside or neighbourRow * LATTICE + neighbourColumn not in cellSet:
                axis.plot([column + xs[0], column + xs[1]], [row + ys[0], row + ys[1]], color=colour, lw=lineWidth,
                          ls=(0, (3, 2)) if dashed else '-', solid_capstyle='butt')


BISTABLE_EDGE = 1.439                                                           # above it a lone cell has only its hyperpolarised state


def drawRingCode(axis, ringValues, vmax, markAboveEdge=True):
    """The held ring values painted on the 40 ring cells (darker purple = higher G_pol / G_ref); the interior left blank. A dot marks a cell
    held above the bistable window's upper edge, where a lone cell can only be dark."""
    image = np.full(LATTICE * LATTICE, np.nan)
    image[RING_CELLS] = ringValues
    axis.imshow(image.reshape(LATTICE, LATTICE), cmap=CODE_MAP, vmin=0, vmax=vmax, interpolation='nearest')
    if markAboveEdge:
        for cell, value in zip(RING_CELLS, ringValues):
            if value > BISTABLE_EDGE:
                axis.plot(cell % LATTICE, cell // LATTICE, 'o', color=OCHRE, ms=6, mec='white', mew=0.8)
    for edge in np.arange(-0.5, LATTICE, 1.0):
        axis.axhline(edge, color='white', lw=0.8)
        axis.axvline(edge, color='white', lw=0.8)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def newFigure(width=13.33, height=7.5):
    return plt.figure(figsize=(width, height))


def save(figure, path, overwrite, dpi=200):
    import os
    if os.path.exists(path) and not overwrite:
        print(f'  exists, kept: {path}')
        plt.close(figure)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    plt.close(figure)
    print(f'  wrote {path}')
