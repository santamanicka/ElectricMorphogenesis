"""The visual essence of the talk "Can bulk pattern formation be canalized from the boundary?": slick figures and movies, almost no numbers. EXPLORATORY.

The picture: a code is held on the boundary of an 11 x 11 tissue like an embryonic organizer (Spemann-Mangold's is the classic case); the tissue responds in two phases (guidance, while the
boundary is held and the code is copied inward; self-organization, once the boundary lets go and the tissue's own machinery completes the pattern); a few waves on the boundary are enough to
guide a whole interior of cells. Boundary cells are drawn in violet (brighter = higher held G_pol), interior cells glow where they are hyperpolarised (cyan for the stripe, amber for the face).
Dark theme; the quantitative companions are in presentation/backup_quantitative/.

    python3 buildCanalizationTalkEssence11x11.py --ffmpeg /path/to/ffmpeg-with-libx264        # everything
    python3 buildCanalizationTalkEssence11x11.py --parts figures                              # figures only
    python3 buildCanalizationTalkEssence11x11.py --parts spatialOrganizer,canals --overwrite  # some

Reads the replays cached by buildCanalizationTalkFigures11x11.py (data/canalizationTalkReplays1888Hold301.npz; run its stages codes, slidingPatterns and slidingAllOrders first)
and replays a few partial holds. Writes presentation/<number>_<name>.png and presentation/movies/<letter>_<name>.mp4 (never overwriting unless --overwrite).
"""
import argparse
import json
import os
import shutil
import subprocess

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import gaussian_filter

import boundaryCodeUtilities as boundary
from canalizationTalkCommon import *

parser = argparse.ArgumentParser()
parser.add_argument('--parts', type=str, default='spatialOrganizer,twoPhases,stripeSwitches,faceNoSinglePart,canals,push,guideThenLetGo,turningTheKnobs,lingering,lingerThenWander,clusters,familySpace,relayKnobs,relayKnobsThresh,relayKnobsThreshGhost')
parser.add_argument('--outputDirectory', type=str, default='presentation')
parser.add_argument('--cachePath', type=str, default='data/canalizationTalkReplays1888Hold301.npz')
parser.add_argument('--partialCachePath', type=str, default='data/canalizationTalkPartialHolds1888Hold301.npz')
parser.add_argument('--familyPath', type=str, default='data/canalizationTalkFamilyTrajectories1888Hold301.npz', help='from analyzeCanalizationTalkFamilyVisits11x11.py')
parser.add_argument('--flagsPath', type=str, default='data/canalizationTalkFamilyFlags1888Hold301.npz')
parser.add_argument('--longHorizonPath', type=str, default='data/canalizationTalkLongHorizon64Trained1888Hold301.npz')
parser.add_argument('--clusterPath', type=str, default='data/canalizationTalkPatternClusters1888Hold301.json', help='from analyzeCanalizationTalkPatternClusters11x11.py')
parser.add_argument('--clusterScoresPath', type=str, default='data/canalizationTalkPatternClusterScores1888Hold301.npz')
parser.add_argument('--distancePath', type=str, default='data/canalizationTalkFamilyDistance1888Hold301.json', help='from analyzeCanalizationTalkFamilyDistance11x11.py')
parser.add_argument('--distanceScoresPath', type=str, default='data/canalizationTalkFamilyDistanceScores1888Hold301.npz')
parser.add_argument('--netStopPatternsPath', type=str, default='data/canalizationTalkNetStopPatterns1888Hold301.npz', help='cache: the tissue at the readout for every simulated knob setting of the relay-net movie')
parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
parser.add_argument('--framesDirectory', type=str, default='canalizationTalkEssenceFrames')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
parts = args.parts.split(',')
cache = dict(np.load(args.cachePath, allow_pickle=True))
partialCache = dict(np.load(args.partialCachePath, allow_pickle=True)) if os.path.exists(args.partialCachePath) else {}
replayer = None

# ------------------------------------------------------------------------------------------------------------------ the look
BACKGROUND, INK_LIGHT, MUTED, FAINT = '#0A111A', '#EAF0F5', '#8796A5', '#2A3947'
BASE = np.array([0x15, 0x22, 0x31]) / 255.0
FAINT_ARRAY = np.array([0x1B, 0x2A, 0x3A]) / 255.0
FAINT_TEXT = '#566676'
VIOLET = np.array([0xB5, 0x8C, 0xFF]) / 255.0
GLOW = {'stripe': np.array([0x35, 0xD0, 0xE6]) / 255.0, 'face': np.array([0xFF, 0xB0, 0x4A]) / 255.0}
GLOW_HEX = {'stripe': '#35D0E6', 'face': '#FFB04A'}
LABEL = {'stripe': 'stripe', 'face': 'face'}
plt.rcParams.update({'figure.facecolor': BACKGROUND, 'axes.facecolor': BACKGROUND, 'savefig.facecolor': BACKGROUND, 'text.color': INK_LIGHT,
                     'axes.edgecolor': FAINT, 'axes.labelcolor': INK_LIGHT, 'xtick.color': MUTED, 'ytick.color': MUTED, 'font.family': 'DejaVu Sans'})
WIDTH, HEIGHT, DPI = 16.0, 9.0, 120                                            # 1920 x 1080


def brightness(vmem):
    return np.clip((-5.0 - np.asarray(vmem, float)) / 55.0, 0.0, 1.0)


def interiorColour(bright, key):
    """Depolarised = the cell's dim slate; hyperpolarised = the target's glow, with a white-hot core at the strongest."""
    mix = bright[..., None] ** 0.85
    colour = BASE * (1 - mix) + GLOW[key] * mix
    return np.clip(colour + 0.32 * (bright[..., None] ** 4) * (1 - colour), 0, 1)


RING_RANGE = {'stripe': (1.1, 1.5), 'face': (0.0, 1.35)}                      # the boundary's colour scale is stretched per target so the stripe's two bumps can be seen


def codeColour(value, key):
    low, high = RING_RANGE[key]
    mix = 0.22 + 0.78 * np.clip((np.asarray(value, float) - low) / (high - low), 0, 1)
    return np.clip(BASE * (1 - mix[..., None]) + VIOLET * mix[..., None], 0, 1)


class Glyph:
    """The tissue as 121 rounded cells over a soft glow: violet boundary cells carry the code, interior cells glow where they are hyperpolarised."""

    def __init__(self, axis, key, targetOutline=False):
        self.axis, self.key = axis, key
        axis.set_xlim(-0.75, 10.75)
        axis.set_ylim(10.75, -0.75)
        axis.set_aspect('equal')
        axis.axis('off')
        self.glow = axis.imshow(np.zeros((110, 110, 4)), extent=(-0.5, 10.5, 10.5, -0.5), zorder=0, interpolation='bilinear')
        self.patches = []
        for row in range(LATTICE):
            for column in range(LATTICE):
                patch = FancyBboxPatch((column - 0.43, row - 0.43), 0.86, 0.86, boxstyle='round,pad=0,rounding_size=0.17', fc=tuple(BASE), ec='none', zorder=2)
                axis.add_patch(patch)
                self.patches.append(patch)
        self.outline = []
        if targetOutline:
            self.outline = self._outline(TARGETS[key]['cells'])

    def _outline(self, cells):
        lines = []
        cellSet = set(int(c) for c in cells)
        for cell in cellSet:
            row, column = divmod(cell, LATTICE)
            for dr, dc, xs, ys in ((-1, 0, (-0.5, 0.5), (-0.5, -0.5)), (1, 0, (-0.5, 0.5), (0.5, 0.5)), (0, -1, (-0.5, -0.5), (-0.5, 0.5)), (0, 1, (0.5, 0.5), (-0.5, 0.5))):
                neighbour = (row + dr) * LATTICE + (column + dc)
                if not (0 <= row + dr < LATTICE and 0 <= column + dc < LATTICE) or neighbour not in cellSet:
                    line, = self.axis.plot([column + xs[0], column + xs[1]], [row + ys[0], row + ys[1]], color=MUTED, lw=1.1, ls=(0, (2, 3)), alpha=0.0, zorder=3)
                    lines.append(line)
        return lines

    def showOutline(self, alpha):
        for line in self.outline:
            line.set_alpha(alpha)

    def update(self, vmem, ringValues=None, ringWeight=1.0, heldCells=None, glowStrength=0.55):
        """vmem: 121 values (mV). ringValues: the 40 code values; ringWeight 0..1 mixes the code colours over the boundary's own (a fade-in or fade-out)."""
        colours = interiorColour(brightness(vmem), self.key)
        if ringValues is not None and ringWeight > 0:
            ringColours = codeColour(ringValues, self.key)
            if heldCells is not None:                                            # only some boundary cells are held: the others stay dim
                held = np.isin(RING_CELLS, heldCells)
                ringColours = np.where(held[:, None], ringColours, BASE * 1.25)
            colours[RING_CELLS] = colours[RING_CELLS] * (1 - ringWeight) + ringColours * ringWeight
        for patch, colour in zip(self.patches, colours):
            patch.set_facecolor(tuple(colour))
        lit = brightness(vmem).reshape(LATTICE, LATTICE) ** 1.5
        lit[[0, -1], :] = 0
        lit[:, [0, -1]] = 0
        blurred = gaussian_filter(np.kron(lit, np.ones((10, 10))), sigma=9)
        rgba = np.zeros((110, 110, 4))
        rgba[..., :3] = GLOW[self.key]
        rgba[..., 3] = np.clip(1.6 * blurred, 0, 1) * glowStrength
        self.glow.set_data(rgba)


def ringRows(coefficients, ceiling=2.0):
    return ringValuesOf(coefficients, ceiling)


def newFigure():
    return plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)


def caption(figure, x, y, text, size=17, colour=MUTED, weight='normal', ha='center'):
    return figure.text(x, y, text, ha=ha, va='center', fontsize=size, color=colour, fontweight=weight)


def arrow(figure, start, end, colour=MUTED, width=2.0, rad=0.0):
    figure.add_artist(FancyArrowPatch(start, end, transform=figure.transFigure, arrowstyle='-|>', mutation_scale=22, lw=width, color=colour, connectionstyle=f'arc3,rad={rad}'))


def savePicture(figure, name):
    path = f'{args.outputDirectory}/{name}'
    if os.path.exists(path) and not args.overwrite:
        print(f'  exists, kept: {path}')
    else:
        os.makedirs(args.outputDirectory, exist_ok=True)
        figure.savefig(path, dpi=DPI)
        print(f'  wrote {path}')
    plt.close(figure)


def trainedCourse(key):
    return cache[f'course_{key}']


def partialHold(key, name, heldCells):
    """The tissue at the target's readout when only some boundary cells are held at the trained code's values (the others follow the model's own rules)."""
    global replayer, partialCache
    storeKey = f'{key}_{name}'
    if storeKey not in partialCache:
        if replayer is None:
            replayer = Replayer()
        target = TARGETS[key]
        partialCache[storeKey] = replayer.readout(ringValuesOf(target['coefficients'], target['ceiling']), target['readIteration'], heldCells=np.array(heldCells, dtype=int)).astype(np.float32)
        np.savez_compressed(args.partialCachePath, **partialCache)
    return partialCache[storeKey]


def drawDial(figure, centreX, centreY, width=0.034, angle=90.0, colour=None, ghostAngles=()):
    """A dial in the style of the movies: 17 ticks over 240 degrees, a long tick at the trained (upright) setting, a pointer. Returns the dial's axes."""
    height = width * 16 / 9                                                                  # a round dial on a 16:9 figure
    axis = figure.add_axes([centreX - width / 2, centreY - height / 2, width, height])
    axis.set_xlim(-1.5, 1.5); axis.set_ylim(-1.5, 1.5); axis.set_aspect('equal'); axis.axis('off')
    axis.add_patch(plt.Circle((0, 0), 1.0, fc=BASE * 1.5, ec=FAINT, lw=1.8))
    for tick in np.linspace(-30, 210, 17):
        a = np.radians(tick)
        isCentre = abs(tick - 90) < 1e-6
        axis.plot([1.12 * np.cos(a), (1.42 if isCentre else 1.28) * np.cos(a)], [1.12 * np.sin(a), (1.42 if isCentre else 1.28) * np.sin(a)], color=INK_LIGHT if isCentre else FAINT, lw=1.8 if isCentre else 1.1)
    for ghost in ghostAngles:                                                                # the dial turned elsewhere, faint
        g = np.radians(ghost)
        axis.plot([0, 0.88 * np.cos(g)], [0, 0.88 * np.sin(g)], color=tuple(VIOLET), lw=2.2, alpha=0.38, solid_capstyle='round')
    a = np.radians(angle)
    axis.plot([0, 0.88 * np.cos(a)], [0, 0.88 * np.sin(a)], color=tuple(VIOLET) if colour is None else colour, lw=3.2, solid_capstyle='round')
    return axis


# ================================================================================================================== 1 the spatial organizer
WAVE_NAMES = ['a level', 'a tilt', 'an oval', 'a trefoil']

if 'spatialOrganizer' in parts:
    figure = newFigure()
    grid = figure.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], left=0.17, right=0.97, top=0.84, bottom=0.07, wspace=0.18, hspace=0.18)
    degrees = np.linspace(-180, 180, 361)
    contentExtent = {}                                                                         # per row: top of the first dial and bottom of the summed wave, in figure coordinates
    for row, key in enumerate(('stripe', 'face')):
        target = TARGETS[key]
        coefficients = target['coefficients']
        count = len(coefficients)
        inner = grid[row, 0].subgridspec(count + 1, 1, hspace=0.35)
        for order in range(count):
            axis = figure.add_subplot(inner[order])
            axis.plot(degrees, coefficients[order] * np.cos(np.radians(order * degrees)), color=VIOLET, lw=2.6, alpha=0.95 - 0.12 * order, solid_capstyle='round')
            for shift in (-0.3, 0.3):                                                         # the same wave with its dial turned down or up (illustration)
                axis.plot(degrees, (coefficients[order] + shift) * np.cos(np.radians(order * degrees)), color=VIOLET, lw=1.5, alpha=0.34, ls=(0, (3, 3)))
            axis.set_ylim(-1.0, 1.7)
            axis.set_xlim(-180, 180)
            axis.axis('off')
            axis.text(-190, 0.35, WAVE_NAMES[order], ha='right', va='center', fontsize=12, color=MUTED)
            axis.text(190, 0.35, '+' if order < count - 1 else '=', ha='left', va='center', fontsize=18, color=MUTED)
            position = axis.get_position()
            setting = 90 - 80 * float(coefficients[order])                                      # the dial shows the size of its wave: up = none, clockwise = positive (80 degrees per unit)
            drawDial(figure, position.x0 - 0.092, position.y0 + position.height * 0.5, width=0.040, angle=setting, ghostAngles=(setting + 24, setting - 24))   # +-0.3 turned down / up
            if order == 0:
                contentExtent[row] = [position.y0 + position.height * 0.5 + 0.034, None]                  # the dial of this wave, at the trained setting
        axis = figure.add_subplot(inner[count])
        total = ringValuesOf(coefficients, target['ceiling'])
        order = np.argsort(RING_ANGLES)
        axis.plot(np.degrees(RING_ANGLES[order]), total[order], color=INK_LIGHT, lw=3.2, solid_capstyle='round')
        axis.set_ylim(-0.1, 1.9)
        axis.set_xlim(-180, 180)
        axis.axis('off')
        totalBox = axis.get_position()
        contentExtent[row][1] = totalBox.y0 + totalBox.height * (float(total.min()) + 0.1) / 2.0
        glyphBoundary = Glyph(figure.add_subplot(grid[row, 1]), key)
        glyphBoundary.update(np.full(121, -5.0), ringValues=total)
        glyphOut = Glyph(figure.add_subplot(grid[row, 2]), key)
        glyphOut.update(trainedCourse(key)[target['readIteration']], ringValues=total, ringWeight=0.0, glowStrength=0.75)
        glyphOut.showOutline(0.55)
    for column, text in enumerate(('a few waves, each with a dial…', '…held on the boundary…', '…guide the tissue')):
        columnBox = grid[0, column].get_position(figure)
        caption(figure, columnBox.x0 + columnBox.width / 2 - (0.06 if column == 0 else 0.0), 0.915, text, size=21, colour=INK_LIGHT)
    figure.text(0.5, 0.968, 'The code is a spatial organizer', ha='center', va='center', fontsize=30, fontweight='bold', color=INK_LIGHT)
    for row, key in enumerate(('stripe', 'face')):                                           # the vertical label: centred on its block of rows, close to the dials
        rowBox = grid[row, 0].get_position(figure)
        figure.text(rowBox.x0 - 0.092 - 0.046, sum(contentExtent[row]) / 2, 'a simple organizer' if key == 'stripe' else 'a complex organizer', ha='center', va='center', fontsize=19,
                    color=GLOW_HEX[key], fontweight='bold', rotation=90)
    figure.text(0.27, 0.032, 'dial = the size of its wave (up: none, clockwise: positive)\nfaint: the dial turned down or up by 0.3, and the wave it would give', ha='center', va='center', fontsize=12, color=FAINT_TEXT, linespacing=1.5)
    savePicture(figure, '1_theSpatialOrganizer.png')

# ================================================================================================================== 2 two steps of reading
if 'twoPhases' in parts:
    columns = [(0, 'hold'), (150, 'hold'), (300, 'hold'), (504, 'release'), (1000, 'release'), (1850, 'release'), (2173, 'release')]
    figure = newFigure()
    grid = figure.add_gridspec(2, len(columns), left=0.04, right=0.985, top=0.74, bottom=0.07, wspace=0.04, hspace=0.12)
    for row, key in enumerate(('stripe', 'face')):
        target = TARGETS[key]
        ring = ringValuesOf(target['coefficients'], target['ceiling'])
        course = trainedCourse(key)
        for column, (iteration, phase) in enumerate(columns):
            glyph = Glyph(figure.add_subplot(grid[row, column]), key)
            glyph.update(course[iteration], ringValues=ring, ringWeight=1.0 if phase == 'hold' else 0.0, glowStrength=0.6)
            if iteration == target['readIteration']:
                for spine_position in ((-0.7, -0.7), ):
                    glyph.axis.add_patch(FancyBboxPatch((-0.62, -0.62), 11.24, 11.24, boxstyle='round,pad=0,rounding_size=0.5', fc='none', ec=GLOW_HEX[key], lw=3.2, alpha=0.9, zorder=5))
    # the two brackets
    left, right = 0.04, 0.985
    unit = (right - left) / len(columns)
    for start, stop, label, colour in ((0, 3, 'guidance: the boundary is held', VIOLET), (3, len(columns), 'self-organization: the boundary lets go', INK_LIGHT)):
        x0, x1 = left + start * unit + 0.006, left + stop * unit - 0.006
        figure.add_artist(plt.Line2D([x0, x1], [0.79, 0.79], transform=figure.transFigure, color=colour, lw=3.5, solid_capstyle='round'))
        figure.text((x0 + x1) / 2, 0.835, label, ha='center', va='center', fontsize=17, color=colour)
    for row, key in enumerate(('stripe', 'face')):
        figure.text(0.012, 0.545 - row * 0.325, 'stripe' if key == 'stripe' else 'face', ha='center', va='center', fontsize=20, color=GLOW_HEX[key], fontweight='bold', rotation=90)
    figure.text(0.5, 0.955, 'Guide, then let go', ha='center', va='center', fontsize=30, fontweight='bold')
    figure.text(0.5, 0.03, 'a simple organizer writes its pattern while it is held; a complex one is finished by the tissue long after it lets go', ha='center', va='center', fontsize=16, color=MUTED)
    savePicture(figure, '2_twoPhases.png')

# ================================================================================================================== 3 the stripe's two switches
if 'stripeSwitches' in parts:
    target = TARGETS['stripe']
    ring = ringValuesOf(target['coefficients'], target['ceiling'])
    endCells = RING_CELLS[ring > BISTABLE_EDGE]
    topEnd, bottomEnd = [int(c) for c in endCells if c < 55], [int(c) for c in endCells if c >= 55]
    variants = [('top end only', topEnd, partialHold('stripe', 'topEnd', topEnd)), ('bottom end only', bottomEnd, partialHold('stripe', 'bottomEnd', bottomEnd)),
                ('both ends', topEnd + bottomEnd, partialHold('stripe', 'bothEnds', topEnd + bottomEnd))]
    figure = newFigure()
    grid = figure.add_gridspec(1, 3, left=0.04, right=0.96, top=0.80, bottom=0.12, wspace=0.10)
    for column, (label, held, pattern) in enumerate(variants):
        glyph = Glyph(figure.add_subplot(grid[0, column]), 'stripe', targetOutline=True)
        glyph.update(pattern, ringValues=ring, heldCells=held, glowStrength=0.7)
        glyph.showOutline(0.5)
        caption(figure, 0.04 + (0.92 / 3) * (column + 0.5), 0.085, label, size=21, colour=INK_LIGHT)
    figure.text(0.5, 0.955, 'Two bumps on the boundary, two switches in the tissue', ha='center', va='center', fontsize=30, fontweight='bold')
    figure.text(0.5, 0.885, 'only the lit boundary cells are held; the rest of the boundary is left to the tissue', ha='center', va='center', fontsize=17, color=MUTED)
    savePicture(figure, '3_stripeTwoSwitches.png')

# ================================================================================================================== 4 the face has no single part
if 'faceNoSinglePart' in parts:
    wall = json.load(open('data/boundaryHarmonicWallCounterfactual1888Hold301FaceMinus60Minus5.json'))
    target = TARGETS['face']
    ring = ringValuesOf(target['coefficients'], target['ceiling'])
    variants = [('the upper wall alone', wall['upperWall']['cells'], partialHold('face', 'upperWall', wall['upperWall']['cells'])),
                ('the lower wall alone', wall['lowerWall']['cells'], partialHold('face', 'lowerWall', wall['lowerWall']['cells'])),
                ('the whole boundary', [int(c) for c in RING_CELLS], partialHold('face', 'wholeRing', [int(c) for c in RING_CELLS]))]
    figure = newFigure()
    grid = figure.add_gridspec(1, 3, left=0.04, right=0.96, top=0.80, bottom=0.12, wspace=0.10)
    for column, (label, held, pattern) in enumerate(variants):
        glyph = Glyph(figure.add_subplot(grid[0, column]), 'face', targetOutline=True)
        glyph.update(pattern, ringValues=ring, heldCells=held, glowStrength=0.7)
        glyph.showOutline(0.5)
        caption(figure, 0.04 + (0.92 / 3) * (column + 0.5), 0.085, label, size=21, colour=INK_LIGHT)
    figure.text(0.5, 0.955, 'No single part of the boundary writes the face', ha='center', va='center', fontsize=30, fontweight='bold')
    figure.text(0.5, 0.885, 'each wall pulls the tissue part-way; only together do they release the whole pattern', ha='center', va='center', fontsize=17, color=MUTED)
    savePicture(figure, '4_faceNoSinglePart.png')

# ================================================================================================================== 5 canals of the dial
if 'canals' in parts:
    offsets = np.round(np.arange(-0.08, 0.0801, 0.002), 6)
    figure = newFigure()
    outer = figure.add_gridspec(2, 1, left=0.08, right=0.97, top=0.84, bottom=0.07, hspace=0.18)
    side = 0.095                                                                      # thumbnail width as a share of the figure; height chosen to make it square
    for row, key in enumerate(('stripe', 'face')):
        target = TARGETS[key]
        patterns = cache[f'slide_{key}_order0_step0.002']
        dark = patterns[:, INTERIOR] < THRESHOLD
        share = np.array([darkCounts(p, target['cells'])[0] for p in patterns]) / len(target['cells'])
        runs, begin = [], 0                                                               # stretches of identical patterns along the dial
        for index in range(1, len(dark) + 1):
            if index == len(dark) or not np.array_equal(dark[index], dark[begin]):
                runs.append((begin, index - 1))
                begin = index
        inner = outer[row].subgridspec(2, 1, height_ratios=[1.35, 1], hspace=0.02)
        holder = figure.add_subplot(inner[0])
        holder.axis('off')
        axis = figure.add_subplot(inner[1])
        axis.set_xlim(-0.083, 0.083)
        axis.set_ylim(-0.42, 1.08)
        axis.axis('off')
        for first, last in runs:                                                          # the staircase: a bright glowing step for every canal
            x0, x1 = offsets[first] - 0.001, offsets[last] + 0.001
            length = last - first + 1
            strength = 0.35 + 0.65 * min(length, 8) / 8.0
            for width, alpha in ((18, 0.10), (10, 0.18), (5, 1.0)):
                axis.plot([x0, x1], [share[first]] * 2, color=GLOW_HEX[key], lw=width * (0.55 + min(length, 12) / 12.0), alpha=alpha * strength, solid_capstyle='round')
        axis.plot(offsets, share, color=INK_LIGHT, lw=0.8, alpha=0.18)
        axis.axvline(0.0, color=MUTED, lw=1.0, ls=(0, (2, 4)), alpha=0.5)
        trainedRun = [r for r in runs if r[0] <= int(np.argmin(np.abs(offsets))) <= r[1]][0]
        others = sorted([r for r in runs if r != trainedRun], key=lambda r: -(r[1] - r[0]))[:4]
        chosen = sorted([trainedRun] + others, key=lambda r: r[0])
        holderBox, axisBox = holder.get_position(), axis.get_position()
        placed = []
        for first, last in chosen:
            middle = (first + last) // 2
            centre = (offsets[first] + offsets[last]) / 2
            xCentre = axisBox.x0 + (centre + 0.083) / 0.166 * axisBox.width
            for taken in placed:                                                          # keep neighbouring thumbnails apart
                if abs(xCentre - taken) < side * 1.12:
                    xCentre = taken + side * 1.12 * (1 if xCentre >= taken else -1)
            placed.append(xCentre)
            height = side * 16 / 9
            small = figure.add_axes([xCentre - side / 2, holderBox.y0 + holderBox.height - height, side, height])
            glyph = Glyph(small, key)
            glyph.update(patterns[middle], glowStrength=0.6)
            if (first, last) == trainedRun:
                small.add_patch(FancyBboxPatch((-0.7, -0.7), 11.4, 11.4, boxstyle='round,pad=0,rounding_size=0.6', fc='none', ec=GLOW_HEX[key], lw=2.5, alpha=0.9))
            xPlateau = axisBox.x0 + (centre + 0.083) / 0.166 * axisBox.width
            yPlateau = axisBox.y0 + (share[middle] + 0.42) / 1.5 * axisBox.height
            figure.add_artist(plt.Line2D([xCentre, xPlateau], [holderBox.y0 + holderBox.height - height, yPlateau + 0.012], transform=figure.transFigure, color=FAINT, lw=1.2))
        longest = max(runs, key=lambda r: r[1] - r[0])
        axis.annotate('', xy=(offsets[longest[0]], share[longest[0]] - 0.12), xytext=(offsets[longest[1]], share[longest[1]] - 0.12), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=1.5))
        axis.text((offsets[longest[0]] + offsets[longest[1]]) / 2, share[longest[0]] - 0.20, 'a canal: many codes, one pattern', ha='center', va='top', fontsize=15, color=MUTED)
        figure.text(0.03, axisBox.y0 + axisBox.height * 0.9, LABEL[key], ha='center', va='center', fontsize=21, color=GLOW_HEX[key], fontweight='bold', rotation=90)
    figure.text(0.5, 0.955, 'Canals: turn the dial and the pattern holds, then jumps', ha='center', va='center', fontsize=30, fontweight='bold')
    figure.text(0.5, 0.895, 'height: how much of the target is present; each flat stretch is a range of codes that give the identical pattern', ha='center', va='center', fontsize=16, color=MUTED)
    figure.text(0.5, 0.025, 'turn the dial  \u2192', ha='center', va='center', fontsize=16, color=MUTED)
    savePicture(figure, '5_canalsOfTheDial.png')

# the regions of the tissue as nodes of the relay net, in lattice coordinates (x across, y down), and each node's mirror twin
NET_POSITIONS = {
    'stripe': dict(ringTop=(5, 0), ringBottom=(5, 10), ringLeft=(0, 5), ringRight=(10, 5), stripeUpper=(5, 3), stripeLower=(5, 7), flankLeftUpper=(2, 3), flankLeftLower=(2, 7),
                   flankRightUpper=(8, 3), flankRightLower=(8, 7)),
    'face': dict(ringTop=(5, 0), ringBottom=(5, 10), ringLeft=(0, 5), ringRight=(10, 5), bgTL=(2.9, 2.9), bgTR=(7.1, 2.9), bgBL=(2.7, 7.5), bgBR=(7.3, 7.5), eyes=(5, 2.5), nose=(5, 5), mouth=(5, 8))}
NET_MIRROR = dict(ringLeft='ringRight', bgTL='bgTR', bgBL='bgBR', flankLeftUpper='flankRightUpper', flankLeftLower='flankRightLower')
NET_MIRROR.update({value: key for key, value in list(NET_MIRROR.items())})

# ================================================================================================================== 6 the push (the relay)
if 'push' in parts:
    stripeNets = json.load(open('data/relayLoopStripesPageData1888Hold301StripesInteriorMinus60Minus5.json'))
    faceNets = json.load(open('data/relayLoopFullNets1888Hold301FaceMinus60Minus5.json'))
    positions, mirror = NET_POSITIONS, NET_MIRROR
    def edgesOf(key):
        if key == 'stripe':
            raw = [(e['from'], e['to'], e['value']) for phase in ('flood', 'clear') for e in stripeNets['trainedTop3Phases'][phase]]
        else:
            trained = faceNets['codes']['trained']
            raw = []
            for phase in range(3):
                values = np.array(trained['field'][phase])
                for index in np.argsort(-np.abs(values))[:3]:
                    pair = faceNets['pairs'][index]
                    raw.append((pair[0], pair[1], float(values[index])) if values[index] > 0 else (pair[1], pair[0], float(-values[index])))
        both = list(raw)
        for sender, receiver, value in raw:
            twin = (mirror.get(sender, sender), mirror.get(receiver, receiver), value)
            if twin[:2] != (sender, receiver):
                both.append(twin)
        return both
    figure = newFigure()
    grid = figure.add_gridspec(1, 2, left=0.04, right=0.96, top=0.80, bottom=0.10, wspace=0.12)
    for column, key in enumerate(('stripe', 'face')):
        target = TARGETS[key]
        axis = figure.add_subplot(grid[0, column])
        glyph = Glyph(axis, key)
        glyph.update(trainedCourse(key)[target['readIteration']], ringValues=ringValuesOf(target['coefficients'], target['ceiling']), glowStrength=0.35)
        edges = edgesOf(key)
        scale = max(e[2] for e in edges)
        for sender, receiver, value in sorted(edges, key=lambda e: e[2]):
            (x0, y0), (x1, y1) = positions[key][sender], positions[key][receiver]
            axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>,head_length=0.55,head_width=0.28', mutation_scale=12, lw=1.2 + 7.0 * value / scale, color=INK_LIGHT, alpha=0.55 + 0.4 * value / scale,
                                           shrinkA=7, shrinkB=8, connectionstyle='arc3,rad=0.12', zorder=6))
        caption(figure, 0.04 + 0.46 * (column + 0.5) + 0.0, 0.055, 'a direct push, from each end of the boundary into the stripe' if key == 'stripe' else 'a loop through the tissue that reverses and keeps circulating', size=17, colour=INK_LIGHT)
    figure.text(0.5, 0.955, 'How the boundary reaches the bulk', ha='center', va='center', fontsize=30, fontweight='bold')
    figure.text(0.5, 0.885, 'the strongest transfers between regions of the tissue; arrows are drawn over the pattern they build', ha='center', va='center', fontsize=17, color=MUTED)
    savePicture(figure, '6_thePush.png')

# ================================================================================================================== 7 the imprint lingers, then the tissue wanders off
FAMILY_NAME = {'stripe': 'stripes', 'face': 'deformed faces'}


def loadFamilyRuns():
    """Dark sets (runs x frames x 81), the family flags and the run names, from analyzeCanalizationTalkFamilyVisits11x11.py."""
    trajectories = np.load(args.familyPath, allow_pickle=True)
    flagFile = np.load(args.flagsPath, allow_pickle=True)
    names, groups = list(trajectories['names']), list(trajectories['groups'])
    dark = np.unpackbits(trajectories['packed'], axis=2)[:, :, :81].astype(bool)
    return names, groups, dark, flagFile, int(trajectories['stride'])


def familyFlag(flagFile, names, name, key):
    row = names.index(name)
    return flagFile['bars'][row] > 0 if key == 'stripe' else flagFile['faceBest'][row].astype(np.float32) >= 0.8


def displayColumns(stride, frames):
    """Time on a broken axis: iterations 0-3000 at full resolution take 60% of the width, 3000-20000 are squeezed into the remaining 40% (a column keeps a visit if any frame in it has one)."""
    early = int(3000 / stride)
    groupsLate = np.array_split(np.arange(early, frames), 400)
    return early, groupsLate


def ribbonRow(flag, early, groupsLate):
    return np.concatenate([flag[:early].astype(float), np.array([flag[g].max() for g in groupsLate])])


def brokenAxisPosition(iterations):
    iterations = np.asarray(iterations, float)
    return np.where(iterations <= 3000, iterations / 3000 * 600, 600 + (iterations - 3000) / 17000 * 400)


def expandToTissue(bits):
    vmem = np.full(121, -5.0)
    vmem[INTERIOR] = np.where(bits, -60.0, -5.0)
    return vmem


if 'lingering' in parts:
    names, groups, dark, flagFile, stride = loadFamilyRuns()
    early, groupsLate = displayColumns(stride, dark.shape[1])
    iterationOf = np.arange(dark.shape[1]) * stride
    figure = newFigure()
    figure.text(0.5, 0.955, 'The imprint lingers, then the tissue wanders off', ha='center', va='center', fontsize=30, fontweight='bold')
    figure.text(0.5, 0.908, 'each row is one tissue after release; it lights up whenever the tissue shows a pattern from its code\u2019s own family', ha='center', va='center', fontsize=16, color=MUTED)
    blockTop = {'stripe': 0.845, 'face': 0.475}
    left, width = 0.145, 0.525
    for key in ('stripe', 'face'):
        trained = f'{key}Trained'
        copies = [n for n, g in zip(names, groups) if g == f'{key}Copy'][:4]
        controls = [n for n, g in zip(names, groups) if g == f'{key}Control'][:5]
        rows = [trained] + copies + controls
        top = blockTop[key]
        ribbonHeight = 0.19
        axis = figure.add_axes([left, top - 0.03 - ribbonHeight, width, ribbonHeight])
        image = np.zeros((len(rows), 1000, 3))
        for r, name in enumerate(rows):
            line = ribbonRow(familyFlag(flagFile, names, name, key), early, groupsLate)
            strength = 1.0 if r == 0 else (0.62 if r <= len(copies) else 0.42)
            image[r] = FAINT_ARRAY[None, :] * (1 - line[:, None]) + (GLOW[key] * strength)[None, :] * line[:, None]
        axis.imshow(image, aspect='auto', interpolation='nearest', extent=(0, 1000, len(rows), 0))
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.axhline(1, color=BACKGROUND, lw=3)
        axis.axhline(1 + len(copies), color=BACKGROUND, lw=3)
        for position in (brokenAxisPosition(301), brokenAxisPosition(3000)):
            axis.axvline(position, color=INK_LIGHT, lw=1.2, alpha=0.7, ls=(0, (3, 3)))
        labelX, unit = left - 0.008, ribbonHeight / len(rows)
        figure.text(labelX, top - 0.03 - unit * 0.5, 'the code', ha='right', va='center', fontsize=14, color=GLOW_HEX[key], fontweight='bold')
        figure.text(labelX, top - 0.03 - unit * (1 + len(copies) / 2), 'nearby codes', ha='right', va='center', fontsize=12, color=MUTED)
        figure.text(labelX, top - 0.03 - unit * (1 + len(copies) + len(controls) / 2), 'other codes,\nsame symmetry', ha='right', va='center', fontsize=12, color=MUTED, linespacing=1.1)
        parent = dark[names.index(trained)]

        def distance(group):
            members = [dark[names.index(n)] for n, g in zip(names, groups) if g == group]
            raw = np.mean([(parent != m).mean(1) for m in members], axis=0)
            return np.convolve(raw, np.ones(20) / 20, mode='same')                           # 100-iteration running mean
        near, far = distance(f'{key}Copy'), distance(f'{key}Control')
        curve = figure.add_axes([left, top - 0.03 - ribbonHeight - 0.015 - 0.085, width, 0.085])
        positions = brokenAxisPosition(iterationOf)
        curve.fill_between(positions, near, far, where=(far > near) & (iterationOf >= 301), color=GLOW[key], alpha=0.22, linewidth=0)
        curve.plot(positions, far, color=MUTED, lw=1.5, alpha=0.9)
        curve.plot(positions, near, color=GLOW_HEX[key], lw=2.4)
        curve.set_xlim(0, 1000); curve.set_ylim(0, 0.45)
        curve.axis('off')
        for position in (brokenAxisPosition(301), brokenAxisPosition(3000)):
            curve.axvline(position, color=INK_LIGHT, lw=1.0, alpha=0.5, ls=(0, (3, 3)))
        for tick in (0, 300, 1000, 3000, 10000, 20000):                                   # time ticks under the lower curve of each block
            figure.text(left + width * float(brokenAxisPosition(tick)) / 1000, top - 0.03 - ribbonHeight - 0.015 - 0.085 - 0.012, f'{tick:,}', ha='center', va='top', fontsize=9.5, color=FAINT_TEXT)
        figure.text(labelX, top - 0.03 - ribbonHeight - 0.015 - 0.045, 'how far a nearby\ncode has drifted', ha='right', va='center', fontsize=11, color=MUTED, linespacing=1.1)
        figure.text(left + width + 0.004, top - 0.03 - ribbonHeight - 0.015 - 0.065, 'other codes', ha='left', va='center', fontsize=9.5, color=MUTED)
        figure.text(left + width + 0.004, top - 0.03 - ribbonHeight - 0.015 - 0.04, 'nearby codes', ha='left', va='center', fontsize=9.5, color=GLOW_HEX[key])
        # gallery: patterns of the family the code visits while it lingers
        inside = (familyFlag(flagFile, names, trained, key)) & (iterationOf >= 301) & (iterationOf < 3000)
        counts = {}
        for k in np.where(inside)[0]:
            counts.setdefault(dark[names.index(trained)][k].tobytes(), []).append(k)
        ordered = sorted(counts.items(), key=lambda item: -len(item[1]))
        placed = []                                                                      # (column, row, item, label)
        if key == 'stripe':
            bars = flagFile['bars'][names.index(trained)]
            for number in (1, 2, 3):
                found = [item for item in ordered if bars[item[1][0]] == number][:2]
                for variant, item in enumerate(found):
                    placed.append((number - 1, variant, item, ['one stripe', 'two stripes', 'three stripes'][number - 1] if variant == 0 else ''))
        else:
            for slot, item in enumerate(ordered[:6]):
                placed.append((slot % 3, slot // 3, item, 'deformed faces' if slot == 0 else ''))
        tile, galleryLeft, spacing = 0.064, 0.752, 0.083
        figure.text(galleryLeft + (2 * spacing + tile) / 2, top + 0.0, 'patterns of the family it visits', ha='center', va='center', fontsize=13, color=INK_LIGHT)
        for column, row, item, label in placed:
            x, y = galleryLeft + column * spacing, top - 0.03 - row * 0.158 - tile * 16 / 9
            small = figure.add_axes([x, y, tile, tile * 16 / 9])
            glyph = Glyph(small, key)
            glyph.update(expandToTissue(dark[names.index(trained)][item[1][0]]), glowStrength=0.6)
            if label:
                figure.text(x + tile / 2, y - 0.012, label, ha='center', va='top', fontsize=10, color=MUTED)
    for position, text in ((brokenAxisPosition(150), 'hold'), (brokenAxisPosition(1650), 'window the code was trained on'), (brokenAxisPosition(11500), 'beyond training')):
        figure.text(left + width * position / 1000, 0.872, text, ha='center', va='center', fontsize=12, color=MUTED)
    figure.text(left + width / 2, 0.052, 'iterations after the start (0 to 3,000 at full scale, then 3,000 to 20,000 squeezed); the tint marks where a nearby code is still closer than an unrelated one', ha='center', va='center', fontsize=11, color=FAINT_TEXT)
    savePicture(figure, '7_theImprintLingers.png')

# ================================================================================================================== movies
def encode(name, frameCount, framesPerSecond):
    directory = f'{args.outputDirectory}/movies'
    os.makedirs(directory, exist_ok=True)
    path = f'{directory}/{name}.mp4'
    if os.path.exists(path) and not args.overwrite:
        print(f'  exists, kept: {path}')
        return
    command = [args.ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(framesPerSecond), '-i', f'{args.framesDirectory}/frame%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
               '-crf', '16', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-movflags', '+faststart', path]
    subprocess.run(command, check=True)
    print(f'  wrote {path} ({frameCount} frames, {frameCount / framesPerSecond:.1f} s)')


def freshFrames():
    shutil.rmtree(args.framesDirectory, ignore_errors=True)
    os.makedirs(args.framesDirectory)


# ================================================================================================================== 8 do the two realms merge? (PCA of the stripe-class and face-class pattern sets)
if 'clusters' in parts:
    from matplotlib.patches import Ellipse
    clusterResults = json.load(open(args.clusterPath))
    scoreFile = np.load(args.clusterScoresPath, allow_pickle=True)
    label, role = scoreFile['label'], scoreFile['role']
    classColour = [GLOW['stripe'], GLOW['face']]
    classHex = [GLOW_HEX['stripe'], GLOW_HEX['face']]
    figure = newFigure()
    figure.text(0.5, 0.955, 'Overlapping realms: the face\u2019s fades, the stripe\u2019s stays', ha='center', va='center', fontsize=30, fontweight='bold')
    figure.text(0.5, 0.908, 'every pattern the stripe-type (cyan) and face-type (amber) codes show, projected on the two main directions of pattern space', ha='center', va='center', fontsize=16, color=MUTED)
    windows_ = [('inSample', 'just after release  (301 to 3,000)'), ('heldOutA', '3,000 to 10,000'), ('heldOutB', '10,000 to 20,000')]
    allPoints = np.concatenate([scoreFile[f'symmetrised_{w}'][:, :2] for w, _ in windows_])
    low, high = np.percentile(allPoints, 0.5, axis=0), np.percentile(allPoints, 99.5, axis=0)
    margin = 0.08 * (high - low)
    for column, (window, title) in enumerate(windows_):
        axis = figure.add_axes([0.06 + column * 0.315, 0.40, 0.28, 0.44])
        points = scoreFile[f'symmetrised_{window}'][:, :2]
        runOf = scoreFile[f'symmetrised_{window}_run']
        for cls in (0, 1):
            for roleName, size, alpha in (('control', 5, 0.28), ('copy', 12, 0.55)):
                chosen = np.isin(runOf, np.where((label == cls) & (role == roleName))[0])
                axis.scatter(points[chosen, 0], points[chosen, 1], s=size, color=classColour[cls], alpha=alpha, linewidths=0)
            members = np.isin(runOf, np.where(label == cls)[0])
            mean, covariance = points[members].mean(0), np.cov(points[members].T)
            values, vectors = np.linalg.eigh(covariance)
            angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
            axis.add_patch(Ellipse(mean, 2 * 1.6 * np.sqrt(values[1]), 2 * 1.6 * np.sqrt(values[0]), angle=angle, fc=classColour[cls], alpha=0.10, ec=classHex[cls], lw=2.2))
        for cls in (0, 1):
            chosen = np.isin(runOf, np.where((label == cls) & (role == 'trained'))[0])
            axis.scatter(points[chosen, 0], points[chosen, 1], s=26, color=classColour[cls], edgecolors=INK_LIGHT, linewidths=0.8, marker='*', alpha=0.95, zorder=5)
        axis.set_xlim(low[0] - margin[0], high[0] + margin[0]); axis.set_ylim(low[1] - margin[1], high[1] + margin[1])
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(FAINT)
        figure.text(0.06 + column * 0.315 + 0.14, 0.865, title, ha='center', va='center', fontsize=16, color=INK_LIGHT)
    figure.text(0.06, 0.375, 'small dots: other codes of the class   \u00b7   larger dots: nearby codes   \u00b7   stars: the trained code   \u00b7   rings: where most of each class lies', ha='left', va='center', fontsize=11.5, color=FAINT_TEXT)
    # how well sets can be told apart, window by window (pooled over each window)
    curve = figure.add_axes([0.10, 0.085, 0.50, 0.225])
    windowNames = ['1000-4000', '4000-10000', '10000-20000']
    series = [('the stripe set from\nthe face set', INK_LIGHT, [clusterResults['pooled']['symmetrised'][w] for w in windowNames], 'accuracy'),
              ('near the trained stripe code,\nfrom other stripe-type codes', GLOW_HEX['stripe'], [clusterResults['within']['symmetrised']['stripe'][w] for w in windowNames], 'balancedAccuracy'),
              ('near the trained face code,\nfrom other face-type codes', GLOW_HEX['face'], [clusterResults['within']['symmetrised']['face'][w] for w in windowNames], 'balancedAccuracy')]
    ceiling = max(r['null95'] for _, _, rows, _ in series for r in rows)
    curve.axhspan(0.44, ceiling, color=MUTED, alpha=0.20, linewidth=0)
    curve.text(2.22, (0.44 + ceiling) / 2, 'chance', ha='right', va='center', fontsize=11, color=MUTED)
    for name, colour, rows, field in series:
        curve.plot(range(3), [r[field] for r in rows], color=colour, lw=3, marker='o', ms=10, zorder=3)
    curve.set_xlim(-0.25, 2.25); curve.set_ylim(0.40, 0.85)
    curve.set_xticks(range(3)); curve.set_xticklabels(['1,000 to 4,000', '4,000 to 10,000', '10,000 to 20,000'], fontsize=12, color=MUTED)
    curve.set_yticks([])
    for spine in ('top', 'right', 'left'):
        curve.spines[spine].set_visible(False)
    curve.spines['bottom'].set_color(FAINT)
    for index, (name, colour, rows, field) in enumerate(sorted(series, key=lambda item: -item[2][-1][item[3]])):          # labels in the order the lines end
        figure.text(0.625, 0.285 - 0.075 * index, name, ha='left', va='center', fontsize=13, color=colour, linespacing=1.1)
    figure.text(0.06, 0.335, 'how well a code the classifier has never seen can be assigned to its set from single patterns (higher = more distinct)', ha='left', va='center', fontsize=13, color=INK_LIGHT)
    figure.text(0.35, 0.03, 'iterations after the start', ha='center', va='center', fontsize=11, color=FAINT_TEXT)
    savePicture(figure, 'backup_quantitative/09_overlappingRealms.png')


# ================================================================================================================== 9 how close is each pattern to the two families?
if 'familySpace' in parts:
    from matplotlib.colors import LinearSegmentedColormap as _Cmap
    distance = json.load(open(args.distancePath))
    scoreFile = np.load(args.distanceScoresPath, allow_pickle=True)
    label, role = scoreFile['label'], scoreFile['role']
    figure = newFigure()
    figure.text(0.5, 0.955, 'Close to their own family? The stripe set, for a while; the face set, hardly', ha='center', va='center', fontsize=27, fontweight='bold')
    figure.text(0.5, 0.908, 'where the patterns of each set fall: how unusually close to the stripe family (across) and to the face family (up)', ha='center', va='center', fontsize=15, color=MUTED)
    windows_ = [('inSample', 'just after release  (301 to 3,000)'), ('heldOutA', '3,000 to 10,000'), ('heldOutB', '10,000 to 20,000')]
    edgesBins = np.linspace(0, 1, 9)
    side, rowBottom = 0.19, {0: 0.505, 1: 0.115}
    heightFraction = side * 16 / 9                                                         # a square panel on a 16:9 figure
    for row, (cls, key) in enumerate(((0, 'stripe'), (1, 'face'))):
        colourMap = _Cmap.from_list(key, [BACKGROUND, GLOW[key] * 0.45, GLOW[key]])
        for column, (window, title) in enumerate(windows_):
            left = 0.085 + column * 0.225
            axis = figure.add_axes([left, rowBottom[row], side, heightFraction])
            x, y, runOf = scoreFile[f'{window}_stripePercentile'], scoreFile[f'{window}_facePercentile'], scoreFile[f'{window}_run']
            members = np.isin(runOf, np.where(label == cls)[0])
            histogram, _, _ = np.histogram2d(x[members], y[members], bins=[edgesBins, edgesBins])
            histogram = histogram / histogram.sum()
            axis.imshow(histogram.T, origin='lower', extent=(0, 1, 0, 1), cmap=colourMap, vmin=0, vmax=0.14, interpolation='nearest', aspect='auto')
            axis.plot([0, 1], [0, 1], color=FAINT_TEXT, lw=1.0, ls=(0, (3, 3)))
            axis.axvline(0.5, color=FAINT, lw=0.8); axis.axhline(0.5, color=FAINT, lw=0.8)
            nearMask = np.isin(runOf, np.where((label == cls) & (role != 'control'))[0])
            trainedMask = np.isin(runOf, np.where((label == cls) & (role == 'trained'))[0])
            axis.scatter([x[nearMask].mean()], [y[nearMask].mean()], s=150, facecolors='none', edgecolors=INK_LIGHT, linewidths=2.2, zorder=5)
            axis.scatter([x[trainedMask].mean()], [y[trainedMask].mean()], s=70, color=INK_LIGHT, marker='*', zorder=6)
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(FAINT)
            if row == 0:
                figure.text(left + side / 2, rowBottom[0] + heightFraction + 0.025, title, ha='center', va='center', fontsize=14, color=INK_LIGHT)
        figure.text(0.04, rowBottom[row] + heightFraction / 2, f'the {key} set', ha='center', va='center', fontsize=17, color=GLOW_HEX[key], fontweight='bold', rotation=90)
    figure.text(0.085 + 0.225 + side / 2, 0.085, 'closer to the stripe family  \u2192', ha='center', va='center', fontsize=12, color=GLOW_HEX['stripe'])
    figure.text(0.07, 0.5, 'closer to the face family  \u2192', ha='center', va='center', fontsize=12, color=GLOW_HEX['face'], rotation=90)
    figure.text(0.085 + 0.225 * 2 + side / 2, 0.052, 'ring: nearby codes  \u00b7  star: trained code  \u00b7  dashed line: equally close to both', ha='center', va='center', fontsize=10.5, color=FAINT_TEXT)
    # the direct test: closeness of the nearby codes to their own family and to the other family, through time
    edges = np.array(distance['timeCourse']['edges'])
    centre = (edges[:-1] + edges[1:]) / 2
    for row, (key, other) in enumerate((('stripe', 'face'), ('face', 'stripe'))):
        axis = figure.add_axes([0.775, rowBottom[row] + 0.04, 0.20, heightFraction - 0.07])
        course = distance['timeCourse'][key]
        axis.plot(centre, np.array(course['copy'][f'{key}Percentile']) * 100, color=GLOW_HEX[key], lw=3.0, label='to its own family')
        axis.plot(centre, np.array(course['copy'][f'{other}Percentile']) * 100, color=MUTED, lw=2.4, ls=(0, (3, 2)), label=f'to the {other} family')
        axis.axhline(50, color=FAINT, lw=1.0)
        axis.axvline(3000, color=INK_LIGHT, lw=1.0, alpha=0.5, ls=(0, (3, 3)))
        axis.set_ylim(30, 85); axis.set_xlim(0, 20000)
        axis.set_yticks([50]); axis.set_yticklabels(['typical'], fontsize=10, color=MUTED)
        axis.set_xticks([3000, 10000, 20000]); axis.set_xticklabels(['3,000', '10,000', '20,000'], fontsize=10, color=MUTED)
        for spine in ('top', 'right'):
            axis.spines[spine].set_visible(False)
        axis.spines['left'].set_color(FAINT); axis.spines['bottom'].set_color(FAINT)
        axis.legend(frameon=False, fontsize=9.5, loc='upper right', labelcolor=MUTED)
    figure.text(0.875, rowBottom[0] + heightFraction + 0.025, 'nearby codes, through time', ha='center', va='center', fontsize=14, color=INK_LIGHT)
    savePicture(figure, 'backup_quantitative/10_closeToTheFamily.png')


# ================================================================================================================== 10 the relay net as a knob turns (movie D)
# The net is estimated exactly as the pages' "Steer the ring: what the simulations say the network does" sections do: the ring's four region levels are computed from the
# knob settings, and the net is a Gaussian-kernel average, in region-level space, of the three biggest transfers per phase of every simulated ring code (846 for the face,
# 499 for the stripe). An edge is drawn when at least a quarter of the nearby simulated codes carry it; thicker = more of them.
PHASE_TINT = {'flood': '#BFE6FF', 'clear': '#FFE0B0', 'write': '#E2D2FF'}
PHASE_BOW = {'flood': 0.12, 'clear': -0.12, 'write': 0.30}
KNOB_NAMES = ['level', 'tilt', 'oval', 'trefoil']
STEERING_PATH = {'stripe': 'data/relayLoopSteeringData1888Hold301StripesInteriorMinus60Minus5Codes499.json', 'face': 'data/relayLoopSteeringData1888Hold301FaceMinus60Minus5Codes846.json'}
# each knob's sweep, where at least about 7 simulated codes cover the setting: stripe, an offset added to the coefficient; face, a multiplier of the coefficient
KNOB_RANGE = {'stripe': {0: (-0.3, 0.3), 1: (-0.3, 0.3), 2: (-0.3, 0.3)}, 'face': {0: (0.6, 1.4), 1: (0.2, 1.8), 2: (0.2, 1.8), 3: (0.0, 2.0)}}


class SteeringLab:
    """The pages' kernel estimator of the relay net from the simulated codes (a port of the steering lab's JavaScript)."""

    def __init__(self, targetKey):
        data = json.load(open(STEERING_PATH[targetKey]))
        self.key = targetKey
        self.trained = np.array(data['trainedCoefficients'], float)
        self.basis, self.bin = np.array(data['ringBasis'], float), np.array(data['ringBin'])
        self.bandwidth, self.levels, self.edges = data['bandwidth'], np.array(data['columns']['levels'], float), data['edges']
        self.member = np.zeros((data['codes'], len(self.edges)), np.float32)
        for row, ids in enumerate(data['columns']['edgeIds']):
            self.member[row, ids] = 1
        self.edgeIndex = {tuple(edge): index for index, edge in enumerate(self.edges)}
        self.trainedEdges = [tuple(self.edges[index]) for index in data['trainedTopEdges']]          # the trained code's own three biggest transfers per phase (figure 6's edges)
        self.lastProbability = None

    def coefficients(self, order, knob):
        value = self.trained.copy()
        if self.key == 'stripe':
            value[order] += knob
        else:
            value[order] *= knob
        return value

    def ringLevels(self, coefficients):
        clipped = np.clip(self.basis @ coefficients, 0, 2)
        return np.array([clipped[self.bin == r].mean() for r in range(4)])

    def net(self, coefficients):
        """(shown edges as (probability, phase, sender, receiver), effective number of simulated codes behind them)."""
        distance2 = ((self.levels - self.ringLevels(coefficients)) ** 2).sum(1)
        weight = np.exp(-distance2 / (2 * self.bandwidth ** 2))
        weight[weight < 1e-6] = 0
        total = weight.sum()
        if total < 1e-9:
            return [], 0.0
        probability = (weight @ self.member) / total
        self.lastProbability = probability
        best = {}
        for index, (phase, sender, receiver) in enumerate(self.edges):                       # for each node pair and phase, the likelier direction
            pairKey = (phase,) + tuple(sorted([sender, receiver]))
            if pairKey not in best or probability[index] > best[pairKey][0]:
                best[pairKey] = (float(probability[index]), phase, sender, receiver)
        return sorted(best.values()), float(total ** 2 / (weight ** 2).sum())


def knobSetting(rangeOfKnob, restValue, progress):
    """Knob value on a sweep rest -> low end -> high end -> rest, with a smooth start and stop of each leg."""
    low, high = rangeOfKnob
    smooth = lambda f: 3 * f ** 2 - 2 * f ** 3
    if progress < 0.25:
        return restValue + (low - restValue) * smooth(progress / 0.25)
    if progress < 0.75:
        return low + (high - low) * smooth((progress - 0.25) / 0.5)
    return high + (restValue - high) * smooth((progress - 0.75) / 0.25)


def knobAngleOf(rangeOfKnob, restValue, value):
    """Pointer angle: the rest setting points up, the low end fans left to 210 degrees, the high end right to -30."""
    low, high = rangeOfKnob
    return 90 + 120 * (restValue - value) / (restValue - low) if value < restValue else 90 - 120 * (value - restValue) / (high - restValue)


def renderRelayKnobMovie(movieName, threshold, ghosts):
    """Movie D and its variants. `threshold`: the share of nearby simulated codes that must carry an edge for it to be drawn solid (0.25 on the pages); `ghosts`: also draw a faint arrow
    for edges carried by a quarter to `threshold` of them, and a dotted grey arrow for an edge of the trained code (figure 6's) that has dropped below a quarter."""
    freshFrames()
    labs = {key: SteeringLab(key) for key in ('stripe', 'face')}
    rest = {'stripe': 0.0, 'face': 1.0}
    knobCount = {'stripe': 3, 'face': 4}
    steps = 60                                                                               # distinct tissue replays per sweep; the net itself is evaluated at every frame
    patternCache = dict(np.load(args.netStopPatternsPath)) if os.path.exists(args.netStopPatternsPath) else {}

    def tissueAt(key, order, setting):
        """The tissue at the target's readout for the exact knob setting (cached on a fine grid)."""
        global replayer
        grid = round(setting, 3)
        name = f'{key}_{order}_{grid:+.3f}'
        if name not in patternCache:
            if replayer is None:
                replayer = Replayer()
            patternCache[name] = replayer.readout(ringValuesOf(labs[key].coefficients(order, grid), 2.0), TARGETS[key]['readIteration']).astype(np.float32)
            if len(patternCache) % 20 == 0:
                np.savez_compressed(args.netStopPatternsPath, **patternCache)
        return patternCache[name]

    trainedNet = {key: labs[key].net(labs[key].trained) for key in labs}
    figure = newFigure()
    figure.text(0.5, 0.955, 'How the net changes as a knob turns', ha='center', va='center', fontsize=30, fontweight='bold')
    panels, arrowArtists, knobs, coverage = {}, {key: [] for key in labs}, {}, {}
    for key, left in (('stripe', 0.04), ('face', 0.52)):
        axis = figure.add_axes([left, 0.33, 0.44, 0.53])
        panels[key] = Glyph(axis, key)
        panels[key].axis = axis
        figure.text(left + 0.22, 0.895, 'a simple organizer: three knobs' if key == 'stripe' else 'a complex organizer: four knobs', ha='center', va='center', fontsize=21, color=GLOW_HEX[key], fontweight='bold')
        for order in range(knobCount[key]):
            knobAxis = figure.add_axes([left + 0.02 + order * 0.11, 0.07, 0.085, 0.17])
            knobAxis.set_xlim(-1.3, 1.3); knobAxis.set_ylim(-1.1, 1.3); knobAxis.set_aspect('equal'); knobAxis.axis('off')
            knobAxis.add_patch(plt.Circle((0, 0), 1.0, fc=BASE * 1.5, ec=FAINT, lw=2))
            for tick in np.linspace(-30, 210, 17):
                a = np.radians(tick)
                long_ = abs(tick - 90) < 1e-6
                knobAxis.plot([1.1 * np.cos(a), (1.38 if long_ else 1.25) * np.cos(a)], [1.1 * np.sin(a), (1.38 if long_ else 1.25) * np.sin(a)], color=INK_LIGHT if long_ else FAINT, lw=2.2 if long_ else 1.4)
            knobAxis.pointer, = knobAxis.plot([0, 0], [0, 0.9], color=INK_LIGHT, lw=4, solid_capstyle='round')
            knobAxis.text(0, -1.45, KNOB_NAMES[order], ha='center', va='center', fontsize=13, color=MUTED)
            knobs[(key, order)] = knobAxis
    for k, (phase, label) in enumerate((('flood', 'flood'), ('clear', 'clear'), ('write', 'write (face only)'))):
        x = 0.375 + k * 0.095 + (0.015 if k == 2 else 0)
        figure.add_artist(plt.Line2D([x - 0.032, x - 0.012], [0.293, 0.293], transform=figure.transFigure, color=PHASE_TINT[phase], lw=4.5, solid_capstyle='round'))
        figure.text(x - 0.005, 0.293, label, ha='left', va='center', fontsize=13, color=MUTED)
    share = {0.25: 'a quarter', 0.5: 'half'}[threshold]
    footnote = f'arrows: transfers that at least {share} of the nearby simulated codes carry among the three biggest of a phase; thicker = more of them  \u00b7  the tissue is simulated afresh at each setting'
    if ghosts:
        footnote += '\nfaint arrows: carried by a quarter to a half of them  \u00b7  dotted: an edge of the trained code that fewer than a quarter now carry'
    figure.text(0.5, 0.032, footnote, ha='center', va='center', fontsize=11, color=FAINT_TEXT, linespacing=1.5)
    plan = [(-1, 0.0)] * 14
    for order in range(4):
        plan += [(order, i / 119) for i in range(120)] + [(order, 1.0)] * 8
    for index, (order, progress) in enumerate(plan):
        for key in labs:
            active = 0 <= order < knobCount[key]
            value = knobSetting(KNOB_RANGE[key][order], rest[key], progress) if active else rest[key]
            quantised = (round(round(progress * (steps - 1)) / (steps - 1), 6)) if active else 0.0
            tissueValue = knobSetting(KNOB_RANGE[key][order], rest[key], quantised) if active else rest[key]
            coefficients = labs[key].coefficients(order if active else 0, value)
            vmem = tissueAt(key, order if active else 0, tissueValue) if active else tissueAt(key, 0, rest[key])
            panels[key].update(vmem, ringValues=ringValuesOf(coefficients, 2.0), glowStrength=0.35)
            for artist in arrowArtists[key]:
                artist.remove()
            arrowArtists[key] = []
            shown, effective = labs[key].net(coefficients)
            thin = 0.55 + 0.45 * min(1.0, effective / 10.0)                                   # fade the arrows where few simulated codes lie nearby
            probabilities = labs[key].lastProbability
            drawn = set()
            for probability, phase, sender, receiver in sorted(shown):
                visible = float(np.clip((probability - (threshold - 0.05)) / 0.10, 0, 1))
                ghostWeight = float(np.clip((probability - 0.20) / 0.10, 0, 1)) * (1 - visible) if ghosts else 0.0
                if visible <= 0 and ghostWeight <= 0:
                    continue
                for a_, b_ in {(sender, receiver), (NET_MIRROR.get(sender, sender), NET_MIRROR.get(receiver, receiver))}:
                    drawn.add((phase, a_, b_))
                    (x0, y0), (x1, y1) = NET_POSITIONS[key][a_], NET_POSITIONS[key][b_]
                    if visible > 0:
                        arrowArtists[key].append(panels[key].axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>,head_length=0.55,head_width=0.28', mutation_scale=12, lw=(1.4 + 6.4 * probability) * (0.5 + 0.5 * visible),
                                                                                          color=PHASE_TINT[phase], alpha=(0.30 + 0.70 * min(1.0, max(probability - threshold, 0) / (0.85 - threshold))) * visible * thin, shrinkA=7, shrinkB=8,
                                                                                          connectionstyle=f'arc3,rad={PHASE_BOW[phase]}', zorder=6)))
                    if ghostWeight > 0:
                        arrowArtists[key].append(panels[key].axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>,head_length=0.5,head_width=0.24', mutation_scale=10, lw=1.5,
                                                                                          color=PHASE_TINT[phase], alpha=0.28 * ghostWeight * thin, shrinkA=7, shrinkB=8,
                                                                                          connectionstyle=f'arc3,rad={PHASE_BOW[phase]}', zorder=5)))
            if ghosts:                                                                           # the trained code's own edges, once fewer than a quarter of the nearby codes carry them
                for phase, sender, receiver in labs[key].trainedEdges:
                    gone = float(1 - np.clip((probabilities[labs[key].edgeIndex[(phase, sender, receiver)]] - 0.20) / 0.10, 0, 1))
                    if gone <= 0:
                        continue
                    for a_, b_ in {(sender, receiver), (NET_MIRROR.get(sender, sender), NET_MIRROR.get(receiver, receiver))}:
                        (x0, y0), (x1, y1) = NET_POSITIONS[key][a_], NET_POSITIONS[key][b_]
                        arrowArtists[key].append(panels[key].axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>,head_length=0.5,head_width=0.24', mutation_scale=10, lw=2.2, linestyle=(0, (1.2, 2.2)),
                                                                                          color=MUTED, alpha=0.55 * gone, shrinkA=7, shrinkB=8, connectionstyle=f'arc3,rad={PHASE_BOW[phase]}', zorder=4)))
            for knobOrder in range(knobCount[key]):
                isActive = active and knobOrder == order
                angle = np.radians(knobAngleOf(KNOB_RANGE[key][knobOrder], rest[key], value if isActive else rest[key]))
                pointer = knobs[(key, knobOrder)].pointer
                pointer.set_data([0, 0.9 * np.cos(angle)], [0, 0.9 * np.sin(angle)])
                pointer.set_color(tuple(VIOLET) if isActive else MUTED)
                pointer.set_linewidth(5 if isActive else 3)
        figure.savefig(f'{args.framesDirectory}/frame{index:04d}.png', dpi=DPI)
    np.savez_compressed(args.netStopPatternsPath, **patternCache)
    plt.close(figure)
    encode(movieName, len(plan), 20)


for partName, movieName, threshold, ghosts in (('relayKnobs', 'D_theNetAsTheKnobsTurn', 0.25, False), ('relayKnobsThresh', 'D_theNetAsTheKnobsTurn_thresh', 0.5, False),
                                               ('relayKnobsThreshGhost', 'D_theNetAsTheKnobsTurn_thresh_ghost', 0.5, True)):
    if partName in parts:
        renderRelayKnobMovie(movieName, threshold, ghosts)


def longHorizonCourse(key):
    """The trained code's tissue for 20,000 iterations in 64-bit, every 5th iteration (the run the family analysis used); cached."""
    store = dict(np.load(args.longHorizonPath)) if os.path.exists(args.longHorizonPath) else {}
    if key not in store:
        with boundary.fullDoublePrecision(True):
            runner = Replayer()
            frames, _ = runner.run(ringValuesOf(TARGETS[key]['coefficients'], TARGETS[key]['ceiling']), 20000)
        store[key] = frames[::5].astype(np.float32)
        np.savez_compressed(args.longHorizonPath, **store)
    return store[key]


if 'lingerThenWander' in parts:
    freshFrames()
    names, groups, dark, flagFile, stride = loadFamilyRuns()
    courses = {key: longHorizonCourse(key) for key in ('stripe', 'face')}
    rings = {key: ringValuesOf(TARGETS[key]['coefficients'], TARGETS[key]['ceiling']) for key in ('stripe', 'face')}
    early, groupsLate = displayColumns(stride, dark.shape[1])
    ribbons = {key: ribbonRow(familyFlag(flagFile, names, f'{key}Trained', key), early, groupsLate) for key in ('stripe', 'face')}
    flagsByFrame = {key: familyFlag(flagFile, names, f'{key}Trained', key) for key in ('stripe', 'face')}
    barsByFrame = flagFile['bars'][names.index('stripeTrained')]
    schedule = list(range(0, 61, 2)) + list(range(62, 601, 2)) + list(range(640, 4000, 40)) + [3999]
    figure = newFigure()
    status = figure.text(0.5, 0.955, '', ha='center', va='center', fontsize=30, fontweight='bold')
    glyphs, ribbonImages, playheads, frames_, labels = {}, {}, {}, {}, {}
    for key, x0 in (('stripe', 0.05), ('face', 0.52)):
        axis = figure.add_axes([x0, 0.27, 0.43, 0.56])
        glyphs[key] = Glyph(axis, key)
        frames_[key] = FancyBboxPatch((-0.7, -0.7), 11.4, 11.4, boxstyle='round,pad=0,rounding_size=0.6', fc='none', ec=GLOW_HEX[key], lw=4, alpha=0.0, zorder=6)
        axis.add_patch(frames_[key])
        figure.text(x0 + 0.215, 0.87, 'a simple organizer' if key == 'stripe' else 'a complex organizer', ha='center', va='center', fontsize=22, color=GLOW_HEX[key], fontweight='bold')
        labels[key] = figure.text(x0 + 0.215, 0.225, '', ha='center', va='center', fontsize=17, color=GLOW_HEX[key])
        ribbonAxis = figure.add_axes([x0 + 0.02, 0.105, 0.39, 0.055])
        ribbonImages[key] = ribbonAxis.imshow(np.tile(FAINT_ARRAY, (1, 1000, 1)), aspect='auto', interpolation='nearest', extent=(0, 1000, 1, 0))
        ribbonAxis.set_xticks([]); ribbonAxis.set_yticks([])
        for spine in ribbonAxis.spines.values():
            spine.set_visible(False)
        for position in (brokenAxisPosition(301), brokenAxisPosition(3000)):
            ribbonAxis.axvline(position, color=INK_LIGHT, lw=1.2, alpha=0.6, ls=(0, (3, 3)))
        playheads[key] = ribbonAxis.axvline(0, color=INK_LIGHT, lw=3)
        for position, text in ((brokenAxisPosition(150), 'hold'), (brokenAxisPosition(1650), 'trained window'), (brokenAxisPosition(11500), 'beyond training')):
            figure.text(x0 + 0.02 + 0.39 * position / 1000, 0.075, text, ha='center', va='center', fontsize=11, color=MUTED)
    for index, frame in enumerate(schedule):
        iteration = frame * stride
        held = iteration < HOLD
        column = int(brokenAxisPosition(iteration))
        for key in ('stripe', 'face'):
            inFamily = bool(flagsByFrame[key][frame]) and not held
            ringWeight = 1.0 if held else max(0.0, 1.0 - (iteration - HOLD) / 20.0)
            glyphs[key].update(courses[key][frame], ringValues=rings[key], ringWeight=ringWeight, glowStrength=0.95 if inFamily else 0.45)
            frames_[key].set_alpha(0.95 if inFamily else 0.0)
            if inFamily:
                labels[key].set_text(['one stripe', 'two stripes', 'three stripes', 'four stripes'][min(int(barsByFrame[frame]), 4) - 1] if key == 'stripe' else 'a deformed face')
            else:
                labels[key].set_text('')
            revealed = np.tile(FAINT_ARRAY * 0.6, (1000, 1))
            line = ribbons[key]
            revealed[:column + 1] = FAINT_ARRAY[None, :] * (1 - line[:column + 1, None]) + GLOW[key][None, :] * line[:column + 1, None]
            ribbonImages[key].set_data(revealed[None, :, :])
            playheads[key].set_xdata([column, column])
        status.set_text('guidance: the boundary is held' if held else ('the imprint lingers' if iteration < 3000 else 'beyond training: the tissue wanders off'))
        status.set_color(tuple(VIOLET) if held else INK_LIGHT)
        figure.savefig(f'{args.framesDirectory}/frame{index:04d}.png', dpi=DPI)
    plt.close(figure)
    encode('C_theImprintLingers', len(schedule), 24)

if 'guideThenLetGo' in parts:
    freshFrames()
    courses = {key: trainedCourse(key) for key in ('stripe', 'face')}
    rings = {key: ringValuesOf(TARGETS[key]['coefficients'], TARGETS[key]['ceiling']) for key in ('stripe', 'face')}
    # (fade-in of the organizer on the boundary 0..1, iteration shown, resting at the end); the hold is slowed down, and the two readouts are hit exactly
    releaseTimes = sorted(set(range(HOLD, 3000, 25)) | {504, 2173})
    frames = [(i / 29, 0) for i in range(30)] + [(1.0, it) for it in range(0, HOLD, 5)] + [(1.0, it) for it in releaseTimes if it <= 2173] + [(1.0, 2173)] * 44
    figure = newFigure()
    axes = {'stripe': figure.add_axes([0.04, 0.20, 0.44, 0.62]), 'face': figure.add_axes([0.52, 0.20, 0.44, 0.62])}
    glyphs = {key: Glyph(axes[key], key, targetOutline=True) for key in axes}
    for key in axes:
        figure.text(0.26 if key == 'stripe' else 0.74, 0.865, 'a simple organizer' if key == 'stripe' else 'a complex organizer', ha='center', va='center', fontsize=24, color=GLOW_HEX[key], fontweight='bold')
    status = figure.text(0.5, 0.955, '', ha='center', va='center', fontsize=30, fontweight='bold')
    barAxis = figure.add_axes([0.05, 0.075, 0.90, 0.028])
    barAxis.set_xlim(0, 1)
    barAxis.set_ylim(0, 1)
    barAxis.axis('off')
    holdShare = 0.28

    def position(iteration):
        return holdShare * iteration / HOLD if iteration <= HOLD else holdShare + (1 - holdShare) * (iteration - HOLD) / (3000 - HOLD)
    barAxis.plot([0.006, holdShare - 0.006], [0.5, 0.5], lw=15, color=tuple(VIOLET * 0.85), solid_capstyle='round')
    barAxis.plot([holdShare + 0.006, 0.994], [0.5, 0.5], lw=15, color=FAINT, solid_capstyle='round')
    figure.text(0.05 + 0.90 * holdShare / 2, 0.135, 'guidance', ha='center', va='center', fontsize=18, color=tuple(VIOLET))
    figure.text(0.05 + 0.90 * (holdShare + (1 - holdShare) / 2), 0.135, 'self-organization', ha='center', va='center', fontsize=18, color=MUTED)
    playhead, = barAxis.plot([0], [0.5], 'o', ms=15, color=INK_LIGHT, zorder=5)
    marks = {key: barAxis.plot([], [], 'o', ms=11, color=GLOW_HEX[key], zorder=4)[0] for key in axes}
    for index, (intro, iteration) in enumerate(frames):
        held = iteration < HOLD
        for key in axes:
            target = TARGETS[key]
            clockIteration = min(iteration, target['readIteration'])          # each panel stops at its own readout, where the pattern is formed; the tissue keeps moving afterwards
            formed = clockIteration >= target['readIteration']
            ringWeight = 1.0 if held else max(0.0, 1.0 - (iteration - HOLD) / 20.0)
            ringWeight *= intro if iteration == 0 else 1.0
            glyphs[key].update(courses[key][clockIteration], ringValues=rings[key], ringWeight=ringWeight, glowStrength=0.85 if formed else 0.55)
            glyphs[key].showOutline(0.0 if formed else 0.55 * intro)
            if formed:
                marks[key].set_data([position(target['readIteration'])], [0.5])
        status.set_text('the organizer appears on the boundary' if iteration == 0 else ('guidance: the boundary is held' if held else 'self-organization: the boundary lets go'))
        status.set_color(tuple(VIOLET) if held else INK_LIGHT)
        playhead.set_data([position(iteration)], [0.5])
        figure.savefig(f'{args.framesDirectory}/frame{index:04d}.png', dpi=DPI)
    plt.close(figure)
    encode('A_guideThenLetGo', len(frames), 24)

if 'turningTheKnobs' in parts:
    freshFrames()
    offsets33 = np.round(np.arange(-0.08, 0.0801, 0.005), 6)
    offsets81 = np.round(np.arange(-0.08, 0.0801, 0.002), 6)

    def patternAt(key, order, offset):
        if order == 0:
            return cache[f'slide_{key}_order0_step0.002'][int(np.argmin(np.abs(offsets81 - offset)))]
        return cache[f'slide_{key}_order{order}_step0.005'][int(np.argmin(np.abs(offsets33 - offset)))]
    figure = newFigure()
    axes = {'stripe': figure.add_axes([0.04, 0.33, 0.44, 0.53]), 'face': figure.add_axes([0.52, 0.33, 0.44, 0.53])}
    glyphs = {key: Glyph(axes[key], key) for key in axes}
    figure.text(0.5, 0.955, 'Turn a knob on the boundary', ha='center', va='center', fontsize=30, fontweight='bold')
    for key in axes:
        figure.text(0.26 if key == 'stripe' else 0.74, 0.895, 'a simple organizer: three knobs' if key == 'stripe' else 'a complex organizer: four knobs', ha='center', va='center', fontsize=21, color=GLOW_HEX[key], fontweight='bold')
    knobAxes = {}
    for key, count in (('stripe', 3), ('face', 4)):
        base = 0.04 if key == 'stripe' else 0.52
        for order in range(count):
            width = 0.44 / 4
            axis = figure.add_axes([base + 0.02 + order * width * (4 / count if count == 3 else 1.0) * (0.75 if count == 3 else 1.0), 0.07, 0.085, 0.17])
            axis.set_xlim(-1.3, 1.3)
            axis.set_ylim(-1.1, 1.3)
            axis.set_aspect('equal')
            axis.axis('off')
            knobAxes[(key, order)] = axis
            axis.add_patch(plt.Circle((0, 0), 1.0, fc=BASE * 1.5, ec=FAINT, lw=2))
            angles = np.radians(np.linspace(-135, 135, 21) + 90)
            for a in angles:
                axis.plot([1.12 * np.cos(a), 1.25 * np.cos(a)], [1.12 * np.sin(a), 1.25 * np.sin(a)], color=FAINT, lw=1.5)
            axis.pointer, = axis.plot([0, 0], [0, 0.9], color=INK_LIGHT, lw=4, solid_capstyle='round')
            axis.text(0, -1.45, WAVE_NAMES[order].replace('a ', '').replace('an ', ''), ha='center', va='center', fontsize=13, color=MUTED)
    track = []
    for order in range(4):
        sweep = list(np.round(np.linspace(0, -0.08, 8), 6)) + list(np.round(np.linspace(-0.08, 0.08, 24), 6)) + list(np.round(np.linspace(0.08, 0, 8), 6))
        track += [(order, offset) for offset in sweep] + [(order, 0.0)] * 6
    for index, (order, offset) in enumerate(track):
        for key in ('stripe', 'face'):
            target = TARGETS[key]
            count = len(target['coefficients'])
            coefficients = target['coefficients'].copy()
            active = order < count
            if active:
                coefficients[order] += offset
            glyphs[key].update(patternAt(key, order, offset) if active else patternAt(key, 0, 0.0), ringValues=ringValuesOf(coefficients, 2.0), glowStrength=0.6)
        for (key, knob), axis in knobAxes.items():
            isActive = knob == order
            value = offset if isActive else 0.0
            angle = np.radians(90 - value / 0.08 * 120)
            axis.pointer.set_data([0, 0.9 * np.cos(angle)], [0, 0.9 * np.sin(angle)])
            axis.pointer.set_color(tuple(VIOLET) if isActive else MUTED)
            axis.pointer.set_linewidth(5 if isActive else 3)
        figure.savefig(f'{args.framesDirectory}/frame{index:04d}.png', dpi=DPI)
    plt.close(figure)
    encode('B_turningTheKnobs', len(track), 20)

shutil.rmtree(args.framesDirectory, ignore_errors=True)
print('parts done:', ', '.join(parts))
