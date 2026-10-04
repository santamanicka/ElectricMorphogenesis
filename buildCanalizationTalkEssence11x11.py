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
parser.add_argument('--parts', type=str, default='spatialOrganizer,twoPhases,stripeSwitches,faceNoSinglePart,canals,push,guideThenLetGo,turningTheKnobs')
parser.add_argument('--outputDirectory', type=str, default='presentation')
parser.add_argument('--cachePath', type=str, default='data/canalizationTalkReplays1888Hold301.npz')
parser.add_argument('--partialCachePath', type=str, default='data/canalizationTalkPartialHolds1888Hold301.npz')
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


# ================================================================================================================== 1 the spatial organizer
WAVE_NAMES = ['a level', 'a tilt', 'an oval', 'a trefoil']

if 'spatialOrganizer' in parts:
    figure = newFigure()
    grid = figure.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], left=0.13, right=0.97, top=0.84, bottom=0.07, wspace=0.18, hspace=0.18)
    degrees = np.linspace(-180, 180, 361)
    for row, key in enumerate(('stripe', 'face')):
        target = TARGETS[key]
        coefficients = target['coefficients']
        count = len(coefficients)
        inner = grid[row, 0].subgridspec(count + 1, 1, hspace=0.35)
        for order in range(count):
            axis = figure.add_subplot(inner[order])
            axis.plot(degrees, coefficients[order] * np.cos(np.radians(order * degrees)), color=VIOLET, lw=2.6, alpha=0.95 - 0.12 * order, solid_capstyle='round')
            axis.set_ylim(-1.0, 1.7)
            axis.set_xlim(-180, 180)
            axis.axis('off')
            axis.text(-190, 0.35, WAVE_NAMES[order], ha='right', va='center', fontsize=12, color=MUTED)
            axis.text(190, 0.35, '+' if order < count - 1 else '=', ha='left', va='center', fontsize=18, color=MUTED)
        axis = figure.add_subplot(inner[count])
        total = ringValuesOf(coefficients, target['ceiling'])
        order = np.argsort(RING_ANGLES)
        axis.plot(np.degrees(RING_ANGLES[order]), total[order], color=INK_LIGHT, lw=3.2, solid_capstyle='round')
        axis.set_ylim(-0.1, 1.9)
        axis.set_xlim(-180, 180)
        axis.axis('off')
        glyphBoundary = Glyph(figure.add_subplot(grid[row, 1]), key)
        glyphBoundary.update(np.full(121, -5.0), ringValues=total)
        glyphOut = Glyph(figure.add_subplot(grid[row, 2]), key)
        glyphOut.update(trainedCourse(key)[target['readIteration']], ringValues=total, ringWeight=0.0, glowStrength=0.75)
        glyphOut.showOutline(0.55)
    for column, text in enumerate(('a few waves…', '…held on the boundary…', '…guide the tissue')):
        caption(figure, (0.25, 0.555, 0.82)[column], 0.915, text, size=21, colour=INK_LIGHT)
    figure.text(0.5, 0.968, 'The code is a spatial organizer', ha='center', va='center', fontsize=30, fontweight='bold', color=INK_LIGHT)
    for row, key in enumerate(('stripe', 'face')):
        figure.text(0.022, 0.63 - row * 0.40, 'a simple organizer' if key == 'stripe' else 'a complex organizer', ha='center', va='center', fontsize=19, color=GLOW_HEX[key], fontweight='bold', rotation=90)
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

# ================================================================================================================== 6 the push (the relay)
if 'push' in parts:
    stripeNets = json.load(open('data/relayLoopStripesPageData1888Hold301StripesInteriorMinus60Minus5.json'))
    faceNets = json.load(open('data/relayLoopFullNets1888Hold301FaceMinus60Minus5.json'))
    positions = {
        'stripe': dict(ringTop=(5, 0), ringBottom=(5, 10), ringLeft=(0, 5), ringRight=(10, 5), stripeUpper=(5, 3), stripeLower=(5, 7), flankLeftUpper=(2, 3), flankLeftLower=(2, 7),
                       flankRightUpper=(8, 3), flankRightLower=(8, 7)),
        'face': dict(ringTop=(5, 0), ringBottom=(5, 10), ringLeft=(0, 5), ringRight=(10, 5), bgTL=(2.9, 2.9), bgTR=(7.1, 2.9), bgBL=(2.7, 7.5), bgBR=(7.3, 7.5), eyes=(5, 2.5), nose=(5, 5), mouth=(5, 8))}
    mirror = dict(ringLeft='ringRight', bgTL='bgTR', bgBL='bgBR', flankLeftUpper='flankRightUpper', flankLeftLower='flankRightLower')
    mirror.update({value: key for key, value in list(mirror.items())})

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
