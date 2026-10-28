"""The stripe family and the face family of analyzeCanalizationTalkFamilyVisits11x11.py, as functions (that script runs its whole analysis on import, so its definitions are
repeated here verbatim, and checked against its stored flags by analyzeCanalizationTalkRecanalization11x11.py --check).

A pattern is the interior's 81 dark / light cells (dark = below -34.6 mV), row by row. Families, fixed before the first run of that analysis:
  stripe family  at least 6 dark cells, and at least 80% of them in "bars": 4-connected dark components at least 3 cells tall, twice as tall as wide, and 80% filled
  face family    the intersection over union with some deformed face is at least 0.8, a deformed face being any union of at least two of the four face blocks
                 (left eye, right eye, nose, mouth), each moved independently by up to one cell
  exact          the plain intersection over union with the target (the 27-cell centre stripe, the 14 feature cells)
"""
import itertools

import numpy as np
from scipy.ndimage import label

import boundaryCodeUtilities as boundary
from canalizationTalkCommon import FACE_CELLS, INTERIOR, STRIPE_CELLS

interior = np.array(INTERIOR)
interiorMask = lambda cells: np.isin(interior, cells).reshape(9, 9)
exactTargets = {'stripe': np.isin(interior, STRIPE_CELLS), 'face': np.isin(interior, FACE_CELLS)}
blocks = [interiorMask(boundary.featureParts[i]) for i in range(4)]


def shifted(mask, dr, dc):
    out = np.zeros_like(mask)
    rows, columns = np.where(mask)
    rows, columns = rows + dr, columns + dc
    keep = (rows >= 0) & (rows < 9) & (columns >= 0) & (columns < 9)
    out[rows[keep], columns[keep]] = True
    return out


deformedFaces = []
for subset in itertools.chain.from_iterable(itertools.combinations(range(4), r) for r in (2, 3, 4)):
    for moves in itertools.product(itertools.product((-1, 0, 1), repeat=2), repeat=len(subset)):
        deformedFaces.append(np.logical_or.reduce([shifted(blocks[b], *move) for b, move in zip(subset, moves)]).reshape(-1))
deformedFaces = np.unique(np.array(deformedFaces), axis=0).astype(np.float32)


def barCount(bits):
    """(number of bars, share of dark cells in bars) for one 81-cell pattern; a bar is a 4-connected dark component at least 3 tall, twice as tall as wide, 80% filled."""
    square = bits.reshape(9, 9)
    labelled, count = label(square)
    bars, inBars = 0, 0
    for number in range(1, count + 1):
        rows, columns = np.where(labelled == number)
        height, width = rows.max() - rows.min() + 1, columns.max() - columns.min() + 1
        if height >= 3 and height >= 2 * width and len(rows) >= 0.8 * height * width:
            bars += 1
            inBars += len(rows)
    return bars, inBars / max(bits.sum(), 1)


def frameFlags(runBits):
    """Per frame: the number of bars if the frame is in the stripe family (else 0), the best deformed-face IoU, and the exact IoUs with the two targets."""
    flat = runBits.reshape(-1, 81)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    bars = np.zeros(len(unique), int)
    for k, bits in enumerate(unique):
        if bits.sum() >= 6:
            count, share = barCount(bits)
            bars[k] = count if (count >= 1 and share >= 0.8) else 0
    unionSize = unique.sum(1).astype(np.float32)
    faceBest = np.zeros(len(unique), np.float32)
    for start in range(0, len(unique), 4000):
        part = unique[start:start + 4000].astype(np.float32)
        intersection = part @ deformedFaces.T
        union = part.sum(1)[:, None] + deformedFaces.sum(1)[None, :] - intersection
        faceBest[start:start + 4000] = (intersection / np.maximum(union, 1)).max(1)
    exact = {}
    for key, target in exactTargets.items():
        intersection = unique.astype(np.float32) @ target.astype(np.float32)
        exact[key] = (intersection / np.maximum(unionSize + target.sum() - intersection, 1))
    return bars[inverse], faceBest[inverse], {key: value[inverse] for key, value in exact.items()}


def episodes(flag, gap=4):
    """Runs of true frames, runs separated by up to `gap` frames joined; returns (starts, lengths) in frames."""
    indices = np.where(flag)[0]
    if len(indices) == 0:
        return np.array([], int), np.array([], int)
    breaks = np.where(np.diff(indices) > gap + 1)[0]
    starts = np.concatenate([[indices[0]], indices[breaks + 1]])
    ends = np.concatenate([indices[breaks], [indices[-1]]])
    return starts, ends - starts + 1
