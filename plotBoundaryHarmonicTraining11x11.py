"""Build the "Training the Ring" report page (PolyPatterning_Sim.md, Section 12): CMA-ES training of a mirror-symmetric
ring code, orders 0 to N, toward the face.

Input: data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json (analyzeBoundaryHarmonicTraining11x11.py,
from the runs of learnBoundaryHarmonics11x11.py). The page's JSON keys are the ones figures/boundaryHarmonicTrainingTemplate.html
reads. Refuses to overwrite an existing page unless --overwrite is given (for rebuilding this page itself).
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--templatePath', type=str, default='figures/boundaryHarmonicTrainingTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/boundaryHarmonicTraining.html')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

if os.path.exists(args.outputPath) and not args.overwrite:
    raise SystemExit(f'{args.outputPath} exists; pass --overwrite to rebuild it')
summary = json.load(open(args.summaryPath))
page = open(args.templatePath).read().replace('__DATA__', json.dumps(summary, separators=(',', ':')))
open(args.outputPath, 'w').write(page)
print(f"wrote {args.outputPath} ({len(page) / 1e6:.2f} MB)")
