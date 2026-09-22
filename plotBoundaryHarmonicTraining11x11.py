"""Build the "Training the Ring" report page (PolyPatterning_Sim.md, Section 12): CMA-ES training of a mirror-symmetric
ring code, orders 0 to N, toward the face.

Inputs: data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json (the face) and, for the face with its outline,
data/boundaryHarmonicTrainingSummary1888Hold301FaceOutlineMinus60Minus5Ceiling2FromRelease.json and ...From1000.json (scored
from the release and from iteration 1,000), all from analyzeBoundaryHarmonicTraining11x11.py on the runs of
learnBoundaryHarmonics11x11.py. The outline summaries go into the page as D.outline, keyed by scoring window, and the
one-order-at-a-time check (analyzeBoundaryHarmonicSensitivity11x11.py) as D.sensitivity, and its fine grid with
best-moment patterns (--storePatterns) as D.explorer, for the slider figure, and the distance of changed codes' tissues from the trained one over time
(analyzeBoundaryHarmonicDivergence11x11.py) as D.divergence, the knockout test
(analyzeBoundaryHarmonicKnockout11x11.py) as D.knockout, the layer response (analyzeBoundaryHarmonicLayers11x11.py) as
D.layers, the mode-ownership ensembles (analyzeBoundaryHarmonicModeOwnership11x11.py) as D.ownership keyed by condition,
and the outcome clustering (analyzeBoundaryHarmonicOutcomes11x11.py) as D.outcomes. The page's JSON keys are the ones figures/boundaryHarmonicTrainingTemplate.html
reads. Refuses to overwrite an existing page unless --overwrite is given (for rebuilding this page itself).
"""
import argparse
import json
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outlineSummaryPaths', type=str,
                    default='FromRelease:data/boundaryHarmonicTrainingSummary1888Hold301FaceOutlineMinus60Minus5Ceiling2FromRelease.json,'
                            'From1000:data/boundaryHarmonicTrainingSummary1888Hold301FaceOutlineMinus60Minus5Ceiling2From1000.json')
parser.add_argument('--sensitivityPath', type=str, default='data/boundaryHarmonicSensitivity1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--sensitivityPatternsPath', type=str, default='data/boundaryHarmonicSensitivity1888Hold301FaceMinus60Minus5PatternsBothMoments.json')
parser.add_argument('--divergencePath', type=str, default='data/boundaryHarmonicDivergence1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--layersPath', type=str, default='data/boundaryHarmonicLayers1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--ownershipPaths', type=str, default='data/boundaryHarmonicModeOwnership1888Hold301FaceMinus60Minus5Baseline.json,'
                    'data/boundaryHarmonicModeOwnership1888Hold301FaceMinus60Minus5HeldThroughout.json,'
                    'data/boundaryHarmonicModeOwnership1888Hold301FaceMinus60Minus5FieldOff.json,'
                    'data/boundaryHarmonicModeOwnership1888Hold301FaceMinus60Minus5Orders0to6.json,'
                    'data/boundaryHarmonicModeOwnership1888Hold301FaceMinus60Minus5BaselineInterior.json')
parser.add_argument('--readoutPath', type=str, default='data/boundaryHarmonicReadout1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--correlationLengthPath', type=str, default='data/boundaryHarmonicCorrelationLength1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--fieldRolePath', type=str, default='data/boundaryHarmonicFieldRole1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outcomesPath', type=str, default='data/boundaryHarmonicOutcomes1888Hold301FaceMinus60Minus5Random.json')
parser.add_argument('--knockoutPath', type=str, default='data/boundaryHarmonicKnockout1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--templatePath', type=str, default='figures/boundaryHarmonicTrainingTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/boundaryHarmonicTraining.html')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

if os.path.exists(args.outputPath) and not args.overwrite:
    raise SystemExit(f'{args.outputPath} exists; pass --overwrite to rebuild it')
summary = json.load(open(args.summaryPath))
summary['outline'] = {key: json.load(open(path)) for key, path in (entry.split(':', 1) for entry in args.outlineSummaryPaths.split(','))
                      if os.path.exists(path)}
summary['sensitivity'] = json.load(open(args.sensitivityPath))
summary['explorer'] = json.load(open(args.sensitivityPatternsPath))
summary['divergence'] = json.load(open(args.divergencePath))
summary['knockout'] = json.load(open(args.knockoutPath)) if os.path.exists(args.knockoutPath) else None
summary['layers'] = json.load(open(args.layersPath)) if os.path.exists(args.layersPath) else None
def ownershipKey(path):
    """The run's name from its file, so that variants never overwrite one another."""
    stem = os.path.basename(path).split('Minus5')[-1].replace('.json', '')
    return stem[0].lower() + stem[1:]


summary['ownership'] = {ownershipKey(path): json.load(open(path))
                        for path in args.ownershipPaths.split(',') if os.path.exists(path)} or None
summary['outcomes'] = json.load(open(args.outcomesPath)) if os.path.exists(args.outcomesPath) else None
summary['fieldRole'] = json.load(open(args.fieldRolePath)) if os.path.exists(args.fieldRolePath) else None
summary['correlationLength'] = json.load(open(args.correlationLengthPath)) if os.path.exists(args.correlationLengthPath) else None
summary['readout'] = json.load(open(args.readoutPath)) if os.path.exists(args.readoutPath) else None
page = open(args.templatePath).read()
if not summary['outline']:
    # the face-with-outline sections are left out until their summaries exist
    page = re.sub(r'<!--OUTLINE-->.*?<!--/OUTLINE-->', '', page, flags=re.S)
if not summary['knockout']:
    page = re.sub(r'<!--KNOCKOUT-->.*?<!--/KNOCKOUT-->', '', page, flags=re.S)
if not summary['layers']:
    page = re.sub(r'<!--LAYERS-->.*?<!--/LAYERS-->', '', page, flags=re.S)
if not summary['ownership']:
    page = re.sub(r'<!--MODES-->.*?<!--/MODES-->', '', page, flags=re.S)
if not (summary['ownership'] and summary['fieldRole']):
    page = re.sub(r'<!--SIGNATURE-->.*?<!--/SIGNATURE-->', '', page, flags=re.S)
if not summary['fieldRole']:
    page = re.sub(r'<!--FIELDROLE-->.*?<!--/FIELDROLE-->', '', page, flags=re.S)
if not summary['correlationLength']:
    page = re.sub(r'<!--CORRELATION-->.*?<!--/CORRELATION-->', '', page, flags=re.S)
if not summary['readout']:
    page = re.sub(r'<!--READOUT-->.*?<!--/READOUT-->', '', page, flags=re.S)
page = page.replace('__DATA__', json.dumps(summary, separators=(',', ':')))
open(args.outputPath, 'w').write(page)
print(f"wrote {args.outputPath} ({len(page) / 1e6:.2f} MB)")
