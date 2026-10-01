"""Build the Relay Loop page from its template and the relay data.

Runs assembleRelayLoopFiveLevelData.py (slider, grid, tracked-edge, Vmem, conductance, single-mode and trained-top-3
blocks, all read from the committed data/ files) and splices each block into its __NAME__ placeholder in
figures/relayLoopTemplate.html, with the edge-curvature table data/relayLoopEdgeLanes.json (optimizeRelayLoopEdgeLanes.py). The page is a self-contained HTML file with the data embedded; test it with
checkRelayLoop.js.

    python3 buildRelayLoopArtifact.py [--overwrite]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--templatePath', type=str, default='figures/relayLoopTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/relayLoop.html')
parser.add_argument('--lanesPath', type=str, default='data/relayLoopEdgeLanes.json')
parser.add_argument('--steeringDataPath', type=str, default=None,
                    help='assembleRelayLoopSteeringData11x11.py output; default is the file with the most simulated codes')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

if os.path.exists(args.outputPath) and not args.overwrite:
    raise SystemExit(f'{args.outputPath} exists; pass --overwrite to rebuild it')

BLOCKS = dict(VARIANT_DATA='variantData', TRAINED_TOP3_PHASES='trainedTop3Phases', SLIDER_DATA='sliderData',
              GRID_DATA='gridData', TRACKED_DATA='trackedData', VMEM_DATA='vmemData', CONDUCTANCE_DATA='conductanceData')

with tempfile.TemporaryDirectory() as directory:
    subprocess.run([sys.executable, 'assembleRelayLoopFiveLevelData.py', '--outputDirectory', directory], check=True)
    page = open(args.templatePath).read()
    for placeholder, name in BLOCKS.items():
        assert page.count(f'__{placeholder}__') == 1, placeholder
        page = page.replace(f'__{placeholder}__', open(f'{directory}/{name}.json').read())
steeringPath = args.steeringDataPath or max(glob.glob('data/relayLoopSteeringData*Codes*.json'), key=lambda p: int(p.rsplit('Codes', 1)[1][:-5]))
assert page.count('__STEERING_DATA__') == 1
page = page.replace('__STEERING_DATA__', open(steeringPath).read())
print(f'steering lab: {steeringPath}')
assert page.count('__EDGE_LANES__') == 1
page = page.replace('__EDGE_LANES__', json.dumps(json.load(open(args.lanesPath)), separators=(',', ':')))   # chosen by optimizeRelayLoopEdgeLanes.py

open(args.outputPath, 'w').write(page)
print(f'wrote {args.outputPath} ({len(page) / 1e6:.2f} MB)')
