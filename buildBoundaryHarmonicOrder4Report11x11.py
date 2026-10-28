"""Builds the order-4 report page (presentation/order4/orderFourReport.html) from presentation/order4/orderFourReportTemplate.html and
data/order4ReportData1888Hold301FaceMinus60Minus5.json (collectOrder4ReportData11x11.py). EXPLORATORY: nothing here was predicted or registered.

    python3 buildBoundaryHarmonicOrder4Report11x11.py [--overwrite]
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', help='replace the page (the template is edited until the report is final)')
args = parser.parse_args()
dataPath = 'data/order4ReportData1888Hold301FaceMinus60Minus5.json'
templatePath = 'presentation/order4/orderFourReportTemplate.html'
outputPath = 'presentation/order4/orderFourReport.html'
if os.path.exists(outputPath) and not args.overwrite:
    raise SystemExit(f'{outputPath} exists; not overwriting (pass --overwrite)')
data = json.dumps(json.load(open(dataPath)), separators=(',', ':')).replace('</', '<\\/')
page = open(templatePath).read().replace('__DATA__', data)
open(outputPath, 'w').write(page)
print(f'wrote {outputPath}: {len(page) / 1e6:.2f} MB')
