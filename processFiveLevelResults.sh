#!/bin/bash
# Once runBoundaryHarmonicRelayFiveLevel.sh's array is done: for every key it ran, the phase summary and the tracked
# edges (what the Relay Loop artifact draws from), then the Vmem snapshots and conductance curves for all of them at
# once. Skips any output that already exists, so it is safe to re-run after a partial failure.
set -e
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
SUFFIX=1888Hold301FaceMinus60Minus5
KEYS=$(python3 -c "import json; print(' '.join(v['key'] for v in json.load(open('data/boundaryHarmonicRingCodeFiveLevel${SUFFIX}.json'))['variants']))")
for key in $KEYS; do
  raw=data/boundaryHarmonicRelayFiveLevel_${key}${SUFFIX}Raw.npz
  [ -f "$raw" ] || { echo "MISSING $raw"; exit 1; }
  [ -f data/boundaryHarmonicRelayVariantPhaseSummary_${key}${SUFFIX}.json ] || python3 computeBoundaryHarmonicRelayVariantPhaseSummary11x11.py --relayPath $raw --variantKey $key
  [ -f data/boundaryHarmonicRelayTrackedEdges_${key}${SUFFIX}.json ] || python3 computeBoundaryHarmonicRelayTrackedEdges11x11.py --relayPath $raw --variantKey $key
done
[ -f data/boundaryHarmonicRelayVmemSnapshotsFiveLevel${SUFFIX}.json ] || python3 extractBoundaryHarmonicRelayVmemSnapshots11x11.py --fiveLevel
[ -f data/boundaryHarmonicRelayConductanceCurvesFiveLevel${SUFFIX}.json ] || python3 extractBoundaryHarmonicRelayConductanceCurves11x11.py --fiveLevel
echo "ALL DONE"
