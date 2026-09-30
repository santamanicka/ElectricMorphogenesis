#!/bin/bash
# Once runBoundaryHarmonicRelaySliderGrid.sh's array is done: build each key's phase summary (fast, what the
# Relay Loop artifact draws from) first, then its full movie/resolutions (slower, for later reuse), skipping
# any key whose output already exists so this is safe to re-run after a partial failure.
set -e
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
KEYS=(slider_o0_t025 slider_o0_t050 slider_o0_t075 slider_o0_t100 slider_o1_t025 slider_o1_t050 slider_o1_t075 slider_o2_t025 slider_o2_t050 slider_o2_t075 slider_o3_t025 slider_o3_t050 slider_o3_t075 grid_o0o1_rhalf_chalf grid_o0o1_rhalf_cknock grid_o0o1_rknock_chalf grid_o0o1_rknock_cknock grid_o0o2_rhalf_chalf grid_o0o2_rhalf_cknock grid_o0o2_rknock_chalf grid_o0o2_rknock_cknock grid_o0o3_rhalf_chalf grid_o0o3_rhalf_cknock grid_o0o3_rknock_chalf grid_o0o3_rknock_cknock grid_o1o2_rhalf_chalf grid_o1o2_rhalf_cknock grid_o1o2_rknock_chalf grid_o1o2_rknock_cknock grid_o1o3_rhalf_chalf grid_o1o3_rhalf_cknock grid_o1o3_rknock_chalf grid_o1o3_rknock_cknock grid_o2o3_rhalf_chalf grid_o2o3_rhalf_cknock grid_o2o3_rknock_chalf grid_o2o3_rknock_cknock)

echo "=== phase summaries (fast pass) ==="
for key in "${KEYS[@]}"; do
  out="data/boundaryHarmonicRelayVariantPhaseSummary_${key}1888Hold301FaceMinus60Minus5.json"
  if [ -f "$out" ]; then echo "skip $key (phase summary exists)"; continue; fi
  echo "--- $key ---"
  python3 computeBoundaryHarmonicRelayVariantPhaseSummary11x11.py \
    --relayPath data/boundaryHarmonicRelaySliderGrid_${key}1888Hold301FaceMinus60Minus5Raw.npz \
    --variantKey $key
done

echo "=== full movie + resolutions (slower pass, for later reuse) ==="
for key in "${KEYS[@]}"; do
  out="data/boundaryHarmonicRelayVariantMovie_${key}1888Hold301FaceMinus60Minus5.json"
  if [ -f "$out" ]; then echo "skip $key (movie exists)"; continue; fi
  echo "--- $key ---"
  python3 buildBoundaryHarmonicRelayVariantMovie11x11.py \
    --relayPath data/boundaryHarmonicRelaySliderGrid_${key}1888Hold301FaceMinus60Minus5Raw.npz \
    --variantsPath data/boundaryHarmonicRingCodeSliderGrid1888Hold301FaceMinus60Minus5.json \
    --variantKey $key \
    --outputPath "$out"
done
echo "ALL DONE"
