#!/bin/bash
# The slider (trained->knockout per order) and grid (two orders at once) ring codes, one relay decomposition per
# array task (computeBoundaryHarmonicRelay11x11.py --ringCodeVariantsPath ... --ringCodeKey ...).
# sbatch --array 0-36 --time 1:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicRelaySliderGrid_%a.out runBoundaryHarmonicRelaySliderGrid.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
KEYS=(slider_o0_t025 slider_o0_t050 slider_o0_t075 slider_o0_t100 slider_o1_t025 slider_o1_t050 slider_o1_t075 slider_o2_t025 slider_o2_t050 slider_o2_t075 slider_o3_t025 slider_o3_t050 slider_o3_t075 grid_o0o1_rhalf_chalf grid_o0o1_rhalf_cknock grid_o0o1_rknock_chalf grid_o0o1_rknock_cknock grid_o0o2_rhalf_chalf grid_o0o2_rhalf_cknock grid_o0o2_rknock_chalf grid_o0o2_rknock_cknock grid_o0o3_rhalf_chalf grid_o0o3_rhalf_cknock grid_o0o3_rknock_chalf grid_o0o3_rknock_cknock grid_o1o2_rhalf_chalf grid_o1o2_rhalf_cknock grid_o1o2_rknock_chalf grid_o1o2_rknock_cknock grid_o1o3_rhalf_chalf grid_o1o3_rhalf_cknock grid_o1o3_rknock_chalf grid_o1o3_rknock_cknock grid_o2o3_rhalf_chalf grid_o2o3_rhalf_cknock grid_o2o3_rknock_chalf grid_o2o3_rknock_cknock)
KEY=${KEYS[$SLURM_ARRAY_TASK_ID]}
PYTHONPATH=$PWD python computeBoundaryHarmonicRelay11x11.py \
  --ringCodeVariantsPath data/boundaryHarmonicRingCodeSliderGrid1888Hold301FaceMinus60Minus5.json \
  --ringCodeKey $KEY \
  --baseline extraUpdateOnly \
  --outputPath data/boundaryHarmonicRelaySliderGrid_${KEY}1888Hold301FaceMinus60Minus5Raw.npz
