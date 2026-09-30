#!/bin/bash
# The curated steering/knockout ring-code variants, one relay decomposition per array task
# (computeBoundaryHarmonicRelay11x11.py --ringCodeVariantsPath ... --ringCodeKey ...).
# sbatch --array 0-7 --time 1:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicRelayVariant_%a.out runBoundaryHarmonicRelayVariants.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
KEYS=(steerOrder0 steerOrder1 steerOrder2 steerOrder3 knockoutOrder1 knockoutOrder2 knockoutOrder3 knockoutOrders123)
KEY=${KEYS[$SLURM_ARRAY_TASK_ID]}
PYTHONPATH=$PWD python computeBoundaryHarmonicRelay11x11.py \
  --ringCodeVariantsPath data/boundaryHarmonicRingCodeVariants1888Hold301FaceMinus60Minus5.json \
  --ringCodeKey $KEY \
  --baseline extraUpdateOnly \
  --outputPath data/boundaryHarmonicRelayVariant_${KEY}1888Hold301FaceMinus60Minus5Raw.npz
