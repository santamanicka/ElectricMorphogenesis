#!/bin/bash
# Field action range is quantised by the extracellular grid: integer fieldScreenSize 2 and 3 give
# neighbourhoods of 4.0 and 15.5 grid points per cell with nothing between them, and screenSize 1
# and 2 are the same condition. Fractional values reach one intermediate neighbourhood, 11.7 at
# screenSize 2.3 to 2.9, which is worth having because it sits in the region where addressability
# collapses. The name cannot go through printf '%02d', so the output directory is passed directly.
source ~/.bashrc
myconda
taskId=${SLURM_ARRAY_TASK_ID:-1}
seed=$(( ${seedBase:-2000} + taskId ))
mkdir -p ${screenDir}
python generate_ensemble.py --n ${samplesPerTask:-40} --source ${sourceDat} \
  --fieldScreenSize ${fieldScreenSize} --num_sim_iters ${numSimIters} --clamp_iters 100 \
  --readout_iters 200 --seed ${seed} \
  --output_prefix ${screenDir}/chunk$(printf '%03d' ${taskId})
