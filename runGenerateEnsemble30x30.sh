#!/bin/bash
# Generate the 30x30 random-clamp ensemble as a SLURM array job.
#
# Each array task writes one chunk with its own clamp seed; merge_ensemble_chunks.py combines
# them afterwards. The protocol is fixed by Section 8.7 of PolyPatterning_Sim.md: field action
# range 5, a 100-iteration clamp, 2500 simulation iterations, and a readout averaged over the
# last 200 steps because the tissue never reaches a fixed point.
#
# N = 1000 is set by the PCA rather than by budget: with 183 free clamp parameters, the
# 200-sample ensemble used at 11x11 would cap the measurable participation ratio and understate
# the expansion. At ~35 s per sample this is ~9.8 CPU-hours total, ~23 min per task.
#
#sbatch --export=ALL --time 2:00:00 -p batch --array 1-25 -e Error_%A_%a.err --mem 4G runGenerateEnsemble30x30.sh
#
# Smaller test run first, two tasks of five samples:
#sbatch --export=ALL,samplesPerTask=5 --time 0:30:00 -p batch --array 1-2 -e Error_%A_%a.err --mem 4G runGenerateEnsemble30x30.sh

# myconda (module load miniforge; source activate santa) is defined in ~/.bashrc, which a batch
# shell does not source on its own.
source ~/.bashrc
myconda

sourceDat=${sourceDat:-"data/StigmergicModelParameters_30x30.dat"}
samplesPerTask=${samplesPerTask:-40}
numSimIters=${numSimIters:-2500}
clampIters=${clampIters:-100}
readoutIters=${readoutIters:-200}
# Clamp seeds are seedBase + taskId, so chunks draw independent clamps and a rerun of one task
# reproduces exactly that chunk.
seedBase=${seedBase:-1000}
outputDir=${outputDir:-"data/ensemble30x30"}

taskId=${SLURM_ARRAY_TASK_ID:-1}
mkdir -p ${outputDir}

echo "task ${taskId}: ${samplesPerTask} samples, seed $((seedBase + taskId)), source ${sourceDat}"

python generate_ensemble.py \
  --n ${samplesPerTask} \
  --source ${sourceDat} \
  --num_sim_iters ${numSimIters} \
  --clamp_iters ${clampIters} \
  --readout_iters ${readoutIters} \
  --seed $((seedBase + taskId)) \
  --output_prefix ${outputDir}/chunk$(printf '%03d' ${taskId})
