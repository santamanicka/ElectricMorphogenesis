#!/bin/bash
# Field action range sweep measuring ADDRESSABILITY, not complexity.
#
# Section 6 chose action range 4-6 by maximising Gaussian TSE. TSE, participation ratio and
# spectral entropy are all functions of correlation structure and are blind to whether the
# boundary code organises the pattern -- the same blindness that let PR report 163 effective
# dimensions of unaddressable scatter at 30x30. This sweep measures instead whether similar
# boundary codes produce similar interior patterns, across the band where pattern survives.
#
# Band chosen from the unclamped spatial-structure measurements (interior spatial std):
#   screen  5 -> 14.77 mV    8 -> 10.38 mV    10 -> 6.33 mV    11 -> 5.81 mV    15 -> 2.35 mV
# 15 is excluded: at 2.35 mV the tissue is nearly uniform and there is little pattern to address.
#
# 5 screen sizes x 5 chunks x 40 samples = 200 samples per screen size. The clamp seed depends
# on the chunk ONLY, so every screen size sees the identical 200 clamps and any difference in
# addressability is attributable to reach alone.
#
#sbatch --export=ALL --time 2:00:00 -p batch --array 1-25 -e Error_%A_%a.err --mem 4G runFieldRangeSweep.sh
#
# Smaller test first, two tasks of three samples:
#sbatch --export=ALL,samplesPerTask=3 --time 0:30:00 -p batch --array 1-2 -e Error_%A_%a.err --mem 4G runFieldRangeSweep.sh

source ~/.bashrc
myconda

sourceDat=${sourceDat:-"data/StigmergicModelParameters_30x30.dat"}
samplesPerTask=${samplesPerTask:-40}
numSimIters=${numSimIters:-2500}
clampIters=${clampIters:-100}
readoutIters=${readoutIters:-200}
seedBase=${seedBase:-2000}
outputDir=${outputDir:-"data/fieldRangeSweep"}
chunksPerScreen=${chunksPerScreen:-5}

# Band is overridable so a single screen size can be replicated on independent clamps.
read -r -a screenSizes <<< "${screenSizeList:-5 6 8 10 11}"

taskId=${SLURM_ARRAY_TASK_ID:-1}
screenIndex=$(( (taskId - 1) / chunksPerScreen ))
chunkId=$(( (taskId - 1) % chunksPerScreen + 1 ))
fieldScreenSize=${screenSizes[$screenIndex]}

# Seed depends on chunkId only, so the clamp set is identical across screen sizes.
seed=$(( seedBase + chunkId ))

screenDir=${outputDir}/screen$(printf '%02d' ${fieldScreenSize})
mkdir -p ${screenDir}

echo "task ${taskId}: screen ${fieldScreenSize}, chunk ${chunkId}, seed ${seed}, ${samplesPerTask} samples"

python generate_ensemble.py \
  --n ${samplesPerTask} \
  --source ${sourceDat} \
  --fieldScreenSize ${fieldScreenSize} \
  --num_sim_iters ${numSimIters} \
  --clamp_iters ${clampIters} \
  --readout_iters ${readoutIters} \
  --seed ${seed} \
  --output_prefix ${screenDir}/chunk$(printf '%03d' ${chunkId})
