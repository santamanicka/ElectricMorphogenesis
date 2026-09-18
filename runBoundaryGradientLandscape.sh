#!/bin/bash
# The boundary dial plus a harmonic, every held value in [0, 1.3], on model 1888 with a 301-iteration hold: order 1
# (code = DC + G cos(theta - phi), the default), order 2 (harmonicOrder=2) or both orders together (combinedOrders=1).
# Each array task runs every numTasks-th grid point; merge afterwards with the same order options plus --merge, e.g.
#   python simulateBoundaryGradientLandscape11x11.py --harmonicOrder 2 --merge
#sbatch --export=ALL,numTasks=64 --time 1:00:00 -p batch --array 0-63 --mem 4G -o slurmBoundaryGradient_%A_%a.out runBoundaryGradientLandscape.sh
#sbatch --export=ALL,numTasks=64,harmonicOrder=2 --time 1:00:00 -p batch --array 0-63 --mem 4G -o slurmBoundaryGradient_%A_%a.out runBoundaryGradientLandscape.sh
#sbatch --export=ALL,numTasks=72,combinedOrders=1 --time 1:00:00 -p batch --array 0-71 --mem 4G -o slurmBoundaryGradient_%A_%a.out runBoundaryGradientLandscape.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
orderOptions="--harmonicOrder ${harmonicOrder:-1}"
[ "${combinedOrders:-0}" = 1 ] && orderOptions="--combinedOrders"
python simulateBoundaryGradientLandscape11x11.py --holdIterations ${holdIterations:-301} ${orderOptions} \
  --dialLimit ${dialLimit:-1.3} --gridStep ${gridStep:-0.02} --gradientDirections ${gradientDirections:-0,22.5,45} \
  --numTasks ${numTasks:-64} --taskIndex ${SLURM_ARRAY_TASK_ID}
