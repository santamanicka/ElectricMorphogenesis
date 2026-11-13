#!/bin/bash
# The neighbourhood test of the original code on the aged tissue (EXPLORATORY; analyzeCanalizationTalkRejuvenationNeighbourhood11x11.py): all three targets, three seeds, about 45 minutes.
#   sbatch -p batch -c 4 --mem 8G --time 3:00:00 -o slurmRejuvenationNeighbourhood_%j.out runRejuvenationNeighbourhood.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHONPATH=$PWD python analyzeCanalizationTalkRejuvenationNeighbourhood11x11.py
