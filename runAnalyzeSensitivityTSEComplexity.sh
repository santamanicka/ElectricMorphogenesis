#!/bin/bash
python analyzeSenstivityTSEComplexity.py
#sbatch --export=ALL --time 1-00:00:00 -p batch --mem 10G runAnalyzeSenstivityTSEComplexity.sh

