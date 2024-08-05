#!/bin/bash
python plotAnalysisData.py --analysisMode $1
#sbatch --time 0-02:00:00 -p batch --mem 2G runPlotAnalysisData.sh "patternability"
