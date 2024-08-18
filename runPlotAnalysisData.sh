#!/bin/bash
python plotAnalysisData.py --analysisMode $analysisMode --characteristicNames $1
#sbatch --export-ALL,analysisMode=fixBiasSweepWeightScreenGJ --time 0-01:00:00 -p batch --mem 2G runPlotAnalysisData.sh "['Covariance']"
