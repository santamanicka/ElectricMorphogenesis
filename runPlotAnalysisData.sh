#!/bin/bash
python plotAnalysisData.py --analysisMode $analysisMode --characteristicNames $1 --sample $sample
#sbatch --export=ALL,analysisMode=fixBiasSweepWeightScreenGJ,sample=Homogenous --time 0-01:00:00 -p batch --mem 2G runPlotAnalysisData.sh "['eVAggVmemDimensionMI']"
