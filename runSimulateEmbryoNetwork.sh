#!/bin/bash
dims="(10,10)"
python simulateEmbryoNetwork.py --dims $dims --teratogenExposure $teratogenExposure --modelNumEmbryo $modelNumEmbryo --modelNumATP $modelNumATP --save $save
#sbatch --export=ALL,teratogenExposure=True,modelNumEmbryo=253,modelNumATP=262,save=True --time 2-00:00:00 -p batch --mem 4G runSimulateEmbryoNetwork.sh