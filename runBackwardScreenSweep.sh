#!/bin/bash
# One screen per array task, so the sweep survives a lost login session and the screens run at once.
# SLURM_ARRAY_TASK_ID selects the screen from the list rather than being used as the screen directly,
# so the set can be changed without renumbering the array.
source ~/.bashrc
myconda
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
export PYTHONPATH=/cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
SCREENS=(2 4 6 8 10 12)
screen=${SCREENS[$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))]}
model=${model:-./data/bestModelParameters_fieldVector_30x30_616.dat}
steps=${jointSteps:-50}
echo "screen ${screen}, jointSteps ${steps}, model ${model}"
# The control comes first: its own produced pattern has a predecessor by construction, so if that
# fails the screen cannot be inverted at all and the target result below means nothing.
echo -n "CONTROL screen ${screen}: "
python analyzeBackwardReachability.py --parameterfile ${model} --fieldScreenSize ${screen} \
  --targetSource trajectory --jointSteps ${steps} --innerIters ${innerIters:-1000} \
  --lr 0.05 --optimiser adam 2>&1 | grep -oE "residual [0-9.]+ mV"
echo -n "TARGET  screen ${screen}: "
python analyzeBackwardReachability.py --parameterfile ${model} --fieldScreenSize ${screen} \
  --targetSource target --jointSteps ${steps} --innerIters ${innerIters:-1000} \
  --lr 0.05 --optimiser adam 2>&1 | grep -oE "residual [0-9.]+ mV"
echo "done screen ${screen}"
