#!/bin/bash
# The order-4 face code's first batch (registration data/boundaryHarmonicOrder4ReplicationPredictions1888Hold301FaceMinus60Minus5.json):
#   task 0  order 3, baseline extraUpdateOnly, written to data/order4Verification/ -- checks that the --windowStates edit left the
#           default behaviour alone (compare with the committed boundaryHarmonicRingOnlyRelay...Raw.npz)
#   task 1  order 4, baseline free             -> the Relay registration's counterpart
#   task 2  order 4, baseline extraUpdateOnly  -> the ring-only relay (the Relay Loop's baseline)
#   task 3  order 4, field role: field on/off x released/held throughout
# The order-4 code is the summary's best order 4 (the population-64 restart 05). Landmarks of its interior mean conductance, replayed in
# float64: trough at recorded 584, second peak at recorded 1687, so the transfer windows are cut at states 302,585,1001,1501,1688
# (the order-3 cuts 302,586,1001,1501,1766 with the order-4 trough and write peak).
# sbatch --array 0-3 --time 2:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicRelayOrder4_%a.out runBoundaryHarmonicRelayOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=$PWD
SUFFIX=1888Hold301FaceMinus60Minus5
ORDER4=data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5Population64/order4_restart05.npz
ORDER4ARGS="--order 4 --primaryIteration 1687 --windowStates 302,585,1001,1501,1688"
mkdir -p data/order4Verification

refuse() { [ -e "$1" ] && { echo "$1 exists; not overwriting"; exit 1; }; }

case $SLURM_ARRAY_TASK_ID in
  0) OUT=data/order4Verification/relayOrder3ExtraUpdateOnlyRaw.npz; refuse $OUT
     python computeBoundaryHarmonicRelay11x11.py --baseline extraUpdateOnly --outputPath $OUT ;;
  1) OUT=data/boundaryHarmonicRelay${SUFFIX/FaceMinus60Minus5/FaceMinus60Minus5Order4}Raw.npz; refuse $OUT
     python computeBoundaryHarmonicRelay11x11.py $ORDER4ARGS --baseline free --outputPath $OUT ;;
  2) OUT=data/boundaryHarmonicRingOnlyRelay${SUFFIX/FaceMinus60Minus5/FaceMinus60Minus5Order4}Raw.npz; refuse $OUT
     python computeBoundaryHarmonicRelay11x11.py $ORDER4ARGS --baseline extraUpdateOnly --outputPath $OUT ;;
  3) python analyzeBoundaryHarmonicFieldRole11x11.py --trainedRunPath $ORDER4 --outputSuffix Order4 ;;
esac
