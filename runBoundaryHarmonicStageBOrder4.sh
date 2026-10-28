#!/bin/bash
# Stage B of the order-4 replication's third batch: the reduced-program analyses (latch, nucleators and recruits, program ensemble, steering,
# set point, field coupling, recruitment necessity). Settings are the order-3 scripts' own (constants inside the scripts), with the order-4 code
# and the order-4 tag via BOUNDARY_FACE_ORDER=4. Records are the Stage A intermediates in $INTERMEDIATES; the one new record is the ensemble
# around the trained code (the order-3 slice ensemble: 1024 codes within +-0.1 of every coefficient, seed 23 -- recovered from the stored codes).
#   slice        record ensemble, 1024 codes, --sliceWidth 0.1
#   program      ProgramEnsemble on the scattered ensemble (1024, seed 17) + the slice ensemble
#   steering     SteeringSweep on the steering record
#   setpoint     SetPointSweep runner (field set point x 0.6 0.8 0.9 1.0 1.1 1.2 1.4) then its scoring
#   latch        LatchIntervention runner (block 14 feature cells, force 14 background cells, 4 no-ops; window 1250..best) then its scoring
#   aggregate    AggregateNucleation runner (nucleators/recruits from the order-4 switch-rule events; baseline, all, ladder 2/4/6/8, 20+20 random, no-op) then scoring
#   coupling     FieldCoupling on the trained record
#   necessity    RecruitmentNecessity on the trained record and the latch intervention
# usage: sbatch ... runBoundaryHarmonicStageBOrder4.sh <slice|program|steering|setpoint|latch|aggregate|coupling|necessity>
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$PWD BOUNDARY_FACE_ORDER=4
I=/cluster/tufts/levinlab/smanic02/scratchFaceOrder4
refuse() { [ -e "$1" ] && { echo "$1 exists; not overwriting"; exit 1; }; }
case $1 in
  slice) refuse $I/recordSliceOrder4.npz
         python -u recordBoundaryHarmonicRuns11x11.py --mode ensemble --numCodes 1024 --sliceWidth 0.1 --outputPath $I/recordSliceOrder4.npz ;;
  program) python -u analyzeBoundaryHarmonicProgramEnsemble11x11.py --recordPath $I/recordEnsembleOrder4.npz --slicePath $I/recordSliceOrder4.npz ;;
  steering) python -u analyzeBoundaryHarmonicSteeringSweep11x11.py --recordPath $I/recordSteeringOrder4.npz ;;
  setpoint) refuse $I/setPointRunOrder4.npz
            python -u runBoundaryHarmonicSetPointSweep11x11.py $I/setPointRunOrder4.npz
            python -u analyzeBoundaryHarmonicSetPointSweep11x11.py --runPath $I/setPointRunOrder4.npz ;;
  latch) refuse $I/latchRunOrder4.npz
         python -u runBoundaryHarmonicLatchIntervention11x11.py $I/latchRunOrder4.npz $I/recordTrainedOrder4.npz
         python -u analyzeBoundaryHarmonicLatchIntervention11x11.py --interventionPath $I/latchRunOrder4.npz --recordPath $I/recordTrainedOrder4.npz ;;
  aggregate) refuse $I/aggregateRunOrder4.npz
         python -u runBoundaryHarmonicAggregateNucleation11x11.py $I/aggregateRunOrder4.npz $I/recordTrainedOrder4.npz
         python -u analyzeBoundaryHarmonicAggregateNucleation11x11.py --runPath $I/aggregateRunOrder4.npz ;;
  coupling) python -u analyzeBoundaryHarmonicFieldCoupling11x11.py --recordPath $I/recordTrainedOrder4.npz ;;
  necessity) python -u analyzeBoundaryHarmonicRecruitmentNecessity11x11.py --recordPath $I/recordTrainedOrder4.npz --interventionPath $I/latchRunOrder4.npz ;;
esac
