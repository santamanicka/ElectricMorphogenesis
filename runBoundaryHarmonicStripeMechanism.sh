#!/bin/bash
# The stripe code's mechanism analyses, as one dependency chain on the cluster (nothing here is run on the login node):
#   record      replay the best code of every size in the summary, Vmem and G_pol at every iteration   (recordBoundaryHarmonicRuns11x11.py --mode trained)
#   branches    each cell's stable voltage roots at every moment                                         (computeBoundaryHarmonicBranches11x11.py)
#   switchRule  the latching-switch rule and its one-step accuracy                                       (analyzeBoundaryHarmonicSwitchRule11x11.py)
#   program     P1-P5 of the registration: trough, selectivity, who is dark when                          (analyzeBoundaryHarmonicStripeProgram11x11.py)
#   relay       the exact decomposition, trained minus the extraUpdateOnly baseline, to the best moment    (computeBoundaryHarmonicRelay11x11.py --target stripesInterior)
#   relayScore  V1, V3, R1-R4 of the registration                                                          (analyzeBoundaryHarmonicStripeRelay11x11.py)
#   movie       the relay movie and its ten-block re-cut for the report                                    (buildBoundaryHarmonicRelayVariantMovie11x11.py)
# The registration (data/boundaryHarmonicStripeMechanismPredictions...json) was committed before this was run.
#
#   bash runBoundaryHarmonicStripeMechanism.sh data/boundaryHarmonicTrainingSummary1888Hold301StripesInteriorMinus60Minus5Ceiling2Pilot.json 2
#
# Arguments: the training summary, and the stripe code's size (default 2). The replay and the branches are intermediates and go to
# $INTERMEDIATES (default the scratch folder beside the repository); the analyses write to data/ and never overwrite.
set -e
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
SUMMARY=${1:?the training summary JSON}
ORDER=${2:-2}
INTERMEDIATES=${INTERMEDIATES:-/cluster/tufts/levinlab/smanic02/scratchStripesDev}
SUFFIX=1888Hold301StripesInteriorMinus60Minus5
NAME=$(basename "$SUMMARY" .json | sed 's/^boundaryHarmonicTrainingSummary//')                       # e.g. 1888Hold301StripesInteriorMinus60Minus5Ceiling2Pilot
WRAP='source ~/.bashrc; myconda; cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; export PYTHONPATH=$PWD;'
SUBMIT="sbatch --parsable -p batch -c 4 --mem 16G"
mkdir -p "$INTERMEDIATES"
RECORD=$INTERMEDIATES/record_$NAME.npz
BRANCHES=$INTERMEDIATES/branches_$NAME.npz
SWITCHRULE=data/boundaryHarmonicSwitchRule$NAME.json
RELAYRAW=data/boundaryHarmonicRingOnlyRelay${SUFFIX}Raw.npz

JR=$($SUBMIT -J stripeRecord --time 1:00:00 -o slurmBoundaryHarmonicStripeRecord_%A.out --wrap "$WRAP python -u recordBoundaryHarmonicRuns11x11.py --mode trained --summaryPath $SUMMARY --outputPath $RECORD")
JB=$($SUBMIT -J stripeBranches --dependency=afterok:$JR --time 2:00:00 -o slurmBoundaryHarmonicStripeBranches_%A.out --wrap "$WRAP python -u computeBoundaryHarmonicBranches11x11.py $RECORD $BRANCHES")
JS=$($SUBMIT -J stripeSwitchRule --dependency=afterok:$JB --time 1:00:00 -o slurmBoundaryHarmonicStripeSwitchRule_%A.out --wrap "$WRAP python -u analyzeBoundaryHarmonicSwitchRule11x11.py --recordPath $RECORD --branchPath $BRANCHES --summaryPath $SUMMARY")
JP=$($SUBMIT -J stripeProgram --dependency=afterok:$JS --time 1:00:00 -o slurmBoundaryHarmonicStripeProgram_%A.out --wrap "$WRAP python -u analyzeBoundaryHarmonicStripeProgram11x11.py --recordPath $RECORD --switchRulePath $SWITCHRULE --summaryPath $SUMMARY --order $ORDER")
JL=$($SUBMIT -J stripeRelay --time 1:00:00 -o slurmBoundaryHarmonicStripeRelay_%A.out --wrap "$WRAP python -u computeBoundaryHarmonicRelay11x11.py --baseline extraUpdateOnly --target stripesInterior --order $ORDER --summaryPath $SUMMARY --outputPath $RELAYRAW")
JC=$($SUBMIT -J stripeRelayScore --dependency=afterok:$JL --time 1:00:00 -o slurmBoundaryHarmonicStripeRelayScore_%A.out --wrap "$WRAP python -u analyzeBoundaryHarmonicStripeRelay11x11.py --relayPath $RELAYRAW --outputPath data/boundaryHarmonicStripeRelay$SUFFIX.json")
JM=$($SUBMIT -J stripeMovie --dependency=afterok:$JL --time 1:00:00 -o slurmBoundaryHarmonicStripeMovie_%A.out --wrap "$WRAP python -u buildBoundaryHarmonicRelayVariantMovie11x11.py --relayPath $RELAYRAW --variantsPath none --variantKey trained --outputPath data/boundaryHarmonicRelayVariantMovie_trained$SUFFIX.json")
echo "record $JR -> branches $JB -> switchRule $JS -> program $JP | relay $JL -> relayScore $JC, movie $JM"
