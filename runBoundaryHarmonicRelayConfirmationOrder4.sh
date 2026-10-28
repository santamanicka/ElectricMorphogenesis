#!/bin/bash
# The order-4 modules-confirmation codes (Amendment 5: 320 codes), relayed exactly as the sweep (runBoundaryHarmonicRelaySweepOrder4.sh with VARIANTS and RECORD_DIR set).
# sbatch --array 0-319%32 --time 1:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicRelayConfirmationOrder4_%a.out runBoundaryHarmonicRelayConfirmationOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
export VARIANTS=data/boundaryHarmonicRingCodeModulesConfirmation1888Hold301FaceMinus60Minus5Order4.json RECORD_DIR=data/relayLoopConfirmationOrder4
exec bash runBoundaryHarmonicRelaySweepOrder4.sh
