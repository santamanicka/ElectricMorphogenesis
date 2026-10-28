#!/bin/bash
# The order-4 five-order page set (Amendment 6: 121 codes), relayed exactly as the sweep (runBoundaryHarmonicRelaySweepOrder4.sh with VARIANTS and RECORD_DIR set).
# sbatch --array 0-120%32 --time 1:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicRelayPageSetOrder4_%a.out runBoundaryHarmonicRelayPageSetOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
export VARIANTS=data/boundaryHarmonicRingCodePageSet1888Hold301FaceMinus60Minus5Order4.json RECORD_DIR=data/relayLoopPageSetOrder4
exec bash runBoundaryHarmonicRelaySweepOrder4.sh
