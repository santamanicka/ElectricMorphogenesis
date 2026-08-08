#!/bin/bash
# Rebuild every figure the documentation cites, from the committed measurement files.
#
# The documentation references figures by path rather than embedding a copy, and this script is
# what keeps those paths honest. During this investigation a figure was quoted in discussion four
# times after the metric underneath it had changed, and two conclusions were drawn from the stale
# render before the mismatch was noticed. Regenerating is cheap -- the analysis is already done and
# stored -- so there is no reason for a picture and a number to disagree.
#
# Usage:  bash runRegenerateFigures.sh [--reanalyse]
#
# By default this replots from the stored .npz files, which takes seconds. With --reanalyse it
# recomputes the metrics from the raw ensembles first, which takes about fifteen minutes and is
# only necessary when cx_ct_metrics.py has changed.
set -e
cd "$(dirname "$0")"

if [ "$1" == "--reanalyse" ]; then
  echo "recomputing metrics from the raw ensembles..."
  python evaluate_cx_ct_metrics.py --sweepDir data/fieldRangeSweepDense \
    --ranges "[2,2.5,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" \
    --sourceDat data/StigmergicModelParameters_30x30.dat --outputPrefix data/cxCt30x30Dense
  python evaluate_cx_ct_metrics.py --sweepDir data/fieldRangeSweep11x11Dense \
    --ranges "[2,2.5,3,4,5,6,7,8,9,10]" \
    --sourceDat data/StigmergicModelParameters.dat --outputPrefix data/cxCt11x11Dense
fi

python plot_cx_ct_tradeoff.py
python replot_unclamped_evolution.py
echo "figures regenerated"
