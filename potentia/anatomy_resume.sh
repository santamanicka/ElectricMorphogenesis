#!/bin/sh
# Resumable: each stage skips runs already in anatomy/feat/results.jsonl.
cd "$(dirname "$0")"
python3 anatomy_feature_search.py pinch >> anatomy/feat_pinch.log 2>&1
python3 anatomy_feature_search.py bigpreset >> anatomy/feat_bigpreset.log 2>&1
echo ALLDONE >> anatomy/feat_bigpreset.log
