#!/bin/sh
cd "$(dirname "$0")"
python3 anatomy_local_search.py stage1 >> anatomy/local_stage1.log 2>&1
echo STAGE1DONE >> anatomy/local_stage1.log
python3 anatomy_local_search.py stage2 15,20 >> anatomy/local_stage2.log 2>&1
echo STAGE2DONE >> anatomy/local_stage2.log
