#!/bin/sh
cd "$(dirname "$0")"
python3 anatomy_local_search.py stage1 >> anatomy/local_stage1.log 2>&1
echo STAGE1DONE >> anatomy/local_stage1.log
