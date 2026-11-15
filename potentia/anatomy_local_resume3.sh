#!/bin/sh
cd "$(dirname "$0")"
python3 anatomy_local_search.py stage3 >> anatomy/local_stage3.log 2>&1
echo STAGE3DONE >> anatomy/local_stage3.log
