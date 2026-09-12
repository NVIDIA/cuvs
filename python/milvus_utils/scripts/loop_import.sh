#! /bin/bash

export IMPORT_JOBID=464889424385847890
export MILVUS_IP=10.185.121.71
import_exit_code=2
date
while [ $import_exit_code -eq 2 ]; do
    python3 scripts/wait_import_complete.py
    import_exit_code=$?
    sleep 600 # 10 minutes
done
date