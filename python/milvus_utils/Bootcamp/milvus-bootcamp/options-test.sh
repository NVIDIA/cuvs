#!/bin/bash

is_failure(){
  if [ $? -ne 0 ]; then
      exit 1
  fi
}

./index_cycle.sh DISKANN L2 short; is_failure
./index_cycle.sh HNSW L2 short; is_failure
./index_cycle.sh IVF_FLAT IP short; is_failure

./index_cycle.sh DISKANN IP long; is_failure
./index_cycle.sh HNSW IP long; is_failure
./index_cycle.sh IVF_FLAT L2 long; is_failure
