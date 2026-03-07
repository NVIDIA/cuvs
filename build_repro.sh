#!/bin/bash

# nvcc -std=c++17 --extended-lambda --expt-relaxed-constexpr \
#   -Xcompiler -Wno-deprecated-declarations -Xcompiler -fopenmp \
#   -o repro_kmeans_mg ../repro_kmeans_mg.cu \
#   -I$CONDA_PREFIX/include \
#   -I$CONDA_PREFIX/include/rapids \
#   -I$CONDA_PREFIX/include/rapids/libcudacxx \
#   -L$CONDA_PREFIX/lib \
#   -lcuvs -lrmm -lnccl -lucp -lucs -lucxx -lgomp \
#   -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
#   -Xlinker -rpath=$CONDA_PREFIX/lib \
#   -gencode arch=compute_100,code=sm_100

nvcc -std=c++17 --extended-lambda --expt-relaxed-constexpr \
  -Xcompiler -Wno-deprecated-declarations -Xcompiler -fopenmp \
  -o raft_discrete ../raft_discrete.cu \
  -I$CONDA_PREFIX/include \
  -I$CONDA_PREFIX/include/rapids \
  -I$CONDA_PREFIX/include/rapids/libcudacxx \
  -L$CONDA_PREFIX/lib \
  -lcuvs -lrmm -lnccl -lucp -lucs -lucxx -lgomp \
  -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
  -Xlinker -rpath=$CONDA_PREFIX/lib \
  -gencode arch=compute_100,code=sm_100
