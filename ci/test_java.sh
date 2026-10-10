#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Runs cuvs-java's own IT test suite against the classes from a prior amd64 build job,
# without recompiling (or running jextract) on this host. cuvs-java is always built on
# amd64; this verifies that the resulting jextract-generated Panama bindings also work
# correctly against a native libcuvs_c.so on this host's architecture.

CUVS_JAVA_ARTIFACT="cuvs-java-cuda${RAPIDS_CUDA_VERSION}"

rapids-logger "Testing the amd64-built cuvs-java artifact on $(arch)"

if [ -e "/opt/conda/etc/profile.d/conda.sh" ]; then
  . /opt/conda/etc/profile.d/conda.sh
fi

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
CUVS_JAVA_DIR=$(rapids-download-from-github "${CUVS_JAVA_ARTIFACT}")

rapids-logger "Generate Java testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key java \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java
set -u

rapids-print-env

# libcuvs comes from the conda environment here, matching the architecture this script is
# running on.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

rapids-logger "Restore the amd64-built target/ directory so Maven can run the tests without recompiling"

rm -rf java/cuvs-java/target
mkdir -p java/cuvs-java/target
cp -a "${CUVS_JAVA_DIR}/." java/cuvs-java/target/

EXITCODE=0
set +e

# [DO NOT MERGE] Reproduce the flaky cuvs-java failures (JVM abort with exit code 134 in
# HnswRandomizedIT, and CagraRandomizedIT top-k mismatches). Every Maven run is a fresh JVM, so
# a native abort only ends that run, and the loop goes on.
pushd java/cuvs-java
LOGDIR="$(mktemp -d)"
run_it() {
  local tag=$1 cls=$2 iters=$3
  local log="${LOGDIR}/${tag}.log"
  mvn --batch-mode verify -Dskip.compile=true -Dit.test="${cls}" -Dtest=NONE \
    -Dsurefire.failIfNoSpecifiedTests=false -Dtests.iters="${iters}" > "${log}" 2>&1
  local rc=$?
  local summary
  summary=$(grep -h "Tests run:.*in com" "${log}" | tail -1)
  echo "RUN ${tag} rc=${rc} ${summary}"
  if [ ${rc} -ne 0 ]; then
    EXITCODE=1
    echo "::group::FAILED ${tag}"
    grep -h -E "Exit Code|terminate called|what\(\)|\[warning\]|tiny dataset|MISMATCH|DIAG|<<< (FAILURE|ERROR)|AssertionError|Exception|seed#" "${log}" | head -200
    echo "::endgroup::"
  else
    grep -h "tiny dataset" "${log}"
  fi
}

rapids-logger "Deterministic reproducer: CAGRA build on a single row"
run_it single-row CagraSingleRowIT 1

rapids-logger "HnswRandomizedIT loop"
END=$((SECONDS + 40 * 60))
N=0
while [ ${SECONDS} -lt ${END} ]; do
  N=$((N + 1))
  run_it "hnsw-${N}" HnswRandomizedIT 10
done

rapids-logger "CagraRandomizedIT loop"
END=$((SECONDS + 30 * 60))
N=0
while [ ${SECONDS} -lt ${END} ]; do
  N=$((N + 1))
  run_it "cagra-${N}" CagraRandomizedIT 2
done

rapids-logger "Summary"
grep -h "Tests run:.*in com" "${LOGDIR}"/*.log | sed -E 's/Time elapsed: [0-9.]+ s //' | sort | uniq -c
grep -l "Exit Code: 134" "${LOGDIR}"/*.log
grep -h "MISMATCH" "${LOGDIR}"/*.log
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
