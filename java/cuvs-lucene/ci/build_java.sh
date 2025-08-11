/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

# TODO: Remove this argument-handling when build and test workflows are separated,
#       and test_java.sh no longer calls build_java.sh
#       ref: https://github.com/rapidsai/cuvs/issues/868
EXTRA_BUILD_ARGS=()
if [[ "${1:-}" == "--run-java-tests" ]]; then
  EXTRA_BUILD_ARGS+=("--run-java-tests")
fi

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Java testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key java \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java
set -u

rapids-print-env

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run Java build"

bash ./build.sh ${EXTRA_BUILD_ARGS[@]}"

rapids-logger "Build script exiting with value: $EXITCODE"
exit ${EXITCODE}