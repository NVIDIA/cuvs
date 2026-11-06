#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install the built cuVS Conda packages with only their runtime dependencies
# and import them on a CPU-only runner. This replaces the import tests that
# used to run inside each recipe build, taking them off the critical path of
# the GPU test jobs, and checks that nothing links against the CUDA driver or
# initializes CUDA at import time.

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading built Conda packages"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cuvs cuvs --stable --cuda "$RAPIDS_CUDA_VERSION")")

# Create an environment holding the given built packages and their runtime
# dependencies, then import the given Python modules in it.
#
# usage: import_packages <env name> <package glob> [<package glob> ...] -- <module> [<module> ...]
import_packages() {
    local env_name="$1"
    shift
    local packages=()
    while [[ "$1" != "--" ]]; do
        packages+=("$1")
        shift
    done
    shift

    rapids-logger "Install ${packages[*]} with only their runtime dependencies"
    rapids-mamba-retry create --yes --name "${env_name}" --strict-channel-priority \
        --override-channels \
        --channel "${PYTHON_CHANNEL}" --channel "${CPP_CHANNEL}" \
        --channel rapidsai-nightly --channel conda-forge \
        "python=${RAPIDS_PY_VERSION}" "${packages[@]}"

    set +u
    conda activate "${env_name}"
    set -u
    rapids-print-env

    for module in "$@"; do
        rapids-logger "Import ${module} on a CPU-only runner"
        python -c "import ${module}"
    done

    rapids-logger "Check that cuvs_bench ships its dataset configuration"
    python -c "import cuvs_bench, pathlib; path = pathlib.Path(cuvs_bench.__file__).parent / 'config' / 'datasets' / 'datasets.yaml'; assert path.is_file(), path"

    set +u
    conda deactivate
    set -u
}

# cuvs-bench and cuvs-bench-cpu both provide the cuvs_bench module, so they
# are checked in separate environments.
import_packages python_imports \
    "${PYTHON_CHANNEL}"/linux-64/cuvs-[0-9]*.conda \
    "${PYTHON_CHANNEL}"/linux-64/cuvs-bench-[0-9]*.conda \
    -- \
    cuvs \
    cuvs.cluster \
    cuvs.common \
    cuvs.distance \
    cuvs.neighbors \
    cuvs.preprocessing \
    cuvs_bench \
    cuvs_bench.generate_groundtruth \
    cuvs_bench.get_dataset \
    cuvs_bench.plot \
    cuvs_bench.run \
    cuvs_bench.split_groundtruth

# cuvs-bench-cpu is only built for one CUDA major version (see ci/build_python.sh).
if compgen -G "${PYTHON_CHANNEL}/linux-64/cuvs-bench-cpu-[0-9]*.conda" > /dev/null; then
    import_packages python_imports_cpu \
        "${PYTHON_CHANNEL}"/linux-64/cuvs-bench-cpu-[0-9]*.conda \
        -- \
        cuvs_bench \
        cuvs_bench.generate_groundtruth \
        cuvs_bench.get_dataset \
        cuvs_bench.plot \
        cuvs_bench.run \
        cuvs_bench.split_groundtruth
else
    rapids-logger "No cuvs-bench-cpu package was built for CUDA ${RAPIDS_CUDA_VERSION}; skipping"
fi
