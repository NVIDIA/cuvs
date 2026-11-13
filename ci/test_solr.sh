#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUVS_DIR=${CUVS_SOURCE_DIR:-${CI_ROOT}}
SOLR_SOURCE_DIR=${SOLR_SOURCE_DIR:-${CI_ROOT}/solr-local}
SOLR_WORK_DIR=${SOLR_WORK_DIR:-${CI_ROOT}/solr-work}
SOLR_RESULTS_DIR=${SOLR_RESULTS_DIR:-${CI_ROOT}/artifacts/solr}
export CUVS_SOLR_CONFIG="${SOLR_WORK_DIR}/nightly.json"
mkdir -p "${SOLR_WORK_DIR}" "${SOLR_RESULTS_DIR}"

PHASE=${1:?Usage: test_solr.sh setup|patch|build|test|report}
case "${PHASE}" in setup|patch|build|test|report) ;; *) echo "Unknown phase: ${PHASE}" >&2; exit 1 ;; esac
exec > >(tee -a "${SOLR_RESULTS_DIR}/${PHASE}.log") 2>&1

if [ "${PHASE}" = report ]; then
  if [ -d "${SOLR_SOURCE_DIR}/solr/modules/cuvs/build/test-results" ]; then
    cp -a "${SOLR_SOURCE_DIR}/solr/modules/cuvs/build/test-results" "${SOLR_RESULTS_DIR}/"
  fi
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      echo '### Solr / nightly cuVS compatibility'
      if [ -f "${SOLR_RESULTS_DIR}/success.json" ]; then
        echo 'Solr distribution built and GPU indexing test passed.'
      else
        echo '**Compatibility check did not pass.** See the solr-cuvs artifact for phase logs and test reports.'
      fi
      if [ -f "${SOLR_RESULTS_DIR}/revisions.json" ]; then
        echo '```json'
        cat "${SOLR_RESULTS_DIR}/revisions.json"
        echo '```'
      fi
    } >> "${GITHUB_STEP_SUMMARY}"
  fi
  exit 0
fi

if [ -e /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
fi

if [ "${PHASE}" = setup ]; then
  rm -f "${SOLR_RESULTS_DIR}/success.json"
  export RAPIDS_SHA
  RAPIDS_SHA=$(git -C "${CUVS_DIR}" rev-parse HEAD)
  BUILD_RUN_ID=${SOLR_BUILD_RUN_ID:-$(rapids-github-run-id)}
  if [[ ! "${BUILD_RUN_ID}" =~ ^[0-9]+$ ]]; then
    echo "Invalid build run ID: ${BUILD_RUN_ID}" >&2
    exit 1
  fi
  # Resolve once: all downloads must come from the same completed producer run.
  gh api "repos/${RAPIDS_REPOSITORY}/actions/runs/${BUILD_RUN_ID}" > "${SOLR_RESULTS_DIR}/build-run.json"
  jq -e --arg sha "${RAPIDS_SHA}" \
    '.head_sha == $sha and .path == ".github/workflows/build.yaml" and .status == "completed"' \
    "${SOLR_RESULTS_DIR}/build-run.json" > /dev/null
  jq -n --arg cuvs "${RAPIDS_SHA}" --arg ci "$(git -C "${CI_ROOT}" rev-parse HEAD)" \
    --arg solr "$(git -C "${SOLR_SOURCE_DIR}" rev-parse HEAD)" \
    --arg run "${BUILD_RUN_ID}" --arg cuda "${RAPIDS_CUDA_VERSION}" \
    '{ci_sha: $ci, cuvs_sha: $cuvs, solr_sha: $solr, build_run_id: $run, cuda: $cuda}' \
    > "${SOLR_RESULTS_DIR}/revisions.json"
  CPP_CHANNEL="${SOLR_WORK_DIR}/native"
  JAVA_DIR="${SOLR_WORK_DIR}/java"
  LUCENE_DIR="${SOLR_WORK_DIR}/lucene"
  for kind in native java lucene; do
    case "${kind}" in
      native) artifact=$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "${RAPIDS_CUDA_VERSION}") ;;
      java) artifact="cuvs-java-cuda${RAPIDS_CUDA_VERSION}" ;;
      lucene) artifact="cuvs-lucene-cuda${RAPIDS_CUDA_VERSION}" ;;
    esac
    mkdir -p "${SOLR_WORK_DIR}/${kind}"
    gh run download "${BUILD_RUN_ID}" --repo "${RAPIDS_REPOSITORY}" \
      --name "${artifact}" --dir "${SOLR_WORK_DIR}/${kind}"
  done

  cd "${CUVS_DIR}"
  conda config --set channel_priority strict
  rapids-dependency-file-generator --output conda --file-key java \
    --prepend-channel "${CPP_CHANNEL}" \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
    | tee "${SOLR_RESULTS_DIR}/environment.yaml"
  rapids-mamba-retry env create --yes --prefix "${SOLR_WORK_DIR}/env" \
    -f "${SOLR_RESULTS_DIR}/environment.yaml"
fi

set +u
conda activate "${SOLR_WORK_DIR}/env"
set -u
# Resolve Conda's bin/java symlink to the actual JDK home used by Gradle.
export JAVA_HOME
JAVA_HOME=$(dirname "$(dirname "$(readlink -f "${CONDA_PREFIX}/bin/java")")")
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GRADLE_USER_HOME="${SOLR_WORK_DIR}/gradle"
export npm_config_cache="${SOLR_WORK_DIR}/npm"

if [ "${PHASE}" = patch ]; then
  rm -f "${SOLR_RESULTS_DIR}/success.json"
  # Keep the manifest and exact diffs with the test results for reproducibility.
  cp -a "${CI_ROOT}/ci/solr/patches" "${SOLR_RESULTS_DIR}/"
  git -C "${SOLR_SOURCE_DIR}" rev-parse HEAD > "${SOLR_RESULTS_DIR}/solr-base-sha.txt"
  PATCH_EXITCODE=0
  cmake -S "${CI_ROOT}/ci/solr" -B "${SOLR_WORK_DIR}/patch-build" \
    -DSOLR_SOURCE_DIR="${SOLR_SOURCE_DIR}" || PATCH_EXITCODE=$?
  if [ -d "${SOLR_WORK_DIR}/patch-build/rapids-cmake/patches/solr" ]; then
    mkdir -p "${SOLR_RESULTS_DIR}/patch-status"
    rm -f "${SOLR_RESULTS_DIR}/patch-status/log" "${SOLR_RESULTS_DIR}/patch-status/err"
    cp -a "${SOLR_WORK_DIR}/patch-build/rapids-cmake/patches/solr/." "${SOLR_RESULTS_DIR}/patch-status/"
  fi
  git -C "${SOLR_SOURCE_DIR}" diff --binary > "${SOLR_RESULTS_DIR}/solr-source.diff"
  exit "${PATCH_EXITCODE}"
fi

if [ "${PHASE}" = setup ]; then
  nvidia-smi
  java -version
  conda list --explicit > "${SOLR_RESULTS_DIR}/conda-explicit.txt"
  # Channel priority alone is insufficient if the build artifact is incomplete.
  python - "${CONDA_PREFIX}" "${CPP_CHANNEL}" <<'PY'
import json
import pathlib
import sys
prefix, channel = map(pathlib.Path, sys.argv[1:])
records = [json.loads(p.read_text()) for p in (prefix / 'conda-meta').glob('libcuvs-*.json')]
records = [r for r in records if r['name'] == 'libcuvs']
if len(records) != 1 or not records[0].get('url', '').startswith(channel.as_uri() + '/'):
    raise SystemExit('libcuvs was not installed from the selected build artifact')
PY
  python "${CI_ROOT}/ci/solr/artifacts.py" stage --solr "${SOLR_SOURCE_DIR}" \
    --java "${JAVA_DIR}" --lucene "${LUCENE_DIR}" \
    --repository "${SOLR_WORK_DIR}/maven" --reports "${SOLR_RESULTS_DIR}" > "${CUVS_SOLR_CONFIG}"
  cp "${CUVS_SOLR_CONFIG}" "${SOLR_RESULTS_DIR}/nightly.json"
  java --enable-native-access=ALL-UNNAMED \
    --class-path "$(jq -r '.java.jar' "${CUVS_SOLR_CONFIG}")" "${CI_ROOT}/ci/solr/CheckGpu.java"
  exit 0
fi

cd "${SOLR_SOURCE_DIR}"
GRADLE_ARGS=(--no-daemon --console=plain --stacktrace --init-script "${CI_ROOT}/ci/solr/nightly.init.gradle")
if [ "${PHASE}" = build ]; then
  ./gradlew "${GRADLE_ARGS[@]}" :solr:packaging:installFullDist
  # The packaging task owns the version-dependent output path.
  python "${CI_ROOT}/ci/solr/artifacts.py" verify-distribution "${CUVS_SOLR_CONFIG}" \
    "$(cat "${SOLR_RESULTS_DIR}/distribution-path.txt")"
else
  rm -f "${SOLR_RESULTS_DIR}/success.json"
  rm -rf "${SOLR_SOURCE_DIR}/solr/modules/cuvs/build/test-results/test"
  ./gradlew "${GRADLE_ARGS[@]}" :solr:modules:cuvs:test \
    --tests org.apache.solr.cuvs.TestCuVSCodecSupportIT \
    -Ptests.jvms=1 -Ptests.neverUpToDate=true -Ptests.useSecurityManager=false
  python "${CI_ROOT}/ci/solr/artifacts.py" verify-tests \
    "${SOLR_SOURCE_DIR}/solr/modules/cuvs/build/test-results/test" > "${SOLR_RESULTS_DIR}/success.json.tmp"
  mv "${SOLR_RESULTS_DIR}/success.json.tmp" "${SOLR_RESULTS_DIR}/success.json"
fi
