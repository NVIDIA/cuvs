#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUVS_DIR=${CUVS_SOURCE_DIR:-${CI_ROOT}}
SOLR_SOURCE_DIR=${SOLR_SOURCE_DIR:-${CI_ROOT}/solr-local}
SOLR_WORK_DIR=${SOLR_WORK_DIR:-${CI_ROOT}/solr-work}
SOLR_RESULTS_DIR=${SOLR_RESULTS_DIR:-${CI_ROOT}/artifacts/solr}
export SOLR_RESULTS_DIR
export SOLR_TEST_BUNDLE="${SOLR_WORK_DIR}/bundle"
export CUVS_SOLR_CONFIG="${SOLR_WORK_DIR}/nightly.json"
mkdir -p "${SOLR_WORK_DIR}" "${SOLR_RESULTS_DIR}"

PHASE=${1:?Usage: test_solr.sh setup|patch|build|restore|test|report}
case "${PHASE}" in
  setup|patch|build|restore|test|report) ;;
  *) echo "Unknown phase: ${PHASE}" >&2; exit 1 ;;
esac
exec > >(tee -a "${SOLR_RESULTS_DIR}/${PHASE}.log") 2>&1

activate_environment() {
  set +u
  conda activate "${SOLR_WORK_DIR}/env"
  set -u
  # Resolve Conda's bin/java symlink to the actual JDK home used by Gradle.
  export JAVA_HOME
  JAVA_HOME=$(dirname "$(dirname "$(readlink -f "${CONDA_PREFIX}/bin/java")")")
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export GRADLE_USER_HOME="${SOLR_WORK_DIR}/gradle"
  export npm_config_cache="${SOLR_WORK_DIR}/npm"
}

run_setup() {
  local BUILD_RUN_ID artifact
  local CPP_CHANNEL="${SOLR_WORK_DIR}/native"
  local JAVA_DIR="${SOLR_WORK_DIR}/java"
  local LUCENE_DIR="${SOLR_WORK_DIR}/lucene"
  rm -f "${SOLR_RESULTS_DIR}/success.json" "${SOLR_RESULTS_DIR}/build-success.json"
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
  CONDA_OVERRIDE_CUDA="${RAPIDS_CUDA_VERSION%.*}" rapids-mamba-retry env create --yes --prefix "${SOLR_WORK_DIR}/env" \
    -f "${SOLR_RESULTS_DIR}/environment.yaml"

  activate_environment
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
}

run_patch() {
  local PATCH_EXITCODE=0
  local PATCH_DIR="${CI_ROOT}/ci/solr/patches"
  local patches file reason
  rm -f "${SOLR_RESULTS_DIR}/success.json"
  # Keep the manifest and exact diffs with the test results for reproducibility.
  cp -a "${PATCH_DIR}" "${SOLR_RESULTS_DIR}/"
  git -C "${SOLR_SOURCE_DIR}" rev-parse HEAD > "${SOLR_RESULTS_DIR}/solr-base-sha.txt"
  patches=$(jq -r '.patches[] | [.file, .issue] | @tsv' "${PATCH_DIR}/solr.json")
  while IFS=$'\t' read -r file reason; do
    [ -n "${file}" ] || continue
    echo "${file}: ${reason}"
    if git -C "${SOLR_SOURCE_DIR}" apply --reverse --check "${PATCH_DIR}/${file}" 2>/dev/null; then
      echo 'Already applied.'
    elif git -C "${SOLR_SOURCE_DIR}" apply "${PATCH_DIR}/${file}"; then
      echo 'Applied.'
    else
      echo "Required Solr patch failed: ${file}" >&2
      PATCH_EXITCODE=1
      break
    fi
  done <<< "${patches}"
  git -C "${SOLR_SOURCE_DIR}" diff --binary > "${SOLR_RESULTS_DIR}/solr-source.diff"
  return "${PATCH_EXITCODE}"
}

run_build() {
  activate_environment
  local DISTRIBUTION
  cd "${SOLR_SOURCE_DIR}"
  ./gradlew --no-daemon --console=plain --stacktrace \
    --init-script "${CI_ROOT}/ci/solr/nightly.init.gradle" \
    --init-script "${CI_ROOT}/ci/solr/export-tests.init.gradle" \
    -Ptests.jvms=1 -Ptests.useSecurityManager=false \
    :solr:packaging:installFullDist :solr:modules:cuvs:prepareCuVSGpuTest
  # The packaging task owns the version-dependent output path.
  DISTRIBUTION=$(cat "${SOLR_RESULTS_DIR}/distribution-path.txt")
  python "${CI_ROOT}/ci/solr/artifacts.py" verify-distribution "${CUVS_SOLR_CONFIG}" "${DISTRIBUTION}"
  python "${CI_ROOT}/ci/solr/artifacts.py" verify-distribution "${CUVS_SOLR_CONFIG}" "${SOLR_TEST_BUNDLE}/classpath"
  mkdir -p "${SOLR_TEST_BUNDLE}/distribution" "${SOLR_TEST_BUNDLE}/provenance" "${SOLR_TEST_BUNDLE}/probe"
  cp -a "${DISTRIBUTION}/." "${SOLR_TEST_BUNDLE}/distribution/"
  cp "${CI_ROOT}/ci/solr/gpu-test.gradle" "${SOLR_TEST_BUNDLE}/build.gradle"
  echo "rootProject.name = 'solr-cuvs-gpu-test'" > "${SOLR_TEST_BUNDLE}/settings.gradle"
  cp "${CUVS_SOLR_CONFIG}" "${SOLR_TEST_BUNDLE}/nightly.json"
  cp "${SOLR_RESULTS_DIR}/revisions.json" "${SOLR_RESULTS_DIR}/build-run.json" "${SOLR_TEST_BUNDLE}/provenance/"
  javac --class-path "$(jq -r '.java.jar' "${CUVS_SOLR_CONFIG}")" \
    -d "${SOLR_TEST_BUNDLE}/probe" "${CI_ROOT}/ci/solr/CheckGpu.java"
  python - "${SOLR_WORK_DIR}/native" "${SOLR_RESULTS_DIR}/conda-explicit.txt" \
    > "${SOLR_TEST_BUNDLE}/environment.txt" <<'PYTHON'
from pathlib import Path
import sys
native, spec = map(Path, sys.argv[1:])
print(spec.read_text().replace(native.resolve().as_uri(), '@CUVS_NATIVE_CHANNEL@'), end='')
PYTHON
  tar -czf "${SOLR_WORK_DIR}/solr-test-bundle.tar.gz" -C "${SOLR_TEST_BUNDLE}" .
  echo '{"distribution_built": true, "tests_compiled": true}' > "${SOLR_RESULTS_DIR}/build-success.json"
}

run_restore() {
  local BUILD_RUN_ID
  mkdir -p "${SOLR_TEST_BUNDLE}" "${SOLR_WORK_DIR}/native"
  tar -xzf "${SOLR_WORK_DIR}/solr-test-bundle.tar.gz" -C "${SOLR_TEST_BUNDLE}"
  cp -a "${SOLR_TEST_BUNDLE}/provenance/." "${SOLR_RESULTS_DIR}/"
  cp "${SOLR_TEST_BUNDLE}/nightly.json" "${CUVS_SOLR_CONFIG}"
  BUILD_RUN_ID=$(jq -r '.build_run_id' "${SOLR_RESULTS_DIR}/revisions.json")
  jq -e --arg cuda "${RAPIDS_CUDA_VERSION}" '.cuda == $cuda' "${SOLR_RESULTS_DIR}/revisions.json" > /dev/null
  # Reuse the CPU job's frozen producer and exact environment, even if main moves.
  gh run download "${BUILD_RUN_ID}" --repo "${RAPIDS_REPOSITORY}" \
    --name "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "${RAPIDS_CUDA_VERSION}")" \
    --dir "${SOLR_WORK_DIR}/native"
  python - "${SOLR_TEST_BUNDLE}" "${SOLR_WORK_DIR}/native" > "${SOLR_RESULTS_DIR}/gpu-environment.txt" <<'PYTHON'
from pathlib import Path
import sys
bundle, native = map(Path, sys.argv[1:])
spec = (bundle / 'environment.txt').read_text()
print(spec.replace('@CUVS_NATIVE_CHANNEL@', native.resolve().as_uri()), end='')
PYTHON
  rapids-mamba-retry create --yes --prefix "${SOLR_WORK_DIR}/env" \
    --file "${SOLR_RESULTS_DIR}/gpu-environment.txt"

  activate_environment
  python "${CI_ROOT}/ci/solr/artifacts.py" verify-distribution "${CUVS_SOLR_CONFIG}" "${SOLR_TEST_BUNDLE}/distribution"
  python "${CI_ROOT}/ci/solr/artifacts.py" verify-distribution "${CUVS_SOLR_CONFIG}" "${SOLR_TEST_BUNDLE}/classpath"
}

run_test() {
  activate_environment
  local JAVA_JAR
  rm -f "${SOLR_RESULTS_DIR}/success.json"
  rm -rf "${SOLR_RESULTS_DIR}/test-results"
  nvidia-smi
  JAVA_JAR=$(find "${SOLR_TEST_BUNDLE}/classpath" -name "$(basename "$(jq -r '.java.jar' "${CUVS_SOLR_CONFIG}")")")
  java --enable-native-access=ALL-UNNAMED --class-path "${SOLR_TEST_BUNDLE}/probe:${JAVA_JAR}" CheckGpu
  # The standalone project only executes precompiled tests; it cannot rebuild Solr.
  bash "${SOLR_TEST_BUNDLE}/gradle/bin/gradle" -p "${SOLR_TEST_BUNDLE}" \
    --offline --no-daemon --console=plain --stacktrace test
  python "${CI_ROOT}/ci/solr/artifacts.py" verify-tests \
    "${SOLR_RESULTS_DIR}/test-results" > "${SOLR_RESULTS_DIR}/success.json.tmp"
  mv "${SOLR_RESULTS_DIR}/success.json.tmp" "${SOLR_RESULTS_DIR}/success.json"
}

run_report() {
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      echo '### Solr / nightly cuVS compatibility'
      if [ "${SOLR_REPORT_STAGE:-test}" = build ] && [ -f "${SOLR_RESULTS_DIR}/build-success.json" ]; then
        echo 'Solr distribution and integration-test bundle built on CPU. See the GPU job for test results.'
      elif [ -f "${SOLR_RESULTS_DIR}/success.json" ]; then
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
}

if [[ "${PHASE}" != report && "${PHASE}" != patch ]] && [ -e /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
fi

"run_${PHASE}"
