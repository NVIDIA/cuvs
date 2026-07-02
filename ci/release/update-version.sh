#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
########################
# CUVS Version Updater #
########################

## Usage
# Primary interface: ./ci/release/update-version.sh --run-context=main|release <new_version>
# Fallback: Environment variable support for automation needs
# NOTE: Must be run from the root of the repository
#
# CLI args take precedence when both are provided
# If neither RUN_CONTEXT nor --run-context is provided, defaults to main
#
# Examples:
#   ./ci/release/update-version.sh --run-context=main 25.12.00
#   ./ci/release/update-version.sh --run-context=release 25.12.00
#   RAPIDS_RUN_CONTEXT=main ./ci/release/update-version.sh 25.12.00

# Verify we're running from the repository root
if [[ ! -f "VERSION" ]] || [[ ! -f "ci/release/update-version.sh" ]] || [[ ! -d "python" ]]; then
    echo "Error: This script must be run from the root of the cuvs repository"
    echo ""
    echo "Usage:"
    echo "  cd /path/to/cuvs"
    echo "  ./ci/release/update-version.sh --run-context=main|release <new_version>"
    echo ""
    echo "Example:"
    echo "  ./ci/release/update-version.sh --run-context=main 25.12.00"
    exit 1
fi

# Parse command line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --run-context=*)
      CLI_RUN_CONTEXT="${1#*=}"
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Determine RUN_CONTEXT with precedence: CLI > Environment > Default
if [[ -n "${CLI_RUN_CONTEXT:-}" ]]; then
    RUN_CONTEXT="${CLI_RUN_CONTEXT}"
    echo "Using run-context from CLI: ${RUN_CONTEXT}"
elif [[ -n "${RAPIDS_RUN_CONTEXT:-}" ]]; then
    RUN_CONTEXT="${RAPIDS_RUN_CONTEXT}"
    echo "Using RUN_CONTEXT from environment: ${RUN_CONTEXT}"
else
    RUN_CONTEXT="main"
    echo "Using default run-context: ${RUN_CONTEXT}"
fi

# Validate RUN_CONTEXT
if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
    echo "Error: Invalid run-context '${RUN_CONTEXT}'. Must be 'main' or 'release'"
    exit 1
fi

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo "$CURRENT_TAG" | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo "$CURRENT_TAG" | awk '{split($0, a, "."); print a[2]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_PATCH=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")
PATCH_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_PATCH}'))")

# Determine branch name based on context
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    RAPIDS_BRANCH_NAME="main"
    echo "Preparing development branch update ${CURRENT_TAG} => ${NEXT_FULL_TAG} (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
    echo "Preparing release branch update ${CURRENT_TAG} => ${NEXT_FULL_TAG} (targeting release/${NEXT_SHORT_TAG} branch)"
fi

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION
echo "${RAPIDS_BRANCH_NAME}" > RAPIDS_BRANCH

DEPENDENCIES=(
  dask-cuda
  cuvs
  cuvs-bench
  libcuvs
  libcuvs-tests
  libraft
  librmm
  pylibraft
  rmm
  rapids-dask-dependency
)
for FILE in dependencies.yaml conda/environments/*.yaml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
done
for FILE in python/*/pyproject.toml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" "${FILE}"
  done
done

# CI files - context-aware branch references
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s|@.*|@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done

# Documentation and code references - context-aware
if [[ "${RUN_CONTEXT}" == "main" ]]; then
  # In main context, keep documentation on main (no changes needed)
  :
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
  # In release context, use release branch for documentation links (word boundaries to avoid partial matches)
  sed_runner "/rapidsai\\/cuvs/ s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" fern/pages/developer_guide.md
  sed_runner "s|\\bmain\\b|release/${NEXT_SHORT_TAG}|g" README.md
  # Only update the GitHub URL, not the main() function
  sed_runner "s|/cuvs/blob/\\bmain\\b/|/cuvs/blob/release/${NEXT_SHORT_TAG}/|g" python/cuvs_bench/cuvs_bench/plot/__main__.py
fi

# Update cuvs-bench Docker image references (version-only, not branch-related)
sed_runner "s|rapidsai/cuvs-bench:[0-9][0-9].[0-9][0-9]|rapidsai/cuvs-bench:${NEXT_SHORT_TAG}|g" fern/pages/cuvs_bench/index.md

# Version references (not branch-related)
sed_runner "s|=[0-9][0-9].[0-9][0-9]|=${NEXT_SHORT_TAG}|g" README.md
sed_runner "s|@v[0-9][0-9].[0-9][0-9].[0-9][0-9]|@v${NEXT_FULL_TAG}|g" examples/go/README.md

# rust can't handle leading 0's in the major/minor/patch version - remove
NEXT_FULL_RUST_TAG=$(printf "%d.%d.%d" $((10#$NEXT_MAJOR)) $((10#$NEXT_MINOR)) $((10#$NEXT_PATCH)))
sed_runner "s/version = \".*\"/version = \"${NEXT_FULL_RUST_TAG}\"/g" rust/Cargo.toml

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/ucx:[0-9.]*@rapidsai/devcontainers/features/ucx:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/cuda:[0-9.]*@rapidsai/devcontainers/features/cuda:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-${CURRENT_SHORT_TAG}@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done

# Update Java API version
NEXT_FULL_JAVA_TAG="${NEXT_SHORT_TAG}.${PATCH_PEP440}"
sed_runner "s/VERSION=\".*\"/VERSION=\"${NEXT_FULL_JAVA_TAG}\"/g" java/build.sh
for FILE in java/*/pom.xml; do
  sed_runner "/<!--CUVS_JAVA#VERSION_UPDATE_MARKER_START-->.*<!--CUVS_JAVA#VERSION_UPDATE_MARKER_END-->/s//<!--CUVS_JAVA#VERSION_UPDATE_MARKER_START--><version>${NEXT_FULL_JAVA_TAG}<\/version><!--CUVS_JAVA#VERSION_UPDATE_MARKER_END-->/g" "${FILE}"
done

sed_runner "s| CuVS [[:digit:]]\{2\}\.[[:digit:]]\{2\} | CuVS ${NEXT_SHORT_TAG} |g" java/README.md
sed_runner "s|-[[:digit:]]\{2\}\.[[:digit:]]\{2\}\.[[:digit:]]\{1,2\}\.jar|-${NEXT_FULL_JAVA_TAG}\.jar|g" java/examples/README.md
sed_runner "s|/[[:digit:]]\{2\}\.[[:digit:]]\{2\}\.[[:digit:]]\{1,2\}/|/${NEXT_FULL_JAVA_TAG}/|g" java/examples/README.md
