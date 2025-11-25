#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

########################
# CUVS Version Updater #
########################

## Usage
# Primary interface:   bash update-version.sh <new_version> [--run-context=main|release]
# Fallback interface:  [RAPIDS_RUN_CONTEXT=main|release] bash update-version.sh <new_version>
# CLI arguments take precedence over environment variables
# Defaults to main when no run-context is specified

set -euo pipefail

# Verify we're running from the repository root
if [[ ! -d ".git" ]]; then
    echo "Error: This script must be run from the repository root directory."
    echo "Expected to find: .git/"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Parse command line arguments
CLI_RUN_CONTEXT=""
VERSION_ARG=""

for arg in "$@"; do
    case $arg in
        --run-context=*)
            CLI_RUN_CONTEXT="${arg#*=}"
            shift
            ;;
        *)
            if [[ -z "$VERSION_ARG" ]]; then
                VERSION_ARG="$arg"
            fi
            ;;
    esac
done

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG="$VERSION_ARG"

# Determine RUN_CONTEXT with CLI precedence over environment variable, defaulting to main
if [[ -n "$CLI_RUN_CONTEXT" ]]; then
    RUN_CONTEXT="$CLI_RUN_CONTEXT"
    echo "Using run-context from CLI: $RUN_CONTEXT"
elif [[ -n "${RAPIDS_RUN_CONTEXT}" ]]; then
    RUN_CONTEXT="$RAPIDS_RUN_CONTEXT"
    echo "Using run-context from environment: $RUN_CONTEXT"
else
    RUN_CONTEXT="main"
    echo "No run-context provided, defaulting to: $RUN_CONTEXT"
fi

# Validate RUN_CONTEXT value
if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
    echo "Error: Invalid run-context value '${RUN_CONTEXT}'"
    echo "Valid values: main, release"
    exit 1
fi

# Validate version argument
if [[ -z "$NEXT_FULL_TAG" ]]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <new_version> [--run-context=<context>]"
    echo "   or: [RAPIDS_RUN_CONTEXT=<context>] $0 <new_version>"
    echo "Note: Defaults to main when run-context is not specified"
    exit 1
fi

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_PATCH=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Strip leading 0s in versions, so e.g. '25.10.00' becomes '25.10.0'
PATCH_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_PATCH}'))")

# Log update context
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    RAPIDS_BRANCH_NAME="main"
    echo "Preparing development branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
    echo "Preparing release branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting release/${NEXT_SHORT_TAG} branch)"
fi

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION
echo "${RAPIDS_BRANCH_NAME}" > RAPIDS_BRANCH

# Update Java version
NEXT_FULL_JAVA_TAG="${NEXT_SHORT_TAG}.${PATCH_PEP440}"
sed_runner "s/VERSION=\".*\"/VERSION=\"${NEXT_FULL_JAVA_TAG}\"/g" build.sh
sed_runner "/<!--CUVS_LUCENE#VERSION_UPDATE_MARKER_START-->.*<!--CUVS_LUCENE#VERSION_UPDATE_MARKER_END-->/s//<!--CUVS_LUCENE#VERSION_UPDATE_MARKER_START--><version>${NEXT_FULL_JAVA_TAG}<\/version><!--CUVS_LUCENE#VERSION_UPDATE_MARKER_END-->/g" pom.xml

sed_runner "s| CuVS [[:digit:]]\{2\}\.[[:digit:]]\{2\} | CuVS ${NEXT_SHORT_TAG} |g" README.md

for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "s/libcuvs==.*/libcuvs==${NEXT_SHORT_TAG}.*/g" "${FILE}"
done

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s|@.*|@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done
