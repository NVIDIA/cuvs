#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.
########################
# CUVS Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_PATCH=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Strip leading 0s in versions, so e.g. '25.10.00' becomes '25.10.0'
PATCH_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_PATCH}'))")

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION

# Update Java version
NEXT_FULL_JAVA_TAG="${NEXT_SHORT_TAG}.${PATCH_PEP440}"
sed_runner "s/VERSION=\".*\"/VERSION=\"${NEXT_FULL_JAVA_TAG}\"/g" build.sh
sed_runner "/<!--CUVS_LUCENE#VERSION_UPDATE_MARKER_START-->.*<!--CUVS_LUCENE#VERSION_UPDATE_MARKER_END-->/s//<!--CUVS_LUCENE#VERSION_UPDATE_MARKER_START--><version>${NEXT_FULL_JAVA_TAG}<\/version><!--CUVS_LUCENE#VERSION_UPDATE_MARKER_END-->/g" pom.xml

sed_runner "s| CuVS [[:digit:]]\{2\}\.[[:digit:]]\{2\} | CuVS ${NEXT_SHORT_TAG} |g" README.md

for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "s/libcuvs==.*/libcuvs==${NEXT_SHORT_TAG}.*/g" "${FILE}"
done

for FILE in .github/workflows/*.yaml; do
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done
