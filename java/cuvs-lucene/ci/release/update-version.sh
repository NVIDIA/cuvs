#!/bin/bash
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
########################
# CUVS Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


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
NEXT_UCXX_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_UCXX_SHORT_TAG}'))")
PATCH_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_PATCH}'))")

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Update Java version
NEXT_FULL_JAVA_TAG="${NEXT_SHORT_TAG}.${PATCH_PEP440}"
sed_runner "s/VERSION=\".*\"/VERSION=\"${NEXT_FULL_JAVA_TAG}\"/g" build.sh
sed_runner "/<!--CUVS_JAVA#VERSION_UPDATE_MARKER_START-->.*<!--CUVS_JAVA#VERSION_UPDATE_MARKER_END-->/s//<!--CUVS_JAVA#VERSION_UPDATE_MARKER_START--><version>${NEXT_FULL_JAVA_TAG}<\/version><!--CUVS_JAVA#VERSION_UPDATE_MARKER_END-->/g" pom.xml

sed_runner "s| CuVS [[:digit:]]\{2\}\.[[:digit:]]\{2\} | CuVS ${NEXT_SHORT_TAG} |g" README.md
sed_runner "s|-[[:digit:]]\{2\}\.[[:digit:]]\{2\}\.[[:digit:]]\{1,2\}\.jar|-${NEXT_FULL_JAVA_TAG}\.jar|g" java/examples/README.md
sed_runner "s|/[[:digit:]]\{2\}\.[[:digit:]]\{2\}\.[[:digit:]]\{1,2\}/|/${NEXT_FULL_JAVA_TAG}/|g" java/examples/README.md
