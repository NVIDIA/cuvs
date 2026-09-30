#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

ARGS="$*"
NUMARGS=$#

VERSION="26.12.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Resolve paths from this script's location so it can be invoked from anywhere.
LUCENE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPODIR="$(cd "${LUCENE_DIR}"/../.. && pwd)"

# cuvs-lucene compiles against the cuvs-java artifact built by this repo, which
# './build.sh java' installs into the local Maven repository.
MAVEN_LOCAL_REPO="${MAVEN_LOCAL_REPO:-${HOME}/.m2/repository}"
if [ ! -f "${MAVEN_LOCAL_REPO}/com/nvidia/cuvs/cuvs-java/${VERSION}/cuvs-java-${VERSION}.jar" ]; then
    echo "com.nvidia.cuvs:cuvs-java:${VERSION} was not found in ${MAVEN_LOCAL_REPO}."
    echo "Please build it first (ex. '${REPODIR}/build.sh libcuvs java') if it is not already installed."
fi

# The tests load libcuvs_c.so through the JVM, so make a local libcuvs build
# discoverable. In CI, libcuvs comes from the conda environment instead.
CUVS_LIB_DIR="${CMAKE_PREFIX_PATH:-${REPODIR}/cpp/build}"
if [ -d "${CUVS_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${CUVS_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# A test failure in one module doesn't stop the others: every module is tested and reported,
# and the build still fails at the end.
MAVEN_INSTALL_ARGS=("--fail-at-end")
if ! hasArg --run-java-tests; then
    MAVEN_INSTALL_ARGS+=("-DskipTests")
fi

cd "${LUCENE_DIR}"

# Builds every lucene-X.Y module (one artifact per supported Lucene release), runs their tests when
# asked to, and installs them into the local Maven repository. Each module's target/ also gets a
# standalone pom.xml, so that it holds all the files needed to publish that module. When the tests
# run, the build also writes their coverage report to each module's target/site/jacoco.
mvn clean install "${MAVEN_INSTALL_ARGS[@]}"

# Build the cuvs-lucene examples against the jar just installed above, to catch drift between the
# examples and the cuvs-lucene API.
if hasArg --build-java-examples; then
    mvn -f "${REPODIR}/examples/java/cuvs-lucene/pom.xml" package
fi
