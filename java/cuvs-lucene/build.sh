#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

ARGS="$*"
NUMARGS=$#

VERSION="26.10.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
GROUP_ID="com.nvidia.cuvs.lucene"

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

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
    MAVEN_VERIFY_ARGS=("-DskipTests")
fi

cd "${LUCENE_DIR}"

mvn clean verify "${MAVEN_VERIFY_ARGS[@]}"

# Coverage data only exists when the tests actually ran.
if hasArg --run-java-tests; then
    mvn jacoco:report
fi

mvn install:install-file -Dfile=./target/cuvs-lucene-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-lucene -Dversion=$VERSION -Dpackaging=jar
cp pom.xml ./target/
