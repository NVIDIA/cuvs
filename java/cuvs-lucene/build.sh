#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

ARGS="$*"
NUMARGS=$#

VERSION="26.02.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
GROUP_ID="com.nvidia.cuvs"

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

mvn verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn install:install-file -Dfile=./target/cuvs-lucene-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-lucene -Dversion=$VERSION -Dpackaging=jar \
  && cp pom.xml ./target/
