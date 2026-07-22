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

if [ -z "${CMAKE_PREFIX_PATH:=}" ]; then
  CMAKE_PREFIX_PATH="$(pwd)/../../cpp/build"
  export CMAKE_PREFIX_PATH
fi

if [ -z ${LD_LIBRARY_PATH+x} ]; then
  export LD_LIBRARY_PATH=$CMAKE_PREFIX_PATH
else
  export LD_LIBRARY_PATH=$CMAKE_PREFIX_PATH:${LD_LIBRARY_PATH}
fi

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

mvn clean verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn jacoco:report \
  && mvn install:install-file -Dfile=./target/cuvs-lucene-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-lucene -Dversion=$VERSION -Dpackaging=jar \
  && cp pom.xml ./target/
