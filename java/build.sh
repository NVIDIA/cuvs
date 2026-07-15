#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

ARGS="$*"
NUMARGS=$#

VERSION="26.08.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
GROUP_ID="com.nvidia.cuvs"

# Identify CUDA major version.
CUDA_VERSION_FROM_NVCC=$(nvcc --version | grep -oP 'release [0-9]+' | awk '{print $2}')
CUDA_MAJOR_VERSION=${CUDA_VERSION_FROM_NVCC:-12}

# Identify architecture.
ARCH=$(uname -m)

BUILD_PROFILE="$ARCH-cuda$CUDA_MAJOR_VERSION"

if [ -z "${CMAKE_PREFIX_PATH:=}" ]; then
  CMAKE_PREFIX_PATH="$(pwd)/../cpp/build"
  export CMAKE_PREFIX_PATH
fi

cmake -B ./internal/build -S ./internal
cmake --build ./internal/build

# Generate Panama FFM API bindings and update (if any of them changed)
./panama-bindings/generate-bindings.sh

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

# Build the java layer
if [ -z ${LD_LIBRARY_PATH+x} ]
then export LD_LIBRARY_PATH=$CMAKE_PREFIX_PATH
else export LD_LIBRARY_PATH=$CMAKE_PREFIX_PATH:${LD_LIBRARY_PATH}
fi

cd cuvs-java
mvn clean verify "${MAVEN_VERIFY_ARGS[@]}" -P "$BUILD_PROFILE" \
  && mvn install:install-file -Dfile=./target/cuvs-java-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-java -Dversion=$VERSION -Dpackaging=jar \
  && mvn install:install-file -Dfile=./target/cuvs-java-$VERSION-"$BUILD_PROFILE".jar -DgroupId=$GROUP_ID -DartifactId=cuvs-java -Dversion=$VERSION -Dclassifier="$BUILD_PROFILE" -Dpackaging=jar \
  && cp pom.xml ./target/
