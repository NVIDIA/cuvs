#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

ARGS="$*"
NUMARGS=$#

VERSION="26.04.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
GROUP_ID="com.nvidia.cuvs.lucene"

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg --build-cuvs-java; then
  CUVS_WORKDIR="cuvs-workdir"
  CUVS_GIT_REPO="https://github.com/rapidsai/cuvs.git"
  if [[ -d "$CUVS_WORKDIR" && -n "$(ls -A "$CUVS_WORKDIR")" ]]; then
    echo "Directory '$CUVS_WORKDIR' exists and is not empty."
    pushd $CUVS_WORKDIR
    git pull
  else
    BRANCH=$(cat "RAPIDS_BRANCH")
    echo "Directory '$CUVS_WORKDIR' does not exist or is empty. Cloning the cuvs's '$BRANCH' branch."
    # Correct branch selection is crucial to avoid version mismatch issues when testing.
    git clone --branch "$BRANCH" $CUVS_GIT_REPO $CUVS_WORKDIR
    pushd $CUVS_WORKDIR
  fi
  ./build.sh java
  popd
fi

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

mvn clean verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn jacoco:report \
  && mvn install:install-file -Dfile=./target/cuvs-lucene-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-lucene -Dversion=$VERSION -Dpackaging=jar \
  && cp pom.xml ./target/
