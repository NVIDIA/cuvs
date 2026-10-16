#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Runs cuvs-lucene's test suite on JDK 21 against the classes from a prior amd64 build job,
# without recompiling. cuvs-lucene is compiled with --release 21 so that applications on JDK 21
# (Solr, for example) can load it, but cuvs-java needs JDK 22+ to reach the GPU. On JDK 21 the
# GPU tests skip themselves, and what runs checks that the codecs still load, read existing
# indexes and fall back to the CPU. None of that depends on the GPU or on the Lucene release, so
# this runs on a CPU node and tests only the newest Lucene module.

CUVS_JAVA_ARTIFACT="cuvs-java-cuda${RAPIDS_CUDA_VERSION}"
CUVS_LUCENE_ARTIFACT="cuvs-lucene-cuda${RAPIDS_CUDA_VERSION}"

if [ -e "/opt/conda/etc/profile.d/conda.sh" ]; then
  . /opt/conda/etc/profile.d/conda.sh
fi

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CUVS_JAVA_DIR=$(rapids-download-from-github "${CUVS_JAVA_ARTIFACT}")
CUVS_LUCENE_DIR=$(rapids-download-from-github "${CUVS_LUCENE_ARTIFACT}")

rapids-logger "Generate JDK 21 testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key java_jdk21 \
  --matrix "" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java_jdk21

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java_jdk21
set -u

rapids-print-env

# Maven, and the JVM surefire forks for the tests, run on the JDK of JAVA_HOME rather than the java
# on PATH, so check the one Maven reports.
MAVEN_JAVA_VERSION="$(cd java/cuvs-lucene && mvn --batch-mode -v | sed -n 's/^Java version: \([0-9]*\).*/\1/p')"
if [ "${MAVEN_JAVA_VERSION}" != "21" ]; then
  echo "Expected Maven to run on JDK 21, but it runs on JDK '${MAVEN_JAVA_VERSION}'" >&2
  exit 1
fi

rapids-logger "Install the amd64-built cuvs-java artifact into the local Maven repository"

CUVS_JAVA_POM="${CUVS_JAVA_DIR}/pom.xml"
if [ ! -f "${CUVS_JAVA_POM}" ]; then
  echo "Could not find pom.xml in the cuvs-java artifact at ${CUVS_JAVA_DIR}" >&2
  exit 1
fi

mapfile -t CUVS_JAVA_JARS < <(find "${CUVS_JAVA_DIR}" -maxdepth 1 -name 'cuvs-java-*.jar' \
  ! -name '*-sources.jar' ! -name '*-javadoc.jar' ! -name '*-tests.jar' ! -name '*-cuda*.jar')
if [ "${#CUVS_JAVA_JARS[@]}" -ne 1 ]; then
  echo "Expected exactly one cuvs-java jar in ${CUVS_JAVA_DIR}, found: ${CUVS_JAVA_JARS[*]:-none}" >&2
  exit 1
fi

# cd is needed to pick up pom.xml in order to avoid rate limit of main maven repo
pushd java/cuvs-lucene
mvn --batch-mode install:install-file -Dfile="${CUVS_JAVA_JARS[0]}" -DpomFile="${CUVS_JAVA_POM}"
popd

rapids-logger "Restore the amd64-built cuvs-lucene target/ directories so Maven can run the tests without recompiling"

rm -rf java/cuvs-lucene/*/target
cp -a "${CUVS_LUCENE_DIR}/." java/cuvs-lucene/

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run cuvs-lucene tests on JDK 21 against the amd64-built classes"

# -Dskip.compile disables all compilation, see ci/test_lucene.sh.
pushd java/cuvs-lucene
NEWEST_MODULE="$(find . -maxdepth 1 -type d -name 'lucene-*' -printf '%f\n' | sort -V | tail -1)"
rapids-logger "Testing ${NEWEST_MODULE}"
mvn --batch-mode test -Dskip.compile=true -pl "${NEWEST_MODULE}"
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
