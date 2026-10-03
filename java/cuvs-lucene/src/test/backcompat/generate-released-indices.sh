#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Writes the back-compat indexes of a released cuvs-lucene version into src/test/resources/backcompat,
# using the jars published to Maven Central. Released cuvs-lucene versions up to 26.08 were a single
# com.nvidia.cuvs.lucene:cuvs-lucene artifact built for one Lucene release.
#
# Usage: generate-released-indices.sh <cuvs-lucene version>
# Needs a GPU, a JDK 22+ and the native libcuvs libraries on LD_LIBRARY_PATH.

set -euo pipefail

VERSION="${1:?Usage: $0 <cuvs-lucene version>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HERE}/../resources/backcompat"
CENTRAL="https://repo.maven.apache.org/maven2"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

fetch() { # group/path artifact version
  curl -sfL -o "${WORK}/$2-$3.jar" "${CENTRAL}/$1/$2/$3/$2-$3.jar"
  echo "${WORK}/$2-$3.jar"
}

POM="$(curl -sfL "${CENTRAL}/com/nvidia/cuvs/lucene/cuvs-lucene/${VERSION}/cuvs-lucene-${VERSION}.pom")"
dependency_version() { # artifactId
  echo "${POM}" | grep -A2 "<artifactId>$1</artifactId>" | grep -o '<version>[^<]*' | head -1 | sed 's/<version>//'
}
LUCENE_VERSION="$(dependency_version lucene-core)"
CUVS_JAVA_VERSION="$(dependency_version cuvs-java)"

CP="$(fetch com/nvidia/cuvs/lucene cuvs-lucene "${VERSION}")"
CP="${CP}:$(fetch com/nvidia/cuvs cuvs-java "${CUVS_JAVA_VERSION}")"
CP="${CP}:$(fetch org/apache/lucene lucene-core "${LUCENE_VERSION}")"
CP="${CP}:$(fetch org/apache/lucene lucene-backward-codecs "${LUCENE_VERSION}")"

javac -cp "${CP}" -d "${WORK}" "${HERE}/GenerateReleasedIndices.java"
java --enable-native-access=ALL-UNNAMED -cp "${CP}:${WORK}" GenerateReleasedIndices "${OUT}" "${VERSION}"
