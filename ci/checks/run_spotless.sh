#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pre-commit hook wrapper that runs 'spotless:apply' to format the Java sources of every Maven
# project under java/.
#
# Most cuvs contributors do not work on the Java client and do not have Maven installed. For them,
# running 'pre-commit run --all-files' matches every Java source file in the repo regardless of
# whether they touched any of it, so this skips gracefully when Maven is missing and there are no
# actual local changes to Java sources. In CI, and for anyone who has actually modified Java
# sources locally, Maven is expected to be available and its absence is treated as an error.

set -euo pipefail

# Keep these in sync with the spotless-fmt hook's 'files'/'exclude' entries in
# .pre-commit-config.yaml.
JAVA_SRC_PATTERN='^java/(cuvs-java|cuvs-lucene)/([^/]+/)?src/.*\.java$'
JAVA_SRC_EXCLUDE='.*/panama/.*'

java_sources_modified() {
  git status --porcelain --untracked-files=all -- java/cuvs-java java/cuvs-lucene |
    cut -c4- |
    grep -Ev "${JAVA_SRC_EXCLUDE}" |
    grep -Eq "${JAVA_SRC_PATTERN}"
}

if ! command -v mvn >/dev/null 2>&1; then
  if [ "${CI:-false}" = "true" ]; then
    echo "spotless-fmt: 'mvn' is required in CI but was not found on PATH." >&2
    exit 1
  fi
  if java_sources_modified; then
    echo "spotless-fmt: 'mvn' is required to format modified Java sources but was not found on PATH." >&2
    exit 1
  fi
  echo "spotless-fmt: 'mvn' was not found on PATH and no Java sources were modified, skipping Java formatting." >&2
  exit 0
fi

POMS=(
  java/cuvs-java/pom.xml
  java/cuvs-lucene/pom.xml
  java/cuvs-lucene/bench/pom.xml
  java/cuvs-lucene/examples/pom.xml
)

for pom in "${POMS[@]}"; do
  mvn --batch-mode --quiet -f "${pom}" spotless:apply
done
