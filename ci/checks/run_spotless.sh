#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pre-commit hook wrapper that runs 'spotless:apply' to format the Java sources of every Maven
# project under java/.
#
# Most cuvs contributors do not work on the Java client and do not have Maven installed. For them
# (running outside CI without Maven) this skips gracefully, so that 'pre-commit run --all-files'
# does not require every contributor to install Maven. In CI, Maven is expected to be available and
# its absence is treated as an error.

set -euo pipefail

if ! command -v mvn >/dev/null 2>&1; then
  if [ "${CI:-false}" = "true" ]; then
    echo "spotless-fmt: 'mvn' is required in CI but was not found on PATH." >&2
    exit 1
  fi
  echo "spotless-fmt: skipping Java formatting ('mvn' not installed and not running in CI)." >&2
  exit 0
fi

POMS=(
  java/cuvs-java/pom.xml
  java/cuvs-lucene/pom.xml
)

for pom in "${POMS[@]}"; do
  mvn --batch-mode --quiet -f "${pom}" spotless:apply
done
