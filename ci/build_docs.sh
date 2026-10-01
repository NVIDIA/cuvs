#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create docs conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs

# seeing failures on activating the environment here on unbound locals
# apply workaround from https://github.com/conda/conda/issues/8186#issuecomment-532874667
set +eu
conda activate docs
set -eu

rapids-print-env

rapids-logger "Validate Fern docs"

find_pr_number() {
  local ref
  for ref in "${RAPIDS_REF_NAME:-}" "${GITHUB_REF:-}" "${GITHUB_REF_NAME:-}"; do
    if [[ "${ref}" =~ (^|/)pull-request/([0-9]+)$ ]]; then
      echo "${BASH_REMATCH[2]}"
      return 0
    fi
    if [[ "${ref}" =~ ^refs/pull/([0-9]+)/ ]]; then
      echo "${BASH_REMATCH[1]}"
      return 0
    fi
    if [[ "${ref}" =~ ^([0-9]+)/merge$ ]]; then
      echo "${BASH_REMATCH[1]}"
      return 0
    fi
  done
}

FERN_DOCS_MODE="${FERN_DOCS_MODE:-check}"
FERN_DOCS_ARGS=()

if [[ "${FERN_DOCS_MODE}" == "preview" ]]; then
  FERN_PREVIEW_ID="${FERN_DOCS_PREVIEW_ID:-}"
  if [[ -z "${FERN_PREVIEW_ID}" ]]; then
    PR_NUMBER="$(find_pr_number || true)"
    if [[ -n "${PR_NUMBER}" ]]; then
      FERN_PREVIEW_ID="pr-${PR_NUMBER}"
    fi
  fi
  if [[ -n "${FERN_PREVIEW_ID}" ]]; then
    FERN_DOCS_ARGS+=(--id "${FERN_PREVIEW_ID}")
  fi
fi

fern/build_docs.sh "${FERN_DOCS_MODE}" "${FERN_DOCS_ARGS[@]}"
