#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Project dependency sources belong in rapids-cpm-versions.json. That allows
# RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE to replace them without source patches.
set -euo pipefail

# git_tag_alias documents the named Git ref that was resolved to git_tag. rapids-cmake ignores it.
catalog=cpp/cmake/thirdparty/rapids-cpm-versions.json

if (( $# == 0 )); then
  set -- cpp/cmake/thirdparty/get_*.cmake examples/cmake/thirdparty/get_*.cmake
fi

for cmake_file in "$@"; do
  if grep -nE '^[[:space:]]*GIT_(REPOSITORY|TAG|SHALLOW)[[:space:]]' "${cmake_file}"; then
    echo "ERROR: ${cmake_file} declares a direct Git source." >&2
    echo "Move its source metadata to cpp/cmake/thirdparty/rapids-cpm-versions.json and use cuvs_cpm_project_package_info()." >&2
    exit 1
  fi
done

invalid_git_tags="$(grep -nE '"git_tag"' "${catalog}" | grep -vE '"git_tag"[[:space:]]*:[[:space:]]*"[a-f0-9]{40}"[[:space:]]*,?[[:space:]]*$' || true)"
if [[ -n "${invalid_git_tags}" ]]; then
  printf '%s\n' "${invalid_git_tags}" >&2
  echo "ERROR: ${catalog} must use 40-character commit hashes for git_tag." >&2
  exit 1
fi
