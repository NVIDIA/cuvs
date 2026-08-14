#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pass CI-specific environment variables to the standalone tarball build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUVS_TARBALL_DOCKER_ENV_VARS="AWS_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN RAPIDS_AUX_SECRET_1 RAPIDS_ARTIFACTS_DIR RAPIDS_BUILD_TYPE RAPIDS_DATETIME_STRING RAPIDS_REPOSITORY RAPIDS_SHA RAPIDS_REF_NAME RAPIDS_NIGHTLY_DATE"

exec "${REPO_ROOT}/build.sh" tarball "$@"
