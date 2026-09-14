# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(cpm_project_package_info_test LANGUAGES NONE)

set(CUVS_REPOSITORY_DIR
    ""
    CACHE PATH "Path to the cuVS repository root"
)
if(NOT CUVS_REPOSITORY_DIR)
  cmake_path(SET CUVS_REPOSITORY_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../../../..")
endif()

set(EXPECTED_DLPACK_VERSION
    "0.8"
    CACHE STRING "Expected resolved DLPack version"
)

include("${CUVS_REPOSITORY_DIR}/cmake/rapids_config.cmake")
include("${rapids-cmake-dir}/cpm/init.cmake")
rapids_cpm_init()
# cmake-lint: disable=C0111
function(assert_project_package_version package_name expected_version)
  include("${CUVS_REPOSITORY_DIR}/cpp/cmake/thirdparty/rapids_cpm_project_package_info.cmake")
  cuvs_cpm_project_package_info(${package_name} VERSION_VAR resolved_version)
  if(NOT resolved_version STREQUAL expected_version)
    message(
      FATAL_ERROR "Expected ${package_name} version ${expected_version}, got ${resolved_version}"
    )
  endif()
endfunction()

# Resolve packages from separate function scopes to verify that the catalog path remains available
# after the helper's global include guard suppresses subsequent includes.
assert_project_package_version(dlpack "${EXPECTED_DLPACK_VERSION}")
assert_project_package_version(kvikio "${RAPIDS_VERSION_MAJOR_MINOR}.00")
