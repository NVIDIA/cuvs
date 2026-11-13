# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)
project(cuvs_solr_patches LANGUAGES NONE)

if(NOT EXISTS "${SOLR_SOURCE_DIR}/build.gradle")
  message(FATAL_ERROR "SOLR_SOURCE_DIR must name the Solr checkout")
endif()

# Use the same RAPIDS CMake bootstrap as cuVS, without enabling C++ or CUDA.
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/rapids_config.cmake")
include("${rapids-cmake-dir}/cpm/package_override.cmake")
include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")

# Evaluate fixed_in against the selected checkout, not a hardcoded Solr version.
file(STRINGS "${SOLR_SOURCE_DIR}/build.gradle" solr_version_line REGEX "String baseVersion =")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" solr_version "${solr_version_line}")
if(NOT solr_version)
  message(FATAL_ERROR "Could not read Solr baseVersion from build.gradle")
endif()

# Solr is already checked out by Actions and is built by Gradle. Register the JSON solely to
# generate the RAPIDS patch command for this existing checkout.
set(CPM_solr_SOURCE "${SOLR_SOURCE_DIR}")
rapids_cpm_package_override("${CMAKE_CURRENT_LIST_DIR}/patches/solr.json")
rapids_cpm_generate_patch_command(solr "${solr_version}" patch_command build_patch_only)
if(patch_command)
  # Drop the FetchContent PATCH_COMMAND keyword before executing its command.
  list(REMOVE_AT patch_command 0)
  file(REMOVE "${CMAKE_BINARY_DIR}/rapids-cmake/patches/solr/log"
       "${CMAKE_BINARY_DIR}/rapids-cmake/patches/solr/err"
  )
  execute_process(
    COMMAND ${patch_command} WORKING_DIRECTORY "${SOLR_SOURCE_DIR}" COMMAND_ERROR_IS_FATAL ANY
  )
  # Required-patch errors are stored by the command and raised by this helper.
  set(solr_ADDED TRUE)
  rapids_cpm_display_patch_status(solr)
endif()
