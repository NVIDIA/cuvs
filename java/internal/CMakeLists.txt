# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)
rapids_cpm_init()

project(
  cuvs-java
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX C
)

find_package(OpenMP)
find_package(Threads)

option(FIND_CUVS_CPP "Search for existing CUVS C++ installations before defaulting to local files"
       ON
)

if(FIND_CUVS_CPP)
  find_package(cuvs "${RAPIDS_VERSION}" REQUIRED COMPONENTS c_api)
endif()

if(NOT cuvs_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_C_LIBRARY ON)
  add_subdirectory(../../cpp cuvs-cpp EXCLUDE_FROM_ALL)
endif()

include(get_dlpack.cmake)

# We are going to copy all the cuvs_c and dlpack headers into a staging location
if(TARGET cuvs_c)
  get_target_property(cuvs_c_headers cuvs_c INCLUDE_DIRECTORIES)
else()
  get_target_property(cuvs_c_headers cuvs::c_api INTERFACE_INCLUDE_DIRECTORIES)
endif()

file(
  INSTALL "${DLPACK_INCLUDE_DIR}"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/bindings/"
  FILES_MATCHING
  PATTERN "*.h"
)

foreach(cuvs_c_dir IN LISTS cuvs_c_headers)
  if(EXISTS "${cuvs_c_dir}/cuvs/")
    file(
      INSTALL "${cuvs_c_dir}/cuvs"
      DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/bindings/include/"
      FILES_MATCHING
      PATTERN "*.h"
    )
  endif()
endforeach()
