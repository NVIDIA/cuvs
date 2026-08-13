# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

include(rapids-cuda)
rapids_cuda_init_architectures(libcuvs-python)

project(
  libcuvs-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

option(USE_NCCL_RUNTIME_WHEEL "Use the NCCL wheel at runtime instead of the system library" OFF)

# Check if cuVS is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuvs package.
find_package(cuvs "${RAPIDS_VERSION}")

if(cuvs_FOUND)
  return()
endif()

unset(cuvs_FOUND)

# --- CUDA --- #
set(CUDA_STATIC_MATH_LIBRARIES OFF)

# --- RAFT ---#
set(CUVS_STATIC_RAPIDS_LIBRARIES OFF)

# --- cuVS ---#
set(BUILD_TESTS OFF)
set(BUILD_C_LIBRARY ON)
set(CUVS_COMPILE_DYNAMIC_ONLY ON)

add_subdirectory(../../cpp cuvs-cpp)

# assumes libcuvs.so is installed 2 levels deep, e.g. site-packages/libcuvs/lib64/libcuvs.so
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  # starting with CTK 13 wheels, all libraries are grouped in this 'nvidia/cu13/lib' directory
  set(rpaths "$ORIGIN/../../nvidia/cu13/lib")
else()
  set(rpaths
      "$ORIGIN/../../nvidia/cublas/lib"
      "$ORIGIN/../../nvidia/curand/lib"
      "$ORIGIN/../../nvidia/cusolver/lib"
      "$ORIGIN/../../nvidia/cusparse/lib"
      "$ORIGIN/../../nvidia/nvjitlink/lib"
  )
endif()

if(USE_NCCL_RUNTIME_WHEEL)
  list(APPEND rpaths "$ORIGIN/../../nvidia/nccl/lib")
endif()

set_property(
  TARGET cuvs
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
set_property(
  TARGET cuvs_c
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
