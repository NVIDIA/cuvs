# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(test_cuvs_c)

project(test_cuvs_c LANGUAGES C CXX CUDA)

# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY ON)
include(../cmake/thirdparty/get_dlpack.cmake)
include(../cmake/thirdparty/get_cuvs.cmake)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")

add_executable(CAGRA_C_EXAMPLE src/cagra_c_example.c)
target_include_directories(CAGRA_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(
  CAGRA_C_EXAMPLE PRIVATE cuvs::c_api CUDA::cudart_static $<TARGET_NAME_IF_EXISTS:conda_env>
)

add_executable(L2_C_EXAMPLE src/L2_c_example.c)
target_include_directories(L2_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(
  L2_C_EXAMPLE PRIVATE cuvs::c_api CUDA::cudart_static $<TARGET_NAME_IF_EXISTS:conda_env>
)

add_executable(IVF_FLAT_C_EXAMPLE src/ivf_flat_c_example.c)
target_include_directories(IVF_FLAT_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(
  IVF_FLAT_C_EXAMPLE PRIVATE cuvs::c_api CUDA::cudart_static $<TARGET_NAME_IF_EXISTS:conda_env>
)

add_executable(IVF_PQ_C_EXAMPLE src/ivf_pq_c_example.c)
target_include_directories(IVF_PQ_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(
  IVF_PQ_C_EXAMPLE PRIVATE cuvs::c_api CUDA::cudart_static $<TARGET_NAME_IF_EXISTS:conda_env>
)

add_executable(BRUTEFORCE_C_EXAMPLE src/bruteforce_c_example.c)
target_include_directories(BRUTEFORCE_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(
  BRUTEFORCE_C_EXAMPLE PRIVATE cuvs::c_api CUDA::cudart_static $<TARGET_NAME_IF_EXISTS:conda_env>
)
