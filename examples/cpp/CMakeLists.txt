# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

rapids_cuda_init_architectures(test_cuvs)

project(test_cuvs LANGUAGES CXX CUDA)
find_package(Threads)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY OFF)
include(../cmake/thirdparty/get_cuvs.cmake)
include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco()

# -------------- compile tasks ----------------- #
add_executable(BRUTE_FORCE_EXAMPLE src/brute_force_bitmap.cu)
add_executable(BALANCED_KMEANS_EXAMPLE src/balanced_kmeans_example.cu)
add_executable(CAGRA_EXAMPLE src/cagra_example.cu)
add_executable(CAGRA_FILTER_UDF_EXAMPLE src/cagra_filter_udf_example.cu)
add_executable(CAGRA_BLOOM_FILTER_EXAMPLE src/cagra_bloom_filter_example.cu)
add_executable(CAGRA_HNSW_ACE_BUILD_EXAMPLE src/cagra_hnsw_ace_build.cu)
add_executable(CAGRA_HNSW_ACE_EXAMPLE src/cagra_hnsw_ace_example.cu)
add_executable(CAGRA_PERSISTENT_EXAMPLE src/cagra_persistent_example.cu)
add_executable(DYNAMIC_BATCHING_EXAMPLE src/dynamic_batching_example.cu)
add_executable(HNSW_ACE_EXAMPLE src/hnsw_ace_example.cu)
add_executable(HNSW_OPENAI_EXAMPLE src/hnsw_openai_example.cu)
add_executable(IVF_FLAT_EXAMPLE src/ivf_flat_example.cu)
add_executable(IVF_PQ_EXAMPLE src/ivf_pq_example.cu)
add_executable(VAMANA_EXAMPLE src/vamana_example.cu)
add_executable(SCANN_EXAMPLE src/scann_example.cu)

# `$<TARGET_NAME_IF_EXISTS:conda_env>` is a generator expression that ensures that targets are
# installed in a conda environment, if one exists
target_link_libraries(BRUTE_FORCE_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(BALANCED_KMEANS_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(
  CAGRA_FILTER_UDF_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>
)
target_link_libraries(
  CAGRA_BLOOM_FILTER_EXAMPLE PRIVATE cuvs::cuvs cuco::cuco $<TARGET_NAME_IF_EXISTS:conda_env>
)
target_link_libraries(
  CAGRA_HNSW_ACE_BUILD_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>
)
target_link_libraries(CAGRA_HNSW_ACE_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(
  CAGRA_PERSISTENT_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env> Threads::Threads
)
target_link_libraries(
  DYNAMIC_BATCHING_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env> Threads::Threads
)
target_link_libraries(HNSW_ACE_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(HNSW_OPENAI_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)

target_link_libraries(IVF_PQ_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(IVF_FLAT_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(VAMANA_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(SCANN_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
