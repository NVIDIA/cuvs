# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)
include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

option(BUILD_CPU_ONLY "Build CPU only components. Applies to CUVS benchmarks currently" OFF)

# workaround for rapids_cuda_init_architectures not working for arch detection with
# enable_language(CUDA)
set(lang_list "CXX" "C")

if(NOT BUILD_CPU_ONLY)
  include(rapids-cuda)
  rapids_cuda_init_architectures(CUVS)
  list(APPEND lang_list "CUDA")
endif()

project(
  CUVS
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES ${lang_list}
)
set(CMAKE_INSTALL_MESSAGE LAZY)

# Disable C++20 module scanning (not using modules, just C++20 features)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Write the version header
rapids_cmake_write_version_file(include/cuvs/version_config.h)

# ##################################################################################################
# * build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##################################################################################################
# * User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuvs shared libraries" ON)
option(BUILD_TESTS "Build cuvs unit-tests" ON)
option(BUILD_C_LIBRARY "Build cuVS C API library" ON)
option(BUILD_CUVS_BENCH "Build cuVS ann benchmarks" OFF)
option(BUILD_CAGRA_HNSWLIB "Build CAGRA+hnswlib interface" ON)
option(BUILD_MG_ALGOS "Build with multi-GPU support" ON)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF
)
option(CUDA_STATIC_MATH_LIBRARIES "Statically link the CUDA math libraries" OFF)
option(CUVS_STATIC_RAPIDS_LIBRARIES "Build and statically link RAPIDS libraries" OFF)
option(CUDA_LOG_COMPILE_TIME "Write a log of compilation times to nvcc_compile_log.csv" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable deprecaction warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(CUVS_COMPILE_DYNAMIC_ONLY "Only build the shared library and skip the static library." OFF)
option(CUVS_NVTX "Enable nvtx markers" OFF)
option(CUVS_RAFT_CLONE_ON_PIN "Explicitly clone RAFT branch when pinned to non-feature branch" ON)

if(BUILD_CPU_ONLY)
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTS OFF)
  set(BUILD_C_LIBRARY OFF)
  set(BUILD_CAGRA_HNSWLIB OFF)
  set(INSTALL_CPP_HEADERS OFF)
elseif(NOT BUILD_SHARED_LIBS)
  set(BUILD_TESTS OFF)
  set(BUILD_C_LIBRARY OFF)
  set(BUILD_CAGRA_HNSWLIB OFF)
endif()

if(NOT BUILD_SHARED_LIBS AND CUVS_COMPILE_DYNAMIC_ONLY)
  message(
    VERBOSE
    "cuVS: Ignoring `BUILD_SHARED_LIBS` and forcing shared library due to `CUVS_COMPILE_DYNAMIC_ONLY`"
  )
endif()

set(cuvs_compile_mode "both")
set(_cuvs_lib_targets cuvs cuvs_static)
if(CUVS_COMPILE_DYNAMIC_ONLY)
  set(cuvs_compile_mode "shared_only")
  set(_cuvs_lib_targets cuvs)
elseif(NOT BUILD_SHARED_LIBS)
  set(cuvs_compile_mode "static_only")
  set(_cuvs_lib_targets cuvs_static)
endif()

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value across all builds so that cache hits occur
set(THREADS_PREFER_PTHREAD_FLAG ON)

include(CMakeDependentOption)
# cmake_dependent_option( CUVS_USE_FAISS_STATIC "Build and statically link the FAISS library for
# nearest neighbors search on GPU" ON CUVS_COMPILE_LIBRARY OFF )

message(VERBOSE "cuVS: Build cuVS unit-tests: ${BUILD_TESTS}")
message(VERBOSE "cuVS: Build CPU only components: ${BUILD_CPU_ONLY}")
message(VERBOSE "cuVS: Build ANN benchmarks: ${BUILD_CUVS_BENCH}")
message(VERBOSE "cuVS: Build only the shared library: ${CUVS_COMPILE_DYNAMIC_ONLY}")
message(VERBOSE "cuVS: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "cuVS: Disable deprecation warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "cuVS: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "cuVS: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "cuVS: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "cuVS: Enable nvtx markers: ${CUVS_NVTX}")
message(VERBOSE "cuVS: Statically link the CUDA math libraries: ${CUDA_STATIC_MATH_LIBRARIES}")
message(VERBOSE "cuVS: Build and statically link RAPIDS libraries: ${CUVS_STATIC_RAPIDS_LIBRARIES}")

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "cuVS: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

# ##################################################################################################
# * Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(
      STATUS "cuVS: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}"
    )
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

# ##################################################################################################
# * compiler options ----------------------------------------------------------

set(_ctk_static_suffix "")
if(CUDA_STATIC_MATH_LIBRARIES)
  set(_ctk_static_suffix "_static")
endif()

if(NOT BUILD_CPU_ONLY)
  # CUDA runtime
  rapids_cuda_init_runtime(USE_STATIC ON)
  # * find CUDAToolkit package
  # * determine GPU architectures
  # * enable the CMake CUDA language
  # * set other CUDA compilation flags
  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET cuvs-exports
    INSTALL_EXPORT_SET cuvs-exports
  )
else()
  add_compile_definitions(BUILD_CPU_ONLY)
endif()

if(NOT DISABLE_OPENMP)
  rapids_find_package(
    OpenMP REQUIRED
    BUILD_EXPORT_SET cuvs-exports
    INSTALL_EXPORT_SET cuvs-exports
  )
  if(OPENMP_FOUND)
    message(VERBOSE "cuVS: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif()
endif()

include(cmake/modules/ConfigureCUDA.cmake)

# ##################################################################################################
# * Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

if(NOT BUILD_CPU_ONLY)
  # We must find CCCL ourselves before raft so that we get the right version.
  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  rapids_cpm_cccl()
  include(cmake/thirdparty/get_raft.cmake)
  include(cmake/thirdparty/get_kvikio.cmake)
  include(cmake/thirdparty/get_cutlass.cmake)
  include(${rapids-cmake-dir}/cpm/cuco.cmake)
  rapids_cpm_cuco()
endif()

if(BUILD_TESTS OR BUILD_C_TESTS)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(BUILD_STATIC)
endif()

if(BUILD_CUVS_BENCH)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench(BUILD_STATIC)
endif()

if(BUILD_CAGRA_HNSWLIB)
  include(cmake/thirdparty/get_hnswlib.cmake)
endif()

include(cmake/modules/generate_inst_matrix.cmake)
include(cmake/modules/generate_string_matrix.cmake)

# ##################################################################################################
# * cuvs ---------------------------------------------------------------------
if(NOT BUILD_CPU_ONLY)

  add_library(cuvs_cpp_headers INTERFACE)
  add_library(cuvs::cuvs_cpp_headers ALIAS cuvs_cpp_headers)
  target_include_directories(
    cuvs_cpp_headers
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
              "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
              "$<INSTALL_INTERFACE:include>"
  )
  target_link_libraries(cuvs_cpp_headers INTERFACE raft::raft rmm::rmm)

  generate_inst_matrix(
    cagra_search_inst_files
    MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/cagra_search_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/cagra_search_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/cagra_search_inst_data_@data_abbrev@.cu"
  )

  generate_inst_matrix(
    cagra_compute_distance_standard_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance_standard_matrix.json"
    INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance_standard_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/detail/cagra/compute_distance_standard_inst_data_@data_abbrev@_index_@index_abbrev@_distance_@distance_abbrev@_metric_@metric@_team_@team_size@_dim_@dim@.cu"
  )
  generate_string_matrix(
    cagra_compute_distance_standard_selector_template_params
    ITEM_FORMAT
    "\nstandard_descriptor_spec<DistanceType::@metric@, @team_size@, @dim@, @data_type@, @index_type@, @distance_type@>"
    GLUE
    ","
    MATRIX_JSON_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance_standard_matrix.json"
  )
  generate_string_matrix(
    cagra_compute_distance_standard_template_inst
    ITEM_FORMAT
    "extern template struct standard_descriptor_spec<DistanceType::@metric@, @team_size@, @dim@, @data_type@, @index_type@, @distance_type@>@semicolon@"
    GLUE
    "\n"
    MATRIX_JSON_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance_standard_matrix.json"
  )

  generate_inst_matrix(
    cagra_compute_distance_vpq_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance_vpq_matrix.json"
    INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance_vpq_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/detail/cagra/compute_distance_vpq_inst_data_@data_abbrev@_index_@index_abbrev@_distance_@distance_abbrev@_codebook_@codebook_abbrev@_metric_@metric@_team_@team_size@_dim_@dim@_pq_bits_@pq_bits@_pq_len_@pq_len@_smem_@smem_abbrev@.cu"
  )
  generate_string_matrix(
    cagra_compute_distance_vpq_selector_template_params
    ITEM_FORMAT
    "\nvpq_descriptor_spec<DistanceType::@metric@, @team_size@, @dim@, @pq_bits@, @pq_len@, @codebook_type@, @data_type@, @index_type@, @distance_type@, @smem_dtype@>"
    GLUE
    ","
    MATRIX_JSON_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance_vpq_matrix.json"
  )
  generate_string_matrix(
    cagra_compute_distance_vpq_template_inst
    ITEM_FORMAT
    "extern template struct vpq_descriptor_spec<DistanceType::@metric@, @team_size@, @dim@, @pq_bits@, @pq_len@, @codebook_type@, @data_type@, @index_type@, @distance_type@, @smem_dtype@>@semicolon@"
    GLUE
    "\n"
    MATRIX_JSON_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance_vpq_matrix.json"
  )

  generate_inst_matrix(
    cagra_search_multi_cta_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/search_multi_cta_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/search_multi_cta_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/detail/cagra/search_multi_cta_inst_data_@data_abbrev@.cu"
  )
  generate_inst_matrix(
    cagra_search_single_cta_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/search_single_cta_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/search_single_cta_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/detail/cagra/search_single_cta_inst_data_@data_abbrev@.cu"
  )

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance-ext.cuh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/detail/cagra/compute_distance-ext.cuh" @ONLY
  )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance.cu.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/detail/cagra/compute_distance.cu" @ONLY
  )

  set(cuvs_cagra_search_cuda_inst_files
      ${cagra_search_inst_files}
      ${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/detail/cagra/compute_distance.cu
      ${cagra_compute_distance_standard_inst_files}
      ${cagra_compute_distance_vpq_inst_files}
      ${cagra_search_multi_cta_inst_files}
      ${cagra_search_single_cta_inst_files}
  )

  if(BUILD_MG_ALGOS)
    generate_inst_matrix(
      CUVS_MG_ALGOS
      MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/mg/mg_flat_matrix.json"
      INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/mg/mg_flat_inst.cu.in"
      OUTPUT_FILE_FORMAT
        "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/mg/mg_flat_inst_data_@data_abbrev@.cu"
    )
    generate_inst_matrix(
      CUVS_MG_ALGOS
      MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/mg/mg_pq_matrix.json"
      INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/mg/mg_pq_inst.cu.in"
      OUTPUT_FILE_FORMAT
        "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/mg/mg_pq_inst_data_@data_abbrev@.cu"
    )
    generate_inst_matrix(
      CUVS_MG_ALGOS
      MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/mg/mg_cagra_matrix.json"
      INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/mg/mg_cagra_inst.cu.in"
      OUTPUT_FILE_FORMAT
        "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/mg/mg_cagra_inst_data_@data_abbrev@.cu"
    )
  endif()

  set(JIT_LTO_TARGET_ARCHITECTURE "70-real")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(JIT_LTO_TARGET_ARCHITECTURE "75-real")
  endif()

  # Generate interleaved scan kernel files at build time
  include(cmake/modules/generate_jit_lto_kernels.cmake)

  add_library(jit_lto_kernel_usage_requirements INTERFACE)
  target_include_directories(
    jit_lto_kernel_usage_requirements
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
              "${CMAKE_CURRENT_SOURCE_DIR}/../c/include"
  )
  target_compile_options(
    jit_lto_kernel_usage_requirements INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
                                                "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
  )
  target_compile_features(jit_lto_kernel_usage_requirements INTERFACE cuda_std_20)
  target_link_libraries(jit_lto_kernel_usage_requirements INTERFACE rmm::rmm raft::raft CCCL::CCCL)

  block(PROPAGATE jit_lto_files)
  set(jit_lto_files)
  set(CMAKE_CUDA_ARCHITECTURES ${JIT_LTO_TARGET_ARCHITECTURE})
  set(neighbors_ns "cuvs::neighbors::detail")
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "sample_filter_@filter_name@_bitset_@bitset_abbrev@_index_@index_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/jit_lto_kernels/filter_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/jit_lto_kernels/filter_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${neighbors_ns}::fragment_tag_sample_filter<${neighbors_ns}::tag_bitset_@bitset_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${neighbors_ns}::tag_filter_@filter_name@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/common/filter"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  set(ivf_flat_ns "cuvs::neighbors::ivf_flat::detail")
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "ivf_flat_interleaved_scan_capacity_@capacity@_@ascending_descending@_data_@data_abbrev@_acc_@acc_abbrev@_index_@index_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/interleaved_scan_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/interleaved_scan_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_flat_ns}::fragment_tag_interleaved_scan<${neighbors_ns}::tag_@data_abbrev@, ${ivf_flat_ns}::tag_acc_@acc_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, @capacity@, @ascending_value@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_flat/interleaved_scan_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_flat/interleaved_scan"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "ivf_flat_load_and_compute_dist_data_@data_abbrev@_acc_@acc_abbrev@_@compute_norm_name@_veclen_@veclen@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/load_and_compute_dist_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/load_and_compute_dist_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_flat_ns}::fragment_tag_load_and_compute_dist<${neighbors_ns}::tag_@data_abbrev@, ${ivf_flat_ns}::tag_acc_@acc_abbrev@, @compute_norm_value@, @veclen@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_flat/interleaved_scan_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_flat/load_and_compute_dist"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_flat_metric_@metric_name@_veclen_@veclen@_data_@data_abbrev@_acc_@acc_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/metric_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/metric_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_flat_ns}::fragment_tag_metric<${neighbors_ns}::tag_@data_abbrev@, ${ivf_flat_ns}::tag_acc_@acc_abbrev@, ${ivf_flat_ns}::tag_metric_@metric_name@, @veclen@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_flat/interleaved_scan_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_flat/metric"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_flat_filter_@filter_name@_index_@index_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/filter_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/filter_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_flat_ns}::fragment_tag_filter<${neighbors_ns}::tag_index_@index_abbrev@, ${neighbors_ns}::tag_filter_@filter_name@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_flat/interleaved_scan_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_flat/filter"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_flat_post_process_@post_process_name@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/post_process_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/post_process_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_flat_ns}::fragment_tag_post_lambda<${ivf_flat_ns}::tag_post_process_@post_process_name@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_flat/interleaved_scan_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_flat/post_process"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  set(ivf_sq_ns "cuvs::neighbors::ivf_sq::detail")
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_sq_scan_capacity_@capacity@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/scan_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/scan_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_sq_ns}::fragment_tag_ivf_sq_scan<@capacity@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_sq/scan_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_sq/scan"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_sq_setup_invariant_smem_metric_@metric_name@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/setup_invariant_smem_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/setup_invariant_smem_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_sq_ns}::fragment_tag_setup_invariant_smem<${ivf_sq_ns}::tag_metric_@metric_name@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_sq/scan_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_sq/setup_invariant_smem"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_sq_setup_per_probe_smem_metric_@metric_name@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/setup_per_probe_smem_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/setup_per_probe_smem_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_sq_ns}::fragment_tag_setup_per_probe_smem<${ivf_sq_ns}::tag_metric_@metric_name@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_sq/scan_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_sq/setup_per_probe_smem"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_sq_accumulate_distance_metric_@metric_name@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/accumulate_distance_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/accumulate_distance_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_sq_ns}::fragment_tag_accumulate_distance<${ivf_sq_ns}::tag_metric_@metric_name@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_sq/scan_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_sq/accumulate_distance"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_sq_finalize_distance_metric_@metric_name@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/finalize_distance_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/finalize_distance_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_sq_ns}::fragment_tag_finalize_distance<${ivf_sq_ns}::tag_metric_@metric_name@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_sq/scan_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_sq/finalize_distance"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_sq_filter_@filter_name@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/filter_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/filter_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_sq_ns}::fragment_tag_ivf_sq_filter<${neighbors_ns}::tag_filter_@filter_name@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_sq/scan_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_sq/filter"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  set(ivf_pq_ns "cuvs::neighbors::ivf_pq::detail")
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_pq_compute_similarity_out_@out_abbrev@_lut_@lut_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/compute_similarity_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/compute_similarity_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_pq_ns}::fragment_tag_compute_similarity<${ivf_pq_ns}::tag_out_@out_abbrev@, ${ivf_pq_ns}::tag_lut_@lut_abbrev@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/compute_similarity"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_pq_prepare_lut_lut_@lut_abbrev@_@enable_smem_lut_str@_@pq_bits@pq"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/prepare_lut_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/prepare_lut_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_pq_ns}::fragment_tag_prepare_lut<${ivf_pq_ns}::tag_lut_@lut_abbrev@, @enable_smem_lut@, @pq_bits@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/prepare_lut"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_pq_store_calculated_distances_out_@out_abbrev@_@k_manage_local_topk_str@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/store_calculated_distances_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/store_calculated_distances_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_pq_ns}::fragment_tag_store_calculated_distances<${ivf_pq_ns}::tag_out_@out_abbrev@, @k_manage_local_topk@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/store_calculated_distances"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_pq_precompute_base_diff_metric_@metric@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/precompute_base_diff_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/precompute_base_diff_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_pq_ns}::fragment_tag_precompute_base_diff<${ivf_pq_ns}::tag_metric_@metric@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/precompute_base_diff"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "ivf_pq_create_lut_lut_@lut_abbrev@_@precomp_base_diff_str@_@pq_bits@pq_metric_@metric@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/create_lut_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/create_lut_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_pq_ns}::fragment_tag_create_lut<${ivf_pq_ns}::tag_lut_@lut_abbrev@, ${ivf_pq_ns}::tag_metric_@metric@, @precomp_base_diff@, @pq_bits@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/create_lut"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_pq_compute_distances_out_@out_abbrev@_lut_@lut_abbrev@_capacity_@capacity@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/compute_distances_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/compute_distances_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_pq_ns}::fragment_tag_compute_distances<${ivf_pq_ns}::tag_out_@out_abbrev@, ${ivf_pq_ns}::tag_lut_@lut_abbrev@, @capacity@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/compute_distances"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_pq_get_early_stop_limit_out_@out_abbrev@_metric_@metric@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/get_early_stop_limit_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/get_early_stop_limit_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_pq_ns}::fragment_tag_get_early_stop_limit<${ivf_pq_ns}::tag_out_@out_abbrev@, ${ivf_pq_ns}::tag_metric_@metric@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/get_early_stop_limit"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_pq_sample_filter_@filter_name@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/sample_filter_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/sample_filter_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_pq_ns}::fragment_tag_sample_filter<${neighbors_ns}::tag_filter_@filter_name@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/sample_filter"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_pq_get_line_width_@pq_bits@pq"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/get_line_width_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/get_line_width_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_pq_ns}::fragment_tag_get_line_width<@pq_bits@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/get_line_width"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_pq_compute_score_out_@out_abbrev@_lut_@lut_abbrev@_@pq_bits@pq"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/compute_score_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/compute_score_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_pq_ns}::fragment_tag_compute_score<${ivf_pq_ns}::tag_out_@out_abbrev@, ${ivf_pq_ns}::tag_lut_@lut_abbrev@, @pq_bits@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/compute_score"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_pq_increment_score_out_@out_abbrev@_@increment_str@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/increment_score_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/increment_score_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_pq_ns}::fragment_tag_increment_score<${ivf_pq_ns}::tag_out_@out_abbrev@, @increment@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_pq/increment_score"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  set(ivf_rabitq_ns "cuvs::neighbors::ivf_rabitq::detail")
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_compute_inner_products_with_lut"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_lut_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_lut_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_rabitq_ns}::fragment_tag_compute_inner_products_with_lut"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/compute_inner_products_with_lut"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_lut_emit_distances_@with_ex_descriptor@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/lut_emit_distances_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/lut_emit_distances_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_rabitq_ns}::fragment_tag_lut_emit_distances<@with_ex_value@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/lut_emit_distances"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_compute_inner_products_with_lut_block_sort"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_lut_block_sort_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_lut_block_sort_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_rabitq_ns}::fragment_tag_compute_inner_products_with_lut_block_sort"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/compute_inner_products_with_lut_block_sort"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_lut_block_sort_emit_topk_@with_ex_descriptor@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/lut_block_sort_emit_topk_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/lut_block_sort_emit_topk_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_rabitq_ns}::fragment_tag_lut_block_sort_emit_topk<@with_ex_value@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/lut_block_sort_emit_topk"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_extract_code_@ex_bits@bit"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/extract_code_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/extract_code_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_rabitq_ns}::fragment_tag_extract_code<@ex_bits@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/extract_code"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_compute_lut_ip_for_vec_@lut_dtype_descriptor@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_lut_ip_for_vec_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_lut_ip_for_vec_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_rabitq_ns}::fragment_tag_compute_lut_ip_for_vec<${ivf_rabitq_ns}::tag_lut_dtype_@lut_dtype_descriptor@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/compute_lut_ip_for_vec"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_compute_bitwise_quantized_ip_for_vec_@num_bits@bit"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_bitwise_quantized_ip_for_vec_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_bitwise_quantized_ip_for_vec_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_rabitq_ns}::fragment_tag_compute_bitwise_quantized_ip_for_vec<@num_bits@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/compute_bitwise_quantized_ip_for_vec"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_compute_inner_products_with_lut16_opt"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_lut16_opt_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_lut16_opt_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_rabitq_ns}::fragment_tag_compute_inner_products_with_lut16_opt"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/compute_inner_products_with_lut16_opt"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_lut16_opt_emit_distances_@with_ex_descriptor@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/lut16_opt_emit_distances_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/lut16_opt_emit_distances_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_rabitq_ns}::fragment_tag_lut16_opt_emit_distances<@with_ex_value@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/lut16_opt_emit_distances"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_compute_inner_products_with_lut16_opt_block_sort_@with_ex_descriptor@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_lut16_opt_block_sort_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_lut16_opt_block_sort_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_rabitq_ns}::fragment_tag_compute_inner_products_with_lut16_opt_block_sort<@with_ex_value@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/compute_inner_products_with_lut16_opt_block_sort"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_compute_inner_products_with_bitwise"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_bitwise_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_bitwise_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_rabitq_ns}::fragment_tag_compute_inner_products_with_bitwise"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/compute_inner_products_with_bitwise"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_bitwise_emit_distances_@with_ex_descriptor@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/bitwise_emit_distances_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/bitwise_emit_distances_kernel.cu.in"
    FRAGMENT_TAG_FORMAT "${ivf_rabitq_ns}::fragment_tag_bitwise_emit_distances<@with_ex_value@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/bitwise_emit_distances"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_compute_inner_products_with_bitwise_block_sort"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_bitwise_block_sort_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/compute_inner_products_with_bitwise_block_sort_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_rabitq_ns}::fragment_tag_compute_inner_products_with_bitwise_block_sort"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/compute_inner_products_with_bitwise_block_sort"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "ivf_rabitq_bitwise_block_sort_emit_topk_@with_ex_descriptor@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/bitwise_block_sort_emit_topk_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/bitwise_block_sort_emit_topk_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${ivf_rabitq_ns}::fragment_tag_bitwise_block_sort_emit_topk<@with_ex_value@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/ivf_rabitq/bitwise_block_sort_emit_topk"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  set(cagra_ns "cuvs::neighbors::cagra::detail")
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "cagra_setup_workspace@pq_prefix@_team_size_@team_size@_dataset_block_dim_@dataset_block_dim@_@pq_bits@pq_@pq_len@subd_data_@data_abbrev@_query_@query_abbrev@_smem_@smem_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/setup_workspace_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/setup_workspace_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_setup_workspace<${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${cagra_ns}::tag_dist_@distance_abbrev@, ${neighbors_ns}::tag_@query_abbrev@, ${cagra_ns}::tag_codebook_@codebook_abbrev@, @team_size@, @dataset_block_dim@, @pq_bits@, @pq_len@, ${cagra_ns}::tag_smem_@smem_abbrev@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/setup_workspace"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
    KERNEL_EXTRA_COMPILE_OPTIONS --maxrregcount=64
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "cagra_compute_distance@pq_prefix@_team_size_@team_size@_dataset_block_dim_@dataset_block_dim@_@pq_bits@pq_@pq_len@subd_data_@data_abbrev@_query_@query_abbrev@_smem_@smem_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_compute_distance<${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${cagra_ns}::tag_dist_@distance_abbrev@, ${neighbors_ns}::tag_@query_abbrev@, ${cagra_ns}::tag_codebook_@codebook_abbrev@, @team_size@, @dataset_block_dim@, @pq_bits@, @pq_len@, ${cagra_ns}::tag_smem_@smem_abbrev@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/compute_distance"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
    KERNEL_EXTRA_COMPILE_OPTIONS --maxrregcount=64
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "cagra_dist_op_@metric_tag@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/dist_op_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/dist_op_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_dist_op<${neighbors_ns}::tag_@query_abbrev@, ${cagra_ns}::tag_dist_@distance_abbrev@, ${cagra_ns}::@jit_metric_tag@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/dist_op"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "cagra_apply_normalization_standard_@norm_kind@_team_size_@team_size@_dataset_block_dim_@dataset_block_dim@_data_@data_abbrev@_query_@query_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/apply_normalization_standard_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/apply_normalization_standard_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_apply_normalization_standard<${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${cagra_ns}::tag_dist_@distance_abbrev@, ${neighbors_ns}::tag_@query_abbrev@, @team_size@, @dataset_block_dim@, ${cagra_ns}::tag_norm_@norm_kind@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/apply_normalization_standard"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "cagra_search_single_cta_@topk_by_bitonic_sort_str@_@bitonic_sort_and_merge_multi_warps_str@_data_@data_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_search_single_cta<${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_index_@source_index_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${cagra_ns}::tag_dist_@distance_abbrev@, @topk_by_bitonic_sort@, @bitonic_sort_and_merge_multi_warps@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/search_single_cta"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "cagra_search_single_cta_p_@topk_by_bitonic_sort_str@_@bitonic_sort_and_merge_multi_warps_str@_data_@data_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_p_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_p_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_search_single_cta_p<${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_index_@source_index_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${cagra_ns}::tag_dist_@distance_abbrev@, @topk_by_bitonic_sort@, @bitonic_sort_and_merge_multi_warps@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/search_single_cta_p"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "cagra_search_multi_cta_data_@data_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_multi_cta_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/search_multi_cta_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_search_multi_cta<${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_index_@source_index_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${cagra_ns}::tag_dist_@distance_abbrev@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/search_multi_cta"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "cagra_random_pickup_data_@data_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/random_pickup_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/random_pickup_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_random_pickup<${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${cagra_ns}::tag_dist_@distance_abbrev@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/random_pickup"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "cagra_compute_distance_to_child_nodes_data_@data_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance_to_child_nodes_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/compute_distance_to_child_nodes_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_compute_distance_to_child_nodes<${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${cagra_ns}::tag_dist_@distance_abbrev@, ${neighbors_ns}::tag_index_@source_index_abbrev@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/compute_distance_to_child_nodes"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "cagra_apply_filter"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/apply_filter_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/apply_filter_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_apply_filter_kernel<${neighbors_ns}::tag_index_@index_abbrev@, ${cagra_ns}::tag_dist_@distance_abbrev@, ${neighbors_ns}::tag_index_@source_index_abbrev@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/apply_filter"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT "sample_filter_@filter_name@_index_@source_index_abbrev@"
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/sample_filter_matrix.json"
    KERNEL_INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/sample_filter_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${cagra_ns}::fragment_tag_sample_filter<${neighbors_ns}::tag_bitset_@bitset_abbrev@, ${neighbors_ns}::tag_index_@source_index_abbrev@, ${neighbors_ns}::tag_filter_@filter_name@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/cagra/filter"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  set(distance_ns "cuvs::distance::detail")
  set(pairwise_matrix_jit_dir
      "${CMAKE_CURRENT_SOURCE_DIR}/src/distance/detail/pairwise_matrix/jit_lto_kernels"
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "pairwise_matrix_@distance_abbrev@_data_@data_abbrev@_acc_@acc_abbrev@_out_@out_abbrev@_index_@index_abbrev@_layout_@layout_abbrev@_veclen_@veclen@"
    MATRIX_JSON_FILE "${pairwise_matrix_jit_dir}/compute_distance_epilog_matrix.json"
    KERNEL_INPUT_FILE "${pairwise_matrix_jit_dir}/pairwise_matrix_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${distance_ns}::fragment_tag_pairwise_matrix<${distance_ns}::tag_distance_@distance_abbrev@, ${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_@acc_abbrev@, ${neighbors_ns}::tag_@out_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${distance_ns}::tag_fin_op_identity, ${distance_ns}::tag_layout_@layout_abbrev@, @veclen@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/pairwise_matrix/pairwise_matrix_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/distance/pairwise_matrix/kernel"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "pairwise_matrix_@distance_abbrev@_data_@data_abbrev@_acc_@acc_abbrev@_out_@out_abbrev@_index_@index_abbrev@_fin_op_rbf_layout_@layout_abbrev@_veclen_@veclen@"
    MATRIX_JSON_FILE "${pairwise_matrix_jit_dir}/compute_distance_epilog_rbf_matrix.json"
    KERNEL_INPUT_FILE "${pairwise_matrix_jit_dir}/pairwise_matrix_rbf_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${distance_ns}::fragment_tag_pairwise_matrix<${distance_ns}::tag_distance_@distance_abbrev@, ${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_@acc_abbrev@, ${neighbors_ns}::tag_@out_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${distance_ns}::tag_fin_op_rbf, ${distance_ns}::tag_layout_@layout_abbrev@, @veclen@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/pairwise_matrix/pairwise_matrix_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/distance/pairwise_matrix/kernel_rbf"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "pairwise_matrix_compute_distance_@distance_abbrev@_data_@data_abbrev@_acc_@acc_abbrev@_index_@index_abbrev@"
    MATRIX_JSON_FILE "${pairwise_matrix_jit_dir}/compute_distance_matrix.json"
    KERNEL_INPUT_FILE "${pairwise_matrix_jit_dir}/compute_distance_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${distance_ns}::fragment_tag_compute_distance<${distance_ns}::tag_distance_@distance_abbrev@, ${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_@acc_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/pairwise_matrix/pairwise_matrix_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/distance/pairwise_matrix/compute_distance"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "pairwise_matrix_compute_distance_@distance_abbrev@_data_@data_abbrev@_acc_@acc_abbrev@_index_@index_abbrev@"
    MATRIX_JSON_FILE "${pairwise_matrix_jit_dir}/compute_distance_rbf_matrix.json"
    KERNEL_INPUT_FILE "${pairwise_matrix_jit_dir}/compute_distance_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${distance_ns}::fragment_tag_compute_distance<${distance_ns}::tag_distance_@distance_abbrev@, ${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_@acc_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/pairwise_matrix/pairwise_matrix_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/distance/pairwise_matrix/compute_distance_rbf"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "pairwise_matrix_compute_distance_epilog_@distance_abbrev@_data_@data_abbrev@_acc_@acc_abbrev@_index_@index_abbrev@_layout_@layout_abbrev@_veclen_@veclen@"
    MATRIX_JSON_FILE "${pairwise_matrix_jit_dir}/compute_distance_epilog_matrix.json"
    KERNEL_INPUT_FILE "${pairwise_matrix_jit_dir}/compute_distance_epilog_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${distance_ns}::fragment_tag_compute_distance_epilog<${distance_ns}::tag_distance_@distance_abbrev@, ${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_@acc_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${distance_ns}::tag_layout_@layout_abbrev@, @veclen@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/pairwise_matrix/pairwise_matrix_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/distance/pairwise_matrix/compute_distance_epilog"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  generate_jit_lto_kernels(
    jit_lto_files
    NAME_FORMAT
      "pairwise_matrix_compute_distance_epilog_@distance_abbrev@_data_@data_abbrev@_acc_@acc_abbrev@_index_@index_abbrev@_layout_@layout_abbrev@_veclen_@veclen@"
    MATRIX_JSON_FILE "${pairwise_matrix_jit_dir}/compute_distance_epilog_rbf_matrix.json"
    KERNEL_INPUT_FILE "${pairwise_matrix_jit_dir}/compute_distance_epilog_kernel.cu.in"
    FRAGMENT_TAG_FORMAT
      "${distance_ns}::fragment_tag_compute_distance_epilog<${distance_ns}::tag_distance_@distance_abbrev@, ${neighbors_ns}::tag_@data_abbrev@, ${neighbors_ns}::tag_@acc_abbrev@, ${neighbors_ns}::tag_index_@index_abbrev@, ${distance_ns}::tag_layout_@layout_abbrev@, @veclen@>"
    FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/pairwise_matrix/pairwise_matrix_fragments.hpp>"
                              "<cuvs/detail/jit_lto/common_fragments.hpp>"
    OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/distance/pairwise_matrix/compute_distance_epilog_rbf"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  endblock()

  # Note that this matrix contains an `arch_includes` placeholder, since we don't currently have a
  # way to do an item-wise transform on a list after computing the matrix product and before
  # configuring the file
  generate_inst_matrix(
    pairwise_matrix_dispatch_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/distance/detail/pairwise_matrix/dispatch_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/distance/detail/pairwise_matrix/dispatch_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/distance/detail/pairwise_matrix/dispatch_inst_data_@data_abbrev@_acc_@acc_abbrev@_out_@out_abbrev@_index_@index_abbrev@_op_@op_abbrev@.cu"
  )
  generate_inst_matrix(
    pairwise_matrix_dispatch_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/distance/detail/pairwise_matrix/dispatch_rbf_matrix.json"
    INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/distance/detail/pairwise_matrix/dispatch_rbf_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/distance/detail/pairwise_matrix/dispatch_rbf_inst_data_@data_abbrev@_acc_@acc_abbrev@_out_@out_abbrev@_index_@index_abbrev@_op_@op_abbrev@.cu"
  )
  generate_inst_matrix(
    cagra_build_inst_files
    MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/cagra_build_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/cagra_build_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/cagra_build_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    cagra_extend_inst_files
    MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/cagra_extend_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/cagra_extend_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/cagra_extend_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    cagra_serialize_inst_files
    MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/cagra_serialize_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/cagra_serialize_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/cagra_serialized_inst_data_@data_abbrev@.cu"
  )
  generate_inst_matrix(
    cagra_merge_inst_files
    MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/cagra_merge_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/cagra_merge_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/cagra_merge_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    iface_cagra_inst_files
    MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/iface/iface_cagra_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/iface/iface_cagra_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/iface/iface_cagra_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    iface_flat_inst_files
    MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/iface/iface_flat_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/iface/iface_flat_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/iface/iface_flat_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    iface_pq_inst_files
    MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/iface/iface_pq_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/iface/iface_pq_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/iface/iface_pq_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    ivf_flat_build_extend_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/ivf_flat_build_extend_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/ivf_flat_build_extend_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/ivf_flat/ivf_flat_build_extend_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    ivf_flat_search_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/ivf_flat_search_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/ivf_flat_search_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/ivf_flat/ivf_flat_search_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    ivf_flat_interleaved_scan_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/ivf_flat_interleaved_scan_matrix.json"
    INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/ivf_flat_interleaved_scan_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/ivf_flat/ivf_flat_interleaved_scan_inst_data_@data_abbrev@_index_@index_abbrev@_filter_@filter_name@.cu"
  )
  generate_inst_matrix(
    ivf_flat_serialize_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/ivf_flat_serialize_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/ivf_flat_serialize_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/ivf_flat/ivf_flat_serialize_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    ivf_pq_build_extend_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_build_extend_matrix.json"
    INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_build_extend_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_build_extend_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    ivf_pq_compute_similarity_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_select_matrix.json"
    INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_select_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_select_inst_out_@out_abbrev@_lut_@lut_abbrev@_filter_@filter_name@_metric_@metric@_@increment_score_str@.cu"
  )
  generate_inst_matrix(
    ivf_pq_compute_similarity_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_run_matrix.json"
    INPUT_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_run_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_run_inst_out_@out_abbrev@_lut_@lut_abbrev@.cu"
  )
  generate_inst_matrix(
    ivf_pq_search_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_search_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_search_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_search_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    ivf_pq_transform_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_transform_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_transform_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/ivf_pq/detail/ivf_pq_transform_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    nn_descent_inst_files
    MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/nn_descent_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/nn_descent_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/nn_descent_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    refine_device_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/refine/detail/refine_device_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/refine/detail/refine_device_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/refine/detail/refine_device_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )
  generate_inst_matrix(
    refine_host_inst_files
    MATRIX_JSON_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/refine/detail/refine_host_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/refine/detail/refine_host_inst.cpp.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/refine/detail/refine_host_inst_data_@data_abbrev@_index_@index_abbrev@.cpp"
  )
  generate_inst_matrix(
    select_k_inst_files
    MATRIX_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/selection/select_k_matrix.json"
    INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/selection/select_k_inst.cu.in"
    OUTPUT_FILE_FORMAT
      "${CMAKE_CURRENT_BINARY_DIR}/src/selection/select_k_inst_data_@data_abbrev@_index_@index_abbrev@.cu"
  )

  add_library(
    cuvs_objs OBJECT
    src/cluster/detail/minClusterDistanceCompute.cu
    src/cluster/agglomerative.cu
    src/cluster/kmeans_cluster_cost.cu
    src/cluster/kmeans_fit_mg_float.cu
    src/cluster/kmeans_fit_mg_double.cu
    src/cluster/kmeans_fit_double.cu
    src/cluster/kmeans_fit_float.cu
    src/cluster/kmeans_auto_find_k_float.cu
    src/cluster/kmeans_fit_predict_double.cu
    src/cluster/kmeans_fit_predict_float.cu
    src/cluster/kmeans_predict_double.cu
    src/cluster/kmeans_predict_float.cu
    src/cluster/kmeans_balanced_fit_float.cu
    src/cluster/kmeans_balanced_fit_half.cu
    src/cluster/kmeans_balanced_fit_predict_float.cu
    src/cluster/kmeans_balanced_predict_float.cu
    src/cluster/kmeans_balanced_predict_half.cu
    src/cluster/kmeans_balanced_fit_int8.cu
    src/cluster/kmeans_balanced_fit_uint8.cu
    src/cluster/kmeans_balanced_fit_predict_int8.cu
    src/cluster/kmeans_balanced_predict_int8.cu
    src/cluster/kmeans_balanced_predict_uint8.cu
    src/cluster/kmeans_transform_double.cu
    src/cluster/kmeans_transform_float.cu
    src/cluster/single_linkage_float.cu
    src/cluster/spectral.cu
    src/core/bitset.cu
    src/core/omp_wrapper.cpp
    src/util/file_io.cpp
    src/util/host_memory.cpp
    src/detail/jit_lto/AlgorithmLauncher.cpp
    src/detail/jit_lto/AlgorithmPlanner.cpp
    src/detail/jit_lto/FragmentEntry.cpp
    src/detail/jit_lto/nvjitlink_checker.cpp
    src/detail/jit_lto/NVRTCLTOFragmentCompiler.cpp
    src/distance/detail/kernels/gram_matrix.cu
    src/distance/detail/kernels/kernel_factory.cu
    src/distance/detail/kernels/kernel_matrices.cu
    ${pairwise_matrix_dispatch_inst_files}
    src/distance/distance.cu
    src/distance/kde.cu
    src/distance/pairwise_distance.cu
    src/distance/sparse_distance.cu
    src/neighbors/all_neighbors/all_neighbors.cu
    src/neighbors/ball_cover.cu
    src/neighbors/epsilon_neighborhood.cu
    src/neighbors/ball_cover/detail/ball_cover/registers_eps_pass_euclidean.cu
    src/neighbors/ball_cover/detail/ball_cover/registers_pass_one.cu
    src/neighbors/ball_cover/detail/ball_cover/registers_pass_two.cu
    src/neighbors/brute_force.cu
    src/neighbors/brute_force_serialize.cu
    src/neighbors/cagra.cpp
    ${cagra_build_inst_files}
    ${cagra_extend_inst_files}
    src/neighbors/cagra_optimize.cu
    ${cagra_serialize_inst_files}
    ${cagra_merge_inst_files}
    ${iface_cagra_inst_files}
    ${iface_flat_inst_files}
    ${iface_pq_inst_files}
    src/neighbors/detail/cagra/topk_for_cagra/topk.cu
    ${cuvs_cagra_search_cuda_inst_files}
    src/neighbors/detail/cagra/cagra_helpers.cpp
    src/neighbors/dynamic_batching.cu
    src/neighbors/composite/index.cu
    $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:src/neighbors/cagra.cpp>
    $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:src/neighbors/hnsw.cpp>
    src/neighbors/ivf_common.cu
    src/neighbors/ivf_flat_index.cpp
    ${ivf_flat_build_extend_inst_files}
    ${ivf_flat_search_inst_files}
    src/neighbors/ivf_flat/ivf_flat_helpers.cu
    ${ivf_flat_interleaved_scan_inst_files}
    ${ivf_flat_serialize_inst_files}
    src/neighbors/ivf_pq_index.cu
    src/neighbors/ivf_pq/ivf_pq_build_common.cu
    src/neighbors/ivf_pq/ivf_pq_serialize.cu
    src/neighbors/ivf_pq/ivf_pq_deserialize.cu
    ${ivf_pq_build_extend_inst_files}
    src/neighbors/ivf_pq/detail/ivf_pq_build_precomputed_int64_t.cu
    ${ivf_pq_compute_similarity_inst_files}
    src/neighbors/ivf_pq/detail/ivf_pq_contiguous_list_data.cu
    src/neighbors/ivf_pq/detail/ivf_pq_list_data.cu
    src/neighbors/ivf_pq/detail/ivf_pq_process_and_fill_codes.cu
    ${ivf_pq_search_inst_files}
    ${ivf_pq_transform_inst_files}
    src/neighbors/ivf_rabitq.cu
    src/neighbors/ivf_rabitq/gpu_index/ivf_gpu.cu
    src/neighbors/ivf_rabitq/gpu_index/initializer_gpu.cu
    src/neighbors/ivf_rabitq/gpu_index/quantizer_gpu.cu
    src/neighbors/ivf_rabitq/gpu_index/rotator_gpu.cu
    src/neighbors/ivf_rabitq/gpu_index/searcher_gpu.cu
    src/neighbors/ivf_rabitq/gpu_index/searcher_gpu_shared_mem_opt.cu
    src/neighbors/ivf_rabitq/gpu_index/searcher_gpu_quantize_query.cu
    src/neighbors/ivf_rabitq/utils/searcher_gpu_utils.cu
    src/neighbors/ivf_sq_index.cpp
    src/neighbors/ivf_sq/ivf_sq_build_extend_float_uint8_t_int64_t.cu
    src/neighbors/ivf_sq/ivf_sq_build_extend_half_uint8_t_int64_t.cu
    src/neighbors/ivf_sq/ivf_sq_search_uint8_t_int64_t.cu
    src/neighbors/ivf_sq/ivf_sq_serialize_uint8_t.cu
    src/neighbors/knn_merge_parts.cu
    src/neighbors/nn_descent.cu
    ${nn_descent_inst_files}
    src/neighbors/nn_descent_gnnd_inst.cu
    src/neighbors/nn_descent_index.cpp
    ${refine_device_inst_files}
    ${refine_host_inst_files}
    src/neighbors/scann/scann_build_float.cu
    src/neighbors/scann/scann_serialize_float.cu
    src/neighbors/sample_filter.cu
    src/neighbors/tiered_index.cu
    src/neighbors/sparse_brute_force.cu
    src/neighbors/vamana_build_float.cu
    src/neighbors/vamana_build_uint8.cu
    src/neighbors/vamana_build_int8.cu
    src/neighbors/vamana_codebooks_float.cu
    src/neighbors/vamana_serialize_float.cu
    src/neighbors/vamana_serialize_uint8.cu
    src/neighbors/vamana_serialize_int8.cu
    src/preprocessing/quantize/scalar.cu
    src/preprocessing/quantize/binary.cu
    src/preprocessing/quantize/pq.cu
    src/preprocessing/spectral/spectral_embedding.cu
    src/preprocessing/pca/pca.cu
    ${select_k_inst_files}
    src/stats/silhouette_score.cu
    src/stats/trustworthiness_score.cu
    ${CUVS_MG_ALGOS}
    ${jit_lto_files}
  )

  set_target_properties(
    cuvs_objs
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               CXX_VISIBILITY_PRESET hidden
               CUDA_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
  )
  target_compile_options(
    cuvs_objs
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${CUVS_DEBUG_CUDA_FLAGS}>"
  )

  target_compile_definitions(
    cuvs_objs PRIVATE $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:CUVS_BUILD_CAGRA_HNSWLIB>
                      $<$<BOOL:${CUVS_NVTX}>:NVTX_ENABLED>
  )

  target_link_libraries(
    cuvs_objs
    PRIVATE cuvs::cuvs_cpp_headers
            cuco::cuco
            nvidia::cutlass::cutlass
            ${CUVS_CTK_MATH_DEPENDENCIES}
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:hnswlib::hnswlib>
            kvikio::kvikio
  )

  target_include_directories(
    cuvs_objs
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../c/include>"
    INTERFACE "$<INSTALL_INTERFACE:include>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/src"
  )

  # Endian detection
  include(TestBigEndian)
  test_big_endian(BIG_ENDIAN)
  if(BIG_ENDIAN)
    target_compile_definitions(cuvs_objs PRIVATE CUVS_SYSTEM_LITTLE_ENDIAN=0)
  else()
    target_compile_definitions(cuvs_objs PRIVATE CUVS_SYSTEM_LITTLE_ENDIAN=1)
  endif()

  if(BUILD_MG_ALGOS)
    rapids_find_generate_module(
      NCCL
      HEADER_NAMES nccl.h
      LIBRARY_NAMES nccl
    )
    find_package(NCCL REQUIRED)
    target_link_libraries(cuvs_objs PUBLIC $<BUILD_LOCAL_INTERFACE:NCCL::NCCL>)

    target_compile_definitions(cuvs_objs PUBLIC CUVS_BUILD_MG_ALGOS)
  endif()

  set(CUVS_CUSOLVER_DEPENDENCY CUDA::cusolver${_ctk_static_suffix})
  set(CUVS_CUBLAS_DEPENDENCY CUDA::cublas${_ctk_static_suffix})
  set(CUVS_CURAND_DEPENDENCY CUDA::curand${_ctk_static_suffix})
  set(CUVS_CUSPARSE_DEPENDENCY CUDA::cusparse${_ctk_static_suffix})

  set(CUVS_CTK_MATH_DEPENDENCIES ${CUVS_CUBLAS_DEPENDENCY} ${CUVS_CUSOLVER_DEPENDENCY}
                                 ${CUVS_CUSPARSE_DEPENDENCY} ${CUVS_CURAND_DEPENDENCY}
  )

  if(NOT cuvs_compile_mode STREQUAL "static_only")
    add_library(cuvs SHARED $<TARGET_OBJECTS:cuvs_objs>)
    add_library(cuvs::cuvs ALIAS cuvs)
    set_target_properties(
      cuvs
      PROPERTIES BUILD_RPATH "\$ORIGIN"
                 INSTALL_RPATH "\$ORIGIN"
                 CXX_STANDARD 20
                 CXX_STANDARD_REQUIRED ON
                 CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 INTERFACE_POSITION_INDEPENDENT_CODE ON
                 POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(
      cuvs
      INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda
                --expt-relaxed-constexpr>
      PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
              "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
              "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${CUVS_DEBUG_CUDA_FLAGS}>"
    )
    target_compile_definitions(
      cuvs PUBLIC $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:CUVS_BUILD_CAGRA_HNSWLIB>
                  $<$<BOOL:${CUVS_NVTX}>:NVTX_ENABLED>
    )

    target_link_libraries(
      cuvs
      PUBLIC rmm::rmm
             raft::raft
             cuvs::cuvs_cpp_headers
             ${CUVS_CTK_MATH_DEPENDENCIES}
             $<BUILD_LOCAL_INTERFACE:$<TARGET_NAME_IF_EXISTS:NCCL::NCCL>>
             $<BUILD_LOCAL_INTERFACE:$<TARGET_NAME_IF_EXISTS:hnswlib::hnswlib>>
             $<$<BOOL:${CUVS_NVTX}>:CUDA::nvtx3>
      PRIVATE $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
              $<COMPILE_ONLY:nvidia::cutlass::cutlass>
              $<COMPILE_ONLY:cuco::cuco>
              CUDA::nvJitLink
              CUDA::nvrtc
              $<BUILD_LOCAL_INTERFACE:kvikio::kvikio>
    )

    # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
    file(
      WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
      [=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=]
    )
    target_link_options(cuvs PRIVATE $<HOST_LINK:${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld>)

  endif()

  if(NOT cuvs_compile_mode STREQUAL "shared_only")
    add_library(cuvs_static STATIC $<TARGET_OBJECTS:cuvs_objs>)
    add_library(cuvs::cuvs_static ALIAS cuvs_static)

    set_target_properties(
      cuvs_static
      PROPERTIES BUILD_RPATH "\$ORIGIN"
                 INSTALL_RPATH "\$ORIGIN"
                 CXX_STANDARD 20
                 CXX_STANDARD_REQUIRED ON
                 CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 POSITION_INDEPENDENT_CODE ON
                 INTERFACE_POSITION_INDEPENDENT_CODE ON
                 EXPORT_NAME cuvs_static
    )

    target_compile_options(cuvs_static PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>")
    target_compile_definitions(
      cuvs_static PUBLIC $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:CUVS_BUILD_CAGRA_HNSWLIB>
                         $<$<BOOL:${CUVS_NVTX}>:NVTX_ENABLED>
    )

    target_include_directories(cuvs_static INTERFACE "$<INSTALL_INTERFACE:include>")

    # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
    target_link_options(cuvs_static PRIVATE $<HOST_LINK:${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld>)

    target_link_libraries(
      cuvs_static
      PUBLIC rmm::rmm
             raft::raft
             cuvs::cuvs_cpp_headers
             ${CUVS_CTK_MATH_DEPENDENCIES}
             $<TARGET_NAME_IF_EXISTS:NCCL::NCCL> # needs to be public for DT_NEEDED
             $<BUILD_LOCAL_INTERFACE:$<TARGET_NAME_IF_EXISTS:hnswlib::hnswlib>> # header only
      PRIVATE $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
              CUDA::nvJitLink
              CUDA::nvrtc
              $<$<BOOL:${CUVS_NVTX}>:CUDA::nvtx3>
              $<COMPILE_ONLY:nvidia::cutlass::cutlass>
              $<COMPILE_ONLY:cuco::cuco>
              $<BUILD_LOCAL_INTERFACE:kvikio::kvikio>
    )
  endif()

  # ################################################################################################
  # * NVTX support in cuvs -----------------------------------------------------
  if(NOT CUVS_NVTX)
    # Allow enable NVTX downstream if not set here. This creates a new option at build/install time,
    # which is set by default to OFF, but can be enabled in the dependent project.
    get_property(
      nvtx_option_help_string
      CACHE CUVS_NVTX
      PROPERTY HELPSTRING
    )
    string(
      CONCAT
        nvtx_export_string
        "option(CUVS_NVTX \""
        ${nvtx_option_help_string}
        "\" OFF)"
        [=[

target_link_libraries(cuvs::cuvs INTERFACE $<$<BOOL:${CUVS_NVTX}>:CUDA::nvtx3>)
target_compile_definitions(cuvs::cuvs INTERFACE $<$<BOOL:${CUVS_NVTX}>:NVTX_ENABLED>)

  ]=]
    )
  endif()

  # ################################################################################################
  # * cuvs_c -------------------------------------------------------------------------------
  if(BUILD_C_LIBRARY)
    add_subdirectory(../c c)
  endif()

  # ################################################################################################
  # * install targets-----------------------------------------------------------
  rapids_cmake_install_lib_dir(lib_dir)
  include(GNUInstallDirs)
  include(CPack)

  # The standalone C artifact links libcuvs_static into libcuvs_c, so private shared dependencies of
  # the static archive must be bundled explicitly. Keep this component out of normal cuVS installs,
  # where package managers provide the KvikIO runtime dependency.
  if(TARGET kvikio::kvikio)
    get_target_property(_cuvs_kvikio_aliased_target kvikio::kvikio ALIASED_TARGET)
    if(_cuvs_kvikio_aliased_target)
      set(_cuvs_standalone_kvikio_target "${_cuvs_kvikio_aliased_target}")
    else()
      set(_cuvs_standalone_kvikio_target kvikio::kvikio)
    endif()
    install(
      FILES "$<TARGET_FILE:${_cuvs_standalone_kvikio_target}>"
      DESTINATION ${lib_dir}
      COMPONENT cuvs_standalone_runtime
      EXCLUDE_FROM_ALL
    )
  endif()

  set(target_names cuvs cuvs_static cuvs_cpp_headers cuvs_c)
  set(component_names cuvs_shared cuvs_static cuvs_cpp_headers c_api)
  set(export_names cuvs-shared-exports cuvs-static-exports cuvs-cpp-headers-exports cuvs-c-exports)
  foreach(target component export IN ZIP_LISTS target_names component_names export_names)
    if(TARGET ${target})
      install(
        TARGETS ${target}
        DESTINATION ${lib_dir}
        COMPONENT ${component}
        EXPORT ${export}
      )

      list(APPEND cuvs_components ${component})
      list(APPEND cuvs_export_sets ${export})
    endif()
  endforeach()

  if(TARGET cuvs_cpp_headers)
    install(
      DIRECTORY include/cuvs
      COMPONENT cuvs_cpp_headers
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(
      FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cuvs/version_config.h
      COMPONENT cuvs_cpp_headers
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuvs
    )
  endif()

  if(TARGET cuvs_c)
    install(
      DIRECTORY ../c/include/cuvs/
      COMPONENT c_api
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuvs
    )
    install(
      FILES ${CUVS_C_BINARY_DIR}/include/cuvs/core/c_config.h
      COMPONENT c_api
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuvs/core/
    )
  endif()

  list(REMOVE_DUPLICATES cuvs_components)
  list(REMOVE_DUPLICATES cuvs_export_sets)
  include(cmake/modules/generate_cuvs_export.cmake)
  generate_cuvs_export(COMPONENTS ${cuvs_components} EXPORT_SETS ${cuvs_export_sets})

endif()

# ##################################################################################################
# * build test executable ----------------------------------------------------

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(internal)
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * build ann benchmark executable -----------------------------------------------

if(BUILD_CUVS_BENCH)
  add_subdirectory(bench/ann/)
endif()
