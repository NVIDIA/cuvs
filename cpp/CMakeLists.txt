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

# Find CMake packages provided by installed wheels.
find_package(Python COMPONENTS Interpreter)
if(Python_FOUND)
  include(${rapids-cmake-dir}/cython-core/find_prefix_paths.cmake)
  rapids_cython_find_prefix_paths("${Python_EXECUTABLE}" wheel_prefix_paths)
  set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)
  list(PREPEND CMAKE_PREFIX_PATH ${wheel_prefix_paths})
endif()

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
option(BUILD_MNMG_TESTS
       "Build multi-node multi-GPU (MNMG) tests that require distributed (UCXX) dependencies" OFF
)
# The MNMG tests are only meaningful when tests are being built. Force the flag OFF otherwise so the
# distributed components are not pulled in unnecessarily.
if(NOT BUILD_TESTS AND BUILD_MNMG_TESTS)
  set(BUILD_MNMG_TESTS OFF)
endif()
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
  include(cmake/thirdparty/get_rtcx.cmake)
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

include(cmake/modules/generate_inst_matrix_collection.cmake)
include(cmake/modules/generate_jit_lto_kernel_collection.cmake)
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

  # Each JSON file describes the instantiations of one family. The `${CMAKE_CURRENT_BINARY_DIR}`
  # references of the output files are expanded.
  generate_inst_matrix_collection(
    cuvs_inst_files
    JSON_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/inst_matrix.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/inst_matrix.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/distance/detail/pairwise_matrix/inst_matrix.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/iface/inst_matrix.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/inst_matrix.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/inst_matrix.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/cluster/inst_matrix.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/refine/detail/inst_matrix.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/selection/inst_matrix.json"
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


  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance-ext.cuh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/detail/cagra/compute_distance-ext.cuh" @ONLY
  )
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/compute_distance.cu.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/detail/cagra/compute_distance.cu" @ONLY
  )

  if(BUILD_MG_ALGOS)
    generate_inst_matrix_collection(
      CUVS_MG_ALGOS
      JSON_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/mg/inst_matrix.json"
    )
  endif()

  set(JIT_LTO_TARGET_ARCHITECTURE "70-real")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(JIT_LTO_TARGET_ARCHITECTURE "75-real")
  endif()

  # Generate interleaved scan kernel files at build time

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
  target_link_libraries(
    jit_lto_kernel_usage_requirements INTERFACE rmm::rmm raft::raft CCCL::CCCL cuco::cuco
  )

  block(PROPAGATE jit_lto_files)
  set(jit_lto_files)
  set(CMAKE_CUDA_ARCHITECTURES ${JIT_LTO_TARGET_ARCHITECTURE})
  set(neighbors_ns "cuvs::neighbors::detail")
  set(ivf_flat_ns "cuvs::neighbors::ivf_flat::detail")
  set(ivf_sq_ns "cuvs::neighbors::ivf_sq::detail")
  set(ivf_pq_ns "cuvs::neighbors::ivf_pq::detail")
  set(ivf_rabitq_ns "cuvs::neighbors::ivf_rabitq::detail")
  set(cagra_ns "cuvs::neighbors::cagra::detail")
  set(distance_ns "cuvs::distance::detail")

  # Each JSON file describes the kernels of one family. The namespace variables above are replaced
  # in the fragment tags of the kernels, which refer to them as `@name@`.
  generate_jit_lto_kernel_collection(
    jit_lto_files
    JSON_FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/jit_lto_kernels/kernels.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/detail/jit_lto_kernels/kernels.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_sq/detail/jit_lto_kernels/kernels.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_pq/detail/jit_lto_kernels/kernels.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_rabitq/jit_lto_kernels/kernels.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/detail/cagra/jit_lto_kernels/kernels.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/distance/detail/pairwise_matrix/jit_lto_kernels/kernels.json"
    KERNEL_LINK_LIBRARIES jit_lto_kernel_usage_requirements
  )
  endblock()

  include(cmake/modules/generate_cutile_kernels.cmake)
  # Tile IR supports sm_103 starting with version 13.1 and sm_107 starting with 13.4. Conservatively
  # gate by the CUDA compiler version, as in the existing sm_107 support.
  set(CUVS_CUTILE_SM103_ENABLED 0)
  set(CUVS_CUTILE_SM107_ENABLED 0)
  set(CUVS_CUTILE_EXCLUDED_GPU_CODES "")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.1)
    set(CUVS_CUTILE_SM103_ENABLED 1)
  else()
    list(APPEND CUVS_CUTILE_EXCLUDED_GPU_CODES "sm_103")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.4)
    set(CUVS_CUTILE_SM107_ENABLED 1)
  else()
    list(APPEND CUVS_CUTILE_EXCLUDED_GPU_CODES "sm_107")
  endif()
  cuvs_cutile_setup(CUVS_CUTILE_ENABLED cutile_python cutile_bin2c)
  if(CUVS_CUTILE_ENABLED)
    set(cutile_smoke_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/jit_lto/cutile_smoke")
    set(cutile_smoke_generated_dir
        "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/detail/jit_lto/cutile_smoke"
    )
    generate_cutile_kernels(
      cutile_smoke_files
      KERNEL_DIR "${cutile_smoke_dir}"
      KERNEL_BASENAME "cutile_smoke"
      KERNEL_PYTHON "smoke_kernel.py"
      EXPORT_SCRIPT "export_smoke.py"
      OUTPUT_DIRECTORY "${cutile_smoke_generated_dir}"
      MATRIX_JSON_FILE "${cutile_smoke_dir}/cutile_smoke_matrix.json" PYTHON "${cutile_python}"
                                                                      BIN2C "${cutile_bin2c}"
      FRAGMENT_TAG_FORMAT_CUBIN
        "cuvs::detail::jit_lto::fragment_tag_cutile_smoke_add_cubin<cuvs::detail::jit_lto::@arch_tag@>"
      FRAGMENT_TAG_HEADER_FILES "<cuvs/detail/jit_lto/cutile_arch_tags.hpp>"
                                "<cuvs/detail/jit_lto/cutile_smoke_fragments.hpp>"
    )

    set(fused_1nn_cutile_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/src/distance/detail/fused_distance_nn/cutile"
    )
    set(cutile_fused_1nn_generated_dir
        "${CMAKE_CURRENT_BINARY_DIR}/generated_kernels/distance/fused_1nn/cutile"
    )
    set(cutile_fused_1nn_tiles "${cutile_fused_1nn_generated_dir}/fused_1nn_cutile_tiles.hpp")
    generate_cutile_kernels(
      cutile_fused_1nn_files
      KERNEL_DIR "${fused_1nn_cutile_dir}"
      KERNEL_BASENAME "fused_1nn"
      KERNEL_PYTHON "fused_1nn_kernel.py"
      EXPORT_SCRIPT "export_fused_1nn.py"
      OUTPUT_DIRECTORY "${cutile_fused_1nn_generated_dir}"
      MATRIX_JSON_FILE "${fused_1nn_cutile_dir}/fused_1nn_cutile_matrix.json"
                       PYTHON "${cutile_python}" BIN2C "${cutile_bin2c}"
      FRAGMENT_TAG_FORMAT_CUBIN
        "cuvs::distance::detail::fragment_tag_fused_1nn_cubin<cuvs::neighbors::detail::tag_@data_abbrev@, cuvs::neighbors::detail::tag_index_@index_abbrev@, cuvs::distance::detail::cutile_tile_config<@tile_m@, @tile_n@, @tile_k@>, cuvs::distance::detail::@abi_tag@, cuvs::detail::jit_lto::@arch_tag@>"
      FRAGMENT_TAG_HEADER_FILES
        "<cuvs/detail/jit_lto/fused_distance_nn/fused_1nn_fragments.hpp>"
        "<cuvs/detail/jit_lto/cutile_arch_tags.hpp>" "<cuvs/detail/jit_lto/common_fragments.hpp>"
    )
    generate_string_matrix(
      cutile_fused_1nn_tile_aliases
      MATRIX_JSON_FILE
      "${fused_1nn_cutile_dir}/fused_1nn_cutile_matrix.json"
      ITEM_FORMAT
      "using fused_1nn_matrix_tile_@data_abbrev@_@arch_tag@_@abi_abbrev@ = cutile_tile_config<@tile_m@, @tile_n@, @tile_k@>@semicolon@"
      GLUE
      "\n"
    )
    set(CUTILE_TILE_METADATA_INCLUDE
        "#include <cuvs/detail/jit_lto/fused_distance_nn/fused_1nn_fragments.hpp>"
    )
    set(CUTILE_TILE_METADATA_NAMESPACE "cuvs::distance::detail")
    set(CUTILE_TILE_METADATA_ALIASES "${cutile_fused_1nn_tile_aliases}")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/cutile_tile_metadata.hpp.in"
      "${cutile_fused_1nn_tiles}" @ONLY
    )
    list(APPEND cutile_fused_1nn_files "${cutile_fused_1nn_tiles}")
  endif()
  # Note that this matrix contains an `arch_includes` placeholder, since we don't currently have a
  # way to do an item-wise transform on a list after computing the matrix product and before
  # configuring the file

  add_library(
    cuvs_objs OBJECT
    src/cluster/detail/minClusterDistanceCompute.cu
    src/cluster/agglomerative.cu
    ${cuvs_inst_files}
    src/cluster/kmeans_cluster_cost.cu
    src/cluster/kmeans_fit_double.cu
    src/cluster/kmeans_fit_float.cu
    $<$<BOOL:${BUILD_MG_ALGOS}>:src/cluster/kmeans_fit_mg_float.cu>
    $<$<BOOL:${BUILD_MG_ALGOS}>:src/cluster/kmeans_fit_mg_double.cu>
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
    src/core/bloom_filter.cu
    src/core/roaring_allowlist.cu
    src/neighbors/roaring_bitmap_filter.cu
    src/core/omp_wrapper.cpp
    src/util/file_io.cpp
    src/util/host_memory.cpp
    src/detail/jit_lto/TileAlgorithmPlanner.cpp
    src/distance/detail/kernels/gram_matrix.cu
    src/distance/detail/kernels/kernel_factory.cu
    src/distance/detail/kernels/kernel_matrices.cu
    src/distance/distance.cu
    src/distance/top_1_nn.cu
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
    src/neighbors/cagra_optimize.cu
    src/neighbors/detail/cagra/graph_shared.cu
    src/neighbors/detail/cagra/cagra_merge_scaffold_shared.cu
    src/neighbors/detail/cagra/topk_for_cagra/topk.cu
    ${CMAKE_CURRENT_BINARY_DIR}/src/neighbors/detail/cagra/compute_distance.cu
    src/neighbors/detail/cagra/cagra_helpers.cpp
    src/neighbors/dynamic_batching.cu
    src/neighbors/composite/index.cu
    $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:src/neighbors/cagra.cpp>
    $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:src/neighbors/hnsw.cpp>
    src/neighbors/ivf_binary_index.cpp
    src/neighbors/ivf_binary/ivf_binary.cu
    src/neighbors/ivf_common.cu
    src/neighbors/ivf_flat_index.cpp
    src/neighbors/ivf_flat/ivf_flat_helpers.cu
    src/neighbors/ivf_pq_index.cu
    src/neighbors/ivf_pq/ivf_pq_build_common.cu
    src/neighbors/ivf_pq/ivf_pq_serialize.cu
    src/neighbors/ivf_pq/ivf_pq_deserialize.cu
    src/neighbors/ivf_pq/detail/ivf_pq_build_precomputed_int64_t.cu
    src/neighbors/ivf_pq/detail/ivf_pq_contiguous_list_data.cu
    src/neighbors/ivf_pq/detail/ivf_pq_list_data.cu
    src/neighbors/ivf_pq/detail/ivf_pq_process_and_fill_codes.cu
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
    src/neighbors/nn_descent_gnnd_inst.cu
    src/neighbors/nn_descent_index.cpp
    src/neighbors/scann/scann_build_float.cu
    src/neighbors/scann/scann_serialize_float.cu
    src/neighbors/sample_filter.cu
    src/neighbors/tiered_index.cu
    src/neighbors/sparse_brute_force.cu
    src/neighbors/vamana_codebooks.cu
    src/preprocessing/quantize/scalar.cu
    src/preprocessing/quantize/binary.cu
    src/preprocessing/quantize/pq.cu
    src/preprocessing/quantize/bbq.cu
    src/preprocessing/spectral/spectral_embedding.cu
    src/preprocessing/pca/pca.cu
    src/stats/silhouette_score.cu
    src/stats/trustworthiness_score.cu
    ${CUVS_MG_ALGOS}
    ${jit_lto_files}
  )

  if(CUVS_CUTILE_ENABLED)
    target_sources(
      cuvs_objs
      PRIVATE ${cutile_fused_1nn_files}
              src/distance/detail/fused_distance_nn/cutile/fused_1nn_tile.cu ${cutile_smoke_files}
    )
  endif()

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
    cuvs_objs
    PRIVATE CUVS_CUTILE_ENABLED=${CUVS_CUTILE_ENABLED}
            CUVS_CUTILE_SM103_ENABLED=${CUVS_CUTILE_SM103_ENABLED}
            CUVS_CUTILE_SM107_ENABLED=${CUVS_CUTILE_SM107_ENABLED}
            $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:CUVS_BUILD_CAGRA_HNSWLIB>
            $<$<BOOL:${BUILD_MG_ALGOS}>:CUVS_BUILD_MG_ALGOS>
            $<$<BOOL:${CUVS_NVTX}>:NVTX_ENABLED>
  )

  target_link_libraries(
    cuvs_objs
    PRIVATE cuvs::cuvs_cpp_headers
            cuco::cuco
            nvidia::cutlass::cutlass
            rtcx::rtcx
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
            "${cutile_fused_1nn_generated_dir}" "${cutile_smoke_generated_dir}"
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
      cuvs
      PUBLIC $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:CUVS_BUILD_CAGRA_HNSWLIB>
             $<$<BOOL:${BUILD_MG_ALGOS}>:CUVS_BUILD_MG_ALGOS> $<$<BOOL:${CUVS_NVTX}>:NVTX_ENABLED>
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
      PRIVATE $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX> $<COMPILE_ONLY:nvidia::cutlass::cutlass>
              $<COMPILE_ONLY:cuco::cuco> CUDA::nvJitLink CUDA::nvrtc rtcx::rtcx kvikio::kvikio
      INTERFACE $<LINK_ONLY:kvikio::kvikio>
    )

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
      cuvs_static
      PUBLIC $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:CUVS_BUILD_CAGRA_HNSWLIB>
             $<$<BOOL:${BUILD_MG_ALGOS}>:CUVS_BUILD_MG_ALGOS> $<$<BOOL:${CUVS_NVTX}>:NVTX_ENABLED>
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
              rtcx::rtcx
              $<$<BOOL:${CUVS_NVTX}>:CUDA::nvtx3>
              $<COMPILE_ONLY:nvidia::cutlass::cutlass>
              $<COMPILE_ONLY:cuco::cuco>
              kvikio::kvikio
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
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * build ann benchmark executable -----------------------------------------------

if(BUILD_CUVS_BENCH)
  add_subdirectory(bench/ann/)
endif()
