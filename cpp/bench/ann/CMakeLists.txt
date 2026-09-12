# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

list(APPEND CMAKE_MODULE_PATH "${CUVS_SOURCE_DIR}")

# ##################################################################################################
# * benchmark options ------------------------------------------------------------------------------

option(CUVS_ANN_BENCH_USE_FAISS_GPU_FLAT "Include faiss' brute-force knn algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_FLAT "Include faiss' ivf flat algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_PQ "Include faiss' ivf pq algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_SQ "Include faiss' ivf sq algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_FAISS_GPU_CAGRA "Include faiss' cagra algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_FAISS_GPU_CAGRA_HNSW
       "Include faiss' cagra algorithm for build and hnsw for search in benchmark" ON
)
option(CUVS_ANN_BENCH_USE_FAISS_CPU_FLAT "Include faiss' cpu brute-force algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_FLAT "Include faiss' cpu ivf flat algorithm in benchmark"
       ON
)
option(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_PQ "Include faiss' cpu ivf pq algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_RABITQ
       "Include faiss' cpu ivf rabitq algorithm in benchmark" ON
)
option(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_SQ "Include faiss' cpu ivf sq algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_FAISS_CPU_HNSW_FLAT "Include faiss' hnsw algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT "Include cuVS ivf flat algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_IVF_SQ "Include cuVS ivf sq algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_IVF_PQ "Include cuVS ivf pq algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_CAGRA "Include cuVS CAGRA in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE "Include cuVS brute force knn in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB "Include cuVS CAGRA with HNSW search in benchmark" ON)
option(CUVS_ANN_BENCH_USE_HNSWLIB "Include hnsw algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_GGNN "Include ggnn algorithm in benchmark" OFF)
option(CUVS_ANN_BENCH_USE_DISKANN "Include DISKANN search in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_VAMANA "Include cuVS Vamana with DiskANN search in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_CAGRA_DISKANN "Include cuVS CAGRA with DISKANN search in benchmark"
       ON
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(ARM|arm|aarch64)")
  set(CUVS_ANN_BENCH_USE_DISKANN OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_VAMANA OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_CAGRA_DISKANN OFF)
endif()
option(CUVS_ANN_BENCH_USE_CUVS_MG "Include cuVS ann mg algorithm in benchmark" ${BUILD_MG_ALGOS})
option(CUVS_ANN_BENCH_SINGLE_EXE
       "Make a single executable with benchmark as shared library modules" OFF
)
option(CUVS_KNN_BENCH_USE_CUVS_BRUTE_FORCE "Include cuVS brute force knn in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_IVF_RABITQ "Include cuVS ivf RaBitQ algorithm in benchmark" ON)

# ##################################################################################################
# * Process options ----------------------------------------------------------

find_package(Threads REQUIRED)

# ----- FAISS use in Benchmarks ----
get_cmake_property(_variableNames VARIABLES)

set(CUVS_ANN_BENCH_USE_FAISS OFF)
message(STATUS "Checking for FAISS use in benchmarks...")
foreach(_varName ${_variableNames})
  if(_varName MATCHES "CUVS_ANN_BENCH_USE_FAISS.+")
    if(${_varName})
      set(CUVS_ANN_BENCH_USE_FAISS ON)
      message(STATUS "${_varName} is detected as ON.")
      break()
    endif()
  endif()
endforeach()

if(CUVS_ANN_BENCH_USE_FAISS)
  message(STATUS "CUVS_ANN_BENCH_USE_FAISS is switched ON")
else()
  message(STATUS "CUVS_ANN_BENCH_USE_FAISS is switched OFF")
endif()

# ----------------------------------

set(CUVS_FAISS_ENABLE_GPU ON)
set(CUVS_USE_FAISS_STATIC ON)

if(BUILD_CPU_ONLY)
  set(CUVS_FAISS_ENABLE_GPU OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_IVF_SQ OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_IVF_PQ OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_IVF_RABITQ OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_CAGRA OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB OFF)
  set(CUVS_ANN_BENCH_USE_GGNN OFF)
  set(CUVS_KNN_BENCH_USE_CUVS_BRUTE_FORCE OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_MG OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_VAMANA OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_CAGRA_DISKANN OFF)
else()
  set(CUVS_FAISS_ENABLE_GPU ON)
endif()

set(CUVS_ANN_BENCH_USE_CUVS OFF)
if(CUVS_ANN_BENCH_USE_CUVS_IVF_PQ
   OR CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE
   OR CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT
   OR CUVS_ANN_BENCH_USE_CUVS_IVF_SQ
   OR CUVS_ANN_BENCH_USE_CUVS_IVF_RABITQ
   OR CUVS_ANN_BENCH_USE_CUVS_CAGRA
   OR CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB
   OR CUVS_KNN_BENCH_USE_CUVS_BRUTE_FORCE
   OR CUVS_ANN_BENCH_USE_CUVS_MG
   OR CUVS_ANN_BENCH_USE_CUVS_VAMANA
   OR CUVS_ANN_BENCH_USE_CUVS_CAGRA_DISKANN
)
  set(CUVS_ANN_BENCH_USE_CUVS ON)
endif()

# ##################################################################################################
# * Fetch requirements -------------------------------------------------------------

if(CUVS_ANN_BENCH_USE_HNSWLIB OR CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB)
  include(cmake/thirdparty/get_hnswlib)
endif()

include(cmake/thirdparty/get_nlohmann_json)
include(cmake/thirdparty/get_sqlite)

if(CUVS_ANN_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_ggnn)
endif()

if(CUVS_ANN_BENCH_USE_DISKANN
   OR CUVS_ANN_BENCH_USE_CUVS_VAMANA
   OR CUVS_ANN_BENCH_USE_CUVS_CAGRA_DISKANN
)
  include(cmake/thirdparty/get_diskann)
endif()

if(CUVS_ANN_BENCH_USE_FAISS)
  include(cmake/thirdparty/get_faiss)
endif()

# ##################################################################################################
# * Target function -------------------------------------------------------------

function(ConfigureAnnBench)

  set(oneValueArgs NAME)
  set(multiValueArgs PATH LINKS CXXFLAGS)

  if(NOT BUILD_CPU_ONLY)
    set(GPU_BUILD ON)
  endif()

  cmake_parse_arguments(
    ConfigureAnnBench "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  set(BENCH_NAME ${ConfigureAnnBench_NAME}_ANN_BENCH)

  if(CUVS_ANN_BENCH_SINGLE_EXE)
    add_library(${BENCH_NAME} SHARED ${ConfigureAnnBench_PATH})
    string(TOLOWER ${BENCH_NAME} BENCH_LIB_NAME)
    set_target_properties(${BENCH_NAME} PROPERTIES OUTPUT_NAME ${BENCH_LIB_NAME})
    add_dependencies(${BENCH_NAME} ANN_BENCH)
  else()
    add_executable(${BENCH_NAME} ${ConfigureAnnBench_PATH})
    target_compile_definitions(${BENCH_NAME} PRIVATE ANN_BENCH_BUILD_MAIN>)
    target_link_libraries(
      ${BENCH_NAME} PRIVATE benchmark::benchmark $<$<TARGET_EXISTS:CUDA::nvtx3>:CUDA::nvtx3>
    )
  endif()

  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE ${ConfigureAnnBench_LINKS}
            nlohmann_json::nlohmann_json
            sqlite3
            Threads::Threads
            $<$<BOOL:${GPU_BUILD}>:CUDA::cudart_static>
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  set_target_properties(
    ${BENCH_NAME}
    PROPERTIES # set target compile options
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
  )

  set(${ConfigureAnnBench_CXXFLAGS} ${CUVS_CXX_FLAGS} ${ConfigureAnnBench_CXXFLAGS})

  target_compile_options(
    ${BENCH_NAME}
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${ConfigureAnnBench_CXXFLAGS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${CUVS_DEBUG_CUDA_FLAGS}>"
  )

  if(CUVS_ANN_BENCH_USE_${ConfigureAnnBench_NAME})
    target_compile_definitions(
      ${BENCH_NAME}
      PUBLIC
        CUVS_ANN_BENCH_USE_${ConfigureAnnBench_NAME}=CUVS_ANN_BENCH_USE_${ConfigureAnnBench_NAME}
    )
  endif()

  target_include_directories(
    ${BENCH_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/../c/include>"
    PRIVATE ${ConfigureAnnBench_INCLUDES}
  )

  install(
    TARGETS ${BENCH_NAME}
    COMPONENT ann_bench
    DESTINATION bin/ann
  )

  add_dependencies(CUVS_ANN_BENCH_ALL ${BENCH_NAME})
endfunction()

# ##################################################################################################
# * Configure benchmark targets -------------------------------------------------------------

if(NOT TARGET CUVS_ANN_BENCH_ALL)
  add_custom_target(CUVS_ANN_BENCH_ALL)
endif()

if(CUVS_ANN_BENCH_USE_HNSWLIB)
  ConfigureAnnBench(NAME HNSWLIB PATH src/hnswlib/hnswlib_benchmark.cpp LINKS hnswlib::hnswlib)

endif()

if(CUVS_ANN_BENCH_USE_CUVS_IVF_PQ)
  ConfigureAnnBench(
    NAME CUVS_IVF_PQ PATH src/cuvs/cuvs_benchmark.cu src/cuvs/cuvs_ivf_pq.cu LINKS cuvs
  )
endif()

if(CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT)
  ConfigureAnnBench(
    NAME CUVS_IVF_FLAT PATH src/cuvs/cuvs_benchmark.cu src/cuvs/cuvs_ivf_flat.cu LINKS cuvs
  )
endif()

if(CUVS_ANN_BENCH_USE_CUVS_IVF_RABITQ)
  ConfigureAnnBench(
    NAME CUVS_IVF_RABITQ PATH src/cuvs/cuvs_benchmark.cu src/cuvs/cuvs_ivf_rabitq.cu LINKS cuvs
  )
endif()

if(CUVS_ANN_BENCH_USE_CUVS_IVF_SQ)
  ConfigureAnnBench(
    NAME CUVS_IVF_SQ PATH src/cuvs/cuvs_benchmark.cu src/cuvs/cuvs_ivf_sq.cu LINKS cuvs
  )
endif()

if(CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE)
  ConfigureAnnBench(NAME CUVS_BRUTE_FORCE PATH src/cuvs/cuvs_benchmark.cu LINKS cuvs)
endif()

if(CUVS_KNN_BENCH_USE_CUVS_BRUTE_FORCE)
  ConfigureAnnBench(
    NAME CUVS_KNN_BRUTE_FORCE PATH
    $<$<BOOL:${CUVS_KNN_BENCH_USE_CUVS_BRUTE_FORCE}>:src/cuvs/cuvs_brute_force_knn.cu> LINKS cuvs
  )
endif()

# Cluster assignment benchmark: brute force vs CAGRA for assigning vectors to clusters (IVF
# training)
if(CUVS_ANN_BENCH_USE_CUVS_CAGRA)
  add_executable(CUVS_CLUSTER_ASSIGNMENT_BENCH src/cuvs/cuvs_cluster_assignment_bench.cu)
  target_link_libraries(
    CUVS_CLUSTER_ASSIGNMENT_BENCH
    PRIVATE cuvs benchmark::benchmark $<$<TARGET_EXISTS:CUDA::nvtx3>:CUDA::nvtx3>
            $<COMPILE_ONLY:nvidia::cutlass::cutlass>
  )
  target_include_directories(
    CUVS_CLUSTER_ASSIGNMENT_BENCH
    PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/../c/include>"
    PRIVATE "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/src>"
  )
  set_target_properties(
    CUVS_CLUSTER_ASSIGNMENT_BENCH
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  install(
    TARGETS CUVS_CLUSTER_ASSIGNMENT_BENCH
    COMPONENT ann_bench
    DESTINATION bin/ann
  )
  add_dependencies(CUVS_ANN_BENCH_ALL CUVS_CLUSTER_ASSIGNMENT_BENCH)

  # IVF-PQ build benchmarks: k-means fit and extend cluster-assignment (brute vs CAGRA)
  add_executable(CUVS_IVFPQ_BUILD_BENCH src/cuvs/cuvs_ivf_pq_build_bench.cu)
  target_link_libraries(CUVS_IVFPQ_BUILD_BENCH PRIVATE cuvs benchmark::benchmark)
  target_include_directories(
    CUVS_IVFPQ_BUILD_BENCH PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/include>"
                                  "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/../c/include>"
  )
  set_target_properties(
    CUVS_IVFPQ_BUILD_BENCH
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  install(
    TARGETS CUVS_IVFPQ_BUILD_BENCH
    COMPONENT ann_bench
    DESTINATION bin/ann
  )
  add_dependencies(CUVS_ANN_BENCH_ALL CUVS_IVFPQ_BUILD_BENCH)

  add_executable(CUVS_FIT_ANN_REBUILD_TUNING_BENCH src/cuvs/cuvs_fit_ann_rebuild_tuning_bench.cu)
  target_link_libraries(
    CUVS_FIT_ANN_REBUILD_TUNING_BENCH
    PRIVATE cuvs benchmark::benchmark $<$<TARGET_EXISTS:CUDA::nvtx3>:CUDA::nvtx3>
            $<COMPILE_ONLY:nvidia::cutlass::cutlass>
  )
  target_include_directories(
    CUVS_FIT_ANN_REBUILD_TUNING_BENCH
    PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/../c/include>"
    PRIVATE "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/src>"
  )
  set_target_properties(
    CUVS_FIT_ANN_REBUILD_TUNING_BENCH
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  install(
    TARGETS CUVS_FIT_ANN_REBUILD_TUNING_BENCH
    COMPONENT ann_bench
    DESTINATION bin/ann
  )
  add_dependencies(CUVS_ANN_BENCH_ALL CUVS_FIT_ANN_REBUILD_TUNING_BENCH)
endif()

if(CUVS_ANN_BENCH_USE_CUVS_CAGRA)
  ConfigureAnnBench(
    NAME
    CUVS_CAGRA
    PATH
    src/cuvs/cuvs_benchmark.cu
    src/cuvs/cuvs_cagra_float.cu
    src/cuvs/cuvs_cagra_half.cu
    src/cuvs/cuvs_cagra_int8_t.cu
    src/cuvs/cuvs_cagra_uint8_t.cu
    LINKS
    cuvs
  )
endif()

if(CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB)
  ConfigureAnnBench(NAME CUVS_CAGRA_HNSWLIB PATH src/cuvs/cuvs_cagra_hnswlib.cu LINKS cuvs)
endif()

if(CUVS_ANN_BENCH_USE_CUVS_MG)
  ConfigureAnnBench(
    NAME
    CUVS_MG
    PATH
    src/cuvs/cuvs_benchmark.cu
    $<$<BOOL:${CUVS_ANN_BENCH_USE_CUVS_MG}>:src/cuvs/cuvs_mg_ivf_flat.cu>
    $<$<BOOL:${CUVS_ANN_BENCH_USE_CUVS_MG}>:src/cuvs/cuvs_mg_ivf_pq.cu>
    $<$<BOOL:${CUVS_ANN_BENCH_USE_CUVS_MG}>:src/cuvs/cuvs_mg_cagra.cu>
    LINKS
    cuvs
    nccl
  )
endif()

message("CUVS_FAISS_TARGETS: ${CUVS_FAISS_TARGETS}")
message("CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
if(CUVS_ANN_BENCH_USE_FAISS_CPU_FLAT)
  ConfigureAnnBench(
    NAME FAISS_CPU_FLAT PATH src/faiss/faiss_cpu_benchmark.cpp LINKS ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_FLAT)
  ConfigureAnnBench(
    NAME FAISS_CPU_IVF_FLAT PATH src/faiss/faiss_cpu_benchmark.cpp LINKS ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_PQ)
  ConfigureAnnBench(
    NAME FAISS_CPU_IVF_PQ PATH src/faiss/faiss_cpu_benchmark.cpp LINKS ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_RABITQ)
  ConfigureAnnBench(
    NAME FAISS_CPU_IVF_RABITQ PATH src/faiss/faiss_cpu_benchmark.cpp LINKS ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_SQ)
  ConfigureAnnBench(
    NAME FAISS_CPU_IVF_SQ PATH src/faiss/faiss_cpu_benchmark.cpp LINKS ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_CPU_HNSW_FLAT)
  ConfigureAnnBench(
    NAME FAISS_CPU_HNSW_FLAT PATH src/faiss/faiss_cpu_benchmark.cpp LINKS ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_FLAT AND CUVS_FAISS_ENABLE_GPU)
  ConfigureAnnBench(
    NAME FAISS_GPU_IVF_FLAT PATH src/faiss/faiss_gpu_benchmark.cu LINKS ${CUVS_FAISS_TARGETS}
    raft::raft
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_PQ AND CUVS_FAISS_ENABLE_GPU)
  ConfigureAnnBench(
    NAME FAISS_GPU_IVF_PQ PATH src/faiss/faiss_gpu_benchmark.cu LINKS ${CUVS_FAISS_TARGETS}
    raft::raft
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_SQ AND CUVS_FAISS_ENABLE_GPU)
  ConfigureAnnBench(
    NAME FAISS_GPU_IVF_SQ PATH src/faiss/faiss_gpu_benchmark.cu LINKS ${CUVS_FAISS_TARGETS}
    raft::raft
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_GPU_FLAT AND CUVS_FAISS_ENABLE_GPU)
  ConfigureAnnBench(
    NAME FAISS_GPU_FLAT PATH src/faiss/faiss_gpu_benchmark.cu LINKS ${CUVS_FAISS_TARGETS}
    raft::raft
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_GPU_CAGRA AND CUVS_FAISS_ENABLE_GPU)
  ConfigureAnnBench(
    NAME FAISS_GPU_CAGRA PATH src/faiss/faiss_gpu_benchmark.cu LINKS ${CUVS_FAISS_TARGETS}
    raft::raft
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_GPU_CAGRA_HNSW AND CUVS_FAISS_ENABLE_GPU)
  ConfigureAnnBench(
    NAME FAISS_GPU_CAGRA_HNSW PATH src/faiss/faiss_gpu_benchmark.cu LINKS ${CUVS_FAISS_TARGETS}
    raft::raft
  )
endif()

if(CUVS_ANN_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_glog)
  ConfigureAnnBench(
    NAME GGNN PATH src/ggnn/ggnn_benchmark.cu LINKS glog::glog ggnn::ggnn CUDA::curand
  )
endif()

if(CUVS_ANN_BENCH_USE_DISKANN)
  ConfigureAnnBench(
    NAME DISKANN_MEMORY PATH src/diskann/diskann_benchmark.cpp LINKS diskann::diskann aio
  )
  ConfigureAnnBench(
    NAME DISKANN_SSD PATH src/diskann/diskann_benchmark.cpp LINKS diskann::diskann aio
  )
endif()

if(CUVS_ANN_BENCH_USE_CUVS_VAMANA)
  ConfigureAnnBench(NAME CUVS_VAMANA PATH src/cuvs/cuvs_vamana.cu LINKS cuvs diskann::diskann aio)
endif()

if(CUVS_ANN_BENCH_USE_CUVS_CAGRA_DISKANN)
  ConfigureAnnBench(
    NAME CUVS_CAGRA_DISKANN PATH src/cuvs/cuvs_cagra_diskann.cu LINKS cuvs diskann::diskann aio
  )
endif()

# ##################################################################################################
# * Dynamically-loading ANN_BENCH executable -------------------------------------------------------
if(CUVS_ANN_BENCH_SINGLE_EXE)
  add_executable(ANN_BENCH src/common/benchmark.cpp)

  target_include_directories(ANN_BENCH PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  target_link_libraries(
    ANN_BENCH PRIVATE raft::raft nlohmann_json::nlohmann_json sqlite3 benchmark::benchmark dl
                      $<$<TARGET_EXISTS:CUDA::nvtx3>:CUDA::nvtx3>
  )
  set_target_properties(
    ANN_BENCH
    PROPERTIES # set target compile options
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
  )
  target_compile_definitions(
    ANN_BENCH
    PRIVATE
      $<$<BOOL:${CUDAToolkit_FOUND}>:ANN_BENCH_LINK_CUDART="libcudart.so.${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}.${CUDAToolkit_VERSION_PATCH}">
  )

  target_link_options(ANN_BENCH PRIVATE -export-dynamic)

  install(
    TARGETS ANN_BENCH
    COMPONENT ann_bench
    DESTINATION bin/ann
    EXCLUDE_FROM_ALL
  )
endif()
