# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

list(APPEND CMAKE_MODULE_PATH "${CUVS_SOURCE_DIR}")

find_package(Threads REQUIRED)

# ##################################################################################################
# * Fetch requirements -------------------------------------------------------------

include(cmake/thirdparty/get_nlohmann_json)

# ##################################################################################################
# * Target function -------------------------------------------------------------

function(ConfigureClusterBench)

  set(oneValueArgs NAME)
  set(multiValueArgs PATH LINKS)

  cmake_parse_arguments(ConfigureClusterBench "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(BENCH_NAME ${ConfigureClusterBench_NAME}_CLUSTER_BENCH)

  add_executable(${BENCH_NAME} ${ConfigureClusterBench_PATH})

  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE ${ConfigureClusterBench_LINKS}
            benchmark::benchmark
            nlohmann_json::nlohmann_json
            rmm::rmm
            Threads::Threads
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<$<TARGET_EXISTS:CUDA::nvtx3>:CUDA::nvtx3>
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

  target_compile_options(
    ${BENCH_NAME}
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${CUVS_DEBUG_CUDA_FLAGS}>"
  )

  install(
    TARGETS ${BENCH_NAME}
    COMPONENT cluster_bench
    DESTINATION bin/cluster
  )

  add_dependencies(CUVS_CLUSTER_BENCH_ALL ${BENCH_NAME})
endfunction()

# ##################################################################################################
# * Configure benchmark targets -------------------------------------------------------------

if(NOT TARGET CUVS_CLUSTER_BENCH_ALL)
  add_custom_target(CUVS_CLUSTER_BENCH_ALL)
endif()

ConfigureClusterBench(NAME CUVS_KMEANS PATH src/cuvs/cuvs_benchmark.cu LINKS cuvs)
