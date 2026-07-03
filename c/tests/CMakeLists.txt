# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# ##################################################################################################
# enable testing ################################################################################
# ##################################################################################################

include(rapids-test)

if(PROJECT_IS_TOP_LEVEL)
  rapids_test_init()
endif()

rapids_cmake_install_lib_dir(lib_dir)
include(GNUInstallDirs)

include(${rapids-cmake-dir}/cpm/gtest.cmake)
rapids_cpm_gtest(BUILD_STATIC)

function(ConfigureTest)

  set(options)
  set(oneValueArgs NAME GPUS PERCENT)
  set(multiValueArgs PATH)

  cmake_parse_arguments(_CUVS_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT DEFINED _CUVS_TEST_GPUS AND NOT DEFINED _CUVS_TEST_PERCENT)
    set(_CUVS_TEST_GPUS 1)
    set(_CUVS_TEST_PERCENT 30)
  endif()
  if(NOT DEFINED _CUVS_TEST_GPUS)
    set(_CUVS_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _CUVS_TEST_PERCENT)
    set(_CUVS_TEST_PERCENT 100)
  endif()

  set(TEST_NAME ${_CUVS_TEST_NAME})

  add_executable(${TEST_NAME} ${_CUVS_TEST_PATH})
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE cuvs::c_api
            GTest::gtest
            GTest::gtest_main
            $<IF:$<BOOL:${CUVSC_STATIC_CUVS_LIBRARY}>,$<COMPILE_ONLY:cuvs::cuvs_static>,cuvs::cuvs>
            $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUVS_C_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../${lib_dir}"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )

  target_include_directories(
    ${TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${CUVS_C_SOURCE_DIR}/tests>"
                         "$<BUILD_INTERFACE:${CUVS_C_SOURCE_DIR}/../cpp/tests>"
  )

  rapids_test_add(
    NAME ${TEST_NAME}
    COMMAND ${TEST_NAME}
    GPUS ${_CUVS_TEST_GPUS}
    PERCENT ${_CUVS_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )
endfunction()

ConfigureTest(NAME INTEROP_TEST PATH core/interop.cu)
ConfigureTest(
  NAME DISTANCE_C_TEST PATH distance/run_pairwise_distance_c.c distance/pairwise_distance_c.cu
)
ConfigureTest(NAME KMEANS_C_TEST PATH cluster/kmeans_c.cu)
ConfigureTest(NAME SELECT_K_C_TEST PATH selection/select_k_c.cu)
if(BUILD_MG_ALGOS)
  ConfigureTest(NAME KMEANS_MG_C_TEST PATH cluster/kmeans_mg_c.cu)
endif()
ConfigureTest(NAME BRUTEFORCE_C_TEST PATH neighbors/run_brute_force_c.c neighbors/brute_force_c.cu)
ConfigureTest(NAME IVF_FLAT_C_TEST PATH neighbors/run_ivf_flat_c.c neighbors/ann_ivf_flat_c.cu)
ConfigureTest(NAME IVF_PQ_C_TEST PATH neighbors/run_ivf_pq_c.c neighbors/ann_ivf_pq_c.cu)
ConfigureTest(NAME IVF_SQ_C_TEST PATH neighbors/run_ivf_sq_c.c neighbors/ann_ivf_sq_c.cu)
ConfigureTest(NAME CAGRA_C_TEST PATH neighbors/ann_cagra_c.cu)
ConfigureTest(NAME MG_C_TEST PATH neighbors/run_mg_c.c neighbors/ann_mg_c.cu)
ConfigureTest(
  NAME ALL_NEIGHBORS_C_TEST PATH neighbors/run_all_neighbors_c.c neighbors/all_neighbors_c.cu
)

ConfigureTest(NAME PCA_C_TEST PATH preprocessing/pca_c.cu)

if(BUILD_CAGRA_HNSWLIB)
  ConfigureTest(NAME HNSW_C_TEST PATH neighbors/ann_hnsw_c.cu)
  target_link_libraries(HNSW_C_TEST PRIVATE hnswlib::hnswlib)
  target_compile_definitions(HNSW_C_TEST PUBLIC CUVS_BUILD_CAGRA_HNSWLIB)
endif()

# ##################################################################################################
# * Validate C API ---------------------------------------------------------------

ConfigureTest(NAME cuvs_c_headers PATH core/headers.c)
ConfigureTest(NAME cuvs_c_test PATH core/c_api.c)
target_link_libraries(cuvs_c_test PRIVATE CUDA::cudart_static)
ConfigureTest(NAME cuvs_c_neighbors_test PATH neighbors/c_api.c)

# ##################################################################################################
# * Validate installed headers are listed in `all.h`

include(cmake/header_check.cmake)
cuvs_c_add_header_check("${CUVS_C_SOURCE_DIR}" "cuvs/core/all.h" INSTALL_COMPONENT_SET testing)

# ##################################################################################################
# * Install Tests ---------------------------------------------------------------

if(PROJECT_IS_TOP_LEVEL)
  rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuvs)
endif()
