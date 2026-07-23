# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# ##################################################################################################
# enable testing ################################################################################
# ##################################################################################################
enable_testing()
include(rapids-test)
rapids_test_init()

function(ConfigureTest)

  set(options NOCUDA FETCH_CODEBOOKS RUN_SERIAL)
  set(oneValueArgs NAME GPUS PERCENT)
  set(multiValueArgs PATH ADDITIONAL_DEP)

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

  if(_CUVS_TEST_NOCUDA)
    set(TEST_NAME "${_CUVS_TEST_NAME}_NOCUDA")
  else()
    set(TEST_NAME ${_CUVS_TEST_NAME})
  endif()

  add_executable(${TEST_NAME} ${_CUVS_TEST_PATH})
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE cuvs
            cuvs::cuvs
            raft::raft
            GTest::gtest
            GTest::gtest_main
            Threads::Threads
            ${CUVS_CTK_MATH_DEPENDENCIES}
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
            ${_CUVS_TEST_ADDITIONAL_DEP}
            $<TARGET_NAME_IF_EXISTS:nvidia::cutlass::cutlass>
  )
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUVS_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_compile_options(
    ${TEST_NAME}
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
            "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${CUVS_DEBUG_CUDA_FLAGS}>"
  )

  if(_CUVS_TEST_NOCUDA)
    target_compile_definitions(${TEST_NAME} PRIVATE "CUVS_DISABLE_CUDA")
  endif()

  if(_CUVS_TEST_FETCH_CODEBOOKS)
    set(CODEBOOKS 384_int8_pq_pivots.bin 384_int8_pq_pivots.bin_rotation_matrix.bin
                  64_float_pq_pivots.bin 64_float_pq_pivots.bin_rotation_matrix.bin
    )

    foreach(CODEBOOK ${CODEBOOKS})
      file(
        DOWNLOAD
        https://data.rapids.ai/cuvs/tests/data/neighbors/ann_vamana/randomized_codebooks/${CODEBOOK}
        ${CUVS_BINARY_DIR}/tests/data/neighbors/ann_vamana/randomized_codebooks/${CODEBOOK}
      )
    endforeach()

    target_compile_definitions(${TEST_NAME} PRIVATE TEST_DATA_DIR="${CUVS_BINARY_DIR}/tests/data/")
  endif()

  target_include_directories(${TEST_NAME} PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/tests>")

  rapids_test_add(
    NAME ${TEST_NAME}
    COMMAND ${TEST_NAME}
    GPUS ${_CUVS_TEST_GPUS}
    PERCENT ${_CUVS_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )

  if(_CUVS_TEST_RUN_SERIAL)
    set_tests_properties(${TEST_NAME} PROPERTIES RUN_SERIAL ON)
  endif()
endfunction()

# ##################################################################################################
# test sources ##################################################################################
# ##################################################################################################

ConfigureTest(
  NAME NEIGHBORS_TEST
  PATH neighbors/brute_force.cu neighbors/brute_force_prefiltered.cu neighbors/sparse_brute_force.cu
       neighbors/refine.cu neighbors/distance_nn.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_TIERED_INDEX_TEST
  PATH neighbors/tiered_index.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME CLUSTER_TEST
  PATH cluster/kmeans.cu cluster/kmeans_balanced.cu cluster/kmeans_find_k.cu cluster/linkage.cu
       cluster/connect_knn.cu cluster/spectral.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_IVF_FLAT_UDF_TEST
  PATH neighbors/ann_ivf_flat/test_udf.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_IVF_FLAT_TEST
  PATH neighbors/ann_ivf_flat/test_float_int64_t.cu neighbors/ann_ivf_flat/test_half_int64_t.cu
       neighbors/ann_ivf_flat/test_int8_t_int64_t.cu neighbors/ann_ivf_flat/test_uint8_t_int64_t.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_IVF_SQ_TEST
  PATH neighbors/ann_ivf_sq/test_float_int64_t.cu neighbors/ann_ivf_sq/test_half_int64_t.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_IVF_PQ_TEST
  PATH neighbors/ann_ivf_pq/test_float_int64_t.cu neighbors/ann_ivf_pq/test_int8_t_int64_t.cu
       neighbors/ann_ivf_pq/test_uint8_t_int64_t.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_IVF_RABITQ_TEST
  PATH neighbors/ann_ivf_rabitq/test_float_int64_t.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_SCANN_TEST
  PATH neighbors/ann_scann/test_float_int64_t.cu
  GPUS 1
  PERCENT 1
)

ConfigureTest(
  NAME NEIGHBORS_ANN_BRUTE_FORCE_TEST
  PATH neighbors/ann_brute_force/test_float.cu neighbors/ann_brute_force/test_half.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_CAGRA_TEST_BUGS
  PATH neighbors/ann_cagra/bug_extreme_inputs_oob.cu
       neighbors/ann_cagra/bug_multi_cta_crash.cu
       neighbors/ann_cagra/bug_graph_smaller_than_dataset.cu
       neighbors/ann_cagra/bug_iterative_cagra_build.cu
       neighbors/ann_cagra/bug_issue_93_reproducer.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST
  PATH neighbors/ann_cagra/test_float_uint32_t.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST
  PATH neighbors/ann_cagra/test_filter_udf.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_CAGRA_HELPERS_TEST
  PATH neighbors/ann_cagra/test_optimize_uint32_t.cu neighbors/ann_cagra/test_batch_load_iterator.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST
  PATH neighbors/ann_cagra/test_half_uint32_t.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST
  PATH neighbors/ann_cagra/test_int8_t_uint32_t.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST
  PATH neighbors/ann_cagra/test_uint8_t_uint32_t.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_NN_DESCENT_TEST
  PATH neighbors/ann_nn_descent/test_float_uint32_t.cu
       neighbors/ann_nn_descent/test_int8_t_uint32_t.cu
       neighbors/ann_nn_descent/test_uint8_t_uint32_t.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ANN_VAMANA_TEST
  PATH neighbors/ann_vamana/test_float_uint32_t.cu neighbors/ann_vamana/test_half_uint32_t.cu
       neighbors/ann_vamana/test_int8_t_uint32_t.cu neighbors/ann_vamana/test_uint8_t_uint32_t.cu
  GPUS 1
  PERCENT 100
  FETCH_CODEBOOKS
)

ConfigureTest(
  NAME NEIGHBORS_BALL_COVER_TEST
  PATH neighbors/ball_cover.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_EPSILON_NEIGHBORHOOD_TEST
  PATH neighbors/epsilon_neighborhood.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_ALL_NEIGHBORS_TEST
  PATH neighbors/all_neighbors/test_float.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME NEIGHBORS_DYNAMIC_BATCHING_TEST
  PATH neighbors/dynamic_batching/test_brute_force.cu neighbors/dynamic_batching/test_cagra.cu
       neighbors/dynamic_batching/test_ivf_flat.cu neighbors/dynamic_batching/test_ivf_pq.cu
  GPUS 1
  PERCENT 100
)

if(BUILD_CAGRA_HNSWLIB)
  ConfigureTest(
    NAME NEIGHBORS_HNSW_TEST
    PATH neighbors/hnsw.cu
    GPUS 1
    PERCENT 100
  )
  target_link_libraries(NEIGHBORS_HNSW_TEST PRIVATE hnswlib::hnswlib)
  target_compile_definitions(NEIGHBORS_HNSW_TEST PUBLIC CUVS_BUILD_CAGRA_HNSWLIB)

  ConfigureTest(
    NAME NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST
    PATH neighbors/ann_hnsw_ace/test_float_uint32_t.cu
    GPUS 1
    PERCENT 100
  )
  target_link_libraries(NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST PRIVATE hnswlib::hnswlib)
  target_compile_definitions(
    NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST PUBLIC CUVS_BUILD_CAGRA_HNSWLIB
  )

  ConfigureTest(
    NAME NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST
    PATH neighbors/ann_hnsw_ace/test_half_uint32_t.cu
    GPUS 1
    PERCENT 100
  )
  target_link_libraries(NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST PRIVATE hnswlib::hnswlib)
  target_compile_definitions(
    NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST PUBLIC CUVS_BUILD_CAGRA_HNSWLIB
  )

  ConfigureTest(
    NAME NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST
    PATH neighbors/ann_hnsw_ace/test_int8_t_uint32_t.cu
    GPUS 1
    PERCENT 100
  )
  target_link_libraries(NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST PRIVATE hnswlib::hnswlib)
  target_compile_definitions(
    NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST PUBLIC CUVS_BUILD_CAGRA_HNSWLIB
  )

  ConfigureTest(
    NAME NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST
    PATH neighbors/ann_hnsw_ace/test_uint8_t_uint32_t.cu
    GPUS 1
    PERCENT 100
  )
  target_link_libraries(NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST PRIVATE hnswlib::hnswlib)
  target_compile_definitions(
    NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST PUBLIC CUVS_BUILD_CAGRA_HNSWLIB
  )
endif()

if(BUILD_MG_ALGOS)
  ConfigureTest(
    NAME NEIGHBORS_MG_TEST
    PATH neighbors/mg/test_float.cu
    GPUS 1
    PERCENT 100
    ADDITIONAL_DEP NCCL::NCCL RUN_SERIAL
  )

  ConfigureTest(
    NAME CLUSTER_KMEANS_MG_TEST
    PATH cluster/kmeans_mg.cu
    GPUS 1
    PERCENT 100
    ADDITIONAL_DEP NCCL::NCCL RUN_SERIAL
  )

  # The MNMG (multi-node multi-GPU) kmeans test depends on the distributed (UCXX) dependencies.
  if(BUILD_MNMG_TESTS)
    if(NOT TARGET ucx::ucp)
      find_package(ucx REQUIRED)
    endif()

    if(NOT TARGET ucxx::ucxx)
      find_package(ucxx REQUIRED)
    endif()

    ConfigureTest(
      NAME CLUSTER_KMEANS_MNMG_TEST
      PATH cluster/kmeans_mnmg.cu
      GPUS 1
      PERCENT 100
      ADDITIONAL_DEP
        NCCL::NCCL
        ucx::ucp
        ucx::ucs
        ucxx::ucxx
        RUN_SERIAL
    )
  endif()
endif()

ConfigureTest(
  NAME DISTANCE_TEST
  PATH distance/dist_canberra.cu
       distance/dist_correlation.cu
       distance/dist_cos.cu
       distance/dist_hamming.cu
       distance/dist_hellinger.cu
       distance/dist_inner_product.cu
       distance/dist_jensen_shannon.cu
       distance/dist_kl_divergence.cu
       distance/dist_l1.cu
       distance/dist_l2_exp.cu
       distance/dist_l2_sqrt_exp.cu
       distance/dist_l_inf.cu
       distance/dist_lp_unexp.cu
       distance/dist_russell_rao.cu
       distance/gram.cu
       distance/kde.cu
       distance/masked_nn.cu
       distance/sparse_distance.cu
       sparse/gram.cu
       sparse/neighbors/cross_component_nn.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME PREPROCESSING_TEST
  PATH preprocessing/scalar_quantization.cu
       preprocessing/binary_quantization.cu
       preprocessing/spectral_embedding.cu
       preprocessing/product_quantization.cu
       preprocessing/pca.cu
  GPUS 1
  PERCENT 100
)

ConfigureTest(
  NAME STATS_TEST
  PATH stats/trustworthiness.cu stats/silhouette_score.cu
  GPUS 1
  PERCENT 100
)

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuvs)
