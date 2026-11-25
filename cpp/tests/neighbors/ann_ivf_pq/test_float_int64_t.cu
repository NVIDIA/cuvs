/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_ivf_pq.cuh"

#include <raft/random/rng.cuh>

#include <algorithm>
#include <vector>

namespace cuvs::neighbors::ivf_pq {

using f32_f32_i64             = ivf_pq_test<float, float, int64_t>;
using f32_f32_i64_flat_layout = ivf_pq_flat_layout_test<float, float, int64_t>;
using f32_f32_i64_filter      = ivf_pq_filter_test<float, float, int64_t>;

TEST_BUILD_HOST_INPUT_SEARCH(f32_f32_i64)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_f32_i64)
TEST_BUILD_EXTEND_SEARCH(f32_f32_i64)
TEST_BUILD_SERIALIZE_SEARCH(f32_f32_i64)
TEST_BUILD_PRECOMPUTED(f32_f32_i64)
INSTANTIATE(f32_f32_i64,
            defaults() + small_dims() + big_dims_moderate_lut() + enum_variety_l2() +
              enum_variety_l2sqrt() + enum_variety_ip() + enum_variety_cosine());

TEST_FLAT_LAYOUT_CODES(f32_f32_i64_flat_layout)
INSTANTIATE(f32_f32_i64_flat_layout, flat_layout_tests());

TEST_BUILD_SEARCH(f32_f32_i64_filter)
INSTANTIATE(f32_f32_i64_filter,
            defaults() + small_dims() + big_dims_moderate_lut() + enum_variety_l2() +
              enum_variety_l2sqrt() + enum_variety_ip() + enum_variety_cosine());

// Empty lists must not take probes from the non-empty ones: with n_probes equal to the number of
// non-empty lists, every indexed row is scanned, as when all lists are probed.
TEST(IvfPqTest, EmptyListsAreNotProbed)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int64_t n_rows    = 10000;
  constexpr int64_t n_indexed = 64;
  constexpr int64_t n_queries = 500;
  constexpr int64_t dim       = 8;
  constexpr uint32_t n_lists  = 256;
  constexpr uint32_t k        = 16;

  auto data    = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
  auto queries = raft::make_device_matrix<float, int64_t>(handle, n_queries, dim);
  raft::random::RngState rng(1234ULL);
  raft::random::uniform(handle, rng, data.data_handle(), data.size(), 0.1f, 2.0f);
  raft::random::uniform(handle, rng, queries.data_handle(), queries.size(), 0.1f, 2.0f);
  auto indexed =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), n_indexed, dim);

  auto search_with = [&](const index<int64_t>& idx, search_params sparams) {
    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(handle, n_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(handle, n_queries, k);
    search(handle,
           sparams,
           idx,
           raft::make_const_mdspan(queries.view()),
           neighbors.view(),
           distances.view());
    std::vector<int64_t> neighbors_h(n_queries * k);
    std::vector<float> distances_h(n_queries * k);
    raft::update_host(neighbors_h.data(), neighbors.data_handle(), neighbors_h.size(), stream);
    raft::update_host(distances_h.data(), distances.data_handle(), distances_h.size(), stream);
    raft::resource::sync_stream(handle);
    return std::make_pair(neighbors_h, distances_h);
  };

  for (auto metric : {cuvs::distance::DistanceType::L2Expanded,
                      cuvs::distance::DistanceType::InnerProduct,
                      cuvs::distance::DistanceType::CosineExpanded}) {
    index_params params;
    params.n_lists           = n_lists;
    params.metric            = metric;
    params.add_data_on_build = false;
    auto idx                 = build(handle, params, raft::make_const_mdspan(data.view()));
    extend(handle, indexed, std::nullopt, &idx);

    std::vector<uint32_t> list_sizes(n_lists);
    raft::update_host(list_sizes.data(), idx.list_sizes().data_handle(), n_lists, stream);
    raft::resource::sync_stream(handle);
    auto n_non_empty = static_cast<uint32_t>(
      std::count_if(list_sizes.begin(), list_sizes.end(), [](uint32_t s) { return s > 0; }));
    ASSERT_LE(n_non_empty, n_indexed);

    for (auto coarse_search_dtype : {CUDA_R_32F, CUDA_R_16F, CUDA_R_8I}) {
      SCOPED_TRACE(::testing::Message() << "metric " << int(metric) << ", coarse_search_dtype "
                                        << int(coarse_search_dtype));
      search_params sparams;
      sparams.coarse_search_dtype                   = coarse_search_dtype;
      sparams.n_probes                              = n_lists;
      auto [expected_neighbors, expected_distances] = search_with(idx, sparams);
      sparams.n_probes                              = n_non_empty;
      auto [neighbors, distances]                   = search_with(idx, sparams);

      EXPECT_TRUE(eval_neighbours(
        expected_neighbors, neighbors, expected_distances, distances, n_queries, k, 0.001, 0.99));
    }
  }
}

}  // namespace cuvs::neighbors::ivf_pq
