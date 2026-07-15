/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "naive_knn.cuh"
#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/linalg/add.cuh>
#include <raft/matrix/gather.cuh>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <thrust/sequence.h>

namespace cuvs::neighbors::ivf_pq {

struct test_ivf_sample_filter {
  static constexpr unsigned offset = 300;
};

struct ivf_pq_inputs {
  uint32_t num_db_vecs             = 4096;
  uint32_t num_queries             = 1024;
  uint32_t dim                     = 64;
  uint32_t k                       = 32;
  std::optional<double> min_recall = std::nullopt;

  cuvs::neighbors::ivf_pq::index_params index_params;
  cuvs::neighbors::ivf_pq::search_params search_params;

  // Set some default parameters for tests
  ivf_pq_inputs()
  {
    index_params.n_lists                  = max(32u, min(1024u, num_db_vecs / 128u));
    index_params.kmeans_trainset_fraction = 1.0;
  }
};

inline auto operator<<(std::ostream& os, const ivf_pq::codebook_gen& p) -> std::ostream&
{
  switch (p) {
    case ivf_pq::codebook_gen::PER_CLUSTER: os << "codebook_gen::PER_CLUSTER"; break;
    case ivf_pq::codebook_gen::PER_SUBSPACE: os << "codebook_gen::PER_SUBSPACE"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

inline auto operator<<(std::ostream& os, const ivf_pq_inputs& p) -> std::ostream&
{
  ivf_pq_inputs dflt;
  bool need_comma = false;
#define PRINT_DIFF_V(spec, val)       \
  do {                                \
    if (dflt spec != p spec) {        \
      if (need_comma) { os << ", "; } \
      os << #spec << " = " << val;    \
      need_comma = true;              \
    }                                 \
  } while (0)
#define PRINT_DIFF(spec) PRINT_DIFF_V(spec, p spec)

  os << "ivf_pq_inputs {";
  PRINT_DIFF(.num_db_vecs);
  PRINT_DIFF(.num_queries);
  PRINT_DIFF(.dim);
  PRINT_DIFF(.k);
  PRINT_DIFF_V(.min_recall, p.min_recall.value_or(0));
  PRINT_DIFF_V(.index_params.metric,
               cuvs::neighbors::print_metric{
                 static_cast<cuvs::distance::DistanceType>((int)p.index_params.metric)});
  PRINT_DIFF(.index_params.metric_arg);
  PRINT_DIFF(.index_params.add_data_on_build);
  PRINT_DIFF(.index_params.n_lists);
  PRINT_DIFF(.index_params.kmeans_n_iters);
  PRINT_DIFF(.index_params.kmeans_trainset_fraction);
  PRINT_DIFF(.index_params.pq_bits);
  PRINT_DIFF(.index_params.pq_dim);
  PRINT_DIFF(.index_params.codebook_kind);
  PRINT_DIFF(.index_params.force_random_rotation);
  PRINT_DIFF(.search_params.n_probes);
  PRINT_DIFF_V(.search_params.lut_dtype, cuvs::neighbors::print_dtype{p.search_params.lut_dtype});
  PRINT_DIFF_V(.search_params.internal_distance_dtype,
               cuvs::neighbors::print_dtype{p.search_params.internal_distance_dtype});
  PRINT_DIFF_V(.search_params.coarse_search_dtype,
               cuvs::neighbors::print_dtype{p.search_params.coarse_search_dtype});
  PRINT_DIFF(.search_params.max_internal_batch_size);
  os << "}";
  return os;
}

template <typename T>
void compare_vectors_l2(
  const raft::resources& res, T a, T b, uint32_t label, double compression_ratio, double eps)
{
  auto n_rows = a.extent(0);
  auto dim    = a.extent(1);
  rmm::mr::managed_memory_resource managed_memory;
  auto dist =
    raft::make_device_mdarray<double>(res, managed_memory, raft::make_extents<uint32_t>(n_rows));
  raft::linalg::map_offset(res, dist.view(), [a, b, dim] __device__(uint32_t i) {
    cuvs::spatial::knn::detail::utils::mapping<float> f{};
    double d = 0.0f;
    for (uint32_t j = 0; j < dim; j++) {
      double t = f(a(i, j)) - f(b(i, j));
      d += t * t;
    }
    return sqrt(d / double(dim));
  });
  raft::resource::sync_stream(res);
  for (uint32_t i = 0; i < n_rows; i++) {
    double d = dist(i);
    // The theoretical estimate of the error is hard to come up with,
    // the estimate below is based on experimentation + curse of dimensionality
    ASSERT_LE(d, 1.2 * eps * std::pow(2.0, compression_ratio))
      << " (label = " << label << ", ix = " << i << ", eps = " << eps << ")";
  }
}

template <typename IdxT>
auto min_output_size(const raft::resources& handle,
                     const ivf_pq::index<IdxT>& index,
                     uint32_t n_probes) -> IdxT
{
  auto acc_sizes        = index.accum_sorted_sizes();
  uint32_t last_nonzero = index.n_lists();
  while (last_nonzero > 0 && acc_sizes(last_nonzero - 1) == acc_sizes(last_nonzero)) {
    last_nonzero--;
  }
  return acc_sizes(last_nonzero) - acc_sizes(last_nonzero - std::min(last_nonzero, n_probes));
}

template <typename EvalT, typename DataT, typename IdxT>
class ivf_pq_test : public ::testing::TestWithParam<ivf_pq_inputs> {
 public:
  ivf_pq_test()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<ivf_pq_inputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

  void gen_data()
  {
    database.resize(size_t{ps.num_db_vecs} * size_t{ps.dim}, stream_);
    search_queries.resize(size_t{ps.num_queries} * size_t{ps.dim}, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20));
    }
    raft::resource::sync_stream(handle_);
  }

  void calc_ref()
  {
    size_t queries_size = size_t{ps.num_queries} * size_t{ps.k};
    rmm::device_uvector<EvalT> distances_naive_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
    cuvs::neighbors::naive_knn<EvalT, DataT, int64_t>(
      handle_,
      distances_naive_dev.data(),
      indices_naive_dev.data(),
      search_queries.data(),
      database.data(),
      ps.num_queries,
      ps.num_db_vecs,
      ps.dim,
      ps.k,
      static_cast<cuvs::distance::DistanceType>((int)ps.index_params.metric));
    distances_ref.resize(queries_size);
    raft::update_host(distances_ref.data(), distances_naive_dev.data(), queries_size, stream_);
    indices_ref.resize(queries_size);
    raft::update_host(indices_ref.data(), indices_naive_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);
  }

  auto build_only()
  {
    auto ipams              = ps.index_params;
    ipams.add_data_on_build = true;

    auto index_view =
      raft::make_device_matrix_view<const DataT, int64_t>(database.data(), ps.num_db_vecs, ps.dim);
    return cuvs::neighbors::ivf_pq::build(handle_, ipams, index_view);
  }

  auto build_only_host_input()
  {
    auto ipams              = ps.index_params;
    ipams.add_data_on_build = true;

    auto host_database = raft::make_host_matrix<DataT, int64_t>(ps.num_db_vecs, ps.dim);
    raft::copy(host_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);
    auto index_view = raft::make_host_matrix_view<const DataT, int64_t>(
      host_database.data_handle(), ps.num_db_vecs, ps.dim);
    return cuvs::neighbors::ivf_pq::build(handle_, ipams, index_view);
  }

  auto build_only_host_input_overlap()
  {
    auto ipams              = ps.index_params;
    ipams.add_data_on_build = true;

    size_t n_streams = 1;
    raft::resource::set_cuda_stream_pool(handle_,
                                         std::make_shared<rmm::cuda_stream_pool>(n_streams));
    auto host_database = raft::make_host_matrix<DataT, int64_t>(ps.num_db_vecs, ps.dim);
    raft::copy(host_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);
    auto index_view = raft::make_host_matrix_view<const DataT, int64_t>(
      host_database.data_handle(), ps.num_db_vecs, ps.dim);
    return cuvs::neighbors::ivf_pq::build(handle_, ipams, index_view);
  }

  auto build_2_extends()
  {
    auto db_indices = raft::make_device_vector<IdxT>(handle_, ps.num_db_vecs);
    raft::linalg::map_offset(handle_, db_indices.view(), raft::identity_op{});
    raft::resource::sync_stream(handle_);
    auto size_1 = IdxT(ps.num_db_vecs) / 2;
    auto size_2 = IdxT(ps.num_db_vecs) - size_1;
    auto vecs_1 = database.data();
    auto vecs_2 = database.data() + size_t(size_1) * size_t(ps.dim);
    auto inds_1 = db_indices.data_handle();
    auto inds_2 = db_indices.data_handle() + size_t(size_1);

    auto ipams              = ps.index_params;
    ipams.add_data_on_build = false;

    auto database_view =
      raft::make_device_matrix_view<const DataT, int64_t>(database.data(), ps.num_db_vecs, ps.dim);
    auto idx = cuvs::neighbors::ivf_pq::build(handle_, ipams, database_view);

    auto vecs_2_view = raft::make_device_matrix_view<const DataT, int64_t>(vecs_2, size_2, ps.dim);
    auto inds_2_view = raft::make_device_vector_view<const IdxT, int64_t>(inds_2, size_2);
    cuvs::neighbors::ivf_pq::extend(handle_, vecs_2_view, inds_2_view, &idx);

    auto vecs_1_view =
      raft::make_device_matrix_view<const DataT, int64_t, raft::row_major>(vecs_1, size_1, ps.dim);
    auto inds_1_view = raft::make_device_vector_view<const IdxT, int64_t>(inds_1, size_1);
    cuvs::neighbors::ivf_pq::extend(handle_, vecs_1_view, inds_1_view, &idx);
    return idx;
  }

  auto build_serialize()
  {
    tmp_index_file index_file;
    cuvs::neighbors::ivf_pq::serialize(handle_, index_file.filename, build_only());
    cuvs::neighbors::ivf_pq::index<IdxT> index(handle_);
    cuvs::neighbors::ivf_pq::deserialize(handle_, index_file.filename, &index);
    return index;
  }

  void build_precomputed()
  {
    auto ipams              = ps.index_params;
    ipams.add_data_on_build = false;
    auto database_view =
      raft::make_device_matrix_view<const DataT, int64_t>(database.data(), ps.num_db_vecs, ps.dim);
    const auto& base_index = cuvs::neighbors::ivf_pq::build(handle_, ipams, database_view);

    auto view_index = cuvs::neighbors::ivf_pq::build(handle_,
                                                     ipams,
                                                     base_index.dim(),
                                                     base_index.pq_centers(),
                                                     base_index.centers(),
                                                     base_index.centers_rot(),
                                                     base_index.rotation_matrix());

    ASSERT_EQ(base_index.pq_centers().data_handle(), view_index.pq_centers().data_handle());
    ASSERT_EQ(base_index.centers().data_handle(), view_index.centers().data_handle());
    ASSERT_EQ(base_index.centers_rot().data_handle(), view_index.centers_rot().data_handle());
    ASSERT_EQ(base_index.rotation_matrix().data_handle(),
              view_index.rotation_matrix().data_handle());

    ASSERT_EQ(base_index.pq_centers().extents(), view_index.pq_centers().extents());
    ASSERT_EQ(base_index.centers().extents(), view_index.centers().extents());
    ASSERT_EQ(base_index.centers_rot().extents(), view_index.centers_rot().extents());
    ASSERT_EQ(base_index.rotation_matrix().extents(), view_index.rotation_matrix().extents());

    auto db_indices = raft::make_device_vector<IdxT>(handle_, ps.num_db_vecs);
    raft::linalg::map_offset(handle_, db_indices.view(), raft::identity_op{});

    auto vecs_view =
      raft::make_device_matrix_view<const DataT, int64_t>(database.data(), ps.num_db_vecs, ps.dim);
    auto inds_view =
      raft::make_device_vector_view<const IdxT, int64_t>(db_indices.data_handle(), ps.num_db_vecs);

    cuvs::neighbors::ivf_pq::extend(handle_, vecs_view, inds_view, &view_index);
    cuvs::neighbors::ivf_pq::extend(handle_,
                                    vecs_view,
                                    inds_view,
                                    const_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(&base_index));

    // Verify that both indices have identical list sizes after extension
    ASSERT_TRUE(cuvs::devArrMatch(base_index.list_sizes().data_handle(),
                                  view_index.list_sizes().data_handle(),
                                  base_index.n_lists(),
                                  cuvs::Compare<uint32_t>{}));
  }

  void check_reconstruction(const index<IdxT>& index,
                            double compression_ratio,
                            uint32_t label,
                            uint32_t n_take,
                            uint32_t n_skip)
  {
    // the original data cannot be reconstructed since the dataset was normalized
    if (index.metric() == cuvs::distance::DistanceType::CosineExpanded) { return; }
    auto& rec_list_base = index.lists()[label];
    // If the data is unbalanced the list might be empty, which is actually nullptr
    if (!rec_list_base) { return; }
    auto rec_list = std::static_pointer_cast<list_data_interleaved<IdxT>>(rec_list_base);
    auto dim      = index.dim();
    n_take        = std::min<uint32_t>(n_take, rec_list->size.load());
    n_skip        = std::min<uint32_t>(n_skip, rec_list->size.load() - n_take);

    if (n_take == 0) { return; }

    auto rec_data  = raft::make_device_matrix<DataT>(handle_, n_take, dim);
    auto orig_data = raft::make_device_matrix<DataT>(handle_, n_take, dim);

    ivf_pq::helpers::codepacker::reconstruct_list_data(
      handle_, index, rec_data.view(), label, n_skip);

    raft::matrix::gather(database.data(),
                         IdxT{dim},
                         IdxT{n_take},
                         rec_list->indices.data_handle() + n_skip,
                         IdxT{n_take},
                         orig_data.data_handle(),
                         stream_);

    compare_vectors_l2(handle_, rec_data.view(), orig_data.view(), label, compression_ratio, 0.06);
  }

  void check_reconstruct_extend(index<IdxT>* index, double compression_ratio, uint32_t label)
  {
    // NB: this is not reference, the list is retained; the index will have to create a new list on
    // `erase_list` op.
    auto old_list_base = index->lists()[label];
    // If the data is unbalanced the list might be empty, which is actually nullptr
    if (!old_list_base) { return; }
    auto old_list = std::static_pointer_cast<list_data_interleaved<IdxT>>(old_list_base);
    auto n_rows   = old_list->size.load();
    if (n_rows == 0) { return; }
    if (index->metric() == cuvs::distance::DistanceType::CosineExpanded) { return; }

    auto vectors_1 = raft::make_device_matrix<EvalT>(handle_, n_rows, index->dim());
    auto indices   = raft::make_device_vector<IdxT>(handle_, n_rows);
    raft::copy(indices.data_handle(), old_list->indices.data_handle(), n_rows, stream_);

    ivf_pq::helpers::codepacker::reconstruct_list_data(
      handle_, *index, vectors_1.view(), label, uint32_t(0));
    ivf_pq::helpers::erase_list(handle_, index, label);
    // NB: passing the type parameter because const->non-const implicit conversion of the mdspans
    // breaks type inference
    ivf_pq::helpers::codepacker::extend_list(
      handle_, index, vectors_1.view(), indices.view(), label);

    auto& new_list = index->lists()[label];
    ASSERT_NE(old_list.get(), new_list.get())
      << "The old list should have been shared and retained after ivf_pq index has erased the "
         "corresponding cluster.";

    auto vectors_2 = raft::make_device_matrix<EvalT>(handle_, n_rows, index->dim());
    ivf_pq::helpers::codepacker::reconstruct_list_data(
      handle_, *index, vectors_2.view(), label, uint32_t(0));
    // The code search is unstable, and there's high chance of repeating values of the lvl-2 codes.
    // Hence, encoding-decoding chain often leads to altering both the PQ codes and the
    // reconstructed data.
    compare_vectors_l2(
      handle_, vectors_1.view(), vectors_2.view(), label, compression_ratio, 0.04);  // 0.025);
  }

  void check_packing(index<IdxT>* index, uint32_t label)
  {
    auto old_list_base = index->lists()[label];
    // If the data is unbalanced the list might be empty, which is actually nullptr
    if (!old_list_base) { return; }
    auto old_list = std::static_pointer_cast<list_data_interleaved<IdxT>>(old_list_base);
    auto n_rows   = old_list->size.load();

    if (n_rows == 0) { return; }

    auto codes   = raft::make_device_matrix<uint8_t>(handle_, n_rows, index->pq_dim());
    auto indices = raft::make_device_vector<IdxT>(handle_, n_rows);
    raft::copy(indices.data_handle(), old_list->indices.data_handle(), n_rows, stream_);

    ivf_pq::helpers::codepacker::unpack_list_data(
      handle_, *index, codes.view(), label, uint32_t(0));
    ivf_pq::helpers::erase_list(handle_, index, label);
    ivf_pq::helpers::codepacker::extend_list_with_codes(
      handle_, index, codes.view(), indices.view(), label);

    auto new_list = std::static_pointer_cast<list_data_interleaved<IdxT>>(index->lists()[label]);
    ASSERT_NE(old_list.get(), new_list.get())
      << "The old list should have been shared and retained after ivf_pq index has erased the "
         "corresponding cluster.";
    auto list_data_byte_size = (n_rows / cuvs::neighbors::ivf_pq::kIndexGroupSize) *
                               new_list->data.extent(1) * new_list->data.extent(2) *
                               new_list->data.extent(3);

    ASSERT_TRUE(old_list->data.size() >= list_data_byte_size);
    ASSERT_TRUE(new_list->data.size() >= list_data_byte_size);
    ASSERT_TRUE(cuvs::devArrMatch(old_list->data.data_handle(),
                                  new_list->data.data_handle(),
                                  list_data_byte_size,
                                  cuvs::Compare<uint8_t>{}));

    // Pack a few vectors back to the list.
    int row_offset = 5;
    int n_vec      = 3;
    if (static_cast<decltype(n_rows)>(row_offset + n_vec) > n_rows) {
      RAFT_LOG_INFO(
        "Skipping IVF-PQ check_packing/pack test for label %u due to insufficient data (%u "
        "records)",
        label,
        uint32_t(n_rows));
      return;
    }
    size_t offset      = row_offset * index->pq_dim();
    auto codes_to_pack = raft::make_device_matrix_view<const uint8_t, uint32_t>(
      codes.data_handle() + offset, n_vec, index->pq_dim());
    ivf_pq::helpers::codepacker::pack_list_data(
      handle_, index, codes_to_pack, label, uint32_t(row_offset));
    ASSERT_TRUE(cuvs::devArrMatch(old_list->data.data_handle(),
                                  new_list->data.data_handle(),
                                  list_data_byte_size,
                                  cuvs::Compare<uint8_t>{}));

    // Another test with the API that take list_data directly
    [[maybe_unused]] auto list_data_view =
      std::static_pointer_cast<list_data_interleaved<IdxT>>(index->lists()[label])->data.view();
    uint32_t n_take = 4;
    if (static_cast<decltype(n_rows)>(row_offset + n_take) > n_rows) {
      RAFT_LOG_INFO(
        "Skipping IVF-PQ check_packing/take test for label %u due to insufficient data (%u "
        "records)",
        label,
        uint32_t(n_rows));
      return;
    }
    auto codes2 = raft::make_device_matrix<uint8_t>(handle_, n_take, index->pq_dim());
    ivf_pq::helpers::codepacker::unpack(
      handle_, list_data_view, index->pq_bits(), row_offset, codes2.view());

    // Write it back
    ivf_pq::helpers::codepacker::pack(
      handle_, make_const_mdspan(codes2.view()), index->pq_bits(), row_offset, list_data_view);
    ASSERT_TRUE(cuvs::devArrMatch(old_list->data.data_handle(),
                                  new_list->data.data_handle(),
                                  list_data_byte_size,
                                  cuvs::Compare<uint8_t>{}));
  }

  void check_flat_layout_codes()
  {
    auto database_view =
      raft::make_device_matrix_view<const DataT, int64_t>(database.data(), ps.num_db_vecs, ps.dim);

    // Step 1: Build INTERLEAVED index WITHOUT data (just train the codebooks)
    auto ipams_interleaved              = ps.index_params;
    ipams_interleaved.add_data_on_build = false;
    ipams_interleaved.codes_layout      = list_layout::INTERLEAVED;
    auto index_interleaved =
      cuvs::neighbors::ivf_pq::build(handle_, ipams_interleaved, database_view);

    // Step 2: Build FLAT index using the SAME precomputed centers from the interleaved index
    auto ipams_flat              = ps.index_params;
    ipams_flat.add_data_on_build = false;
    ipams_flat.codes_layout      = list_layout::FLAT;
    auto index_flat              = cuvs::neighbors::ivf_pq::build(handle_,
                                                     ipams_flat,
                                                     index_interleaved.dim(),
                                                     index_interleaved.pq_centers(),
                                                     index_interleaved.centers(),
                                                     index_interleaved.centers_rot(),
                                                     index_interleaved.rotation_matrix());

    ASSERT_EQ(index_interleaved.codes_layout(), list_layout::INTERLEAVED);
    ASSERT_EQ(index_flat.codes_layout(), list_layout::FLAT);

    // Step 3: Extend both indexes with the same data
    auto indices = raft::make_device_vector<IdxT, int64_t>(handle_, ps.num_db_vecs);
    raft::linalg::map_offset(handle_, indices.view(), raft::identity_op{});
    cuvs::neighbors::ivf_pq::extend(
      handle_, database_view, std::make_optional(indices.view()), &index_interleaved);
    cuvs::neighbors::ivf_pq::extend(
      handle_, database_view, std::make_optional(indices.view()), &index_flat);

    ASSERT_TRUE(cuvs::devArrMatch(index_interleaved.list_sizes().data_handle(),
                                  index_flat.list_sizes().data_handle(),
                                  index_interleaved.n_lists(),
                                  cuvs::Compare<uint32_t>{}));

    auto ipams_from_flat              = ps.index_params;
    ipams_from_flat.add_data_on_build = false;
    ipams_from_flat.codes_layout      = list_layout::INTERLEAVED;
    auto index_from_flat              = cuvs::neighbors::ivf_pq::build(handle_,
                                                          ipams_from_flat,
                                                          index_interleaved.dim(),
                                                          index_interleaved.pq_centers(),
                                                          index_interleaved.centers(),
                                                          index_interleaved.centers_rot(),
                                                          index_interleaved.rotation_matrix());

    std::vector<uint32_t> list_sizes_host(index_flat.n_lists());
    raft::update_host(
      list_sizes_host.data(), index_flat.list_sizes().data_handle(), index_flat.n_lists(), stream_);
    raft::resource::sync_stream(handle_);

    // Step 4: Pack flat codes into the new interleaved index
    uint32_t bytes_per_vec =
      raft::div_rounding_up_safe(index_flat.pq_dim() * index_flat.pq_bits(), 8u);

    for (uint32_t label = 0; label < index_flat.n_lists(); label++) {
      uint32_t n_rows = list_sizes_host[label];
      if (n_rows == 0) { continue; }

      auto& list      = index_flat.lists()[label];
      auto codes_view = raft::make_device_matrix_view<const uint8_t, uint32_t>(
        list->data_ptr(), n_rows, bytes_per_vec);
      auto indices_view =
        raft::make_device_vector_view<const IdxT, uint32_t>(list->indices_ptr(), n_rows);

      helpers::codepacker::extend_list_with_contiguous_codes(
        handle_, &index_from_flat, codes_view, indices_view, label);
    }

    size_t queries_size = ps.num_queries * ps.k;
    rmm::device_uvector<IdxT> indices_original(queries_size, stream_);
    rmm::device_uvector<EvalT> distances_original(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_from_flat(queries_size, stream_);
    rmm::device_uvector<EvalT> distances_from_flat(queries_size, stream_);

    auto query_view =
      raft::make_device_matrix_view<DataT, uint32_t>(search_queries.data(), ps.num_queries, ps.dim);
    auto inds_orig_view =
      raft::make_device_matrix_view<IdxT, uint32_t>(indices_original.data(), ps.num_queries, ps.k);
    auto dists_orig_view = raft::make_device_matrix_view<EvalT, uint32_t>(
      distances_original.data(), ps.num_queries, ps.k);
    auto inds_flat_view =
      raft::make_device_matrix_view<IdxT, uint32_t>(indices_from_flat.data(), ps.num_queries, ps.k);
    auto dists_flat_view = raft::make_device_matrix_view<EvalT, uint32_t>(
      distances_from_flat.data(), ps.num_queries, ps.k);

    cuvs::neighbors::ivf_pq::search(
      handle_, ps.search_params, index_interleaved, query_view, inds_orig_view, dists_orig_view);
    cuvs::neighbors::ivf_pq::search(
      handle_, ps.search_params, index_from_flat, query_view, inds_flat_view, dists_flat_view);

    // Copy results to host for eval_neighbours
    std::vector<IdxT> indices_orig_host(queries_size);
    std::vector<IdxT> indices_flat_host(queries_size);
    std::vector<EvalT> distances_orig_host(queries_size);
    std::vector<EvalT> distances_flat_host(queries_size);
    raft::update_host(indices_orig_host.data(), indices_original.data(), queries_size, stream_);
    raft::update_host(indices_flat_host.data(), indices_from_flat.data(), queries_size, stream_);
    raft::update_host(distances_orig_host.data(), distances_original.data(), queries_size, stream_);
    raft::update_host(
      distances_flat_host.data(), distances_from_flat.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);

    // Evaluate recall - should be very high since codes are identical
    ASSERT_TRUE(eval_neighbours(indices_orig_host,
                                indices_flat_host,
                                distances_orig_host,
                                distances_flat_host,
                                ps.num_queries,
                                ps.k,
                                0.001,  // eps
                                0.99))  // min_recall
      << "Recall mismatch between original interleaved and repacked-from-flat indexes";
  }

  template <typename BuildIndex>
  void run(BuildIndex build_index)
  {
    index<IdxT> index = build_index();

    double compression_ratio =
      static_cast<double>(ps.dim * 8) / static_cast<double>(index.pq_dim() * index.pq_bits());

    for (uint32_t label = 0; label < index.n_lists(); label++) {
      switch (label % 3) {
        case 0: {
          // Reconstruct and re-write vectors for one label
          check_reconstruct_extend(&index, compression_ratio, label);
        } break;
        case 1: {
          // Dump and re-write codes for one label
          check_packing(&index, label);
        } break;
        default: {
          // check a small subset of data in a randomly chosen cluster to see if the data
          // reconstruction works well.
          check_reconstruction(index, compression_ratio, label, 100, 7);
        }
      }
    }

    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivf_pq(queries_size);
    std::vector<EvalT> distances_ivf_pq(queries_size);

    rmm::device_uvector<EvalT> distances_ivf_pq_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_ivf_pq_dev(queries_size, stream_);

    auto query_view =
      raft::make_device_matrix_view<DataT, uint32_t>(search_queries.data(), ps.num_queries, ps.dim);
    auto inds_view = raft::make_device_matrix_view<IdxT, uint32_t>(
      indices_ivf_pq_dev.data(), ps.num_queries, ps.k);
    auto dists_view = raft::make_device_matrix_view<EvalT, uint32_t>(
      distances_ivf_pq_dev.data(), ps.num_queries, ps.k);

    cuvs::neighbors::ivf_pq::search(
      handle_, ps.search_params, index, query_view, inds_view, dists_view);

    raft::update_host(distances_ivf_pq.data(), distances_ivf_pq_dev.data(), queries_size, stream_);
    raft::update_host(indices_ivf_pq.data(), indices_ivf_pq_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);

    // A very conservative lower bound on recall
    double min_recall =
      static_cast<double>(ps.search_params.n_probes) / static_cast<double>(ps.index_params.n_lists);
    // Using a heuristic to lower the required recall due to code-packing errors
    min_recall =
      std::min(std::erfc(0.05 * compression_ratio / std::max(min_recall, 0.5)), min_recall);
    // Use explicit per-test min recall value if provided.
    min_recall = ps.min_recall.value_or(min_recall);

    ASSERT_TRUE(cuvs::neighbors::eval_neighbours(indices_ref,
                                                 indices_ivf_pq,
                                                 distances_ref,
                                                 distances_ivf_pq,
                                                 ps.num_queries,
                                                 ps.k,
                                                 0.0001 * compression_ratio,
                                                 min_recall))
      << ps;

    // Test a few extra invariants
    IdxT min_results = min_output_size(handle_, index, ps.search_params.n_probes);
    IdxT max_oob     = ps.k <= min_results ? 0 : ps.k - min_results;
    IdxT found_oob   = 0;
    for (uint32_t query_ix = 0; query_ix < ps.num_queries; query_ix++) {
      for (uint32_t k = 0; k < ps.k; k++) {
        auto flat_i   = query_ix * ps.k + k;
        auto found_ix = indices_ivf_pq[flat_i];
        if (found_ix == cuvs::neighbors::ivf_pq::kOutOfBoundsRecord<IdxT>) {
          found_oob++;
          continue;
        }
        ASSERT_NE(found_ix, cuvs::neighbors::ivf::kInvalidRecord<IdxT>)
          << "got an invalid record at query_ix = " << query_ix << ", k = " << k
          << " (distance = " << distances_ivf_pq[flat_i] << ")";
        ASSERT_LT(found_ix, ps.num_db_vecs)
          << "got an impossible index = " << found_ix << " at query_ix = " << query_ix
          << ", k = " << k << " (distance = " << distances_ivf_pq[flat_i] << ")";
      }
    }
    ASSERT_LE(found_oob, max_oob)
      << "got too many records out-of-bounds (see ivf_pq::kOutOfBoundsRecord<IdxT>).";
    if (found_oob > 0) {
      RAFT_LOG_WARN(
        "Got %zu results out-of-bounds because of large top-k (%zu) and small n_probes (%u) and "
        "small DB size/n_lists ratio (%zu / %u)",
        size_t(found_oob),
        size_t(ps.k),
        ps.search_params.n_probes,
        size_t(ps.num_db_vecs),
        ps.index_params.n_lists);
    }
  }

  void SetUp() override  // NOLINT
  {
    gen_data();
    calc_ref();
  }

  void TearDown() override  // NOLINT
  {
    cudaGetLastError();
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  ivf_pq_inputs ps;                           // NOLINT
  rmm::device_uvector<DataT> database;        // NOLINT
  rmm::device_uvector<DataT> search_queries;  // NOLINT
  std::vector<IdxT> indices_ref;              // NOLINT
  std::vector<EvalT> distances_ref;           // NOLINT
};

/**
 * Separate test fixture for flat layout tests.
 * This ensures flat_layout_tests() parameters only run check_flat_layout_codes(),
 * not all the other tests defined in ivf_pq_test.
 */
template <typename EvalT, typename DataT, typename IdxT>
class ivf_pq_flat_layout_test : public ivf_pq_test<EvalT, DataT, IdxT> {};

template <typename EvalT, typename DataT, typename IdxT>
class ivf_pq_filter_test : public ::testing::TestWithParam<ivf_pq_inputs> {
 public:
  ivf_pq_filter_test()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<ivf_pq_inputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

  void gen_data()
  {
    database.resize(size_t{ps.num_db_vecs} * size_t{ps.dim}, stream_);
    search_queries.resize(size_t{ps.num_queries} * size_t{ps.dim}, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20));
    }
    raft::resource::sync_stream(handle_);
  }

  void calc_ref()
  {
    size_t queries_size = size_t{ps.num_queries} * size_t{ps.k};
    rmm::device_uvector<EvalT> distances_naive_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
    cuvs::neighbors::naive_knn<EvalT, DataT, IdxT>(
      handle_,
      distances_naive_dev.data(),
      indices_naive_dev.data(),
      search_queries.data(),
      database.data() + test_ivf_sample_filter::offset * ps.dim,
      ps.num_queries,
      ps.num_db_vecs - test_ivf_sample_filter::offset,
      ps.dim,
      ps.k,
      static_cast<cuvs::distance::DistanceType>((int)ps.index_params.metric));
    raft::linalg::addScalar(indices_naive_dev.data(),
                            indices_naive_dev.data(),
                            IdxT(test_ivf_sample_filter::offset),
                            queries_size,
                            stream_);
    distances_ref.resize(queries_size);
    raft::update_host(distances_ref.data(), distances_naive_dev.data(), queries_size, stream_);
    indices_ref.resize(queries_size);
    raft::update_host(indices_ref.data(), indices_naive_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);
  }

  auto build_only()
  {
    auto ipams              = ps.index_params;
    ipams.add_data_on_build = true;

    auto index_view =
      raft::make_device_matrix_view<const DataT, IdxT>(database.data(), ps.num_db_vecs, ps.dim);
    return cuvs::neighbors::ivf_pq::build(handle_, ipams, index_view);
  }

  template <typename BuildIndex>
  void run(BuildIndex build_index)
  {
    index<IdxT> index = build_index();

    double compression_ratio =
      static_cast<double>(ps.dim * 8) / static_cast<double>(index.pq_dim() * index.pq_bits());
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivf_pq(queries_size);
    std::vector<EvalT> distances_ivf_pq(queries_size);

    rmm::device_uvector<EvalT> distances_ivf_pq_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_ivf_pq_dev(queries_size, stream_);

    auto query_view =
      raft::make_device_matrix_view<DataT, uint32_t>(search_queries.data(), ps.num_queries, ps.dim);
    auto inds_view = raft::make_device_matrix_view<IdxT, uint32_t>(
      indices_ivf_pq_dev.data(), ps.num_queries, ps.k);
    auto dists_view = raft::make_device_matrix_view<EvalT, uint32_t>(
      distances_ivf_pq_dev.data(), ps.num_queries, ps.k);

    // Create Bitset filter
    auto removed_indices =
      raft::make_device_vector<IdxT, int64_t>(handle_, test_ivf_sample_filter::offset);
    thrust::sequence(
      raft::resource::get_thrust_policy(handle_),
      thrust::device_pointer_cast(removed_indices.data_handle()),
      thrust::device_pointer_cast(removed_indices.data_handle() + test_ivf_sample_filter::offset));
    raft::resource::sync_stream(handle_);

    cuvs::core::bitset<std::uint32_t, IdxT> removed_indices_bitset(
      handle_, removed_indices.view(), ps.num_db_vecs);
    auto bitset_filter_obj =
      cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset.view());
    cuvs::neighbors::ivf_pq::search(
      handle_, ps.search_params, index, query_view, inds_view, dists_view, bitset_filter_obj);

    raft::update_host(distances_ivf_pq.data(), distances_ivf_pq_dev.data(), queries_size, stream_);
    raft::update_host(indices_ivf_pq.data(), indices_ivf_pq_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);

    // A very conservative lower bound on recall
    double min_recall =
      static_cast<double>(ps.search_params.n_probes) / static_cast<double>(ps.index_params.n_lists);
    // Using a heuristic to lower the required recall due to code-packing errors
    min_recall =
      std::min(std::erfc(0.05 * compression_ratio / std::max(min_recall, 0.5)), min_recall);
    // Use explicit per-test min recall value if provided.
    min_recall = ps.min_recall.value_or(min_recall);

    ASSERT_TRUE(cuvs::neighbors::eval_neighbours(indices_ref,
                                                 indices_ivf_pq,
                                                 distances_ref,
                                                 distances_ivf_pq,
                                                 ps.num_queries,
                                                 ps.k,
                                                 0.0001 * compression_ratio,
                                                 min_recall))
      << ps;
  }

  void SetUp() override  // NOLINT
  {
    gen_data();
    calc_ref();
  }

  void TearDown() override  // NOLINT
  {
    cudaGetLastError();
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  ivf_pq_inputs ps;                           // NOLINT
  rmm::device_uvector<DataT> database;        // NOLINT
  rmm::device_uvector<DataT> search_queries;  // NOLINT
  std::vector<IdxT> indices_ref;              // NOLINT
  std::vector<EvalT> distances_ref;           // NOLINT
};

/* Test cases */
using test_cases_t = std::vector<ivf_pq_inputs>;

// concatenate parameter sets for different type
template <typename T>
auto operator+(const std::vector<T>& a, const std::vector<T>& b) -> std::vector<T>
{
  std::vector<T> res = a;
  res.insert(res.end(), b.begin(), b.end());
  return res;
}

inline auto defaults() -> test_cases_t { return {ivf_pq_inputs{}}; }

template <typename B, typename A, typename F>
auto map(const std::vector<A>& xs, F f) -> std::vector<B>
{
  std::vector<B> ys(xs.size());
  std::transform(xs.begin(), xs.end(), ys.begin(), f);
  return ys;
}

inline auto with_dims(const std::vector<uint32_t>& dims) -> test_cases_t
{
  return map<ivf_pq_inputs>(dims, [](uint32_t d) {
    ivf_pq_inputs x;
    x.dim = d;
    return x;
  });
}

/** These will surely trigger the fastest kernel available. */
inline auto small_dims() -> test_cases_t { return with_dims({1, 2, 3, 4, 5, 8, 15, 16, 17}); }

inline auto small_dims_per_cluster() -> test_cases_t
{
  return map<ivf_pq_inputs>(small_dims(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    return y;
  });
}

inline auto big_dims() -> test_cases_t
{
  // with_dims({512, 513, 1023, 1024, 1025, 2048, 2049, 2050, 2053, 6144, 8192, 12288, 16384});
  auto xs = with_dims({512, 513, 1023, 1024, 1025, 2048, 2049, 2050, 2053, 6144});
  return map<ivf_pq_inputs>(xs, [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    uint32_t pq_len       = 2;
    y.index_params.pq_dim = raft::div_rounding_up_safe(x.dim, pq_len);
    // This comes from pure experimentation, also the recall depens a lot on pq_len.
    y.min_recall = 0.48 + 0.028 * std::log2(x.dim);
    return y;
  });
}

/** These will surely trigger no-smem-lut kernel.  */
inline auto big_dims_moderate_lut() -> test_cases_t
{
  return map<ivf_pq_inputs>(big_dims(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    uint32_t pq_len           = 2;
    y.index_params.pq_dim     = raft::round_up_safe(raft::div_rounding_up_safe(x.dim, pq_len), 4u);
    y.index_params.pq_bits    = 6;
    y.search_params.lut_dtype = CUDA_R_16F;
    y.min_recall              = 0.69;
    return y;
  });
}

/** Some of these should trigger no-basediff kernel.  */
inline auto big_dims_small_lut() -> test_cases_t
{
  return map<ivf_pq_inputs>(big_dims(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    uint32_t pq_len           = 8;
    y.index_params.pq_dim     = raft::round_up_safe(raft::div_rounding_up_safe(x.dim, pq_len), 4u);
    y.index_params.pq_bits    = 6;
    y.search_params.lut_dtype = CUDA_R_8U;
    y.min_recall              = 0.21;
    return y;
  });
}

/**
 * Helper function to add test cases with modifications.
 * Note: Marked noinline to work around GCC 14 false positive stringop-overflow warnings
 * when compiling with C++20.
 */
template <typename F>
__attribute__((noinline)) void add_test_case(test_cases_t& xs, F&& modifier)
{
  ivf_pq_inputs temp{};
  modifier(temp);
  xs.push_back(std::move(temp));
}

/**
 * A minimal set of tests to check various enum-like parameters.
 */
inline auto enum_variety() -> test_cases_t
{
  test_cases_t xs;

  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.min_recall                 = 0.86;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.min_recall                 = 0.86;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.index_params.pq_bits       = 4;
    x.min_recall                 = 0.79;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.index_params.pq_bits       = 5;
    x.min_recall                 = 0.83;
  });

  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.index_params.pq_bits = 6;
    x.min_recall           = 0.84;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.index_params.pq_bits = 7;
    x.min_recall           = 0.85;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.index_params.pq_bits = 8;
    x.min_recall           = 0.86;
  });

  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.index_params.force_random_rotation = true;
    x.min_recall                         = 0.86;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.index_params.force_random_rotation = false;
    x.min_recall                         = 0.86;
  });

  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.search_params.lut_dtype = CUDA_R_32F;
    x.min_recall              = 0.86;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.search_params.lut_dtype = CUDA_R_16F;
    x.min_recall              = 0.86;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.search_params.lut_dtype = CUDA_R_8U;
    x.min_recall              = 0.84;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.search_params.coarse_search_dtype = CUDA_R_16F;
    x.min_recall                        = 0.86;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.search_params.coarse_search_dtype = CUDA_R_8I;
    // 8-bit coarse search is experimental and there's no go guarantee of any recall
    // if the data is not normalized. Especially for L2, because we store vector norms alongside the
    // cluster centers.
    x.min_recall = 0.1;
  });

  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.search_params.internal_distance_dtype = CUDA_R_32F;
    x.min_recall                            = 0.86;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.search_params.internal_distance_dtype = CUDA_R_16F;
    x.search_params.lut_dtype               = CUDA_R_16F;
    x.min_recall                            = 0.86;
  });
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.search_params.internal_distance_dtype = CUDA_R_16F;
    x.search_params.lut_dtype               = CUDA_R_16F;
    x.search_params.coarse_search_dtype     = CUDA_R_16F;
    x.min_recall                            = 0.86;
  });

  return xs;
}

inline auto enum_variety_l2() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.metric = distance::DistanceType::L2Expanded;
    return y;
  });
}

inline auto enum_variety_ip() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    if (y.min_recall.has_value()) {
      if (y.search_params.lut_dtype == CUDA_R_8U) {
        // InnerProduct score is signed,
        // thus we're forced to used signed 8-bit representation,
        // thus we have one bit less precision
        y.min_recall = y.min_recall.value() * 0.88;
      } else {
        // In other cases it seems to perform a little bit better, still worse than L2
        y.min_recall = y.min_recall.value() * 0.94;
      }
    }
    y.index_params.metric = distance::DistanceType::InnerProduct;
    return y;
  });
}

inline auto enum_variety_l2sqrt() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.metric = distance::DistanceType::L2SqrtExpanded;
    return y;
  });
}

inline auto enum_variety_cosine() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    if (y.min_recall.has_value()) {
      if (y.search_params.lut_dtype == CUDA_R_8U) {
        // TODO: Increase this recall threshold for 8 bit lut
        // (https://github.com/rapidsai/cuvs/issues/390)
        y.min_recall = y.min_recall.value() * 0.70;
      } else {
        // In other cases it seems to perform a little bit better, still worse than L2
        y.min_recall = y.min_recall.value() * 0.94;
      }
    }
    y.index_params.metric = distance::DistanceType::CosineExpanded;
    return y;
  });
}

/**
 * Try different number of n_probes, some of which may trigger the non-fused version of the search
 * kernel.
 */
inline auto var_n_probes() -> test_cases_t
{
  ivf_pq_inputs dflt;
  std::vector<uint32_t> xs;
  for (auto x = dflt.index_params.n_lists; x >= 1; x /= 2) {
    xs.push_back(x);
  }
  return map<ivf_pq_inputs>(xs, [](uint32_t n_probes) {
    ivf_pq_inputs x;
    x.search_params.n_probes = n_probes;
    return x;
  });
}

/**
 * Try different number of nearest neighbours.
 * Values smaller than 32 test if the code behaves well when Capacity (== 32) does not change,
 * but `k <= Capacity` changes.
 *
 * Values between `32 and ivf_pq::detail::kMaxCapacity` test various instantiations of the
 * main kernel (Capacity-templated)
 *
 * Values above ivf_pq::detail::kMaxCapacity should trigger the non-fused version of the kernel
 * (manage_local_topk = false).
 *
 * Also we test here various values that are close-but-not-power-of-two to catch any problems
 * related to rounding/alignment.
 *
 * Note, we cannot control explicitly which instance of the search kernel to choose, hence it's
 * important to try a variety of different values of `k` to make sure all paths are triggered.
 *
 * Set the log level to DEBUG (5) or above to inspect the selected kernel instances.
 */
inline auto var_k() -> test_cases_t
{
  return map<ivf_pq_inputs, uint32_t>(
    {1, 2, 3, 5, 8, 15, 16, 32, 63, 65, 127, 128, 256, 257, 1023, 2048, 2049}, [](uint32_t k) {
      ivf_pq_inputs x;
      x.k = k;
      // when there's not enough data, try more cluster probes
      x.search_params.n_probes = max(x.search_params.n_probes, min(x.index_params.n_lists, k));
      return x;
    });
}

/**
 * Cases brought up from downstream projects.
 */
inline auto special_cases() -> test_cases_t
{
  test_cases_t xs;

  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.num_db_vecs                = 1183514;
    x.dim                        = 100;
    x.num_queries                = 10000;
    x.k                          = 10;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_dim        = 10;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 1024;
    x.search_params.n_probes     = 50;
  });

  // Test large max_internal_batch_size
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.num_db_vecs                           = 500000;
    x.dim                                   = 100;
    x.num_queries                           = 128 * 1024 * 1024;
    x.k                                     = 10;
    x.index_params.codebook_kind            = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_dim                   = 10;
    x.index_params.pq_bits                  = 8;
    x.index_params.n_lists                  = 1024;
    x.search_params.n_probes                = 50;
    x.search_params.max_internal_batch_size = 64 * 1024 * 1024;
  });

  // Test small max_internal_batch_size
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.num_db_vecs                           = 500000;
    x.dim                                   = 100;
    x.num_queries                           = 128 * 1024 * 1024;
    x.k                                     = 10;
    x.index_params.codebook_kind            = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_dim                   = 10;
    x.index_params.pq_bits                  = 8;
    x.index_params.n_lists                  = 1024;
    x.search_params.n_probes                = 50;
    x.search_params.max_internal_batch_size = 1024 * 1024;
  });

  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.num_db_vecs                = 10000;
    x.dim                        = 16;
    x.num_queries                = 500;
    x.k                          = 128;
    x.index_params.metric        = distance::DistanceType::L2Expanded;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 100;
    x.search_params.n_probes     = 100;
  });

  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.num_db_vecs                = 10000;
    x.dim                        = 16;
    x.num_queries                = 500;
    x.k                          = 129;
    x.index_params.metric        = distance::DistanceType::L2Expanded;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 100;
    x.search_params.n_probes     = 100;
  });

  return xs;
}

/* Test instantiations */

#define TEST_BUILD_SEARCH(type)                         \
  TEST_P(type, build_search) /* NOLINT */               \
  {                                                     \
    this->run([this]() { return this->build_only(); }); \
  }

#define TEST_BUILD_HOST_INPUT_SEARCH(type)                         \
  TEST_P(type, build_host_input_search) /* NOLINT */               \
  {                                                                \
    this->run([this]() { return this->build_only_host_input(); }); \
  }

#define TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(type)                         \
  TEST_P(type, build_host_input_overlap_search) /* NOLINT */               \
  {                                                                        \
    this->run([this]() { return this->build_only_host_input_overlap(); }); \
  }

#define TEST_BUILD_EXTEND_SEARCH(type)                       \
  TEST_P(type, build_extend_search) /* NOLINT */             \
  {                                                          \
    this->run([this]() { return this->build_2_extends(); }); \
  }

#define TEST_BUILD_SERIALIZE_SEARCH(type)                    \
  TEST_P(type, build_serialize_search) /* NOLINT */          \
  {                                                          \
    this->run([this]() { return this->build_serialize(); }); \
  }

#define TEST_BUILD_PRECOMPUTED(type) \
  TEST_P(type, build_precomputed) /* NOLINT */ { this->build_precomputed(); }

#define TEST_FLAT_LAYOUT_CODES(type) \
  TEST_P(type, flat_layout_codes) /* NOLINT */ { this->check_flat_layout_codes(); }

#define INSTANTIATE(type, vals) \
  INSTANTIATE_TEST_SUITE_P(IvfPq, type, ::testing::ValuesIn(vals)); /* NOLINT */

/**
 * Test cases for flat layout comparison.
 * Tests that FLAT layout produces the same PQ codes as INTERLEAVED layout
 * by packing flat codes into a new interleaved index and comparing search results.
 */
inline auto flat_layout_tests() -> test_cases_t
{
  test_cases_t xs;

  // Test with pq_bits = 8 (byte-aligned, 1 code per byte)
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.num_db_vecs          = 1000;
    x.dim                  = 64;
    x.index_params.n_lists = 10;
    x.index_params.pq_bits = 8;
    x.index_params.pq_dim  = 16;
  });

  // Test with pq_bits = 6 (packed codes)
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.num_db_vecs          = 1000;
    x.dim                  = 64;
    x.index_params.n_lists = 10;
    x.index_params.pq_bits = 6;
    x.index_params.pq_dim  = 16;
  });

  // Test with PER_CLUSTER codebook and pq_bits = 8
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.num_db_vecs                = 1000;
    x.dim                        = 64;
    x.index_params.n_lists       = 10;
    x.index_params.pq_bits       = 8;
    x.index_params.pq_dim        = 16;
    x.index_params.codebook_kind = codebook_gen::PER_CLUSTER;
  });

  // Test with PER_CLUSTER codebook and pq_bits = 6
  add_test_case(xs, [](ivf_pq_inputs& x) {
    x.num_db_vecs                = 1000;
    x.dim                        = 64;
    x.index_params.n_lists       = 10;
    x.index_params.pq_bits       = 6;
    x.index_params.pq_dim        = 16;
    x.index_params.codebook_kind = codebook_gen::PER_CLUSTER;
  });

  return xs;
}

}  // namespace cuvs::neighbors::ivf_pq
