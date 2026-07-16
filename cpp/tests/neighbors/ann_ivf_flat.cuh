/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "naive_knn.cuh"

#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/linalg/normalize.cuh>
#include <raft/stats/mean.cuh>
#include <thrust/reduce.h>

#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/linalg/add.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/util/fast_int_div.cuh>
#include <rmm/cuda_stream_pool.hpp>

namespace cuvs::neighbors::ivf_flat {

struct test_ivf_sample_filter {
  static constexpr unsigned offset = 300;
};

template <typename IdxT>
struct AnnIvfFlatInputs {
  IdxT num_queries;
  IdxT num_db_vecs;
  IdxT dim;
  IdxT k;
  IdxT nprobe;
  IdxT nlist;
  cuvs::distance::DistanceType metric;
  bool adaptive_centers;
  bool host_dataset = false;
  // The kernel_copy_overlapping option is only applicable when host dataset is enabled.
  bool kernel_copy_overlapping = false;
};

template <typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const AnnIvfFlatInputs<IdxT>& p)
{
  os << "{ " << p.num_queries << ", " << p.num_db_vecs << ", " << p.dim << ", " << p.k << ", "
     << p.nprobe << ", " << p.nlist << ", "
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << ", " << p.adaptive_centers << "," << p.host_dataset << "," << p.kernel_copy_overlapping
     << '}';
  return os;
}

template <typename T, typename DataT, typename IdxT>
class AnnIVFFlatTest : public ::testing::TestWithParam<AnnIvfFlatInputs<IdxT>> {
 public:
  AnnIVFFlatTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnIvfFlatInputs<IdxT>>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

  void testIVFFlat()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivfflat(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<T> distances_ivfflat(queries_size);
    std::vector<T> distances_naive(queries_size);

    if (ps.kernel_copy_overlapping) {
      size_t n_streams = 1;
      raft::resource::set_cuda_stream_pool(handle_,
                                           std::make_shared<rmm::cuda_stream_pool>(n_streams));
    }

    {
      rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      cuvs::neighbors::naive_knn<T, DataT, IdxT>(handle_,
                                                 distances_naive_dev.data(),
                                                 indices_naive_dev.data(),
                                                 search_queries.data(),
                                                 database.data(),
                                                 ps.num_queries,
                                                 ps.num_db_vecs,
                                                 ps.dim,
                                                 ps.k,
                                                 ps.metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      // unless something is really wrong with clustering, this could serve as a lower bound on
      // recall
      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);

      rmm::device_uvector<T> distances_ivfflat_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_ivfflat_dev(queries_size, stream_);

      {
        cuvs::neighbors::ivf_flat::index_params index_params;
        cuvs::neighbors::ivf_flat::search_params search_params;
        index_params.n_lists          = ps.nlist;
        index_params.metric           = ps.metric;
        index_params.adaptive_centers = ps.adaptive_centers;
        search_params.n_probes        = ps.nprobe;

        index_params.add_data_on_build        = false;
        index_params.kmeans_trainset_fraction = 0.5;
        index_params.metric_arg               = 0;

        cuvs::neighbors::ivf_flat::index<DataT, IdxT> idx(handle_, index_params, ps.dim);
        cuvs::neighbors::ivf_flat::index<DataT, IdxT> index_2(handle_, index_params, ps.dim);

        if (!ps.host_dataset) {
          auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
            (const DataT*)database.data(), ps.num_db_vecs, ps.dim);
          idx = cuvs::neighbors::ivf_flat::build(handle_, index_params, database_view);
          auto vector_indices = raft::make_device_vector<IdxT, IdxT>(handle_, ps.num_db_vecs);
          raft::linalg::map_offset(handle_, vector_indices.view(), raft::identity_op{});
          raft::resource::sync_stream(handle_);

          IdxT half_of_data = ps.num_db_vecs / 2;

          auto half_of_data_view = raft::make_device_matrix_view<const DataT, IdxT>(
            (const DataT*)database.data(), half_of_data, ps.dim);

          const std::optional<raft::device_vector_view<const IdxT, IdxT>> no_opt = std::nullopt;
          index_2 = cuvs::neighbors::ivf_flat::extend(handle_, half_of_data_view, no_opt, idx);

          auto new_half_of_data_view = raft::make_device_matrix_view<const DataT, IdxT>(
            database.data() + half_of_data * ps.dim, IdxT(ps.num_db_vecs) - half_of_data, ps.dim);

          auto new_half_of_data_indices_view = raft::make_device_vector_view<const IdxT, IdxT>(
            vector_indices.data_handle() + half_of_data, IdxT(ps.num_db_vecs) - half_of_data);

          cuvs::neighbors::ivf_flat::extend(
            handle_,
            new_half_of_data_view,
            std::make_optional<raft::device_vector_view<const IdxT, IdxT>>(
              new_half_of_data_indices_view),
            &index_2);
        } else {
          auto host_database = raft::make_host_matrix<DataT, IdxT>(ps.num_db_vecs, ps.dim);
          raft::copy(
            host_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);
          idx =
            ivf_flat::build(handle_, index_params, raft::make_const_mdspan(host_database.view()));

          auto vector_indices = raft::make_host_vector<IdxT>(handle_, ps.num_db_vecs);
          std::iota(vector_indices.data_handle(), vector_indices.data_handle() + ps.num_db_vecs, 0);

          IdxT half_of_data = ps.num_db_vecs / 2;

          auto half_of_data_view = raft::make_host_matrix_view<const DataT, IdxT>(
            (const DataT*)host_database.data_handle(), half_of_data, ps.dim);

          const std::optional<raft::host_vector_view<const IdxT, IdxT>> no_opt = std::nullopt;
          index_2 = ivf_flat::extend(handle_, half_of_data_view, no_opt, idx);

          auto new_half_of_data_view = raft::make_host_matrix_view<const DataT, IdxT>(
            host_database.data_handle() + half_of_data * ps.dim,
            IdxT(ps.num_db_vecs) - half_of_data,
            ps.dim);
          auto new_half_of_data_indices_view = raft::make_host_vector_view<const IdxT, IdxT>(
            vector_indices.data_handle() + half_of_data, IdxT(ps.num_db_vecs) - half_of_data);
          ivf_flat::extend(handle_,
                           new_half_of_data_view,
                           std::make_optional<raft::host_vector_view<const IdxT, IdxT>>(
                             new_half_of_data_indices_view),
                           &index_2);
        }

        auto search_queries_view = raft::make_device_matrix_view<const DataT, IdxT>(
          search_queries.data(), ps.num_queries, ps.dim);
        auto indices_out_view = raft::make_device_matrix_view<IdxT, IdxT>(
          indices_ivfflat_dev.data(), ps.num_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<T, IdxT>(
          distances_ivfflat_dev.data(), ps.num_queries, ps.k);
        tmp_index_file index_file;
        cuvs::neighbors::ivf_flat::serialize(handle_, index_file.filename, index_2);
        cuvs::neighbors::ivf_flat::index<DataT, IdxT> index_loaded(handle_);
        cuvs::neighbors::ivf_flat::deserialize(handle_, index_file.filename, &index_loaded);
        ASSERT_EQ(index_2.size(), index_loaded.size());

        cuvs::neighbors::ivf_flat::search(handle_,
                                          search_params,
                                          index_loaded,
                                          search_queries_view,
                                          indices_out_view,
                                          dists_out_view);

        raft::update_host(
          distances_ivfflat.data(), distances_ivfflat_dev.data(), queries_size, stream_);
        raft::update_host(
          indices_ivfflat.data(), indices_ivfflat_dev.data(), queries_size, stream_);
        raft::resource::sync_stream(handle_);

        // Test the centroid invariants
        if (index_2.adaptive_centers()) {
          // The centers must be up-to-date with the corresponding data
          std::vector<uint32_t> list_sizes(index_2.n_lists());
          std::vector<IdxT*> list_indices(index_2.n_lists());
          rmm::device_uvector<float> centroid(ps.dim, stream_);
          raft::copy(
            list_sizes.data(), index_2.list_sizes().data_handle(), index_2.n_lists(), stream_);
          raft::copy(
            list_indices.data(), index_2.inds_ptrs().data_handle(), index_2.n_lists(), stream_);
          raft::resource::sync_stream(handle_);
          for (uint32_t l = 0; l < index_2.n_lists(); l++) {
            if (list_sizes[l] == 0) continue;
            rmm::device_uvector<float> cluster_data(list_sizes[l] * ps.dim, stream_);
            cuvs::spatial::knn::detail::utils::copy_selected<float>((IdxT)list_sizes[l],
                                                                    (IdxT)ps.dim,
                                                                    database.data(),
                                                                    list_indices[l],
                                                                    (IdxT)ps.dim,
                                                                    cluster_data.data(),
                                                                    (IdxT)ps.dim,
                                                                    stream_);
            raft::stats::mean<true, float, uint32_t>(
              centroid.data(), cluster_data.data(), ps.dim, list_sizes[l], false, stream_);
            ASSERT_TRUE(cuvs::devArrMatch(index_2.centers().data_handle() + ps.dim * l,
                                          centroid.data(),
                                          ps.dim,
                                          cuvs::CompareApprox<float>(0.001),
                                          stream_));
          }
        } else {
          // The centers must be immutable
          ASSERT_TRUE(cuvs::devArrMatch(index_2.centers().data_handle(),
                                        idx.centers().data_handle(),
                                        index_2.centers().size(),
                                        cuvs::Compare<float>(),
                                        stream_));
        }
      }
      float eps = std::is_same_v<DataT, half> ? 0.005 : 0.001;
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ivfflat,
                                  distances_naive,
                                  distances_ivfflat,
                                  ps.num_queries,
                                  ps.k,
                                  eps,
                                  min_recall));
    }
  }

  void testPacker()
  {
    ivf_flat::index_params index_params;
    ivf_flat::search_params search_params;
    index_params.n_lists          = ps.nlist;
    index_params.metric           = ps.metric;
    index_params.adaptive_centers = false;
    search_params.n_probes        = ps.nprobe;

    index_params.add_data_on_build        = false;
    index_params.kmeans_trainset_fraction = 1.0;
    index_params.metric_arg               = 0;

    auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
      (const DataT*)database.data(), ps.num_db_vecs, ps.dim);

    auto idx = ivf_flat::build(handle_, index_params, database_view);

    const std::optional<raft::device_vector_view<const IdxT, IdxT>> no_opt = std::nullopt;
    index<DataT, IdxT> extend_index = ivf_flat::extend(handle_, database_view, no_opt, idx);

    auto list_sizes = raft::make_host_vector<uint32_t>(idx.n_lists());
    raft::update_host(list_sizes.data_handle(),
                      extend_index.list_sizes().data_handle(),
                      extend_index.n_lists(),
                      stream_);
    raft::resource::sync_stream(handle_);

    auto& lists = idx.lists();

    // conservative memory allocation for codepacking
    auto list_device_spec = list_spec<uint32_t, DataT, IdxT>{idx.dim(), false};

    for (uint32_t label = 0; label < idx.n_lists(); label++) {
      uint32_t list_size = list_sizes.data_handle()[label];

      ivf::resize_list(handle_, lists[label], list_device_spec, list_size, 0);
    }

    ivf_flat::helpers::recompute_internal_state(handle_, &idx);

    using interleaved_group = raft::Pow2<kIndexGroupSize>;

    for (uint32_t label = 0; label < idx.n_lists(); label++) {
      uint32_t list_size = list_sizes.data_handle()[label];

      if (list_size > 0) {
        uint32_t padded_list_size = interleaved_group::roundUp(list_size);
        uint32_t n_elems          = padded_list_size * idx.dim();
        auto& list_data           = lists[label]->data;
        auto& list_inds           = extend_index.lists()[label]->indices;

        // fetch the flat codes
        auto flat_codes = raft::make_device_matrix<DataT, uint32_t>(handle_, list_size, idx.dim());

        raft::matrix::gather(
          handle_,
          raft::make_device_matrix_view<const DataT, uint32_t>(
            (const DataT*)database.data(), static_cast<uint32_t>(ps.num_db_vecs), idx.dim()),
          raft::make_device_vector_view<const IdxT, uint32_t>((const IdxT*)list_inds.data_handle(),
                                                              list_size),
          flat_codes.view());

        helpers::codepacker::pack(
          handle_, make_const_mdspan(flat_codes.view()), idx.veclen(), 0, list_data.view());

        {
          auto mask = raft::make_device_vector<bool>(handle_, n_elems);

          raft::linalg::map_offset(
            handle_,
            mask.view(),
            [dim = idx.dim(),
             list_size,
             padded_list_size,
             chunk_size = raft::util::FastIntDiv<int32_t>(
               static_cast<int32_t>(idx.veclen()))] __device__(auto i) {
              uint32_t max_group_offset = interleaved_group::roundDown(list_size);
              if (i < max_group_offset * dim) { return true; }
              uint32_t surplus = (i - max_group_offset * dim);
              uint32_t ingroup_id =
                interleaved_group::mod(static_cast<int32_t>(surplus) / chunk_size);
              return ingroup_id < (list_size - max_group_offset);
            });

          // ensure that the correct number of indices are masked out
          ASSERT_TRUE(thrust::reduce(raft::resource::get_thrust_policy(handle_),
                                     mask.data_handle(),
                                     mask.data_handle() + n_elems,
                                     0) == list_size * ps.dim);

          auto packed_list_data = raft::make_device_vector<DataT, uint32_t>(handle_, n_elems);

          raft::linalg::map_offset(handle_,
                                   packed_list_data.view(),
                                   [mask      = mask.data_handle(),
                                    list_data = list_data.data_handle()] __device__(uint32_t i) {
                                     if (mask[i]) return list_data[i];
                                     return DataT{0};
                                   });

          auto& extend_data         = extend_index.lists()[label]->data;
          auto extend_data_filtered = raft::make_device_vector<DataT, uint32_t>(handle_, n_elems);
          raft::linalg::map_offset(
            handle_,
            extend_data_filtered.view(),
            [mask        = mask.data_handle(),
             extend_data = extend_data.data_handle()] __device__(uint32_t i) {
              if (mask[i]) return extend_data[i];
              return DataT{0};
            });

          ASSERT_TRUE(cuvs::devArrMatch(packed_list_data.data_handle(),
                                        extend_data_filtered.data_handle(),
                                        n_elems,
                                        cuvs::Compare<DataT>(),
                                        stream_));
        }

        auto unpacked_flat_codes =
          raft::make_device_matrix<DataT, uint32_t>(handle_, list_size, idx.dim());

        helpers::codepacker::unpack(
          handle_, list_data.view(), idx.veclen(), 0, unpacked_flat_codes.view());

        ASSERT_TRUE(cuvs::devArrMatch(flat_codes.data_handle(),
                                      unpacked_flat_codes.data_handle(),
                                      list_size * ps.dim,
                                      cuvs::Compare<DataT>(),
                                      stream_));
      }
    }
  }

  void testFilter()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivfflat(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<T> distances_ivfflat(queries_size);
    std::vector<T> distances_naive(queries_size);

    {
      rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      auto* database_filtered_ptr = database.data() + test_ivf_sample_filter::offset * ps.dim;
      cuvs::neighbors::naive_knn<T, DataT, IdxT>(handle_,
                                                 distances_naive_dev.data(),
                                                 indices_naive_dev.data(),
                                                 search_queries.data(),
                                                 database_filtered_ptr,
                                                 ps.num_queries,
                                                 ps.num_db_vecs - test_ivf_sample_filter::offset,
                                                 ps.dim,
                                                 ps.k,
                                                 ps.metric);
      raft::linalg::addScalar(indices_naive_dev.data(),
                              indices_naive_dev.data(),
                              IdxT(test_ivf_sample_filter::offset),
                              queries_size,
                              stream_);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      // unless something is really wrong with clustering, this could serve as a lower bound on
      // recall
      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);

      auto distances_ivfflat_dev = raft::make_device_matrix<T, IdxT>(handle_, ps.num_queries, ps.k);
      auto indices_ivfflat_dev =
        raft::make_device_matrix<IdxT, IdxT>(handle_, ps.num_queries, ps.k);

      {
        ivf_flat::index_params index_params;
        ivf_flat::search_params search_params;
        index_params.n_lists          = ps.nlist;
        index_params.metric           = ps.metric;
        index_params.adaptive_centers = ps.adaptive_centers;
        search_params.n_probes        = ps.nprobe;

        index_params.add_data_on_build        = true;
        index_params.kmeans_trainset_fraction = 0.5;
        index_params.metric_arg               = 0;

        // Create IVF Flat index
        auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
          (const DataT*)database.data(), ps.num_db_vecs, ps.dim);
        auto index = ivf_flat::build(handle_, index_params, database_view);

        // Create Bitset filter
        auto removed_indices =
          raft::make_device_vector<IdxT, int64_t>(handle_, test_ivf_sample_filter::offset);
        raft::linalg::map_offset(handle_, removed_indices.view(), raft::identity_op{});
        raft::resource::sync_stream(handle_);

        cuvs::core::bitset<std::uint32_t, IdxT> removed_indices_bitset(
          handle_, removed_indices.view(), ps.num_db_vecs);
        auto bitset_filter_obj =
          cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset.view());

        // Search with the filter
        auto search_queries_view = raft::make_device_matrix_view<const DataT, IdxT>(
          search_queries.data(), ps.num_queries, ps.dim);
        ivf_flat::search(handle_,
                         search_params,
                         index,
                         search_queries_view,
                         indices_ivfflat_dev.view(),
                         distances_ivfflat_dev.view(),
                         bitset_filter_obj);

        raft::update_host(
          distances_ivfflat.data(), distances_ivfflat_dev.data_handle(), queries_size, stream_);
        raft::update_host(
          indices_ivfflat.data(), indices_ivfflat_dev.data_handle(), queries_size, stream_);
        raft::resource::sync_stream(handle_);
      }
      float eps = std::is_same_v<DataT, half> ? 0.005 : 0.001;
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ivfflat,
                                  distances_naive,
                                  distances_ivfflat,
                                  ps.num_queries,
                                  ps.k,
                                  eps,
                                  min_recall));
    }
  }

  void SetUp() override
  {
    database.resize(ps.num_db_vecs * ps.dim, stream_);
    search_queries.resize(ps.num_queries * ps.dim, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) {
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

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnIvfFlatInputs<IdxT> ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

const std::vector<AnnIvfFlatInputs<int64_t>> inputs = {
  // test various dims (aligned and not aligned to vector sizes)
  {1000, 10000, 1, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 2, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 3, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 3, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 4, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 4, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 5, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 5, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 8, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 8, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 5, 16, 40, 1024, cuvs::distance::DistanceType::L2SqrtExpanded, false},
  {1000, 10000, 5, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 8, 16, 40, 1024, cuvs::distance::DistanceType::L2SqrtExpanded, true},
  {1000, 10000, 8, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},

  // test dims that do not fit into kernel shared memory limits
  {1000, 10000, 2048, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 2048, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 2049, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 2049, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 2050, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 2050, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  // TODO: Re-enable test after adjusting parameters for higher recall. See
  // https://github.com/rapidsai/cuvs/issues/1091
  // {1000, 10000, 2051, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 2051, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 2052, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 2052, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 2053, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2053, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 2056, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2056, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},

  // various random combinations
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::L2Expanded, false},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::CosineExpanded, false},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false},

  // host input data
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::L2Expanded, false, true},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true},

  // // host input data with prefetching for kernel copy overlapping
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded, false, true, true},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false, true, true},

  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::InnerProduct, true},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::CosineExpanded, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded, false},
  {10000, 131072, 8, 10, 50, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {10000, 131072, 8, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, true},

  {1000, 10000, 4096, 20, 50, 1024, cuvs::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 4096, 20, 50, 1024, cuvs::distance::DistanceType::CosineExpanded, false},

  // test splitting the big query batches  (> max gridDim.y) into smaller batches
  {100000, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::InnerProduct, false},
  {100000, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::CosineExpanded, false},
  {1000000, 1024, 32, 10, 256, 256, cuvs::distance::DistanceType::InnerProduct, false},
  {1000000, 1024, 32, 10, 256, 256, cuvs::distance::DistanceType::CosineExpanded, false},
  {98306, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::InnerProduct, true},
  {98306, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::CosineExpanded, true},

  // test radix_sort for getting the cluster selection
  {1000,
   10000,
   16,
   10,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 2,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   cuvs::distance::DistanceType::L2Expanded,
   false},
  {1000,
   10000,
   16,
   10,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   cuvs::distance::DistanceType::InnerProduct,
   false},
  {1000,
   10000,
   16,
   10,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   cuvs::distance::DistanceType::CosineExpanded,
   false},

  // The following two test cases should show very similar recall.
  // num_queries, num_db_vecs, dim, k, nprobe, nlist, metric, adaptive_centers
  {20000, 8712, 3, 10, 51, 66, cuvs::distance::DistanceType::L2Expanded, false},
  {100000, 8712, 3, 10, 51, 66, cuvs::distance::DistanceType::L2Expanded, false}};

}  // namespace cuvs::neighbors::ivf_flat
