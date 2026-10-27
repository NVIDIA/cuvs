/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/selection/select_k.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/select_k.cuh>

#include <cub/device/device_segmented_radix_sort.cuh>

namespace cuvs::selection::detail {

/**
 * Stable selection for any k: sort each row by index, then stable-sort it by key, and keep the
 * first k. Radix sort keeps the input order of equal keys, so equal keys stay in index order:
 * ascending for select-min and descending for select-max, as in the stable warp sort.
 */
template <typename T, typename IdxT>
void stable_sort_select_k(raft::resources const& handle,
                          const T* in_val,
                          const IdxT* in_idx,
                          int64_t batch_size,
                          int64_t len,
                          int k,
                          T* out_val,
                          IdxT* out_idx,
                          bool select_min,
                          const IdxT* len_i)
{
  auto stream  = raft::resource::get_cuda_stream(handle);
  auto mr      = raft::resource::get_workspace_resource_ref(handle);
  int64_t n    = batch_size * len;
  auto ext     = raft::make_extents<int64_t>(n);
  auto keys_a  = raft::make_device_mdarray<T, int64_t>(handle, mr, ext);
  auto keys_b  = raft::make_device_mdarray<T, int64_t>(handle, mr, ext);
  auto idx_a   = raft::make_device_mdarray<IdxT, int64_t>(handle, mr, ext);
  auto idx_b   = raft::make_device_mdarray<IdxT, int64_t>(handle, mr, ext);
  auto row_ext = raft::make_extents<int64_t>(batch_size);
  auto begins  = raft::make_device_mdarray<int64_t, int64_t>(handle, mr, row_ext);
  auto ends    = raft::make_device_mdarray<int64_t, int64_t>(handle, mr, row_ext);

  raft::copy(keys_a.data_handle(), in_val, n, stream);
  raft::linalg::map_offset(handle, idx_a.view(), [in_idx, len] __device__(int64_t i) {
    return in_idx != nullptr ? in_idx[i] : IdxT(i % len);
  });
  raft::linalg::map_offset(handle, begins.view(), [len] __device__(int64_t r) { return r * len; });
  raft::linalg::map_offset(handle, ends.view(), [len, len_i] __device__(int64_t r) {
    return r * len + (len_i != nullptr ? int64_t(len_i[r]) : len);
  });

  auto sort =
    [&](void* tmp, size_t& bytes, auto* keys_in, auto* keys_out, auto* vals_in, auto* vals_out) {
      constexpr int kEndBit = sizeof(*keys_in) * 8;
      if (select_min) {
        cub::DeviceSegmentedRadixSort::SortPairs(tmp,
                                                 bytes,
                                                 keys_in,
                                                 keys_out,
                                                 vals_in,
                                                 vals_out,
                                                 n,
                                                 batch_size,
                                                 begins.data_handle(),
                                                 ends.data_handle(),
                                                 0,
                                                 kEndBit,
                                                 stream);
      } else {
        cub::DeviceSegmentedRadixSort::SortPairsDescending(tmp,
                                                           bytes,
                                                           keys_in,
                                                           keys_out,
                                                           vals_in,
                                                           vals_out,
                                                           n,
                                                           batch_size,
                                                           begins.data_handle(),
                                                           ends.data_handle(),
                                                           0,
                                                           kEndBit,
                                                           stream);
      }
    };
  size_t bytes_by_idx = 0;
  size_t bytes_by_key = 0;
  sort(nullptr,
       bytes_by_idx,
       idx_a.data_handle(),
       idx_b.data_handle(),
       keys_a.data_handle(),
       keys_b.data_handle());
  sort(nullptr,
       bytes_by_key,
       keys_b.data_handle(),
       keys_a.data_handle(),
       idx_b.data_handle(),
       idx_a.data_handle());
  size_t bytes = std::max(bytes_by_idx, bytes_by_key);
  auto tmp = raft::make_device_mdarray<char, size_t>(handle, mr, raft::make_extents<size_t>(bytes));
  sort(tmp.data_handle(),
       bytes,
       idx_a.data_handle(),
       idx_b.data_handle(),
       keys_a.data_handle(),
       keys_b.data_handle());
  sort(tmp.data_handle(),
       bytes,
       keys_b.data_handle(),
       keys_a.data_handle(),
       idx_b.data_handle(),
       idx_a.data_handle());

  // A row shorter than k fills its tail with an empty slot that loses to every element.
  int64_t m        = batch_size * int64_t(k);
  const T* keys    = keys_a.data_handle();
  const IdxT* idxs = idx_a.data_handle();
  T empty_key      = select_min ? raft::upper_bound<T>() : raft::lower_bound<T>();
  IdxT empty_idx   = select_min ? raft::upper_bound<IdxT>() : raft::lower_bound<IdxT>();
  raft::linalg::map_offset(handle,
                           raft::make_device_vector_view<T, int64_t>(out_val, m),
                           [keys, len, k, len_i, empty_key] __device__(int64_t i) {
                             int64_t r = i / k, j = i % k;
                             int64_t row_len = len_i != nullptr ? int64_t(len_i[r]) : len;
                             return j < row_len ? keys[r * len + j] : empty_key;
                           });
  raft::linalg::map_offset(handle,
                           raft::make_device_vector_view<IdxT, int64_t>(out_idx, m),
                           [idxs, len, k, len_i, empty_idx] __device__(int64_t i) {
                             int64_t r = i / k, j = i % k;
                             int64_t row_len = len_i != nullptr ? int64_t(len_i[r]) : len;
                             return j < row_len ? idxs[r * len + j] : empty_idx;
                           });
}

template <typename T, typename IdxT>
void select_k(raft::resources const& handle,
              raft::device_matrix_view<const T, int64_t, raft::row_major> in_val,
              std::optional<raft::device_matrix_view<const IdxT, int64_t, raft::row_major>> in_idx,
              raft::device_matrix_view<T, int64_t, raft::row_major> out_val,
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> out_idx,
              bool select_min,
              bool sorted,
              SelectAlgo algo,
              std::optional<raft::device_vector_view<const IdxT, int64_t>> len_i,
              bool stable)
{
  RAFT_EXPECTS(out_val.extent(1) <= int64_t(std::numeric_limits<int>::max()),
               "output k must fit the int type.");
  auto batch_size = in_val.extent(0);
  auto len        = in_val.extent(1);
  auto k          = int(out_val.extent(1));
  RAFT_EXPECTS(batch_size == out_val.extent(0), "batch sizes must be equal");
  RAFT_EXPECTS(batch_size == out_idx.extent(0), "batch sizes must be equal");
  if (in_idx.has_value()) {
    RAFT_EXPECTS(batch_size == in_idx->extent(0), "batch sizes must be equal");
    RAFT_EXPECTS(len == in_idx->extent(1), "value and index input lengths must be equal");
  }
  RAFT_EXPECTS(int64_t(k) == out_idx.extent(1), "value and index output lengths must be equal");
  if (stable) {
    if (k > raft::matrix::detail::select::warpsort::kMaxCapacity) {
      return stable_sort_select_k<T, IdxT>(handle,
                                           in_val.data_handle(),
                                           in_idx.has_value() ? in_idx->data_handle() : nullptr,
                                           batch_size,
                                           len,
                                           k,
                                           out_val.data_handle(),
                                           out_idx.data_handle(),
                                           select_min,
                                           len_i.has_value() ? len_i->data_handle() : nullptr);
    }
    algo = SelectAlgo::kWarpDistributedShmStable;
  }

  // just delegate implementation to raft - the primary benefit here is to have
  // instantiations only compiled once in cuvs
  return raft::matrix::detail::select_k<T, IdxT>(
    handle,
    in_val.data_handle(),
    in_idx.has_value() ? in_idx->data_handle() : nullptr,
    batch_size,
    len,
    k,
    out_val.data_handle(),
    out_idx.data_handle(),
    select_min,
    sorted,
    algo,
    len_i.has_value() ? len_i->data_handle() : nullptr);
}
}  // namespace cuvs::selection::detail

#define instantiate_cuvs_selection_select_k(T, IdxT)                                      \
  void cuvs::selection::select_k(                                                         \
    raft::resources const& handle,                                                        \
    raft::device_matrix_view<const T, int64_t, raft::row_major> in_val,                   \
    std::optional<raft::device_matrix_view<const IdxT, int64_t, raft::row_major>> in_idx, \
    raft::device_matrix_view<T, int64_t, raft::row_major> out_val,                        \
    raft::device_matrix_view<IdxT, int64_t, raft::row_major> out_idx,                     \
    bool select_min,                                                                      \
    bool sorted,                                                                          \
    SelectAlgo algo,                                                                      \
    std::optional<raft::device_vector_view<const IdxT, int64_t>> len_i,                   \
    bool stable)                                                                          \
  {                                                                                       \
    detail::select_k<T, IdxT>(                                                            \
      handle, in_val, in_idx, out_val, out_idx, select_min, sorted, algo, len_i, stable); \
  }
