/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <neighbors/ivf_pq_impl.hpp>

#include <raft/core/mdspan.hpp>
#include <raft/linalg/map.cuh>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename IdxT>
void transpose_pq_centers(const raft::resources& handle,
                          owning_impl<IdxT>* impl,
                          const float* pq_centers_source)
{
  auto extents = impl->pq_centers().extents();
  static_assert(extents.rank() == 3);
  auto extents_source =
    raft::make_extents<uint32_t>(extents.extent(0), extents.extent(2), extents.extent(1));
  auto span_source = raft::make_mdspan<const float, uint32_t, raft::row_major, false, true>(
    pq_centers_source, extents_source);
  auto pq_centers_view = raft::make_device_vector_view<float, IdxT>(
    impl->pq_centers().data_handle(), impl->pq_centers().size());
  raft::linalg::map_offset(handle, pq_centers_view, [span_source, extents] __device__(size_t i) {
    uint32_t ii[3];
    for (int r = 2; r > 0; r--) {
      ii[r] = i % extents.extent(r);
      i /= extents.extent(r);
    }
    ii[0] = i;
    return span_source(ii[0], ii[2], ii[1]);
  });
}

template void transpose_pq_centers<long>(const raft::resources&, owning_impl<long>*, const float*);

}  // namespace cuvs::neighbors::ivf_pq::detail
