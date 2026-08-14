/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../src/neighbors/detail/ann_utils.cuh"
#include <cuvs/distance/distance.hpp>
#include <raft/matrix/detail/select_k.cuh>
#include <raft/util/cuda_utils.cuh>

#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cuvs::neighbors {

template <typename EvalT, typename DataT, typename IdxT>
RAFT_KERNEL naive_distance_kernel(EvalT* dist,
                                  const DataT* x,
                                  const DataT* y,
                                  IdxT m,
                                  IdxT n,
                                  IdxT k,
                                  cuvs::distance::DistanceType metric)
{
  IdxT midx = IdxT(threadIdx.x) + IdxT(blockIdx.x) * IdxT(blockDim.x);
  if (midx >= m) return;
  IdxT grid_size = IdxT(blockDim.y) * IdxT(gridDim.y);
  for (IdxT nidx = threadIdx.y + blockIdx.y * blockDim.y; nidx < n; nidx += grid_size) {
    EvalT acc   = EvalT(0);
    EvalT normX = EvalT(0);
    EvalT normY = EvalT(0);
    for (IdxT i = 0; i < k; ++i) {
      IdxT xidx = i + midx * k;
      IdxT yidx = i + nidx * k;
      auto xv   = x[xidx];
      auto yv   = y[yidx];
      switch (metric) {
        case cuvs::distance::DistanceType::InnerProduct: {
          acc += static_cast<EvalT>(xv) * static_cast<EvalT>(yv);
        } break;
        case cuvs::distance::DistanceType::CosineExpanded: {
          acc += static_cast<EvalT>(xv) * static_cast<EvalT>(yv);
          normX += static_cast<EvalT>(xv) * static_cast<EvalT>(xv);
          normY += static_cast<EvalT>(yv) * static_cast<EvalT>(yv);
        } break;
        case cuvs::distance::DistanceType::L2SqrtExpanded:
        case cuvs::distance::DistanceType::L2SqrtUnexpanded:
        case cuvs::distance::DistanceType::L2Expanded:
        case cuvs::distance::DistanceType::L2Unexpanded: {
          auto diff = static_cast<EvalT>(xv) - static_cast<EvalT>(yv);
          acc += diff * diff;
        } break;
        case cuvs::distance::DistanceType::BitwiseHamming: {
          if constexpr (std::is_same_v<uint8_t, DataT> || std::is_same_v<int8_t, DataT>) {
            acc += __popc(static_cast<uint32_t>(xv ^ yv) & 0xff);
          }
        } break;
        case cuvs::distance::DistanceType::L1: {
          auto diff = static_cast<EvalT>(xv) - static_cast<EvalT>(yv);
          acc += raft::abs(diff);
        } break;
        case cuvs::distance::DistanceType::Linf: {
          auto diff = static_cast<EvalT>(xv) - static_cast<EvalT>(yv);
          auto ad   = raft::abs(diff);
          acc       = (ad > acc) ? ad : acc;
        } break;
        default: break;
      }
    }
    switch (metric) {
      case cuvs::distance::DistanceType::L2SqrtExpanded:
      case cuvs::distance::DistanceType::L2SqrtUnexpanded: {
        acc = raft::sqrt(acc);
      } break;
      case cuvs::distance::DistanceType::CosineExpanded: {
        acc = 1 - acc / (raft::sqrt(normX) * raft::sqrt(normY));
      }
      default: break;
    }
    dist[midx * n + nidx] = acc;
  }
}

/**
 * Naive, but flexible bruteforce KNN search.
 *
 * TODO: either replace this with brute_force_knn or with distance+select_k
 *       when either distance or brute_force_knn support 8-bit int inputs.
 */
template <typename EvalT, typename DataT, typename IdxT>
void naive_knn(raft::resources const& handle,
               EvalT* dist_topk,
               IdxT* indices_topk,
               const DataT* x,
               const DataT* y,
               size_t n_inputs,
               size_t input_len,
               size_t dim,
               uint32_t k,
               cuvs::distance::DistanceType type)
{
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref();

  auto stream = raft::resource::get_cuda_stream(handle);
  dim3 block_dim(16, 32, 1);
  // maximum reasonable grid size in `y` direction
  auto grid_y =
    static_cast<uint16_t>(std::min<size_t>(raft::ceildiv<size_t>(input_len, block_dim.y), 32768));

  // bound the memory used by this function
  size_t max_batch_size =
    std::min<size_t>(n_inputs, raft::ceildiv<size_t>(size_t(1) << size_t(27), input_len));
  rmm::device_uvector<EvalT> dist(max_batch_size * input_len, stream, mr);

  for (size_t offset = 0; offset < n_inputs; offset += max_batch_size) {
    size_t batch_size = std::min(max_batch_size, n_inputs - offset);
    dim3 grid_dim(raft::ceildiv<size_t>(batch_size, block_dim.x), grid_y, 1);

    naive_distance_kernel<EvalT, DataT, IdxT><<<grid_dim, block_dim, 0, stream>>>(
      dist.data(), x + offset * dim, y, batch_size, input_len, dim, type);

    raft::matrix::detail::select_k<EvalT, IdxT>(handle,
                                                dist.data(),
                                                nullptr,
                                                batch_size,
                                                input_len,
                                                static_cast<int>(k),
                                                dist_topk + offset * k,
                                                indices_topk + offset * k,
                                                cuvs::distance::is_min_close(type));
  }
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

}  // namespace cuvs::neighbors
