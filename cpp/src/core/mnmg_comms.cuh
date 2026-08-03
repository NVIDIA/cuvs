/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/nccl_comm.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cuvs::core::detail {

template <typename T>
inline constexpr bool unsupported_nccl_dtype_v = false;

template <typename T>
inline ncclDataType_t nccl_dtype()
{
  using U = std::remove_cv_t<T>;
  if constexpr (std::is_same_v<U, float>) {
    return ncclFloat;
  } else if constexpr (std::is_same_v<U, double>) {
    return ncclDouble;
  } else if constexpr (std::is_same_v<U, int> || std::is_same_v<U, std::int32_t>) {
    return ncclInt32;
  } else if constexpr (std::is_same_v<U, std::int64_t>) {
    return ncclInt64;
  } else {
    static_assert(unsupported_nccl_dtype_v<U>,
                  "Unsupported NCCL data type for MNMG collectives. Supported "
                  "types are float, double, int/std::int32_t, and std::int64_t.");
    return ncclFloat;
  }
}

inline ncclRedOp_t nccl_op(raft::comms::op_t op)
{
  switch (op) {
    case raft::comms::op_t::SUM: return ncclSum;
    case raft::comms::op_t::PROD: return ncclProd;
    case raft::comms::op_t::MIN: return ncclMin;
    case raft::comms::op_t::MAX: return ncclMax;
  }
  RAFT_FAIL("Unsupported allreduce operation");
}

class mnmg_comms {
 public:
  mnmg_comms(raft::resources const& dev_res, bool use_nccl, ncclComm_t nccl_comm)
    : dev_res_(dev_res),
      use_nccl_(use_nccl),
      nccl_comm_(nccl_comm),
      stream_(raft::resource::get_cuda_stream(dev_res_))
  {
  }

  cudaStream_t stream() const { return stream_; }

  template <typename T>
  void allreduce(T* sendbuf,
                 T* recvbuf,
                 std::size_t count,
                 raft::comms::op_t op = raft::comms::op_t::SUM) const
  {
    if (use_nccl_) {
      RAFT_NCCL_TRY(
        ncclAllReduce(sendbuf, recvbuf, count, nccl_dtype<T>(), nccl_op(op), nccl_comm_, stream_));
    } else {
      const auto& comm = raft::resource::get_comms(dev_res_);
      comm.allreduce(sendbuf, recvbuf, count, op, stream_);
    }
  }

  template <typename T>
  void bcast(T* buf, std::size_t count, int root) const
  {
    if (use_nccl_) {
      RAFT_NCCL_TRY(ncclBroadcast(buf, buf, count, nccl_dtype<T>(), root, nccl_comm_, stream_));
    } else {
      const auto& comm = raft::resource::get_comms(dev_res_);
      comm.bcast(buf, count, root, stream_);
    }
  }

  template <typename T>
  void bcast(const T* sendbuf, T* recvbuf, std::size_t count, int root) const
  {
    if (use_nccl_) {
      RAFT_NCCL_TRY(
        ncclBroadcast(sendbuf, recvbuf, count, nccl_dtype<T>(), root, nccl_comm_, stream_));
    } else {
      const auto& comm = raft::resource::get_comms(dev_res_);
      comm.bcast(sendbuf, recvbuf, count, root, stream_);
    }
  }

  template <typename T>
  void allgather(T* sendbuf, T* recvbuf, std::size_t count) const
  {
    if (use_nccl_) {
      RAFT_NCCL_TRY(ncclAllGather(sendbuf, recvbuf, count, nccl_dtype<T>(), nccl_comm_, stream_));
    } else {
      const auto& comm = raft::resource::get_comms(dev_res_);
      comm.allgather(sendbuf, recvbuf, count, stream_);
    }
  }

  template <typename T>
  void allgatherv(const T* sendbuf,
                  T* recvbuf,
                  const std::size_t* recvcounts,
                  const std::size_t* displs,
                  std::size_t my_count) const
  {
    if (use_nccl_) {
      int my_rank   = 0;
      int num_ranks = 0;
      RAFT_NCCL_TRY(ncclCommUserRank(nccl_comm_, &my_rank));
      RAFT_NCCL_TRY(ncclCommCount(nccl_comm_, &num_ranks));

      RAFT_NCCL_TRY(ncclGroupStart());
      for (int r = 0; r < num_ranks; ++r) {
        if (r == my_rank) { continue; }
        if (my_count > 0) {
          RAFT_NCCL_TRY(ncclSend(sendbuf, my_count, nccl_dtype<T>(), r, nccl_comm_, stream_));
        }
        if (recvcounts[r] > 0) {
          RAFT_NCCL_TRY(
            ncclRecv(recvbuf + displs[r], recvcounts[r], nccl_dtype<T>(), r, nccl_comm_, stream_));
        }
      }
      RAFT_NCCL_TRY(ncclGroupEnd());

      if (my_count > 0 && sendbuf != recvbuf + displs[my_rank]) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(recvbuf + displs[my_rank],
                                      sendbuf,
                                      my_count * sizeof(T),
                                      cudaMemcpyDeviceToDevice,
                                      stream_));
      }
    } else {
      const auto& comm = raft::resource::get_comms(dev_res_);
      comm.allgatherv(sendbuf, recvbuf, recvcounts, displs, stream_);
    }
  }

  template <typename T>
  void reduce(T* sendbuf,
              T* recvbuf,
              std::size_t count,
              int root,
              raft::comms::op_t op = raft::comms::op_t::SUM) const
  {
    if (use_nccl_) {
      RAFT_NCCL_TRY(ncclReduce(
        sendbuf, recvbuf, count, nccl_dtype<T>(), nccl_op(op), root, nccl_comm_, stream_));
    } else {
      const auto& comm = raft::resource::get_comms(dev_res_);
      comm.reduce(sendbuf, recvbuf, count, op, root, stream_);
    }
  }

  void group_start() const
  {
    if (use_nccl_) { RAFT_NCCL_TRY(ncclGroupStart()); }
  }

  void group_end() const
  {
    if (use_nccl_) { RAFT_NCCL_TRY(ncclGroupEnd()); }
  }

 private:
  raft::resources const& dev_res_;
  bool use_nccl_;
  ncclComm_t nccl_comm_;
  cudaStream_t stream_;
};

}  // namespace cuvs::core::detail
