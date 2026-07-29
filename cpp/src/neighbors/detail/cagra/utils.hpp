/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// TODO: This shouldn't be invoking anything from detail outside of neighbors namespace
#include <raft/core/copy.cuh>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {
namespace utils {
template <class DATA_T>
inline cudaDataType_t get_cuda_data_type();
template <>
inline cudaDataType_t get_cuda_data_type<float>()
{
  return CUDA_R_32F;
}
template <>
inline cudaDataType_t get_cuda_data_type<half>()
{
  return CUDA_R_16F;
}
template <>
inline cudaDataType_t get_cuda_data_type<int8_t>()
{
  return CUDA_R_8I;
}
template <>
inline cudaDataType_t get_cuda_data_type<uint8_t>()
{
  return CUDA_R_8U;
}
template <>
inline cudaDataType_t get_cuda_data_type<uint32_t>()
{
  return CUDA_R_32U;
}
template <>
inline cudaDataType_t get_cuda_data_type<uint64_t>()
{
  return CUDA_R_64U;
}

template <class T>
constexpr unsigned size_of();
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::int8_t>()
{
  return 1;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint8_t>()
{
  return 1;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint16_t>()
{
  return 2;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint32_t>()
{
  return 4;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint64_t>()
{
  return 8;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<uint4>()
{
  return 16;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<ulonglong4>()
{
  return 32;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<float>()
{
  return 4;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<half>()
{
  return 2;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<half2>()
{
  return 4;
}

// max values for data types
template <class BS_T, class FP_T>
union fp_conv {
  BS_T bs;
  FP_T fp;
};
template <class T>
_RAFT_HOST_DEVICE constexpr inline T get_max_value();
template <>
_RAFT_HOST_DEVICE constexpr inline float get_max_value<float>()
{
  return FLT_MAX;
};
template <>
_RAFT_HOST_DEVICE constexpr inline half get_max_value<half>()
{
  return fp_conv<std::uint16_t, half>{.bs = 0x7aff}.fp;
};
template <>
_RAFT_HOST_DEVICE constexpr inline std::uint32_t get_max_value<std::uint32_t>()
{
  return 0xffffffffu;
};
template <>
_RAFT_HOST_DEVICE constexpr inline std::uint64_t get_max_value<std::uint64_t>()
{
  return 0xfffffffffffffffflu;
};

template <int A, int B, class = void>
struct constexpr_max {
  static const int value = A;
};

template <int A, int B>
struct constexpr_max<A, B, std::enable_if_t<(B > A), bool>> {
  static const int value = B;
};

template <class IdxT>
struct gen_index_msb_1_mask {
  static constexpr IdxT value = static_cast<IdxT>(1) << (utils::size_of<IdxT>() * 8 - 1);
};
}  // namespace utils

template <typename T>
bool is_ptr_device_accessible(T* ptr)
{
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, ptr));
  return attr.devicePointer != nullptr;
}

template <typename T>
bool is_ptr_host_accessible(T* ptr)
{
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, ptr));
  return attr.hostPointer != nullptr;
}

/**
 * Utility to sync memory from a host_matrix_view to a device_matrix_view
 *
 * In certain situations (UVM/HMM/ATS) host memory might be directly accessible on the
 * device, and no extra allocations need to be performed. This class checks
 * if the host_matrix_view is already accessible on the device, and only creates device
 * memory and copies over if necessary. In memory limited situations this is preferable
 * to having both a host and device copy
 * TODO: once the mdbuffer changes here https://github.com/wphicks/raft/blob/fea-mdbuffer
 * have been merged, we should remove this class and switch over to using mdbuffer for this
 */
template <typename T, typename IdxT>
class device_matrix_view_from_host {
 public:
  device_matrix_view_from_host(raft::resources const& res,
                               raft::host_matrix_view<T, IdxT> host_view)
    : res_(res), host_view_(host_view)
  {
    cudaPointerAttributes attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, host_view.data_handle()));
    device_ptr      = reinterpret_cast<T*>(attr.devicePointer);
    bool needs_copy = (device_ptr == NULL) ||
                      (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged);
    if (needs_copy) {
      // allocate memory and copy over
      // NB: We use the temporary "large" workspace resource here; this structure is supposed to
      // live on stack and not returned to a user.
      // The user may opt to set this resource to managed memory to allow large allocations.
      device_mem_.emplace(raft::make_device_mdarray<T, IdxT>(
        res, raft::resource::get_large_workspace_resource_ref(res), host_view.extents()));
      raft::copy(res, device_mem_->view(), host_view);
      device_ptr = device_mem_->data_handle();
    }
  }

  ~device_matrix_view_from_host() noexcept
  {
    /*
    If there's no copy, there's no allocation owned by this struct.
    If there's no allocation, there's no guarantee that the device pointer is stream-ordered.
    If there's no stream order guarantee, we must synchronize with the stream before the struct is
    destroyed to make sure all GPU operations in that stream finish earlier.
    */
    if (!allocated_memory()) { raft::resource::sync_stream(res_); }
  }

  raft::device_matrix_view<T, IdxT> view()
  {
    return raft::make_device_matrix_view<T, IdxT>(
      device_ptr, host_view_.extent(0), host_view_.extent(1));
  }

  T* data_handle() { return device_ptr; }

  [[nodiscard]] bool allocated_memory() const { return device_mem_.has_value(); }

 private:
  const raft::resources& res_;
  std::optional<raft::device_matrix<T, IdxT>> device_mem_;
  raft::host_matrix_view<T, IdxT> host_view_;
  T* device_ptr;
};

/**
 * Utility to sync memory from a device_matrix_view to a host_matrix_view
 *
 * In certain situations (UVM/HMM/ATS) device memory might be directly accessible on the
 * host, and no extra allocations need to be performed. This class checks
 * if the device_matrix_view is already accessible on the host, and only creates host
 * memory and copies over if necessary. In memory limited situations this is preferable
 * to having both a host and device copy
 * TODO: once the mdbuffer changes here https://github.com/wphicks/raft/blob/fea-mdbuffer
 * have been merged, we should remove this class and switch over to using mdbuffer for this
 */
template <typename T, typename IdxT>
class host_matrix_view_from_device {
 public:
  host_matrix_view_from_device(raft::resources const& res,
                               raft::device_matrix_view<T, IdxT> device_view)
    : device_view_(device_view)
  {
    cudaPointerAttributes attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, device_view.data_handle()));
    host_ptr = reinterpret_cast<T*>(attr.hostPointer);
    if (host_ptr == NULL) {
      // allocate memory and copy over
      host_mem_.emplace(
        raft::make_host_matrix<T, IdxT>(device_view.extent(0), device_view.extent(1)));
      raft::copy(res, host_mem_->view(), device_view);
      host_ptr = host_mem_->data_handle();
    }
  }

  raft::host_matrix_view<T, IdxT> view()
  {
    return raft::make_host_matrix_view<T, IdxT>(
      host_ptr, device_view_.extent(0), device_view_.extent(1));
  }

  T* data_handle() { return host_ptr; }

  bool allocated_memory() const { return host_mem_.has_value(); }

 private:
  std::optional<raft::host_matrix<T, IdxT>> host_mem_;
  raft::device_matrix_view<T, IdxT> device_view_;
  T* host_ptr;
};

// Copy matrix src to dst. pad rows with 0 if necessary to make them 16 byte aligned.
template <typename T, typename data_accessor>
void copy_with_padding(
  raft::resources const& res,
  raft::device_matrix<T, int64_t, raft::row_major>& dst,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, data_accessor> src,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref())
{
  size_t padded_dim = raft::round_up_safe<size_t>(src.extent(1) * sizeof(T), 16) / sizeof(T);

  if ((dst.extent(0) != src.extent(0)) || (static_cast<size_t>(dst.extent(1)) != padded_dim)) {
    // clear existing memory before allocating to prevent OOM errors on large datasets
    if (dst.size()) { dst = raft::make_device_matrix<T, int64_t>(res, 0, 0); }
    dst =
      raft::make_device_mdarray<T>(res, mr, raft::make_extents<int64_t>(src.extent(0), padded_dim));
  }
  if (dst.extent(1) == src.extent(1)) {
    raft::copy(res, dst.view(), src);
  } else {
    // copy with padding
    raft::matrix::fill(res, dst.view(), T(0));
    raft::copy_matrix(dst.data_handle(),
                      dst.extent(1),
                      src.data_handle(),
                      src.extent(1),
                      src.extent(1),
                      src.extent(0),
                      raft::resource::get_cuda_stream(res));
  }
}

/**
 * Utility to create a batched device view from a host view
 *
 * This utility will create a batched device view from a host view and will handle the prefetch and
 * writeback of the data Each batch can be referenced exactlyonce by calling the next_view()
 * function
 *
 * Usage:
 * ```
 * batched_device_view_from_host<float, int32_t> view(res, host_view, batch_size, host_writeback,
 * initialize); while (view.next_view().extent(0) > 0) { auto device_view = view.next_view();
 *   // use device_view
 * }
 * ```
 *
 * The call to next_view() will
 * * synchronize on all previous operations / increments batch_id_
 * * (optionally) write back the data of the previous batch to the host
 * * (optionally) prefetch the data of the next batch
 * * return the view of the current batch
 *
 * @tparam T The type of the data
 * @tparam IdxT The type of the index
 */
template <typename T, typename IdxT>
class batched_device_view_from_host {
 public:
  enum class memory_strategy {
    device_only,   // data is on device only (no copy needed)
    copy_device,   // data is explicitly moved to/from device buffers
    managed_only,  // data is on managed memory (system managed)
  };

  /**
   * Create a batched device view from a host view and will handle the prefetch and
   * writeback of the data. Each batch can be referenced exactly once by calling the next_view()
   * method.
   *
   * @param res The resources to use
   * @param host_view The host view to create the batched device view from
   * @param batch_size The batch size
   * @param host_writeback Whether to write back the data to the host (only for host memory)
   * (default: false)
   * @param initialize Whether to initialize the data (only for managed memory) (default: true)
   */
  batched_device_view_from_host(raft::resources const& res,
                                raft::host_matrix_view<T, IdxT> host_view,
                                uint64_t batch_size,
                                bool host_writeback = false,
                                bool initialize     = true)
    : res_(res),
      host_view_(host_view),
      batch_size_(batch_size),
      offset_(0),
      batch_id_(-2),
      num_buffers_(2),
      host_writeback_(host_writeback),
      initialize_(initialize)
  {
    if (host_view.extent(0) == 0) {
      mem_strategy_ = memory_strategy::device_only;
      return;
    }

    RAFT_EXPECTS(host_writeback_ || initialize_,
                 "At least one of host_writeback or initialize must be true");

    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr_, host_view.data_handle()));
    switch (attr_.type) {
      case cudaMemoryTypeUnregistered:
      case cudaMemoryTypeHost:
      case cudaMemoryTypeManaged: mem_strategy_ = memory_strategy::copy_device; break;
      case cudaMemoryTypeDevice: mem_strategy_ = memory_strategy::device_only; break;
    }

    RAFT_LOG_DEBUG("Memory strategy: %d for type %d, size %zu",
                   static_cast<int>(mem_strategy_),
                   static_cast<int>(attr_.type),
                   host_view.extent(0) * host_view.extent(1) * sizeof(T));

    // buffer allocations
    if (mem_strategy_ == memory_strategy::copy_device) {
      try {
        device_mem_[0].emplace(raft::make_device_mdarray<T, IdxT>(
          res,
          raft::resource::get_workspace_resource_ref(res),
          raft::make_extents<int64_t>(batch_size, host_view.extent(1))));
        device_ptr[0] = device_mem_[0]->data_handle();
        if (batch_size < static_cast<uint64_t>(host_view.extent(0))) {
          device_mem_[1].emplace(raft::make_device_mdarray<T, IdxT>(
            res,
            raft::resource::get_workspace_resource_ref(res),
            raft::make_extents<int64_t>(batch_size, host_view.extent(1))));
          device_ptr[1] = device_mem_[1]->data_handle();
        }
        if (host_writeback_ && initialize_ &&
            batch_size * 2 < static_cast<uint64_t>(host_view.extent(0))) {
          num_buffers_ = 3;
          device_mem_[2].emplace(raft::make_device_mdarray<T, IdxT>(
            res,
            raft::resource::get_workspace_resource_ref(res),
            raft::make_extents<int64_t>(batch_size, host_view.extent(1))));
          device_ptr[2] = device_mem_[2]->data_handle();
        }
      } catch (std::bad_alloc& e) {
        if (attr_.devicePointer != nullptr) {
          RAFT_LOG_DEBUG("Insufficient memory for device buffers, switching to managed memory");
          mem_strategy_ = memory_strategy::managed_only;
        } else {
          throw std::bad_alloc();
        }
      } catch (raft::logic_error& e) {
        if (attr_.devicePointer != nullptr) {
          RAFT_LOG_DEBUG(
            "Insufficient memory for device buffers (logic error), switching to managed memory");
          mem_strategy_ = memory_strategy::managed_only;
        } else {
          throw raft::logic_error("Insufficient memory for device buffers (logic error)");
        }
      }
    }

    // setup stream pool if not already present
    size_t required_streams = host_writeback_ && initialize_ ? 2 : 1;
    if (!res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL) ||
        raft::resource::get_stream_pool_size(res) < required_streams) {
      // always create at least 2 streams to account for subsequent iterator calls.
      // set_cuda_stream_pool now requires a non-const resource; the referenced resource
      // outlives this object, so attaching the pool to it here is safe.
      raft::resource::set_cuda_stream_pool(const_cast<raft::resources&>(res),
                                           std::make_shared<rmm::cuda_stream_pool>(2));
    }
    prefetch_stream_  = raft::resource::get_stream_from_stream_pool(res);
    writeback_stream_ = raft::resource::get_stream_from_stream_pool(res);

    // if data is managed and not for_write_ we can set the attribute on the device ptr
    if (mem_strategy_ == memory_strategy::managed_only) {
      location_.type = cudaMemLocationTypeDevice;
      location_.id   = static_cast<CUdevice>(raft::resource::get_device_id(res_));
      if (!host_writeback_) {
        advise_read_mostly(host_view_.data_handle(),
                           host_view_.extent(0) * host_view_.extent(1) * sizeof(T));
        // TODO maybe also reset upon destruction
      }
    }

    // prefetch next batch (0)
    prefetch_next_batch();
  }

  ~batched_device_view_from_host() noexcept
  {
    raft::resource::sync_stream(res_);

    // if data is on host and for_write --> make sure to copy back last active
    // if data is managed and evict --> evict last active

    // make sure to sync on prefetch stream & res
    switch (mem_strategy_) {
      case memory_strategy::managed_only:
        if (!host_writeback_) {
          uint32_t discard_pos     = batch_id_ % num_buffers_;
          size_t discard_size_rows = actual_batch_size_[discard_pos];
          if (batch_id_ > 0) {
            discard_pos = (batch_id_ - 1) % num_buffers_;
            discard_size_rows += batch_size_;
          }
          discard_managed_region(device_ptr[discard_pos],
                                 discard_size_rows * host_view_.extent(1) * sizeof(T));
          writeback_stream_.synchronize();
        }
        break;
      case memory_strategy::copy_device:
        if (host_writeback_) {
          uint32_t writeback_pos_last = batch_id_ % num_buffers_;
          if (batch_id_ > 0) {
            uint32_t writeback_pos    = (batch_id_ - 1) % num_buffers_;
            uint64_t writeback_offset = (batch_id_ - 1) * batch_size_;
            writeback_from_device_to_host(device_ptr[writeback_pos], writeback_offset, batch_size_);
          }
          {
            uint64_t writeback_offset_last = batch_id_ * batch_size_;
            writeback_from_device_to_host(device_ptr[writeback_pos_last],
                                          writeback_offset_last,
                                          actual_batch_size_[writeback_pos_last]);
          }
          writeback_stream_.synchronize();
        }
        break;
      case memory_strategy::device_only: break;
    }
  }

  /**
   * Returns the next view of the batch
   *
   * This function will ensure the next batch is ready and will trigger the prefetch of the
   * subsequent next batch. If writeback is enabled, the last active batch will be written back to
   * the host.
   *
   * @return The next view of the batch
   */
  raft::device_matrix_view<T, IdxT> next_view()
  {
    bool end_of_data = static_cast<uint64_t>((batch_id_ + 1) * batch_size_) >=
                       static_cast<uint64_t>(host_view_.extent(0));

    // special case for empty host view or last batch surpassed
    if (end_of_data) {
      return raft::make_device_matrix_view<T, IdxT>(nullptr, 0, host_view_.extent(1));
    }

    // trigger prefetch of next batch (also increments batch_id_)
    prefetch_next_batch();

    uint32_t current_pos = batch_id_ % num_buffers_;
    return raft::make_device_matrix_view<T, IdxT>(
      device_ptr[current_pos], actual_batch_size_[current_pos], host_view_.extent(1));
  }

 private:
  /**
   * Prefetch the next batch
   *
   * This function will prefetch the next batch and will handle the writeback of the data.
   *
   * @return True if the next batch exists, false otherwise
   */
  bool prefetch_next_batch()
  {
    batch_id_++;

    // ensure previous batch at position batch_id_ is ready
    if (initialize_) { prefetch_stream_.synchronize(); }
    if (host_writeback_) { writeback_stream_.synchronize(); }

    // this step will
    // * write back data from batch_id_ - 1
    // * prefetch data for batch_id_ + 1

    // if data is on host and host_writeback_ is true we will have to copy it back
    // if data is on host and initialize_ is true we will have to copy it to the device_ptr

    // if data is managed and !host_writeback_ we can discard the data from device memory
    // if data is managed and initialize_ is true we can prefetch it to the device
    // if data is managed and !initialize_ we can discard and prefetch the data location

    // if data is on device only this is almost a noop, just prepping the pointers

    RAFT_EXPECTS(static_cast<int64_t>(offset_) <= host_view_.extent(0), "Offset out of bounds");

    bool next_batch_exists = offset_ < static_cast<uint64_t>(host_view_.extent(0));

    if (next_batch_exists) {
      // synchronize to ensure all previous operations are completed
      // in particular all work on batch_id_ - 1
      raft::resource::sync_stream(res_);

      int32_t prefetch_pos             = (batch_id_ + 1) % num_buffers_;
      actual_batch_size_[prefetch_pos] = min(batch_size_, host_view_.extent(0) - offset_);

      switch (mem_strategy_) {
        case memory_strategy::managed_only:
          if (!host_writeback_ && batch_id_ > 1) {
            uint32_t discard_pos = (batch_id_ - 1) % num_buffers_;
            size_t discard_size  = batch_size_ * host_view_.extent(1) * sizeof(T);
            discard_managed_region(device_ptr[discard_pos], discard_size);
          }
          // prefetch next position
          device_ptr[prefetch_pos] = host_view_.data_handle() + offset_ * host_view_.extent(1);
          prefetch_managed_region(
            device_ptr[prefetch_pos],
            actual_batch_size_[prefetch_pos] * host_view_.extent(1) * sizeof(T));
          break;
        case memory_strategy::copy_device:
          if (host_writeback_ && batch_id_ > 0) {
            // copy back last active
            uint32_t writeback_pos    = (batch_id_ - 1) % num_buffers_;
            uint64_t writeback_offset = (batch_id_ - 1) * batch_size_;
            writeback_from_device_to_host(device_ptr[writeback_pos], writeback_offset, batch_size_);
          }
          if (initialize_) {
            // prefetch next position
            prefetch_from_host_to_device(
              device_ptr[prefetch_pos], offset_, actual_batch_size_[prefetch_pos]);
          }

          break;
        case memory_strategy::device_only:
          // just move pointer to next position
          device_ptr[prefetch_pos] = host_view_.data_handle() + offset_ * host_view_.extent(1);
          break;
      }

      offset_ += actual_batch_size_[prefetch_pos];
    }

    return next_batch_exists;
  }

  void advise_read_mostly(T* ptr, size_t size)
  {
#if CUDA_VERSION >= 13000
    RAFT_CUDA_TRY(cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, location_));
#else
    RAFT_CUDA_TRY(cudaMemAdvise_v2(ptr, size, cudaMemAdviseSetReadMostly, location_));
#endif
  }

  void discard_managed_region(T* dev_ptr, size_t size)
  {
#if CUDA_VERSION >= 13000
    void* dptrs[1]  = {dev_ptr};
    size_t sizes[1] = {size};
    RAFT_CUDA_TRY(cudaMemDiscardBatchAsync(dptrs, sizes, 1, 0, writeback_stream_));
#endif
    // FIXME: CUDA12 does not support discard
  }

  void prefetch_managed_region(T* dev_ptr, size_t size)
  {
#if CUDA_VERSION >= 13000
    if (initialize_) {
      RAFT_CUDA_TRY(cudaMemPrefetchAsync(dev_ptr, size, location_, 0, prefetch_stream_));
    } else {
      void* dptrs[1]  = {dev_ptr};
      size_t sizes[1] = {size};
      RAFT_CUDA_TRY(
        cudaMemDiscardAndPrefetchBatchAsync(dptrs, sizes, 1, location_, 0, prefetch_stream_));
    }
#else
    // FIXME: CUDA12 does not support discard - so we just prefetch
    if (initialize_) {
      RAFT_CUDA_TRY(cudaMemPrefetchAsync_v2(dev_ptr, size, location_, 0, prefetch_stream_));
    } else {
      RAFT_CUDA_TRY(cudaMemPrefetchAsync_v2(dev_ptr, size, location_, 0, prefetch_stream_));
    }
#endif
  }

  void prefetch_from_host_to_device(T* dev_ptr, size_t src_row_offset, size_t num_rows)
  {
    const size_t n_elem  = num_rows * host_view_.extent(1);
    const size_t n_bytes = n_elem * sizeof(T);
    // use memcpy instead of raft::copy to avoid strange behavior with HMM/ATS memory
    RAFT_CUDA_TRY(cudaMemcpyAsync(dev_ptr,
                                  host_view_.data_handle() + src_row_offset * host_view_.extent(1),
                                  n_bytes,
                                  cudaMemcpyHostToDevice,
                                  prefetch_stream_));
  }

  void writeback_from_device_to_host(T* dev_ptr, size_t dst_row_offset, size_t num_rows)
  {
    const size_t n_elem  = num_rows * host_view_.extent(1);
    const size_t n_bytes = n_elem * sizeof(T);
    // use memcpy instead of raft::copy to avoid strange behavior with HMM/ATS memory
    RAFT_CUDA_TRY(cudaMemcpyAsync(host_view_.data_handle() + dst_row_offset * host_view_.extent(1),
                                  dev_ptr,
                                  n_bytes,
                                  cudaMemcpyDeviceToHost,
                                  writeback_stream_));
  }

  // stream pool for local streams
  std::optional<std::shared_ptr<rmm::cuda_stream_pool>> local_stream_pool_;
  rmm::cuda_stream_view prefetch_stream_;
  rmm::cuda_stream_view writeback_stream_;

  // configuration
  memory_strategy mem_strategy_;
  const raft::resources& res_;
  bool initialize_;      // initialize the data on the device
  bool host_writeback_;  // write back the data to the host

  // batch position information
  uint64_t batch_size_;
  int32_t batch_id_;
  uint64_t offset_;

  cudaMemLocation location_;

  // input pointer information
  raft::host_matrix_view<T, IdxT> host_view_;
  cudaPointerAttributes attr_;

  // internal device buffers
  uint64_t num_buffers_;
  std::optional<raft::device_matrix<T, IdxT>> device_mem_[3];
  T* device_ptr[3];
  uint32_t actual_batch_size_[3];
};

}  // namespace cuvs::neighbors::cagra::detail
