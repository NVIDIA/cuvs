/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include <cuvs/version_config.h>

#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/memory_tracking_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rapids_logger/logger.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include "../core/exceptions.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

extern "C" cuvsError_t cuvsResourcesCreate(cuvsResources_t* res)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = new raft::resources{};
    *res         = reinterpret_cast<uintptr_t>(res_ptr);
  });
}

extern "C" cuvsError_t cuvsResourcesSetWorkspacePool(cuvsResources_t res, size_t initial_size_bytes)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    // Create an uncapped pool: pre-warms with initial_size_bytes to avoid cudaMalloc on every
    // query, but can grow beyond that if an allocation exceeds the initial reservation.
    raft::resource::set_workspace_resource(
      *res_ptr,
      rmm::mr::pool_memory_resource{rmm::mr::get_current_device_resource_ref(),
                                    initial_size_bytes});
  });
}

extern "C" cuvsError_t cuvsResourcesCreateWithMemoryTracking(cuvsResources_t* res,
                                                             const char* csv_path,
                                                             int64_t sample_interval_ms)
{
  return cuvs::core::translate_exceptions([=] {
    if (csv_path == nullptr || csv_path[0] == '\0') {
      throw std::invalid_argument("csv_path must be a non-empty string");
    }
    if (sample_interval_ms < 0) {
      throw std::invalid_argument("sample_interval_ms must be >= 0");
    }
    auto res_ptr = new raft::memory_tracking_resources{
      std::string{csv_path}, std::chrono::milliseconds{sample_interval_ms}};
    *res = reinterpret_cast<uintptr_t>(res_ptr);
  });
}

extern "C" cuvsError_t cuvsResourcesDestroy(cuvsResources_t res)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    delete res_ptr;
  });
}

extern "C" cuvsError_t cuvsMultiGpuResourcesCreate(cuvsResources_t* res)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = new raft::device_resources_snmg{};
    *res         = reinterpret_cast<uintptr_t>(res_ptr);
  });
}

extern "C" cuvsError_t cuvsMultiGpuResourcesCreateWithDeviceIds(cuvsResources_t* res,
                                                                DLManagedTensor* device_ids)
{
  return cuvs::core::translate_exceptions([=] {
    // Basic validation
    if (device_ids == nullptr || device_ids->dl_tensor.data == nullptr) {
      throw std::invalid_argument("device_ids cannot be null");
    }

    // Check data type is int32
    if (device_ids->dl_tensor.dtype.code != kDLInt || device_ids->dl_tensor.dtype.bits != 32) {
      throw std::invalid_argument("device_ids must be int32");
    }

    // Check data is on host
    if (device_ids->dl_tensor.device.device_type != kDLCPU) {
      throw std::invalid_argument("device_ids must be on host memory");
    }

    // Cast void* to int* to perform pointer arithmetic
    int* data_ptr = static_cast<int*>(device_ids->dl_tensor.data);
    std::vector<int> ids(data_ptr, data_ptr + device_ids->dl_tensor.shape[0]);

    auto res_ptr = new raft::device_resources_snmg{ids};
    *res         = reinterpret_cast<uintptr_t>(res_ptr);
  });
}

extern "C" cuvsError_t cuvsMultiGpuResourcesDestroy(cuvsResources_t res)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::device_resources_snmg*>(res);
    delete res_ptr;
  });
}

extern "C" cuvsError_t cuvsMultiGpuResourcesSetMemoryPool(cuvsResources_t res,
                                                          int percent_of_free_memory)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::device_resources_snmg*>(res);
    res_ptr->set_memory_pool(percent_of_free_memory);
  });
}

extern "C" cuvsError_t cuvsStreamSet(cuvsResources_t res, cudaStream_t stream)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    raft::resource::set_cuda_stream(*res_ptr, static_cast<rmm::cuda_stream_view>(stream));
  });
}

extern "C" cuvsError_t cuvsStreamGet(cuvsResources_t res, cudaStream_t* stream)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    *stream      = raft::resource::get_cuda_stream(*res_ptr);
  });
}

extern "C" cuvsError_t cuvsStreamSync(cuvsResources_t res)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    raft::resource::sync_stream(*res_ptr);
  });
}

extern "C" cuvsError_t cuvsDeviceIdGet(cuvsResources_t res, int* device_id)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    *device_id   = raft::resource::get_device_id(*res_ptr);
  });
}

extern "C" cuvsError_t cuvsRMMAlloc(cuvsResources_t res, void** ptr, size_t bytes)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    auto stream  = raft::resource::get_cuda_stream(*res_ptr);
    *ptr         = raft::resource::get_workspace_resource_ref(*res_ptr).allocate(stream, bytes);
  });
}

extern "C" cuvsError_t cuvsRMMFree(cuvsResources_t res, void* ptr, size_t bytes)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    auto stream  = raft::resource::get_cuda_stream(*res_ptr);
    raft::resource::get_workspace_resource_ref(*res_ptr).deallocate(stream, ptr, bytes);
  });
}

extern "C" cuvsError_t cuvsRMMPoolMemoryResourceEnable(int initial_pool_size_percent,
                                                       int max_pool_size_percent,
                                                       bool managed)
{
  return cuvs::core::translate_exceptions([=] {
    auto initial_size = rmm::percent_of_free_device_memory(initial_pool_size_percent);
    auto max_size     = rmm::percent_of_free_device_memory(max_pool_size_percent);

    if (managed) {
      rmm::mr::set_current_device_resource(
        rmm::mr::pool_memory_resource{rmm::mr::managed_memory_resource{}, initial_size, max_size});
    } else {
      rmm::mr::set_current_device_resource(
        rmm::mr::pool_memory_resource{rmm::mr::cuda_memory_resource{}, initial_size, max_size});
    }
  });
}

extern "C" cuvsError_t cuvsRMMAsyncMemoryResourceEnable()
{
  return cuvs::core::translate_exceptions([=] {
    rmm::mr::set_current_device_resource(rmm::mr::cuda_async_memory_resource{});
  });
}

extern "C" cuvsError_t cuvsRMMMemoryResourceReset()
{
  return cuvs::core::translate_exceptions([=] {
    rmm::mr::reset_current_device_resource();
  });
}

thread_local std::unique_ptr<rmm::mr::pinned_host_memory_resource> pinned_mr;

extern "C" cuvsError_t cuvsRMMHostAlloc(void** ptr, size_t bytes)
{
  return cuvs::core::translate_exceptions([=] {
    if (pinned_mr == nullptr) { pinned_mr = std::make_unique<rmm::mr::pinned_host_memory_resource>(); }
    *ptr = pinned_mr->allocate_sync(bytes);
  });
}

extern "C" cuvsError_t cuvsRMMHostFree(void* ptr, size_t bytes)
{
  return cuvs::core::translate_exceptions([=] { pinned_mr->deallocate_sync(ptr, bytes); });
}

thread_local std::string last_error_text = "";

extern "C" const char* cuvsGetLastErrorText()
{
  return last_error_text.empty() ? NULL : last_error_text.c_str();
}

extern "C" void cuvsSetLogLevel(cuvsLogLevel_t verbosity)
{
  rapids_logger::level_enum level = rapids_logger::level_enum::trace;
  switch (verbosity) {
    case CUVS_LOG_LEVEL_TRACE:
      level = rapids_logger::level_enum::trace;
      break;
    case CUVS_LOG_LEVEL_DEBUG:
      level = rapids_logger::level_enum::debug;
      break;
    case CUVS_LOG_LEVEL_INFO:
      level = rapids_logger::level_enum::info;
      break;
    case CUVS_LOG_LEVEL_WARN:
      level = rapids_logger::level_enum::warn;
      break;
    case CUVS_LOG_LEVEL_ERROR:
      level = rapids_logger::level_enum::error;
      break;
    case CUVS_LOG_LEVEL_CRITICAL:
      level = rapids_logger::level_enum::critical;
      break;
    case CUVS_LOG_LEVEL_OFF:
      level = rapids_logger::level_enum::off;
      break;
    default: RAFT_FAIL("Unsupported cuvsLogLevel_t value provided");
  }
  raft::default_logger().set_level(level);
}

extern "C" cuvsLogLevel_t cuvsGetLogLevel()
{
  return static_cast<cuvsLogLevel_t>(raft::default_logger().level());
}

extern "C" void cuvsSetLastErrorText(const char* error) { last_error_text = error ? error : ""; }

extern "C" cuvsError_t cuvsVersionGet(uint16_t* major, uint16_t* minor, uint16_t* patch)
{
  *major = CUVS_VERSION_MAJOR;
  *minor = CUVS_VERSION_MINOR;
  *patch = CUVS_VERSION_PATCH;
  return CUVS_SUCCESS;
}

namespace {
template <typename T>
void _copy_matrix(cuvsResources_t res, DLManagedTensor* src_managed, DLManagedTensor* dst_managed)
{
  DLTensor& src = src_managed->dl_tensor;
  DLTensor& dst = dst_managed->dl_tensor;
  auto res_ptr  = reinterpret_cast<raft::resources*>(res);
  auto stream   = raft::resource::get_cuda_stream(*res_ptr);

  if (src.ndim == 2) {
    // use raft::copy_matrix for 2D tensors - this will handle copying from strided to non-strided
    // views well
    int64_t src_row_stride = src.strides == nullptr ? src.shape[1] : src.strides[0];
    int64_t dst_row_stride = dst.strides == nullptr ? dst.shape[1] : dst.strides[0];

    raft::copy_matrix<T>(static_cast<T*>(dst.data),
                         dst_row_stride,
                         static_cast<const T*>(src.data),
                         src_row_stride,
                         src.shape[1],
                         src.shape[0],
                         stream);
  } else {
    // Otherwise use cudaMemcpyAsync - and assert that we don't have strided data
    RAFT_EXPECTS(src.strides == nullptr, "cuvsCopyMatrix only supports strides with 2D inputs");
    RAFT_EXPECTS(dst.strides == nullptr, "cuvsCopyMatrix only supports strides with 2D inputs");

    size_t elements = 1;
    for (int64_t i = 0; i < src.ndim; ++i) {
      elements *= src.shape[i];
    }
    raft::copy<T>(static_cast<T*>(dst.data), static_cast<const T*>(src.data), elements, stream);
  }
}
}  // namespace

extern "C" cuvsError_t cuvsMatrixCopy(cuvsResources_t res,
                                      DLManagedTensor* src_managed,
                                      DLManagedTensor* dst_managed)
{
  return cuvs::core::translate_exceptions([=] {
    DLTensor& src = src_managed->dl_tensor;
    DLTensor& dst = dst_managed->dl_tensor;

    RAFT_EXPECTS(src.ndim == dst.ndim, "src and dst tensors should have the same dimensions");

    for (int64_t i = 0; i < src.ndim; ++i) {
      RAFT_EXPECTS(src.shape[i] == dst.shape[i], "shape mismatch between src and dst tensors");
    }
    RAFT_EXPECTS(src.dtype.code == dst.dtype.code, "dtype mismatch between src and dst tensors");

    // at some point we could probably copy from a float32 to a float16 here, but for the
    // moment this isn't supported
    RAFT_EXPECTS(src.dtype.bits == dst.dtype.bits,
                 "dtype bits width mismatch between src and dst tensors");

    if (src.dtype.code == kDLFloat && src.dtype.bits == 32) {
      _copy_matrix<float>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLFloat && src.dtype.bits == 16) {
      _copy_matrix<half>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLFloat && src.dtype.bits == 64) {
      _copy_matrix<double>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLInt && src.dtype.bits == 8) {
      _copy_matrix<int8_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLInt && src.dtype.bits == 16) {
      _copy_matrix<int16_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLInt && src.dtype.bits == 32) {
      _copy_matrix<int32_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLInt && src.dtype.bits == 64) {
      _copy_matrix<int64_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLUInt && src.dtype.bits == 8) {
      _copy_matrix<uint8_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLUInt && src.dtype.bits == 16) {
      _copy_matrix<uint16_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLUInt && src.dtype.bits == 32) {
      _copy_matrix<uint32_t>(res, src_managed, dst_managed);
    } else if (src.dtype.code == kDLUInt && src.dtype.bits == 64) {
      _copy_matrix<uint64_t>(res, src_managed, dst_managed);
    } else {
      RAFT_FAIL("Unsupported dtype: %d and bits: %d", src.dtype.code, src.dtype.bits);
    }
  });
}

extern "C" void cuvsMatrixDestroy(DLManagedTensor* tensor)
{
  if (tensor->dl_tensor.shape != nullptr) {
    delete[] tensor->dl_tensor.shape;
    tensor->dl_tensor.shape = nullptr;
  }
  if (tensor->dl_tensor.strides != nullptr) {
    delete[] tensor->dl_tensor.strides;
    tensor->dl_tensor.strides = nullptr;
  }
}

extern "C" cuvsError_t cuvsMatrixSliceRows(cuvsResources_t res,
                                           DLManagedTensor* src_managed,
                                           int64_t start,
                                           int64_t end,
                                           DLManagedTensor* dst_managed)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(end >= start, "end index must be greater than start index");

    DLTensor& src = src_managed->dl_tensor;
    DLTensor& dst = dst_managed->dl_tensor;
    RAFT_EXPECTS(src.ndim <= 2, "src should be a 1 or 2 dimensional tensor");
    RAFT_EXPECTS(src.shape != nullptr, "shape should be initialized in the src tensor");

    dst.dtype    = src.dtype;
    dst.device   = src.device;
    dst.ndim     = src.ndim;
    dst.shape    = new int64_t[dst.ndim];
    dst.shape[0] = end - start;

    int64_t row_strides = 1;

    if (dst.ndim == 2) {
      dst.shape[1] = src.shape[1];
      row_strides = dst.shape[1];

      if (src.strides) {
        dst.strides = new int64_t[2];
        row_strides = dst.strides[0] = src.strides[0];
        dst.strides[1]               = src.strides[1];
      }
    }

    dst.data = static_cast<char*>(src.data) + start * row_strides * (dst.dtype.bits / 8);
    dst_managed->deleter = cuvsMatrixDestroy;
  });
}
