/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <raft/core/error.hpp>

namespace cuvs::neighbors::detail {

/** Smem high-water + last CUDA kernel handle for one @p KernelT. Handle as uint64_t bits (not
 *  std::atomic<cudaKernel_t>) for portable lock-free atomics. */
template <typename KernelT>
struct jit_kernel_smem_state {
  std::atomic<std::uint32_t> last_smem_size{0};
  std::atomic<std::uint64_t> last_cuda_kernel_bits{0};
  std::mutex mutex;
};

/** One state object per @p KernelT (Meyers singleton). Avoids `static inline` data members in a
 *  class template, which can pull in 16-byte libatomic helpers under NVCC/host linking. */
template <typename KernelT>
jit_kernel_smem_state<KernelT>& jit_kernel_smem_state_for() noexcept
{
  static jit_kernel_smem_state<KernelT> state;
  return state;
}

/**
 * @brief (Thread-)Safely invoke a kernel with a maximum dynamic shared memory size.
 *
 * Maintains a monotonically growing high-water mark for
 * `cudaFuncAttributeMaxDynamicSharedMemorySize`. When the kernel identity changes, the new kernel
 * is brought up to the current high-water mark; when @p smem_size exceeds the high-water mark, it
 * is grown for the current kernel. This guarantees every kernel's attribute is always >= @p
 * smem_size at the time of launch.
 *
 * This is required because the sequence `cudaFuncSetAttribute` + kernel launch is not executed
 * atomically. Used this way, `cudaFuncAttributeMaxDynamicSharedMemorySize` can only grow and the
 * kernel remains safe to launch.
 *
 * NB: cudaFuncSetAttribute is per kernel handle value, not per C++ type. Multiple template
 * instantiations may share the same @p KernelT (e.g. the same function signature), so we track the
 * last @p cuda_kernel handle (as opaque bits) alongside the smem high-water mark.
 *
 * @tparam KernelT Kernel function type from kernel_def.hpp (keys static state per signature).
 * @tparam KernelLauncherT Type of the launch callable (e.g. lambda calling launcher->dispatch).
 * @param smem_size Dynamic shared memory required for this launch.
 * @param launch Invoked after attributes are set; takes no arguments.
 * @param cuda_kernel Handle passed to cudaFuncSetAttribute (e.g. launcher->get_kernel()).
 */
template <typename KernelT, typename KernelLauncherT>
void safely_launch_kernel_with_smem_size(std::uint32_t smem_size,
                                         KernelLauncherT const& launch,
                                         cudaKernel_t cuda_kernel)
{
  auto& st = jit_kernel_smem_state_for<KernelT>();

  std::uint64_t const current_bits =
    static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(cuda_kernel));

  // Fast path: skip the lock when the kernel handle matches and smem is within bounds.
  // Load order matters: last_smem_size (acquire) before last_cuda_kernel_bits (relaxed). Inside
  // the lock we store in the opposite order: last_cuda_kernel_bits (relaxed) then last_smem_size
  // (release). This way an acquire load of last_smem_size that sees a post-cudaFuncSetAttribute
  // value is guaranteed to also see the corresponding handle bits.
  std::uint32_t observed_smem = st.last_smem_size.load(std::memory_order_acquire);
  std::uint64_t observed_bits = st.last_cuda_kernel_bits.load(std::memory_order_relaxed);
  if (smem_size > observed_smem || current_bits != observed_bits) {
    std::lock_guard<std::mutex> guard(st.mutex);
    // Re-check under the lock: the outside decision can be stale.
    std::uint32_t cur_smem_size = st.last_smem_size.load(std::memory_order_relaxed);
    observed_bits               = st.last_cuda_kernel_bits.load(std::memory_order_relaxed);
    bool need_update            = (current_bits != observed_bits);
    if (smem_size > cur_smem_size) {
      cur_smem_size = smem_size;
      need_update   = true;
    }
    if (need_update) {
      auto launch_status = cudaFuncSetAttribute(
        cuda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cur_smem_size);
      RAFT_EXPECTS(launch_status == cudaSuccess,
                   "Failed to set max dynamic shared memory size to %u bytes",
                   cur_smem_size);
      // Store order matters: handle bits before last_smem_size (release) so the fast-path acquire
      // load of last_smem_size also publishes the handle.
      st.last_cuda_kernel_bits.store(current_bits, std::memory_order_relaxed);
      st.last_smem_size.store(cur_smem_size, std::memory_order_release);
    }
  }
  return launch();
}

}  // namespace cuvs::neighbors::detail
