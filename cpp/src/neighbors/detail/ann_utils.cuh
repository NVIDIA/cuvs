/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_fp16.hpp>

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace cuvs::spatial::knn::detail::utils {

/** Whether pointers are accessible on the device or on the host. */
enum class pointer_residency {
  /** Some of the pointers are on the device, some on the host. */
  mixed,
  /** All pointers accessible from both the device and the host. */
  host_and_device,
  /** All pointers are host accessible. */
  host_only,
  /** All poitners are device accessible. */
  device_only
};

template <typename... Types>
struct pointer_residency_count {};

template <>
struct pointer_residency_count<> {
  static inline auto run() -> std::tuple<int, int> { return std::make_tuple(0, 0); }
};

template <typename Type, typename... Types>
struct pointer_residency_count<Type, Types...> {
  static inline auto run(const Type* ptr, const Types*... ptrs) -> std::tuple<int, int>
  {
    auto [on_device, on_host] = pointer_residency_count<Types...>::run(ptrs...);
    cudaPointerAttributes attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, ptr));
    if (attr.devicePointer || attr.type == cudaMemoryTypeDevice) { ++on_device; }
    if (attr.hostPointer || attr.type == cudaMemoryTypeUnregistered) { ++on_host; }
    return std::make_tuple(on_device, on_host);
  }
};

/** Check if all argument pointers reside on the host or on the device. */
template <typename... Types>
auto check_pointer_residency(const Types*... ptrs) -> pointer_residency
{
  auto [on_device, on_host] = pointer_residency_count<Types...>::run(ptrs...);
  int n_args                = sizeof...(Types);
  if (on_device == n_args && on_host == n_args) { return pointer_residency::host_and_device; }
  if (on_device == n_args) { return pointer_residency::device_only; }
  if (on_host == n_args) { return pointer_residency::host_only; }
  return pointer_residency::mixed;
}

/** RAII helper to access the host data from gpu when necessary. */
template <typename PtrT, typename Action>
struct with_mapped_memory_t {
  with_mapped_memory_t(PtrT ptr, size_t size, Action action) : action_(action)
  {
    if (ptr == nullptr) { return; }
    switch (utils::check_pointer_residency(ptr)) {
      case utils::pointer_residency::device_only:
      case utils::pointer_residency::host_and_device: {
        dev_ptr_ = (void*)ptr;  // NOLINT
      } break;
      default: {
        host_ptr_ = (void*)ptr;  // NOLINT
        RAFT_CUDA_TRY(cudaHostRegister(host_ptr_, size, choose_flags(ptr)));
        RAFT_CUDA_TRY(cudaHostGetDevicePointer(&dev_ptr_, host_ptr_, 0));
      } break;
    }
  }

  ~with_mapped_memory_t()
  {
    if (host_ptr_ != nullptr) { cudaHostUnregister(host_ptr_); }
  }

  auto operator()() { return action_((PtrT)dev_ptr_); }  // NOLINT

 private:
  Action action_;
  void* host_ptr_ = nullptr;
  void* dev_ptr_  = nullptr;

  template <typename T>
  static auto choose_flags(const T*) -> unsigned int
  {
    int dev_id, readonly_supported;
    RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(
      &readonly_supported, cudaDevAttrHostRegisterReadOnlySupported, dev_id));
    if (readonly_supported) {
      return cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    } else {
      return cudaHostRegisterMapped;
    }
  }

  template <typename T>
  static auto choose_flags(T*) -> unsigned int
  {
    return cudaHostRegisterMapped;
  }
};

template <typename T>
struct config {};

template <>
struct config<double> {
  using value_t                    = double;
  static constexpr double kDivisor = 1.0;
};
template <>
struct config<float> {
  using value_t                    = float;
  static constexpr double kDivisor = 1.0;
};
template <>
struct config<half> {
  using value_t                    = half;
  static constexpr double kDivisor = 1.0;
};
template <>
struct config<uint8_t> {
  using value_t                    = uint32_t;
  static constexpr double kDivisor = 256.0;
};
template <>
struct config<int8_t> {
  using value_t                    = int32_t;
  static constexpr double kDivisor = 128.0;
};

/**
 * @brief Converting values between the types taking into account scaling factors
 * for the integral types.
 *
 * @tparam T target type of the mapping.
 */
template <typename T>
struct mapping {
  /**
   * @defgroup
   * @brief Cast and possibly scale a value of the source type `S` to the target type `T`.
   *
   * @tparam S source type
   * @param x source value
   * @{
   */
  template <typename S>
  HDI constexpr auto operator()(const S& x) const -> std::enable_if_t<std::is_same_v<S, T>, T>
  {
    return x;
  };

  template <typename S>
  HDI constexpr auto operator()(const S& x) const -> std::enable_if_t<!std::is_same_v<S, T>, T>
  {
    constexpr double kMult = config<T>::kDivisor / config<S>::kDivisor;
    if constexpr (std::is_floating_point_v<S>) { return static_cast<T>(x * static_cast<S>(kMult)); }
    if constexpr (std::is_floating_point_v<T>) { return static_cast<T>(x) * static_cast<T>(kMult); }
    return static_cast<T>(static_cast<float>(x) * static_cast<float>(kMult));
  };
  /** @} */
};

template <>
template <>
HDI constexpr auto mapping<int8_t>::operator()(const uint8_t& x) const -> int8_t
{
  // Avoid overflows when converting uint8_t -> int_8
  return static_cast<int8_t>(x >> 1);
}

template <>
template <>
HDI constexpr auto mapping<int8_t>::operator()(const float& x) const -> int8_t
{
  // Carefully clamp floats if out-of-bounds.
  return static_cast<int8_t>(std::clamp<float>(x * 128.0f, -128.0f, 127.0f));
}

/**
 * @brief Sets the first num bytes of the block of memory pointed by ptr to the specified value.
 *
 * @param[out] ptr host or device pointer
 * @param[in] value
 * @param[in] n_bytes
 */
template <typename T, typename IdxT>
inline void memzero(T* ptr, IdxT n_elems, rmm::cuda_stream_view stream)
{
  switch (check_pointer_residency(ptr)) {
    case pointer_residency::host_and_device:
    case pointer_residency::device_only: {
      RAFT_CUDA_TRY(cudaMemsetAsync(ptr, 0, n_elems * sizeof(T), stream));
    } break;
    case pointer_residency::host_only: {
      stream.synchronize();
      ::memset(ptr, 0, n_elems * sizeof(T));
    } break;
    default: RAFT_FAIL("memset: unreachable code");
  }
}

template <typename T, typename IdxT>
static __global__ void outer_add_kernel(const T* a, IdxT len_a, const T* b, IdxT len_b, T* c)
{
  IdxT gid = threadIdx.x + blockDim.x * static_cast<IdxT>(blockIdx.x);
  IdxT i   = gid / len_b;
  IdxT j   = gid % len_b;
  if (i >= len_a) return;
  c[gid] = (a == nullptr ? T(0) : a[i]) + (b == nullptr ? T(0) : b[j]);
}

template <typename T, typename IdxT>
static __global__ void block_copy_kernel(const IdxT* in_offsets,
                                         const IdxT* out_offsets,
                                         IdxT n_blocks,
                                         const T* in_data,
                                         T* out_data,
                                         IdxT n_mult)
{
  IdxT i = static_cast<IdxT>(blockDim.x) * static_cast<IdxT>(blockIdx.x) + threadIdx.x;
  // find the source offset using the binary search.
  uint32_t l     = 0;
  uint32_t r     = n_blocks;
  IdxT in_offset = 0;
  if (in_offsets[r] * n_mult <= i) return;
  while (l + 1 < r) {
    uint32_t c = (l + r) >> 1;
    IdxT o     = in_offsets[c] * n_mult;
    if (o <= i) {
      l         = c;
      in_offset = o;
    } else {
      r = c;
    }
  }
  // raft::copy the data
  out_data[out_offsets[l] * n_mult - in_offset + i] = in_data[i];
}

/**
 * raft::copy chunks of data from one array to another at given offsets.
 *
 * @tparam T element type
 * @tparam IdxT index type
 *
 * @param[in] in_offsets
 * @param[in] out_offsets
 * @param n_blocks size of the offset arrays minus one.
 * @param[in] in_data
 * @param[out] out_data
 * @param n_mult constant multiplier for offset values (such as e.g. `dim`)
 * @param stream
 */
template <typename T, typename IdxT>
void block_copy(const IdxT* in_offsets,
                const IdxT* out_offsets,
                IdxT n_blocks,
                const T* in_data,
                T* out_data,
                IdxT n_mult,
                rmm::cuda_stream_view stream)
{
  IdxT in_size;
  update_host(&in_size, in_offsets + n_blocks, 1, stream);
  stream.synchronize();
  dim3 threads(128, 1, 1);
  dim3 blocks(raft::ceildiv<IdxT>(in_size * n_mult, threads.x), 1, 1);
  block_copy_kernel<<<blocks, threads, 0, stream>>>(
    in_offsets, out_offsets, n_blocks, in_data, out_data, n_mult);
}

/**
 * @brief Fill matrix `c` with all combinations of sums of vectors `a` and `b`.
 *
 * NB: device-only function
 *
 * @tparam T    element type
 * @tparam IdxT index type
 *
 * @param[in] a device pointer to a vector [len_a]
 * @param len_a number of elements in `a`
 * @param[in] b device pointer to a vector [len_b]
 * @param len_b number of elements in `b`
 * @param[out] c row-major matrix [len_a, len_b]
 * @param stream
 */
template <typename T, typename IdxT>
void outer_add(const T* a, IdxT len_a, const T* b, IdxT len_b, T* c, rmm::cuda_stream_view stream)
{
  dim3 threads(128, 1, 1);
  dim3 blocks(raft::ceildiv<IdxT>(len_a * len_b, threads.x), 1, 1);
  outer_add_kernel<<<blocks, threads, 0, stream>>>(a, len_a, b, len_b, c);
}

template <typename T, typename S, typename IdxT, typename LabelT>
static __global__ void copy_selected_kernel(
  IdxT n_rows, IdxT n_cols, const S* src, const LabelT* row_ids, IdxT ld_src, T* dst, IdxT ld_dst)
{
  IdxT gid   = threadIdx.x + blockDim.x * static_cast<IdxT>(blockIdx.x);
  IdxT j     = gid % n_cols;
  IdxT i_dst = gid / n_cols;
  if (i_dst >= n_rows) return;
  auto i_src              = static_cast<IdxT>(row_ids[i_dst]);
  dst[ld_dst * i_dst + j] = mapping<T>{}(src[ld_src * i_src + j]);
}

/**
 * @brief raft::copy selected rows of a matrix while mapping the data from the source to the target
 * type.
 *
 * @tparam T      target type
 * @tparam S      source type
 * @tparam IdxT   index type
 * @tparam LabelT label type
 *
 * @param n_rows
 * @param n_cols
 * @param[in] src input matrix [..., ld_src]
 * @param[in] row_ids selection of rows to be copied [n_rows]
 * @param ld_src number of cols in the input (ld_src >= n_cols)
 * @param[out] dst output matrix [n_rows, ld_dst]
 * @param ld_dst number of cols in the output (ld_dst >= n_cols)
 * @param stream
 */
template <typename T, typename S, typename IdxT, typename LabelT>
void copy_selected(IdxT n_rows,
                   IdxT n_cols,
                   const S* src,
                   const LabelT* row_ids,
                   IdxT ld_src,
                   T* dst,
                   IdxT ld_dst,
                   rmm::cuda_stream_view stream)
{
  switch (check_pointer_residency(src, dst, row_ids)) {
    case pointer_residency::host_and_device:
    case pointer_residency::device_only: {
      IdxT block_dim = 128;
      IdxT grid_dim  = raft::ceildiv(n_rows * n_cols, block_dim);
      copy_selected_kernel<T, S>
        <<<grid_dim, block_dim, 0, stream>>>(n_rows, n_cols, src, row_ids, ld_src, dst, ld_dst);
    } break;
    case pointer_residency::host_only: {
      stream.synchronize();
      for (IdxT i_dst = 0; i_dst < n_rows; i_dst++) {
        auto i_src = static_cast<IdxT>(row_ids[i_dst]);
        for (IdxT j = 0; j < n_cols; j++) {
          dst[ld_dst * i_dst + j] = mapping<T>{}(src[ld_src * i_src + j]);
        }
      }
      stream.synchronize();
    } break;
    default: RAFT_FAIL("All pointers must reside on the same side, host or device.");
  }
}

/**
 * Helper that returns the stream to use for non-blocking copies and a flag indicating whether
 * cross-stream pipelining (prefetch / writeback) can be enabled.
 *
 * If `res` has a CUDA stream pool with at least one stream, the first pool stream is used and
 * `true` is returned (prefetch can run concurrently with kernels on res's main stream). Otherwise
 * the main stream itself is returned with `false`, and the caller should treat prefetch as a
 * no-op (no overlap is possible on a single stream).
 */
inline auto get_prefetch_stream(raft::resources const& res)
  -> std::pair<rmm::cuda_stream_view, bool>
{
  if (res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL) &&
      raft::resource::get_stream_pool_size(res) >= 1) {
    return {raft::resource::get_stream_from_stream_pool(res), true};
  }
  return {raft::resource::get_cuda_stream(res), false};
}

/**
 * Iterate a 2D mdspan in row-batches with optional pipelined H2D / D2H copies and kernel work.
 *
 * Strategy is selected at compile time from `MdspanT::accessor_type::is_device_accessible`:
 *   * passthrough: each batch is a row-range of `input_view_` directly; no buffering, no copy.
 *   * copy_device: each batch is staged through one or two internal device buffers via
 *     `cudaMemcpyAsync` on the caller-supplied `copy_stream`. With `prefetch=true`, two buffers
 *     are used as a ring so that the next batch's H2D and the previous batch's D2H can overlap
 *     with the user kernel running on `res`'s main stream.
 *
 * Three orthogonal flags control behavior of the copy_device strategy:
 *   * `prefetch`: if true, allocate two device buffers and pipeline copies using
 *     `prefetch_next_batch()`. If false, copies happen synchronously at `operator*` and one buffer
 *     is allocated. As an optimization, when `n_iters_ <= 1` (i.e. `batch_size >=
 *     input_view.extent(0)`) prefetching is internally downgraded to false and only one buffer is
 *     allocated, since there is no "next batch" to overlap with.
 *   * `initialize`: if true, stage source rows H2D into the buffer before yielding the batch.
 *     If false, the buffer is handed out uninitialized (kernel produces the data from scratch).
 *   * `host_writeback`: if true, queue D2H of every advanced batch back to `input_view_`.
 *     Pending writebacks are flushed on destruction.
 *
 * `initialize` and `host_writeback` are independent: it is legal to skip H2D
 * (`initialize=false, host_writeback=true`) when the kernel produces the result from scratch.
 * At least one of them must be true.
 *
 * Stream model (prefetch=true):
 *   * The user passes `copy_stream`, which should be distinct from `res`'s main stream (use
 *     `get_prefetch_stream(res)`); otherwise no real overlap is possible.
 *   * Writeback is done with a 2-iteration delay so that the D2H of batch `i` is queued in the
 *     `prefetch_next_batch()` of iteration `i+2` -- right before the same slot is overwritten by
 *     the next H2D. This way the D2H reads a slot whose user kernel (`kernel_i`) is two
 *     iterations old and therefore guaranteed finished, without needing CUDA events for
 *     cross-stream synchronization.
 *   * `prefetch_next_batch()` queues, on `copy_stream`, in order: (1) D2H of the prefetch slot's
 *     stale kernel output (if `host_writeback`), then (2) H2D of `pos` into the prefetch slot
 *     (if `initialize`); then it `sync_stream(res)`'s so the host stall for the previous kernel
 *     overlaps with the just-queued copies.
 *   * `operator*` (next iteration's `load()`) swaps the ring slots and `synchronize()`'s
 *     `copy_stream` so the swapped-in slot is fully staged before the user's next kernel runs.
 *
 * Stream model (prefetch=false):
 *   * Single buffer; copies happen synchronously inside `operator*` with `copy_stream` then
 *     `synchronize()`'d from the host. No overlap is attempted.
 *
 * Iteration ends when `operator++` reaches `n_iters_`. The iterator can be reused via `reset()`.
 *
 * Usage with prefetch (matches the legacy `batch_load_iterator` pattern):
 * ```
 * auto [copy_stream, enable_prefetch] = utils::get_prefetch_stream(res);
 * utils::batch_load_iterator iter(res, view, batch_size, copy_stream, mr, enable_prefetch);
 * iter.prefetch_next_batch();
 * for (auto const& batch : iter) {
 *   kernel<<<..., raft::resource::get_cuda_stream(res)>>>(batch.data(), ...);
 *   iter.prefetch_next_batch();
 * }
 * ```
 *
 * Usage with writeback (replaces `batched_device_view`):
 * ```
 * auto [copy_stream, enable_prefetch] = utils::get_prefetch_stream(res);
 * utils::batch_load_iterator iter(res, view, batch_size, copy_stream, mr, enable_prefetch,
 *                                 /\*initialize=*\/false, /\*host_writeback=*\/true);
 * iter.prefetch_next_batch();
 * for (auto& batch : iter) {
 *   kernel<<<..., raft::resource::get_cuda_stream(res)>>>(batch.view());
 *   iter.prefetch_next_batch();
 * }
 * ```
 */
template <typename MdspanT>
struct batch_load_iterator {
  using mdspan_type   = MdspanT;
  using accessor_type = typename MdspanT::accessor_type;
  using element_type  = typename MdspanT::element_type;
  using index_type    = typename MdspanT::index_type;
  using value_type_d  = std::remove_const_t<element_type>;
  using size_type     = size_t;

  static constexpr bool kPassthrough = accessor_type::is_device_accessible;

  // Type returned by `view()` for the passthrough strategy: a 2D submdspan of `input_view_` over a
  // contiguous row range. Built without ever calling `data_handle()` on the input mdspan.
  // (Per the mdspan spec, slicing a `layout_right` with a `tuple{lo, hi}` over the leading dim
  // yields a `layout_stride` mdspan with the input's accessor preserved.)
  using passthrough_view_type =
    decltype(cuda::std::submdspan(std::declval<MdspanT&>(),
                                  std::declval<cuda::std::tuple<index_type, index_type>>(),
                                  cuda::std::full_extent));
  // Type returned by `view()` for the copy_device strategy: a row-major exhaustive device view
  // over the iterator's internal device buffer.
  using copy_view_type  = raft::device_matrix_view<element_type, index_type>;
  using batch_view_type = std::conditional_t<kPassthrough, passthrough_view_type, copy_view_type>;

  /** A single batch of data residing in (or accessible from) device memory. */
  struct batch {
    ~batch() noexcept
    {
      if constexpr (!kPassthrough) {
        if (host_writeback_ && source_ != nullptr) {
          // Two slots may still hold un-flushed kernel output:
          //  * `prefetch_dev_ptr_` for the batch the loop didn't reach (its prefetch_next_batch()
          //    that would have queued the D2H never happened).
          //  * `dev_ptr_` for the most-recently-loaded batch (its writeback is always deferred to
          //    the destructor since no future iteration recycles its slot).
          // The user kernel that wrote either slot may still be in flight on `res`'s main stream
          // when this destructor runs, so host-stall on it before issuing the D2Hs to avoid a
          // read-while-write race.
          const bool has_pending =
            (prefetch_ && prefetch_dirty_pos_.has_value()) || (dirty_cur_ && pos_.has_value());
          if (has_pending) { raft::resource::sync_stream(*res_); }
          if (prefetch_ && prefetch_dirty_pos_.has_value()) {
            queue_d2h(prefetch_dev_ptr_, *prefetch_dirty_pos_);
            prefetch_dirty_pos_.reset();
          }
          if (dirty_cur_ && pos_.has_value()) {
            queue_d2h(dev_ptr_, *pos_);
            dirty_cur_ = false;
          }
        }
      }
      // Stream is shared with the iterator; it must be sync'd before the underlying buffers (or,
      // in the passthrough case, the source mdspan) can be safely reused.
      copy_stream_.synchronize();
    }

    [[nodiscard]] auto row_width() const -> size_type { return row_width_; }
    [[nodiscard]] auto offset() const -> size_type { return pos_.value_or(0) * batch_size_; }
    [[nodiscard]] auto size() const -> size_type { return batch_len_; }
    [[nodiscard]] auto does_copy() const -> bool { return !kPassthrough; }

    /**
     * 2D view of the staged batch.
     *
     * Passthrough: a `cuda::std::submdspan` of `input_view_` over the active row range. The
     * implementation never calls `data_handle()` on `input_view_`; the mdspan's accessor is
     * preserved end-to-end, which is the contract that lets future device mdspans without a raw
     * pointer flow through this code path unchanged.
     *
     * Copy_device: a `device_matrix_view` over the internal device buffer (row-major exhaustive).
     */
    [[nodiscard]] auto view() const -> batch_view_type
    {
      if constexpr (kPassthrough) {
        const index_type row_lo = static_cast<index_type>(pos_.value_or(0) * batch_size_);
        const index_type row_hi = static_cast<index_type>(row_lo + batch_len_);
        return cuda::std::submdspan(
          input_view_, cuda::std::tuple{row_lo, row_hi}, cuda::std::full_extent);
      } else {
        return raft::make_device_matrix_view<element_type, index_type>(
          dev_ptr_, static_cast<index_type>(batch_len_), static_cast<index_type>(row_width_));
      }
    }

    /**
     * Raw device pointer of the staged batch. Provided for backward compatibility with raw-pointer
     * call sites. In passthrough mode this forwards to `view().data_handle()`, which means it
     * relies on the input mdspan's accessor exposing a pointer. Future device mdspans without a
     * raw pointer should call `view()` instead and treat the result as an mdspan.
     */
    [[nodiscard]] auto data() const -> element_type*
    {
      if constexpr (kPassthrough) {
        return view().data_handle();
      } else {
        return dev_ptr_;
      }
    }

   private:
    template <typename>
    friend struct batch_load_iterator;

    // Helper: only call `data_handle()` on the input mdspan in copy_device mode. In passthrough
    // mode we keep `source_` at `nullptr` (it is never read) so the iterator imposes no
    // raw-pointer requirement on the input accessor.
    static auto get_source(MdspanT input_view) noexcept -> element_type*
    {
      if constexpr (kPassthrough) {
        return nullptr;
      } else {
        return input_view.data_handle();
      }
    }

    batch(raft::resources const& res,
          MdspanT input_view,
          size_type batch_size,
          rmm::cuda_stream_view copy_stream,
          rmm::device_async_resource_ref mr,
          bool prefetch,
          bool initialize,
          bool host_writeback)
      : copy_stream_(copy_stream),
        res_(&res),
        input_view_(input_view),
        source_(get_source(input_view)),
        n_rows_(static_cast<size_type>(input_view.extent(0))),
        row_width_(static_cast<size_type>(input_view.extent(1))),
        batch_size_(std::min<size_type>(batch_size, std::max<size_type>(n_rows_, 1))),
        n_iters_(n_rows_ == 0 ? 0 : raft::div_rounding_up_safe(n_rows_, batch_size_)),
        prefetch_(prefetch),
        initialize_(initialize),
        host_writeback_(host_writeback),
        buf_0_(0, copy_stream, mr),
        buf_1_(0, copy_stream, mr)
    {
      if (n_rows_ == 0) { return; }
      RAFT_EXPECTS(initialize_ || host_writeback_,
                   "At least one of initialize or host_writeback must be true");
      RAFT_EXPECTS(!host_writeback_ || !std::is_const_v<element_type>,
                   "host_writeback=true requires a non-const element type");

      if constexpr (!kPassthrough) {
        if (source_ == nullptr) {
          // Null source: yield batches with the right offsets/sizes but data() == nullptr.
          // Skip allocation and never queue copies.
          return;
        }
        buf_0_.resize(row_width_ * batch_size_, copy_stream);
        dev_ptr_ = reinterpret_cast<element_type*>(buf_0_.data());
        // The second buffer is only useful when there is more than one batch to overlap. With
        // n_iters_ <= 1, there is no "next batch" to stage while a kernel runs on the current
        // one, so prefetching offers no benefit. Downgrade `prefetch_` to false to skip the
        // buf_1_ allocation and have `prefetch()` / `load()` take the single-buffer fast path.
        if (prefetch_ && n_iters_ > 1) {
          buf_1_.resize(row_width_ * batch_size_, copy_stream);
          prefetch_dev_ptr_ = reinterpret_cast<element_type*>(buf_1_.data());
        } else {
          prefetch_ = false;
        }
      }
    }

    /**
     * Make this batch represent position `pos`.
     *
     * Passthrough: pure bookkeeping; the per-batch view is recomputed on demand by `view()` via
     * `cuda::std::submdspan`, never via pointer arithmetic on the input mdspan.
     *
     * Copy_device, prefetch=true: swap the ring slots and host-sync `copy_stream` so the
     * swapped-in slot is fully staged before the user's next kernel runs. The slot we swapped
     * out (now `prefetch_dev_ptr_`) carried the just-completed kernel's writes; if `host_writeback`
     * is on, its writeback is recorded for the *next* `prefetch_next_batch()` to flush -- when
     * that slot is about to be overwritten by a new H2D, by which time the kernel that wrote it
     * is two iterations old and guaranteed finished (the legacy 2-iteration-delay model).
     *
     * Copy_device, prefetch=false: synchronously stage H2D into the single buffer and host-sync
     * `copy_stream`.
     *
     * No-op if the buffer already holds `pos`. Iteration end is signaled by `pos >= n_iters_`.
     */
    void load(size_type pos)
    {
      if (n_iters_ == 0) { return; }
      if (pos == pos_) { return; }
      if (pos >= n_iters_) { return; }

      const size_type row_offset = pos * batch_size_;
      const size_type len =
        std::min<size_type>(batch_size_, n_rows_ - std::min(row_offset, n_rows_));

      if constexpr (kPassthrough) {
        // Passthrough: just record the new slice; view() will compute the submdspan.
        pos_.emplace(pos);
        batch_len_ = len;
        return;
      } else {
        if (source_ == nullptr) {
          pos_.emplace(pos);
          batch_len_ = len;
          // dev_ptr_ remains nullptr (or the empty-source buffer state).
          return;
        }

        if (prefetch_ && prefetch_pos_.has_value() && *prefetch_pos_ == pos) {
          // Hand-off from the prefetch slot. Before swapping, transfer the soon-to-be-stale
          // dev_ptr_'s dirty state into prefetch_dirty_pos_: after the swap, the slot that just
          // received `kernel_(prev pos_)`'s writes becomes prefetch_dev_ptr_, and its D2H will be
          // queued by the *next* prefetch_next_batch() right before the slot is overwritten by a
          // new H2D. By that point, the kernel that wrote it is two iterations old -- finished
          // by main-stream FIFO, no events required.
          if (host_writeback_ && dirty_cur_ && pos_.has_value()) {
            prefetch_dirty_pos_ = pos_;
          } else {
            prefetch_dirty_pos_.reset();
          }
          std::swap(dev_ptr_, prefetch_dev_ptr_);
          prefetch_pos_.reset();
          // Ensure prefetch_next_batch()'s queued H2D into this slot (and any prior D2H of the
          // slot from the previous overwrite) finished before the user kernel reads it.
          copy_stream_.synchronize();
        } else {
          // Non-pipelined fast path (prefetch_=false, or prefetch_pos_ didn't match).
          if (host_writeback_ && dirty_cur_ && pos_.has_value()) {
            queue_d2h(dev_ptr_, *pos_);
            dirty_cur_ = false;
          }
          if (initialize_) { queue_h2d(dev_ptr_, row_offset, len); }
          copy_stream_.synchronize();
        }
        pos_.emplace(pos);
        batch_len_ = len;
        if (host_writeback_) {
          // Every advanced batch is implicitly dirty: the user kernel will write to it before
          // the next prefetch_next_batch() (or destructor) recycles the slot.
          dirty_cur_ = true;
        }
      }
    }

    /**
     * Cross-stream pipelining step (called by the user *after* enqueuing the kernel for the
     * current batch).
     *
     * On `copy_stream`, queues -- in this order, into the *prefetch* slot only:
     *   1. D2H of the prefetch slot's stale kernel output (if `host_writeback` and the slot was
     *      written by a kernel two iterations ago). The kernel that produced that output is
     *      already finished by main-stream FIFO (the user has since enqueued the kernel for
     *      the slot currently in `dev_ptr_`), so no event is needed -- the D2H reads a slot the
     *      main stream is no longer touching.
     *   2. H2D of `pos` into the prefetch slot (if `initialize`).
     *
     * Then host-stalls on `res`'s main stream so the host has waited out the just-enqueued user
     * kernel by the time control returns; meanwhile the D2H/H2D queued above runs concurrently
     * on `copy_stream`, overlapping the writeback with the kernel.
     *
     * No-op if prefetch is disabled, source is null, passthrough, or `pos >= n_iters_`.
     */
    void prefetch(size_type pos)
    {
      if constexpr (kPassthrough) { return; }
      if (n_iters_ == 0 || pos >= n_iters_ || source_ == nullptr) { return; }
      if (!prefetch_) {
        // No-op: in non-pipelined mode load() does the staging synchronously in operator*.
        return;
      }

      // 2-iteration-delayed writeback: the prefetch slot still holds kernel output from two
      // iterations ago (from when this slot was last `dev_ptr_`); D2H it now -- right before we
      // overwrite the slot with the next H2D. The kernel that wrote it is past on the main
      // stream, so this D2H runs concurrently with the just-enqueued kernel for `dev_ptr_`
      // without racing (different slots).
      if (host_writeback_ && prefetch_dirty_pos_.has_value()) {
        queue_d2h(prefetch_dev_ptr_, *prefetch_dirty_pos_);
        prefetch_dirty_pos_.reset();
      }

      // H2D of the next batch into the prefetch slot. Sequenced after the D2H on `copy_stream`,
      // so the H2D doesn't clobber the slot before its prior contents have been read out.
      if (initialize_) {
        const size_type row_offset = pos * batch_size_;
        const size_type len =
          std::min<size_type>(batch_size_, n_rows_ - std::min(row_offset, n_rows_));
        queue_h2d(prefetch_dev_ptr_, row_offset, len);
      }
      prefetch_pos_.emplace(pos);

      // Host-stall on the user's main stream: the just-enqueued kernel runs concurrently with
      // the copies above. By the time control returns, the kernel is done and the next load()
      // can safely read its writes (still in `dev_ptr_`'s slot, which becomes prefetch_dev_ptr_
      // after the upcoming swap).
      raft::resource::sync_stream(*res_);
    }

    void queue_h2d(element_type* dst, size_type src_row_offset, size_type num_rows)
    {
      if (num_rows == 0) { return; }
      const size_t n_bytes = num_rows * row_width_ * sizeof(value_type_d);
      // dst is `element_type*` (potentially `const T*`), but it points into a non-const internal
      // buffer (`rmm::device_uvector<value_type_d>`); the const-cast restores the writable view.
      // Use cudaMemcpyAsync directly (rather than raft::copy) to avoid issues with
      // HMM/ATS-mapped host pointers being misclassified.
      RAFT_CUDA_TRY(cudaMemcpyAsync(const_cast<value_type_d*>(dst),
                                    source_ + src_row_offset * row_width_,
                                    n_bytes,
                                    cudaMemcpyHostToDevice,
                                    copy_stream_));
    }

    void queue_d2h(element_type* src, size_type pos)
    {
      const size_type row_offset = pos * batch_size_;
      const size_type num_rows =
        std::min<size_type>(batch_size_, n_rows_ - std::min(row_offset, n_rows_));
      if (num_rows == 0) { return; }
      const size_t n_bytes = num_rows * row_width_ * sizeof(value_type_d);
      RAFT_CUDA_TRY(cudaMemcpyAsync(const_cast<value_type_d*>(source_) + row_offset * row_width_,
                                    src,
                                    n_bytes,
                                    cudaMemcpyDeviceToHost,
                                    copy_stream_));
    }

    rmm::cuda_stream_view copy_stream_;
    raft::resources const* res_;
    MdspanT input_view_;
    element_type* source_;
    size_type n_rows_;
    size_type row_width_;
    size_type batch_size_;
    size_type n_iters_;
    bool prefetch_;
    bool initialize_;
    bool host_writeback_;

    rmm::device_uvector<value_type_d> buf_0_;
    rmm::device_uvector<value_type_d> buf_1_;

    // Slot bookkeeping (only meaningful for !kPassthrough).
    element_type* dev_ptr_          = nullptr;
    element_type* prefetch_dev_ptr_ = nullptr;
    // `pos_`: batch index currently held in `dev_ptr_`'s slot (the user-facing slot).
    // `prefetch_pos_`: batch index pre-staged in `prefetch_dev_ptr_`'s slot via H2D.
    // `prefetch_dirty_pos_`: batch index whose kernel output is sitting in `prefetch_dev_ptr_`'s
    //                       slot and still needs to be D2H'd back to host. Set by `load()`'s
    //                       swap when the slot it just retired held kernel writes; consumed
    //                       (D2H'd) by the *next* `prefetch()` -- the legacy 2-iteration delay.
    std::optional<size_type> pos_;
    std::optional<size_type> prefetch_pos_;
    std::optional<size_type> prefetch_dirty_pos_;
    size_type batch_len_ = 0;
    bool dirty_cur_      = false;
  };

  using value_type = batch;
  using reference  = const value_type&;
  using pointer    = const value_type*;

  /**
   * Construct an iterator over `input_view`.
   *
   * @param res             raft resources (must outlive the iterator)
   * @param input_view      typed mdspan to iterate; row-major; passthrough vs copy is decided
   *                        at compile time from the accessor.
   * @param batch_size      desired batch size in rows. Clamped to n_rows.
   * @param copy_stream     stream used for H2D / D2H copies in copy_device mode. Pass a non-main
   *                        stream (see `get_prefetch_stream`) to enable real overlap.
   * @param mr              memory resource for the internal device buffer(s).
   * @param prefetch        enable 2-buffer pipelining via `prefetch_next_batch()`.
   * @param initialize      stage H2D source rows before yielding each batch (default true).
   * @param host_writeback  queue D2H of every advanced batch back to `input_view`.
   *                        At least one of `initialize` / `host_writeback` must be true for
   *                        non-empty input.
   */
  batch_load_iterator(raft::resources const& res,
                      MdspanT input_view,
                      size_type batch_size,
                      rmm::cuda_stream_view copy_stream,
                      rmm::device_async_resource_ref mr,
                      bool prefetch       = false,
                      bool initialize     = true,
                      bool host_writeback = false)
    : cur_batch_(new value_type(
        res, input_view, batch_size, copy_stream, mr, prefetch, initialize, host_writeback)),
      cur_pos_(0),
      cur_prefetch_pos_(0)
  {
  }

  /** Convenience overload that uses `get_workspace_resource_ref(res)` as the memory resource. */
  batch_load_iterator(raft::resources const& res,
                      MdspanT input_view,
                      size_type batch_size,
                      rmm::cuda_stream_view copy_stream,
                      bool prefetch       = false,
                      bool initialize     = true,
                      bool host_writeback = false)
    : batch_load_iterator(res,
                          input_view,
                          batch_size,
                          copy_stream,
                          raft::resource::get_workspace_resource_ref(res),
                          prefetch,
                          initialize,
                          host_writeback)
  {
  }

  /** Whether iteration copies the data on each step (i.e. not passthrough). */
  [[nodiscard]] auto does_copy() const -> bool { return !kPassthrough; }
  /** Reset the iterator (and prefetch) position to begin(). Reusable iteration. */
  void reset()
  {
    cur_pos_          = 0;
    cur_prefetch_pos_ = 0;
  }
  /** Reset the iterator (and prefetch) position to end(). */
  void reset_to_end()
  {
    cur_pos_          = cur_batch_->n_iters_;
    cur_prefetch_pos_ = cur_batch_->n_iters_;
  }
  [[nodiscard]] auto begin() const -> const batch_load_iterator
  {
    batch_load_iterator x(*this);
    x.reset();
    return x;
  }
  [[nodiscard]] auto end() const -> const batch_load_iterator
  {
    batch_load_iterator x(*this);
    x.reset_to_end();
    return x;
  }
  [[nodiscard]] auto operator*() const -> reference
  {
    cur_batch_->load(cur_pos_);
    return *cur_batch_;
  }
  [[nodiscard]] auto operator->() const -> pointer
  {
    cur_batch_->load(cur_pos_);
    return cur_batch_.get();
  }
  /** Issue the prefetch for the next-but-one batch. See class doc for stream semantics. */
  void prefetch_next_batch() { cur_batch_->prefetch(cur_prefetch_pos_++); }
  friend auto operator==(const batch_load_iterator& x, const batch_load_iterator& y) -> bool
  {
    return x.cur_batch_ == y.cur_batch_ && x.cur_pos_ == y.cur_pos_;
  }
  friend auto operator!=(const batch_load_iterator& x, const batch_load_iterator& y) -> bool
  {
    return x.cur_batch_ != y.cur_batch_ || x.cur_pos_ != y.cur_pos_;
  }
  auto operator++() -> batch_load_iterator&
  {
    ++cur_pos_;
    return *this;
  }
  auto operator++(int) -> batch_load_iterator
  {
    batch_load_iterator x(*this);
    ++cur_pos_;
    return x;
  }
  auto operator--() -> batch_load_iterator&
  {
    --cur_pos_;
    return *this;
  }
  auto operator--(int) -> batch_load_iterator
  {
    batch_load_iterator x(*this);
    --cur_pos_;
    return x;
  }

 private:
  std::shared_ptr<value_type> cur_batch_;
  size_type cur_pos_;
  size_type cur_prefetch_pos_;
};

/**
 * Runtime-dispatched wrapper over `batch_load_iterator<MdspanT>` that takes a raw pointer and
 * picks the host- or device-typed mdspan instantiation based on `cudaPointerGetAttributes`,
 * preserving the legacy "force copy unless cudaMemoryTypeDevice" policy.
 *
 * Use this at call sites that don't know statically whether `ptr` is host or device memory.
 * Sites that already hold a typed mdspan should use `batch_load_iterator<MdspanT>` directly to
 * pick the strategy at compile time.
 */
template <typename T, typename IdxT = int64_t>
class batch_load_iterator_dyn {
  using HostMd     = raft::host_matrix_view<T, IdxT>;
  using DeviceMd   = raft::device_matrix_view<T, IdxT>;
  using HostIter   = batch_load_iterator<HostMd>;
  using DeviceIter = batch_load_iterator<DeviceMd>;

 public:
  using size_type = size_t;

  /** Uniform batch-proxy view across host/device branches. */
  class batch {
   public:
    [[nodiscard]] auto data() const -> T* { return dev_ptr_; }
    [[nodiscard]] auto size() const -> size_type { return batch_len_; }
    [[nodiscard]] auto offset() const -> size_type { return offset_; }
    [[nodiscard]] auto row_width() const -> size_type { return row_width_; }
    [[nodiscard]] auto does_copy() const -> bool { return does_copy_; }
    [[nodiscard]] auto view() const -> raft::device_matrix_view<T, IdxT>
    {
      return raft::make_device_matrix_view<T, IdxT>(
        dev_ptr_, static_cast<IdxT>(batch_len_), static_cast<IdxT>(row_width_));
    }

   private:
    template <typename, typename>
    friend class batch_load_iterator_dyn;
    T* dev_ptr_          = nullptr;
    size_type batch_len_ = 0;
    size_type offset_    = 0;
    size_type row_width_ = 0;
    bool does_copy_      = false;
  };

  using value_type = batch;
  using reference  = const value_type&;
  using pointer    = const value_type*;

  /**
   * Construct via runtime pointer dispatch.
   *
   * If `ptr` is a pure-device pointer (`cudaMemoryTypeDevice` with non-null `devicePointer`),
   * the device-accessor branch is selected (passthrough). Otherwise (host, pinned, managed,
   * unregistered, HMM/ATS, or `nullptr`), the host-accessor branch is selected (copy_device).
   */
  batch_load_iterator_dyn(raft::resources const& res,
                          T* ptr,
                          IdxT n_rows,
                          IdxT row_width,
                          size_type batch_size,
                          rmm::cuda_stream_view copy_stream,
                          rmm::device_async_resource_ref mr,
                          bool prefetch       = false,
                          bool initialize     = true,
                          bool host_writeback = false)
    : impl_(make_impl(res,
                      ptr,
                      n_rows,
                      row_width,
                      batch_size,
                      copy_stream,
                      mr,
                      prefetch,
                      initialize,
                      host_writeback)),
      proxy_(std::make_shared<batch>())
  {
  }

  /** Convenience overload that uses `get_workspace_resource_ref(res)` as the memory resource. */
  batch_load_iterator_dyn(raft::resources const& res,
                          T* ptr,
                          IdxT n_rows,
                          IdxT row_width,
                          size_type batch_size,
                          rmm::cuda_stream_view copy_stream,
                          bool prefetch       = false,
                          bool initialize     = true,
                          bool host_writeback = false)
    : batch_load_iterator_dyn(res,
                              ptr,
                              n_rows,
                              row_width,
                              batch_size,
                              copy_stream,
                              raft::resource::get_workspace_resource_ref(res),
                              prefetch,
                              initialize,
                              host_writeback)
  {
  }

  [[nodiscard]] auto does_copy() const -> bool
  {
    return std::visit([](auto const& it) { return it.does_copy(); }, impl_);
  }
  void reset()
  {
    std::visit([](auto& it) { it.reset(); }, impl_);
  }
  void reset_to_end()
  {
    std::visit([](auto& it) { it.reset_to_end(); }, impl_);
  }
  [[nodiscard]] auto begin() const -> batch_load_iterator_dyn
  {
    batch_load_iterator_dyn x(*this);
    x.reset();
    return x;
  }
  [[nodiscard]] auto end() const -> batch_load_iterator_dyn
  {
    batch_load_iterator_dyn x(*this);
    x.reset_to_end();
    return x;
  }
  [[nodiscard]] auto operator*() const -> reference
  {
    std::visit(
      [this](auto const& it) {
        auto const& b      = *it;
        proxy_->dev_ptr_   = const_cast<T*>(b.data());
        proxy_->batch_len_ = b.size();
        proxy_->offset_    = b.offset();
        proxy_->row_width_ = b.row_width();
        proxy_->does_copy_ = b.does_copy();
      },
      impl_);
    return *proxy_;
  }
  [[nodiscard]] auto operator->() const -> pointer
  {
    (void)**this;
    return proxy_.get();
  }
  void prefetch_next_batch()
  {
    std::visit([](auto& it) { it.prefetch_next_batch(); }, impl_);
  }
  friend auto operator==(const batch_load_iterator_dyn& x, const batch_load_iterator_dyn& y) -> bool
  {
    return x.impl_ == y.impl_;
  }
  friend auto operator!=(const batch_load_iterator_dyn& x, const batch_load_iterator_dyn& y) -> bool
  {
    return !(x == y);
  }
  auto operator++() -> batch_load_iterator_dyn&
  {
    std::visit([](auto& it) { ++it; }, impl_);
    return *this;
  }
  auto operator++(int) -> batch_load_iterator_dyn
  {
    batch_load_iterator_dyn x(*this);
    ++(*this);
    return x;
  }
  auto operator--() -> batch_load_iterator_dyn&
  {
    std::visit([](auto& it) { --it; }, impl_);
    return *this;
  }
  auto operator--(int) -> batch_load_iterator_dyn
  {
    batch_load_iterator_dyn x(*this);
    --(*this);
    return x;
  }

 private:
  std::variant<HostIter, DeviceIter> impl_;
  // Shared proxy: copies of the iterator share storage so that `*it1++` and `*it2` consistently
  // observe the same backing buffer state, mirroring the legacy shared-batch contract.
  std::shared_ptr<batch> proxy_;

  static auto make_impl(raft::resources const& res,
                        T* ptr,
                        IdxT n_rows,
                        IdxT row_width,
                        size_type batch_size,
                        rmm::cuda_stream_view copy_stream,
                        rmm::device_async_resource_ref mr,
                        bool prefetch,
                        bool initialize,
                        bool host_writeback) -> std::variant<HostIter, DeviceIter>
  {
    bool is_pure_device = false;
    if (ptr != nullptr) {
      cudaPointerAttributes attr{};
      RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, ptr));
      is_pure_device = (attr.type == cudaMemoryTypeDevice) && (attr.devicePointer != nullptr);
    }
    if (is_pure_device) {
      return DeviceIter(res,
                        raft::make_device_matrix_view<T, IdxT>(ptr, n_rows, row_width),
                        batch_size,
                        copy_stream,
                        mr,
                        prefetch,
                        initialize,
                        host_writeback);
    }
    return HostIter(res,
                    raft::make_host_matrix_view<T, IdxT>(ptr, n_rows, row_width),
                    batch_size,
                    copy_stream,
                    mr,
                    prefetch,
                    initialize,
                    host_writeback);
  }
};

// Locally-rolled `type_identity_t` so this header compiles in TUs that build with C++17
// (e.g. the cuvs C tests). Equivalent to `std::type_identity_t` (C++20).
namespace detail {
template <typename T>
struct type_identity {
  using type = T;
};
template <typename T>
using type_identity_t = typename type_identity<T>::type;
}  // namespace detail

/**
 * Builder for `batch_load_iterator_dyn<T, IdxT>`. Use at sites that have a raw pointer with
 * unknown memory location and want the legacy "force copy unless cudaMemoryTypeDevice" semantics.
 *
 * `ptr` is taken by `T const*` so callers can pass either a `T*` or a `const T*` (matching the
 * legacy `batch_load_iterator<T>(const T* source, ...)` API). The iterator's element type is `T`
 * (non-const), so `batch.data()` returns `T*` for kernels that expect a non-const view of the
 * source. Const-correctness at the API boundary is the caller's responsibility.
 *
 * `n_rows` / `row_width` are placed in non-deduced contexts so that `IdxT` is taken from the
 * explicit template argument (or its `int64_t` default) and the integer arguments are implicitly
 * converted to `IdxT`, regardless of their incoming integer type.
 */
template <typename T, typename IdxT = int64_t>
auto make_batch_load_iterator(raft::resources const& res,
                              T const* ptr,
                              detail::type_identity_t<IdxT> n_rows,
                              detail::type_identity_t<IdxT> row_width,
                              size_t batch_size,
                              rmm::cuda_stream_view copy_stream,
                              rmm::device_async_resource_ref mr,
                              bool prefetch       = false,
                              bool initialize     = true,
                              bool host_writeback = false) -> batch_load_iterator_dyn<T, IdxT>
{
  return batch_load_iterator_dyn<T, IdxT>(res,
                                          const_cast<T*>(ptr),
                                          n_rows,
                                          row_width,
                                          batch_size,
                                          copy_stream,
                                          mr,
                                          prefetch,
                                          initialize,
                                          host_writeback);
}

/** Convenience overload that uses `get_workspace_resource_ref(res)` as the memory resource. */
template <typename T, typename IdxT = int64_t>
auto make_batch_load_iterator(raft::resources const& res,
                              T const* ptr,
                              detail::type_identity_t<IdxT> n_rows,
                              detail::type_identity_t<IdxT> row_width,
                              size_t batch_size,
                              rmm::cuda_stream_view copy_stream,
                              bool prefetch       = false,
                              bool initialize     = true,
                              bool host_writeback = false) -> batch_load_iterator_dyn<T, IdxT>
{
  return make_batch_load_iterator<T, IdxT>(res,
                                           ptr,
                                           n_rows,
                                           row_width,
                                           batch_size,
                                           copy_stream,
                                           raft::resource::get_workspace_resource_ref(res),
                                           prefetch,
                                           initialize,
                                           host_writeback);
}

}  // namespace cuvs::spatial::knn::detail::utils
