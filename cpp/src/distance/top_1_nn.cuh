/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/fused_distance_nn.cuh"

#include <cuvs/core/export.hpp>

#include <raft/core/kvp.hpp>
#include <raft/core/resources.hpp>

#include <cuda/iterator>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace cuvs::distance {

/** Separate index and distance arrays used by backends with structure-of-arrays output. */
template <typename IdxT, typename DistT>
struct Top1nnOutput {
  IdxT* nearest_idx;
  DistT* nearest_dist;
};

namespace detail {

template <typename DataT>
using top_1_nn_distance_t = std::conditional_t<std::is_same_v<DataT, half>, float, DataT>;

template <typename DataT, typename IdxT>
struct Top1nnOutputTypes {
  using kvp      = raft::KeyValuePair<IdxT, DataT>*;
  using scalar   = DataT*;
  using separate = Top1nnOutput<IdxT, top_1_nn_distance_t<DataT>>;
};

template <typename OutputT>
struct Top1nnResultTraits {
  static constexpr bool supported = false;
};

template <typename IdxT, typename DistT>
struct Top1nnResultTraits<raft::KeyValuePair<IdxT, DistT>*> {
  using index_type     = IdxT;
  using distance_type  = DistT;
  using key_value_type = raft::KeyValuePair<IdxT, DistT>;
  using output_type    = key_value_type*;

  static constexpr bool supported                   = true;
  static constexpr Top1nnOutputLayout output_layout = Top1nnOutputLayout::KeyValuePair;

  struct key_op {
    __host__ __device__ IdxT operator()(const key_value_type& value) const { return value.key; }
  };

  struct value_op {
    __host__ __device__ DistT operator()(const key_value_type& value) const { return value.value; }
  };

  static auto indices(output_type output) { return cuda::transform_iterator(output, key_op{}); }
  static auto distances(output_type output) { return cuda::transform_iterator(output, value_op{}); }
};

template <typename IdxT, typename DistT>
struct Top1nnResultTraits<Top1nnOutput<IdxT, DistT>> {
  using index_type    = IdxT;
  using distance_type = DistT;
  using output_type   = Top1nnOutput<IdxT, DistT>;

  static constexpr bool supported                   = true;
  static constexpr Top1nnOutputLayout output_layout = Top1nnOutputLayout::Separate;

  static IdxT* indices(output_type output) { return output.nearest_idx; }
  static DistT* distances(output_type output) { return output.nearest_dist; }
};

}  // namespace detail

/** Non-owning view of backend-native top-1 NN output. */
template <typename DataT, typename IdxT>
class Top1nnResultView {
 public:
  using distance_type         = detail::top_1_nn_distance_t<DataT>;
  using key_value_type        = raft::KeyValuePair<IdxT, DataT>;
  using key_value_output_type = key_value_type*;
  using separate_output_type  = Top1nnOutput<IdxT, distance_type>;
  using plan_type             = detail::Top1nnPlan<IdxT>;

  template <typename OutputT>
  static Top1nnResultView from_output(const plan_type& plan, IdxT size, OutputT output)
  {
    using output_type = std::remove_cvref_t<OutputT>;
    using traits      = detail::Top1nnResultTraits<output_type>;
    if constexpr (!traits::supported) {
      static_assert(traits::supported, "Unsupported top-1 NN result output type");
    } else {
      static_assert(std::is_same_v<typename traits::index_type, IdxT>,
                    "top-1 NN result index type does not match the plan");
      if constexpr (traits::output_layout == detail::Top1nnOutputLayout::KeyValuePair) {
        static_assert(std::is_same_v<typename traits::distance_type, DataT>,
                      "KVP distance type does not match the top-1 NN data type");
      } else {
        static_assert(std::is_same_v<typename traits::distance_type, distance_type>,
                      "Separate distance type does not match the top-1 NN distance type");
      }

      RAFT_EXPECTS(size >= 0, "top-1 NN result size must not be negative");
      RAFT_EXPECTS(size == plan.m, "top-1 NN result size does not match the invocation plan");
      RAFT_EXPECTS(plan.output_alignment > 0, "top-1 NN output alignment must be positive");
      RAFT_EXPECTS(plan.output_layout == traits::output_layout,
                   "top-1 NN result output type does not match the plan");

      Top1nnResultView result{plan, size};
      if constexpr (traits::output_layout == detail::Top1nnOutputLayout::KeyValuePair) {
        RAFT_EXPECTS(size == 0 || output != nullptr,
                     "top-1 NN KVP result requires non-null output");
        RAFT_EXPECTS(output == nullptr ||
                       reinterpret_cast<std::uintptr_t>(output) % plan.output_alignment == 0,
                     "top-1 NN KVP output does not satisfy the plan alignment");
        result.key_values_ = output;
      } else {
        RAFT_EXPECTS(size == 0 || output.nearest_dist != nullptr,
                     "top-1 NN separate result requires a non-null distance output");
        RAFT_EXPECTS(size == 0 || (output.nearest_idx != nullptr) == plan.store_indices,
                     "top-1 NN index output does not match the plan");
        RAFT_EXPECTS(
          (output.nearest_idx == nullptr ||
           reinterpret_cast<std::uintptr_t>(output.nearest_idx) % plan.output_alignment == 0) &&
            (output.nearest_dist == nullptr ||
             reinterpret_cast<std::uintptr_t>(output.nearest_dist) % plan.output_alignment == 0),
          "top-1 NN separate output does not satisfy the plan alignment");
        result.separate_ = output;
      }
      return result;
    }
  }

  static Top1nnResultView from_storage(const plan_type& plan,
                                       IdxT size,
                                       void* storage,
                                       std::size_t storage_bytes)
  {
    RAFT_EXPECTS(plan.output_alignment > 0, "top-1 NN output alignment must be positive");
    RAFT_EXPECTS(storage_bytes >= plan.output_bytes,
                 "top-1 NN result storage is smaller than required by the plan");
    RAFT_EXPECTS(plan.output_bytes == 0 || storage != nullptr,
                 "top-1 NN result requires non-null storage");
    if (storage != nullptr) {
      RAFT_EXPECTS(reinterpret_cast<std::uintptr_t>(storage) % plan.output_alignment == 0,
                   "top-1 NN result storage does not satisfy the plan alignment");
    }

    if (plan.output_layout == detail::Top1nnOutputLayout::Separate) {
      RAFT_EXPECTS(plan.distance_offset <= plan.output_bytes,
                   "top-1 NN distance offset exceeds its output storage");
      RAFT_EXPECTS(plan.distance_offset % alignof(distance_type) == 0,
                   "top-1 NN distance offset does not satisfy its type alignment");
      auto* bytes = static_cast<char*>(storage);
      auto* distances =
        bytes == nullptr ? nullptr : reinterpret_cast<distance_type*>(bytes + plan.distance_offset);
      auto* indices =
        bytes == nullptr || !plan.store_indices ? nullptr : reinterpret_cast<IdxT*>(bytes);
      return from_output(plan, size, separate_output_type{indices, distances});
    }
    return from_output(plan, size, reinterpret_cast<key_value_output_type>(storage));
  }

  const plan_type& plan() const { return plan_; }
  IdxT size() const { return size_; }
  detail::Top1nnOutputLayout output_layout() const { return plan_.output_layout; }

  key_value_output_type key_values() const
  {
    RAFT_EXPECTS(output_layout() == detail::Top1nnOutputLayout::KeyValuePair,
                 "top-1 NN result does not contain KVP output");
    return key_values_;
  }

  template <typename Fn>
  void visit_indices(Fn&& fn) const
  {
    if (output_layout() == detail::Top1nnOutputLayout::Separate) {
      using traits = detail::Top1nnResultTraits<separate_output_type>;
      std::forward<Fn>(fn)(traits::indices(separate_));
    } else {
      using traits = detail::Top1nnResultTraits<key_value_output_type>;
      std::forward<Fn>(fn)(traits::indices(key_values_));
    }
  }

  template <typename Fn>
  void visit_distances(Fn&& fn) const
  {
    if (output_layout() == detail::Top1nnOutputLayout::Separate) {
      using traits = detail::Top1nnResultTraits<separate_output_type>;
      std::forward<Fn>(fn)(traits::distances(separate_));
    } else {
      using traits = detail::Top1nnResultTraits<key_value_output_type>;
      std::forward<Fn>(fn)(traits::distances(key_values_));
    }
  }

  template <typename SeparateFn, typename KeyValueFn>
  void visit_native(SeparateFn&& separate_fn, KeyValueFn&& key_value_fn) const
  {
    if (output_layout() == detail::Top1nnOutputLayout::Separate) {
      std::forward<SeparateFn>(separate_fn)(separate_.nearest_idx, separate_.nearest_dist);
    } else {
      std::forward<KeyValueFn>(key_value_fn)(key_values_);
    }
  }

 private:
  Top1nnResultView(const plan_type& plan, IdxT size) : plan_{plan}, size_{size} {}

  plan_type plan_{};
  IdxT size_{};
  key_value_output_type key_values_{};
  separate_output_type separate_{};
};

template <typename DataT, typename IdxT, typename OutputT>
Top1nnResultView<DataT, IdxT> make_top_1_nn_result_view(const detail::Top1nnPlan<IdxT>& plan,
                                                        IdxT size,
                                                        OutputT output)
{
  return Top1nnResultView<DataT, IdxT>::from_output(plan, size, output);
}

template <typename DataT, typename IdxT>
Top1nnResultView<DataT, IdxT> bind_top_1_nn_result_view(const detail::Top1nnPlan<IdxT>& plan,
                                                        IdxT size,
                                                        void* storage,
                                                        std::size_t storage_bytes)
{
  return Top1nnResultView<DataT, IdxT>::from_storage(plan, size, storage, storage_bytes);
}

/**
 * Return the workspace bytes required for one top-1 NN call.
 *
 * Callers that batch a larger problem should pass their maximum batch dimensions and reuse one
 * allocation across calls.
 */
template <typename DataT, typename IdxT>
CUVS_EXPORT std::size_t top_1_nn_workspace_size(IdxT m,
                                                IdxT n,
                                                IdxT k,
                                                const detail::Top1nnTuning& tuning,
                                                detail::Top1nnBackend backend,
                                                bool store_indices = true);

/** Resolve one exact invocation or automatic policy and return its native storage requirements. */
template <typename DataT, typename IdxT>
CUVS_EXPORT detail::Top1nnPlan<IdxT> probe_top_1_nn(
  raft::resources const& handle,
  const DataT* x,
  const DataT* y,
  IdxT m,
  IdxT n,
  IdxT k,
  const detail::Top1nnTuning& tuning,
  DistanceType metric,
  detail::Top1nnBackend backend = detail::Top1nnBackend::Auto,
  bool store_indices            = true);

/** Dispatch 1-NN using the native output representation recorded in a reusable plan. */
template <typename DataT, typename IdxT, typename OutputT, typename NormT = DataT>
CUVS_EXPORT void top_1_nn(raft::resources const& handle,
                          OutputT output,
                          const DataT* x,
                          const DataT* y,
                          const NormT* xn,
                          const NormT* yn,
                          IdxT m,
                          IdxT n,
                          IdxT k,
                          const detail::Top1nnTuning& tuning,
                          void* workspace,
                          std::size_t workspace_bytes,
                          bool sqrt,
                          bool init_out_buffer,
                          bool is_row_major,
                          DistanceType metric,
                          float metric_arg,
                          const detail::Top1nnPlan<IdxT>& plan);

/** Convenience overload for an explicit backend or automatic selection policy. */
template <typename DataT, typename IdxT, typename OutputT, typename NormT = DataT>
void top_1_nn(raft::resources const& handle,
              OutputT output,
              const DataT* x,
              const DataT* y,
              const NormT* xn,
              const NormT* yn,
              IdxT m,
              IdxT n,
              IdxT k,
              const detail::Top1nnTuning& tuning,
              void* workspace,
              std::size_t workspace_bytes,
              bool sqrt,
              bool init_out_buffer,
              bool is_row_major,
              DistanceType metric,
              float metric_arg,
              detail::Top1nnBackend backend)
{
  using output_type = std::remove_cvref_t<OutputT>;
  constexpr bool is_separate_output =
    std::is_same_v<output_type, typename detail::Top1nnOutputTypes<DataT, IdxT>::separate>;
  constexpr bool is_kvp_output =
    std::is_same_v<output_type, typename detail::Top1nnOutputTypes<DataT, IdxT>::kvp>;
  bool store_indices = is_kvp_output;
  if constexpr (is_separate_output) { store_indices = output.nearest_idx != nullptr; }
  const auto plan = probe_top_1_nn(handle, x, y, m, n, k, tuning, metric, backend, store_indices);
  RAFT_EXPECTS(plan.available, "No top_1_nn backend is available for this invocation");
  top_1_nn(handle,
           output,
           x,
           y,
           xn,
           yn,
           m,
           n,
           k,
           tuning,
           workspace,
           workspace_bytes,
           sqrt,
           init_out_buffer,
           is_row_major,
           metric,
           metric_arg,
           plan);
}
#define CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(DataT, IdxT)            \
  extern template std::size_t top_1_nn_workspace_size<DataT, IdxT>( \
    IdxT, IdxT, IdxT, const detail::Top1nnTuning&, detail::Top1nnBackend, bool)

CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(float, int);
CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(float, int64_t);
CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(double, int);
CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(double, int64_t);
CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(half, int);
CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(half, int64_t);

#undef CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE

#define CUVS_EXTERN_PROBE_TOP_1_NN(DataT, IdxT)                         \
  extern template detail::Top1nnPlan<IdxT> probe_top_1_nn<DataT, IdxT>( \
    raft::resources const&,                                             \
    const DataT*,                                                       \
    const DataT*,                                                       \
    IdxT,                                                               \
    IdxT,                                                               \
    IdxT,                                                               \
    const detail::Top1nnTuning&,                                        \
    DistanceType,                                                       \
    detail::Top1nnBackend,                                              \
    bool)

CUVS_EXTERN_PROBE_TOP_1_NN(float, int);
CUVS_EXTERN_PROBE_TOP_1_NN(float, int64_t);
CUVS_EXTERN_PROBE_TOP_1_NN(double, int);
CUVS_EXTERN_PROBE_TOP_1_NN(double, int64_t);
CUVS_EXTERN_PROBE_TOP_1_NN(half, int);
CUVS_EXTERN_PROBE_TOP_1_NN(half, int64_t);

#undef CUVS_EXTERN_PROBE_TOP_1_NN

#define CUVS_EXTERN_TOP_1_NN(DataT, IdxT, NormT, OutputKind)                                 \
  extern template void                                                                       \
  top_1_nn<DataT, IdxT, typename detail::Top1nnOutputTypes<DataT, IdxT>::OutputKind, NormT>( \
    raft::resources const&,                                                                  \
    typename detail::Top1nnOutputTypes<DataT, IdxT>::OutputKind,                             \
    const DataT*,                                                                            \
    const DataT*,                                                                            \
    const NormT*,                                                                            \
    const NormT*,                                                                            \
    IdxT,                                                                                    \
    IdxT,                                                                                    \
    IdxT,                                                                                    \
    const detail::Top1nnTuning&,                                                             \
    void*,                                                                                   \
    std::size_t,                                                                             \
    bool,                                                                                    \
    bool,                                                                                    \
    bool,                                                                                    \
    DistanceType,                                                                            \
    float,                                                                                   \
    const detail::Top1nnPlan<IdxT>&)

CUVS_EXTERN_TOP_1_NN(float, int, float, kvp);
CUVS_EXTERN_TOP_1_NN(float, int, float, scalar);
CUVS_EXTERN_TOP_1_NN(float, int, float, separate);
CUVS_EXTERN_TOP_1_NN(float, int64_t, float, kvp);
CUVS_EXTERN_TOP_1_NN(float, int64_t, float, scalar);
CUVS_EXTERN_TOP_1_NN(float, int64_t, float, separate);
CUVS_EXTERN_TOP_1_NN(double, int, double, kvp);
CUVS_EXTERN_TOP_1_NN(double, int, double, scalar);
CUVS_EXTERN_TOP_1_NN(double, int64_t, double, kvp);
CUVS_EXTERN_TOP_1_NN(double, int64_t, double, scalar);
CUVS_EXTERN_TOP_1_NN(half, int, float, separate);
CUVS_EXTERN_TOP_1_NN(half, int64_t, float, separate);

#undef CUVS_EXTERN_TOP_1_NN

}  // namespace cuvs::distance
