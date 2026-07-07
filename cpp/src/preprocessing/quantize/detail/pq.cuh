/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../core/nvtx.hpp"
#include "../../../neighbors/detail/vpq_dataset.cuh"
#include "pq_codepacking.cuh"  // pq_bits-bitfield
#include "vpq_dataset_impl.hpp"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>
#include <raft/core/operators.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>

#include "../../../cluster/kmeans_balanced.cuh"

namespace cuvs::preprocessing::quantize::pq::detail {

inline void fill_missing_params_heuristics(cuvs::preprocessing::quantize::pq::params& params,
                                           size_t n_rows,
                                           size_t dim)
{
  if (params.pq_dim == 0) { params.pq_dim = raft::div_rounding_up_safe(dim, size_t{4}); }
  if (params.vq_n_centers == 0) {
    params.vq_n_centers = raft::round_up_safe<uint32_t>(std::sqrt(n_rows), 8);
  }
}

inline bool is_balanced_kmeans(const cuvs::preprocessing::quantize::pq::params& params)
{
  return std::holds_alternative<cuvs::cluster::kmeans::balanced_params>(params.kmeans_params);
}

inline uint32_t get_kmeans_n_iters(const cuvs::preprocessing::quantize::pq::params& params)
{
  return std::visit(
    [](const auto& kp) -> uint32_t {
      using kmeans_type = std::decay_t<decltype(kp)>;
      if constexpr (std::is_same_v<kmeans_type, cuvs::cluster::kmeans::balanced_params>) {
        return kp.n_iters;
      } else {
        return static_cast<uint32_t>(kp.max_iter);
      }
    },
    params.kmeans_params);
}

inline cuvs::distance::DistanceType get_kmeans_metric(
  const cuvs::preprocessing::quantize::pq::params& params)
{
  return std::visit([](const auto& kp) -> cuvs::distance::DistanceType { return kp.metric; },
                    params.kmeans_params);
}

inline auto to_vpq_params(const cuvs::preprocessing::quantize::pq::params& params)
  -> cuvs::neighbors::vpq_params
{
  auto kmeans_type = is_balanced_kmeans(params) ? cuvs::cluster::kmeans::kmeans_type::KMeansBalanced
                                                : cuvs::cluster::kmeans::kmeans_type::KMeans;
  return cuvs::neighbors::vpq_params{
    .pq_bits                         = params.pq_bits,
    .pq_dim                          = params.pq_dim,
    .vq_n_centers                    = params.vq_n_centers,
    .kmeans_n_iters                  = get_kmeans_n_iters(params),
    .vq_kmeans_trainset_fraction     = 1.0,
    .pq_kmeans_trainset_fraction     = 1.0,
    .pq_kmeans_type                  = kmeans_type,
    .max_train_points_per_pq_code    = params.max_train_points_per_pq_code,
    .max_train_points_per_vq_cluster = params.max_train_points_per_vq_cluster};
}

template <typename MathT, typename DatasetT>
auto train_pq_subspaces(
  const raft::resources& res,
  const cuvs::preprocessing::quantize::pq::params& params,
  const DatasetT& dataset,
  const raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers)
  -> raft::device_matrix<MathT, uint32_t, raft::row_major>
{
  using ix_t              = int64_t;
  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  const ix_t pq_len       = raft::div_rounding_up_safe(dim, pq_dim);
  RAFT_EXPECTS((dim % pq_dim) == 0, "Dimension must be divisible by pq_dim");
  const ix_t n_rows_train =
    std::min<ix_t>(n_rows, params.max_train_points_per_pq_code * pq_n_centers);
  RAFT_EXPECTS(
    n_rows_train >= pq_n_centers,
    "The number of training samples must be equal to or greater than the number of PQ centers");
  auto pq_trainset = raft::make_device_matrix<MathT, ix_t, raft::row_major>(res, 0, 0);
  // Subtract VQ centers
  if (!vq_centers.empty()) {
    pq_trainset    = cuvs::util::subsample(res, dataset, n_rows_train);
    auto vq_labels = raft::make_device_vector<uint32_t, ix_t>(res, pq_trainset.extent(0));
    cuvs::neighbors::detail::predict_vq<uint32_t>(
      res, raft::make_const_mdspan(pq_trainset.view()), vq_centers, vq_labels.view());
    raft::linalg::map_offset(
      res,
      pq_trainset.view(),
      [labels = vq_labels.view(), vq_centers, dim] __device__(ix_t off, MathT x) {
        ix_t i = off / dim;
        ix_t j = off % dim;
        return x - vq_centers(labels(i), j);
      },
      raft::make_const_mdspan(pq_trainset.view()));
  }

  // Train PQ centers for each subspace
  auto sub_dataset = raft::make_device_matrix<MathT, ix_t>(res, n_rows_train, pq_len);

  auto pq_centers =
    raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, pq_dim * pq_n_centers, pq_len);
  auto trainset_ptr     = !vq_centers.empty() ? pq_trainset.data_handle() : dataset.data_handle();
  auto sub_labels       = raft::make_device_vector<uint32_t, ix_t>(res, 0);
  auto pq_cluster_sizes = raft::make_device_vector<uint32_t, ix_t>(res, 0);
  auto device_memory    = raft::resource::get_workspace_resource_ref(res);
  if (is_balanced_kmeans(params)) {
    sub_labels = raft::make_device_mdarray<uint32_t>(
      res, device_memory, raft::make_extents<ix_t>(n_rows_train));
    pq_cluster_sizes = raft::make_device_mdarray<uint32_t>(
      res, device_memory, raft::make_extents<ix_t>(pq_n_centers));
  }

  for (ix_t m = 0; m < pq_dim; m++) {
    raft::copy_matrix(sub_dataset.data_handle(),
                      pq_len,
                      trainset_ptr + m * pq_len,
                      dim,
                      pq_len,
                      n_rows_train,
                      raft::resource::get_cuda_stream(res));
    auto pq_centers_subspace_view = raft::make_device_matrix_view<MathT, uint32_t, raft::row_major>(
      pq_centers.data_handle() + m * pq_n_centers * pq_len, pq_n_centers, pq_len);
    cuvs::neighbors::detail::train_pq_centers<MathT, ix_t>(
      res,
      params.kmeans_params,
      raft::make_const_mdspan(sub_dataset.view()),
      pq_centers_subspace_view,
      sub_labels.view(),
      pq_cluster_sizes.view());
  }
  return pq_centers;
}

template <typename DataT, typename MathT, typename AccessorType>
quantizer<MathT> build(
  raft::resources const& res,
  const cuvs::preprocessing::quantize::pq::params params,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, AccessorType> dataset)
{
  auto n_rows = dataset.extent(0);
  auto dim    = dataset.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::pq::build(%zu, %u, %u, %u)",
    size_t(n_rows),
    dim,
    params.pq_bits,
    params.pq_dim);
  RAFT_EXPECTS(params.pq_bits >= 4 && params.pq_bits <= 16,
               "PQ bits must be within [4, 16], got %u",
               params.pq_bits);
  RAFT_EXPECTS(get_kmeans_metric(params) == cuvs::distance::DistanceType::L2Expanded,
               "KMeans metric must be L2Expanded");
  std::visit(
    [&](auto const& base_kmeans_params) {
      using KP = std::decay_t<decltype(base_kmeans_params)>;
      if constexpr (std::is_same_v<KP, cuvs::cluster::kmeans::params>) {
        RAFT_EXPECTS(base_kmeans_params.init != cuvs::cluster::kmeans::params::InitMethod::Array,
                     "Array initialization is not supported for PQ training");
      }
    },
    params.kmeans_params);
  auto filled_params = params;
  fill_missing_params_heuristics(filled_params, n_rows, dim);
  auto vpq_params   = to_vpq_params(filled_params);
  auto vq_code_book = raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, 0, 0);
  if (filled_params.use_vq) {
    vq_code_book = cuvs::neighbors::detail::train_vq<MathT>(res, vpq_params, dataset);
  }
  auto pq_code_book = raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, 0, 0);
  if (filled_params.use_subspaces) {
    pq_code_book = train_pq_subspaces<MathT>(
      res, filled_params, dataset, raft::make_const_mdspan(vq_code_book.view()));
  } else {
    pq_code_book = cuvs::neighbors::detail::train_pq<MathT>(
      res, filled_params, dataset, raft::make_const_mdspan(vq_code_book.view()));
  }
  std::optional<raft::device_matrix<MathT, uint32_t, raft::row_major>> vq_code_book_opt;
  if (filled_params.use_vq) { vq_code_book_opt = std::move(vq_code_book); }
  return {filled_params,
          vpq_codebooks<MathT>{std::make_unique<vpq_codebooks_owning<MathT>>(
            std::move(pq_code_book), std::move(vq_code_book_opt))}};
}

template <typename MathT>
quantizer<MathT> build_view(
  raft::resources const& res,
  const params& params,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  std::optional<raft::device_matrix_view<const MathT, uint32_t, raft::row_major>> vq_centers =
    std::nullopt)
{
  RAFT_EXPECTS(params.pq_bits >= 4 && params.pq_bits <= 16,
               "PQ bits must be within [4, 16], got %u",
               params.pq_bits);
  RAFT_EXPECTS(params.pq_dim > 0, "pq_dim must be specified for view-type quantizer");

  const uint32_t pq_n_centers = 1u << params.pq_bits;

  if (params.use_subspaces) {
    RAFT_EXPECTS(pq_centers.extent(0) == params.pq_dim * pq_n_centers,
                 "For use_subspaces=true, pq_centers must have shape [pq_dim * pq_n_centers, "
                 "pq_len], got [%u, %u]",
                 pq_centers.extent(0),
                 pq_centers.extent(1));
  } else {
    RAFT_EXPECTS(pq_centers.extent(0) == pq_n_centers,
                 "For use_subspaces=false, pq_centers must have shape [pq_n_centers, pq_len], got "
                 "[%u, %u]",
                 pq_centers.extent(0),
                 pq_centers.extent(1));
  }

  if (params.use_vq) {
    RAFT_EXPECTS(vq_centers.has_value(), "vq_centers must be provided when use_vq=true");
    RAFT_EXPECTS(params.vq_n_centers > 0,
                 "params.vq_n_centers must be > 0 when use_vq=true (got %u)",
                 params.vq_n_centers);
    RAFT_EXPECTS(vq_centers.value().data_handle() != nullptr,
                 "vq_centers data pointer must be non-null when use_vq=true");
    RAFT_EXPECTS(vq_centers.value().extent(0) == params.vq_n_centers,
                 "vq_centers must have shape [vq_n_centers, dim] (vq_n_centers=%u), got "
                 "extent(0)=%u",
                 params.vq_n_centers,
                 vq_centers.value().extent(0));
    return {
      params,
      vpq_codebooks<MathT>{std::make_unique<vpq_codebooks_view<MathT>>(pq_centers, vq_centers)}};
  } else {
    if (vq_centers.has_value()) {
      RAFT_LOG_WARN("vq_centers will be ignored since params.use_vq=false");
    }
    return {params, vpq_codebooks<MathT>{std::make_unique<vpq_codebooks_view<MathT>>(pq_centers)}};
  }
}

template <typename T, typename QuantI, typename AccessorType>
void transform(
  raft::resources const& res,
  const quantizer<T>& quantizer,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, AccessorType> dataset,
  raft::device_matrix_view<QuantI, int64_t> pq_codes_out,
  std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::pq::transform(%zu, %zu, %zu)",
    size_t(dataset.extent(0)),
    size_t(dataset.extent(1)),
    size_t(pq_codes_out.extent(1)));
  RAFT_EXPECTS(pq_codes_out.extent(0) == dataset.extent(0),
               "Output matrix must have the same number of rows as the input dataset");
  RAFT_EXPECTS(pq_codes_out.extent(1) == get_quantized_dim(quantizer.params_quantizer),
               "Output matrix doesn't have the correct number of columns");
  RAFT_EXPECTS(quantizer.params_quantizer.pq_bits >= 4 && quantizer.params_quantizer.pq_bits <= 16,
               "PQ bits must be within [4, 16]");

  // Honor params.use_vq as the source of truth: when it is false, pass an
  // empty view to the kernel regardless of what the codebooks contain
  // (the kernel gates VQ subtraction on vq_centers.empty(), so an empty view
  // guarantees no residual VQ is applied even if a misconfigured quantizer
  // somehow carries non-empty centers). Conversely, when it is true the
  // codebook must be present.
  auto vq_centers_opt = quantizer.codebooks.vq_code_book();
  RAFT_EXPECTS(!quantizer.params_quantizer.use_vq || vq_centers_opt.has_value(),
               "Quantizer has params.use_vq=true but no VQ codebook");
  auto vq_centers =
    quantizer.params_quantizer.use_vq
      ? vq_centers_opt.value()
      : raft::make_device_matrix_view<const T, uint32_t, raft::row_major>(nullptr, 0, 0);
  auto pq_centers     = quantizer.codebooks.pq_code_book();
  auto vq_labels_view = raft::make_device_vector_view<uint32_t, int64_t>(nullptr, 0);
  if (vq_labels.has_value()) { vq_labels_view = vq_labels.value(); }

  if (quantizer.params_quantizer.use_subspaces) {
    cuvs::neighbors::detail::process_and_fill_codes_subspaces<T, int64_t>(
      res,
      to_vpq_params(quantizer.params_quantizer),
      dataset,
      pq_centers,
      vq_centers,
      vq_labels_view,
      pq_codes_out);
  } else {
    cuvs::neighbors::detail::process_and_fill_codes<T, int64_t>(
      res,
      to_vpq_params(quantizer.params_quantizer),
      dataset,
      pq_centers,
      vq_centers,
      vq_labels_view,
      pq_codes_out);
  }
}

template <uint32_t BlockSize,
          uint32_t SubWarpSize,
          typename CodeT,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__launch_bounds__(BlockSize) RAFT_KERNEL reconstruct_vectors_kernel(
  raft::device_matrix_view<const uint8_t, IdxT, raft::row_major> codes,
  raft::device_matrix_view<DataT, IdxT, raft::row_major> dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<const DataT, uint32_t, raft::row_major> vq_centers,
  std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels,
  const uint32_t pq_bits,
  bool use_subspaces)
{
  const uint32_t n_centers = 1 << pq_bits;
  using subwarp_align      = raft::Pow2<SubWarpSize>;
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= dataset.extent(0)) { return; }
  uint32_t lane_id      = subwarp_align::mod(raft::laneId());
  const uint32_t pq_len = raft::div_rounding_up_unsafe(dataset.extent(1), pq_centers.extent(1));

  LabelT vq_label = 0;
  if (vq_labels.has_value()) { vq_label = vq_labels.value()(row_ix); }
  cuvs::preprocessing::quantize::detail::bitfield_view_t code_view{&codes(row_ix, 0), pq_bits};
  for (uint32_t j = lane_id; j < pq_len; j += SubWarpSize) {
    const CodeT code = code_view[j];
    for (uint32_t k = 0; k < pq_centers.extent(1); k++) {
      const auto col                    = j * pq_centers.extent(1) + k;
      const uint32_t pq_subspace_offset = use_subspaces ? n_centers * j : 0;
      if (!vq_centers.empty()) {
        dataset(row_ix, col) = pq_centers(pq_subspace_offset + code, k) + vq_centers(vq_label, col);
      } else {
        dataset(row_ix, col) = pq_centers(pq_subspace_offset + code, k);
      }
    }
  }
}

template <typename DataT, typename MathT, typename IdxT, typename LabelT>
auto reconstruct_vectors(
  const raft::resources& res,
  const params& params,
  raft::device_matrix_view<const uint8_t, IdxT, raft::row_major> codes,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<const DataT, uint32_t, raft::row_major> vq_centers,
  std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels,
  raft::device_matrix_view<DataT, IdxT, raft::row_major> out_vectors,
  bool use_subspaces)
{
  const IdxT n_rows       = out_vectors.extent(0);
  const IdxT pq_bits      = params.pq_bits;
  const IdxT pq_n_centers = IdxT{1} << pq_bits;

  auto stream = raft::resource::get_cuda_stream(res);

  constexpr IdxT kBlockSize  = 256;
  const IdxT threads_per_vec = std::min<IdxT>(raft::WarpSize, pq_n_centers);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    if (pq_bits == 4) {
      return reconstruct_vectors_kernel<kBlockSize, 16, uint8_t, DataT, MathT, IdxT, LabelT>;
    } else if (pq_bits <= 8) {
      return reconstruct_vectors_kernel<kBlockSize,
                                        raft::WarpSize,
                                        uint8_t,
                                        DataT,
                                        MathT,
                                        IdxT,
                                        LabelT>;
    } else if (pq_bits <= 16) {
      return reconstruct_vectors_kernel<kBlockSize,
                                        raft::WarpSize,
                                        uint16_t,
                                        DataT,
                                        MathT,
                                        IdxT,
                                        LabelT>;
    } else {
      RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 16]", pq_bits);
    }
  }(pq_bits);
  dim3 blocks(raft::div_rounding_up_safe<IdxT>(n_rows, kBlockSize / threads_per_vec), 1, 1);
  kernel<<<blocks, threads, 0, stream>>>(
    codes, out_vectors, pq_centers, vq_centers, vq_labels, pq_bits, use_subspaces);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T, typename QuantI = uint8_t>
void inverse_transform(
  raft::resources const& res,
  const quantizer<T>& quantizer,
  raft::device_matrix_view<const QuantI, int64_t> codes,
  raft::device_matrix_view<T, int64_t> out,
  std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels = std::nullopt)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::pq::inverse_transform(%zu, %zu, %zu)",
    size_t(codes.extent(0)),
    size_t(codes.extent(1)),
    size_t(out.extent(1)));
  using label_t = uint32_t;
  using idx_t   = int64_t;
  RAFT_EXPECTS(out.extent(0) == codes.extent(0),
               "Output matrix must have the same number of rows as the input codes");
  RAFT_EXPECTS(codes.extent(1) == get_quantized_dim(quantizer.params_quantizer),
               "Codes matrix doesn't have the correct number of columns");
  RAFT_EXPECTS(quantizer.params_quantizer.pq_bits >= 4 && quantizer.params_quantizer.pq_bits <= 16,
               "PQ bits must be within [4, 16]");

  // Honor params.use_vq strictly (see the matching block in transform()).
  auto vq_centers_opt = quantizer.codebooks.vq_code_book();
  RAFT_EXPECTS(!quantizer.params_quantizer.use_vq || vq_centers_opt.has_value(),
               "Quantizer has params.use_vq=true but no VQ codebook");
  auto vq_centers =
    quantizer.params_quantizer.use_vq
      ? vq_centers_opt.value()
      : raft::make_device_matrix_view<const T, uint32_t, raft::row_major>(nullptr, 0, 0);

  // VQ-label preflight: when use_vq is true the kernel reads vq_labels(row) per
  // row and falls back to label 0 when vq_labels is absent. Without this check every vector can be
  // reconstructed with vq_label 0.
  if (quantizer.params_quantizer.use_vq) {
    RAFT_EXPECTS(vq_labels.has_value(),
                 "When params.use_vq is true, vq_labels must be provided to inverse_transform()");
    RAFT_EXPECTS(vq_labels.value().extent(0) == codes.extent(0),
                 "When params.use_vq is true, vq_labels must have the same number of rows as "
                 "codes (got %zu vs %zu)",
                 size_t(vq_labels.value().extent(0)),
                 size_t(codes.extent(0)));
  }

  reconstruct_vectors<T, T, idx_t, label_t>(res,
                                            quantizer.params_quantizer,
                                            codes,
                                            quantizer.codebooks.pq_code_book(),
                                            vq_centers,
                                            vq_labels,
                                            out,
                                            quantizer.params_quantizer.use_subspaces);
}

template <typename NewMathT, typename OldMathT>
auto vpq_convert_math_type(const raft::resources& res, const vpq_codebooks<OldMathT>& src)
  -> vpq_codebooks<NewMathT>
{
  auto vq_src_opt = src.vq_code_book();
  auto pq_src     = src.pq_code_book();

  auto pq_new = raft::make_device_matrix<NewMathT, uint32_t, raft::row_major>(
    res, pq_src.extent(0), pq_src.extent(1));
  raft::linalg::map(
    res, pq_new.view(), cuvs::spatial::knn::detail::utils::mapping<NewMathT>{}, pq_src);

  std::optional<raft::device_matrix<NewMathT, uint32_t, raft::row_major>> vq_new_opt;
  if (vq_src_opt.has_value()) {
    auto vq_src = vq_src_opt.value();
    auto vq_new = raft::make_device_matrix<NewMathT, uint32_t, raft::row_major>(
      res, vq_src.extent(0), vq_src.extent(1));
    raft::linalg::map(
      res, vq_new.view(), cuvs::spatial::knn::detail::utils::mapping<NewMathT>{}, vq_src);
    vq_new_opt = std::move(vq_new);
  }

  return vpq_codebooks<NewMathT>{
    std::make_unique<vpq_codebooks_owning<NewMathT>>(std::move(pq_new), std::move(vq_new_opt))};
}

inline auto make_pq_params_from_vpq(const cuvs::neighbors::vpq_params& in_params,
                                    const uint64_t n_rows)
  -> cuvs::preprocessing::quantize::pq::params
{
  const uint32_t pq_n_centers              = 1 << in_params.pq_bits;
  uint32_t max_train_points_per_vq_cluster = in_params.max_train_points_per_vq_cluster;
  if (in_params.vq_n_centers > 0) {
    max_train_points_per_vq_cluster =
      std::min<uint32_t>(max_train_points_per_vq_cluster,
                         n_rows * in_params.vq_kmeans_trainset_fraction / in_params.vq_n_centers);
  }
  return cuvs::preprocessing::quantize::pq::params{
    in_params.pq_bits,
    in_params.pq_dim,
    true,
    true,
    in_params.vq_n_centers,
    in_params.kmeans_n_iters,
    in_params.pq_kmeans_type,
    std::min<uint32_t>(in_params.max_train_points_per_pq_code,
                       n_rows * in_params.pq_kmeans_trainset_fraction / pq_n_centers),
    max_train_points_per_vq_cluster};
}

template <typename DatasetT, typename MathT, typename IdxT>
auto vpq_build(const raft::resources& res,
               const cuvs::neighbors::vpq_params& params,
               const DatasetT& dataset) -> vpq_dataset<MathT, IdxT>
{
  using label_t = uint32_t;
  auto ps       = cuvs::neighbors::detail::fill_missing_params_heuristics(params, dataset);

  auto pq_params = make_pq_params_from_vpq(ps, dataset.extent(0));
  // Train codes
  auto vq_code_book = cuvs::neighbors::detail::train_vq<MathT>(res, ps, dataset);
  auto pq_code_book = cuvs::neighbors::detail::train_pq<MathT>(
    res, pq_params, dataset, raft::make_const_mdspan(vq_code_book.view()));

  const IdxT n_rows       = dataset.extent(0);
  const IdxT codes_rowlen = sizeof(label_t) * (1 + raft::div_rounding_up_safe<IdxT>(
                                                     ps.pq_dim * ps.pq_bits, 8 * sizeof(label_t)));

  auto codes = raft::make_device_matrix<uint8_t, IdxT, raft::row_major>(res, n_rows, codes_rowlen);
  cuvs::neighbors::detail::process_and_fill_codes<MathT, IdxT>(
    res,
    ps,
    dataset,
    raft::make_const_mdspan(pq_code_book.view()),
    raft::make_const_mdspan(vq_code_book.view()),
    raft::make_device_vector_view<label_t, IdxT>(nullptr, 0),
    codes.view(),
    true);

  return vpq_dataset<MathT, IdxT>{
    vpq_codebooks<MathT>{std::make_unique<vpq_codebooks_owning<MathT>>(std::move(pq_code_book),
                                                                       std::move(vq_code_book))},
    std::move(codes)};
}

template <typename DatasetT>
auto vpq_build_half(const raft::resources& res,
                    const cuvs::neighbors::vpq_params& params,
                    const DatasetT& dataset) -> vpq_dataset<half, int64_t>
{
  // Build in float, then convert codebooks to half; data (uint8 codes) is moved, not copied.
  auto float_ds       = vpq_build<DatasetT, float, int64_t>(res, params, dataset);
  auto half_codebooks = vpq_convert_math_type<half, float>(res, float_ds.codebooks);
  return vpq_dataset<half, int64_t>{std::move(half_codebooks), std::move(float_ds.data)};
}

}  // namespace cuvs::preprocessing::quantize::pq::detail
