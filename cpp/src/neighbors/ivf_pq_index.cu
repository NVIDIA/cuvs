/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_pq.hpp>

#include "detail/ann_utils.cuh"
#include "ivf_pq_impl.hpp"

#include <raft/core/operators.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/reduce.cuh>

#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::ivf_pq {

template <typename IdxT>
index_impl<IdxT>::index_impl(raft::resources const& handle,
                             cuvs::distance::DistanceType metric,
                             codebook_gen codebook_kind,
                             uint32_t n_lists,
                             uint32_t dim,
                             uint32_t pq_bits,
                             uint32_t pq_dim,
                             bool conservative_memory_allocation,
                             list_layout codes_layout,
                             std::optional<bool> use_ann_for_extend)
  : metric_(metric),
    codebook_kind_(codebook_kind),
    codes_layout_(codes_layout),
    dim_(dim),
    pq_bits_(pq_bits),
    pq_dim_(pq_dim == 0 ? index<IdxT>::calculate_pq_dim(dim) : pq_dim),
    conservative_memory_allocation_(conservative_memory_allocation),
    use_ann_for_extend_(use_ann_for_extend),
    lists_(n_lists),
    list_sizes_{raft::make_device_vector<uint32_t, uint32_t>(handle, n_lists)},
    data_ptrs_{raft::make_device_vector<uint8_t*, uint32_t>(handle, n_lists)},
    inds_ptrs_{raft::make_device_vector<IdxT*, uint32_t>(handle, n_lists)},
    accum_sorted_sizes_{raft::make_host_vector<IdxT, uint32_t>(n_lists + 1)}
{
  check_consistency();
  accum_sorted_sizes_(n_lists) = 0;
}

template <typename IdxT>
cuvs::distance::DistanceType index_impl<IdxT>::metric() const noexcept
{
  return metric_;
}

template <typename IdxT>
codebook_gen index_impl<IdxT>::codebook_kind() const noexcept
{
  return codebook_kind_;
}

template <typename IdxT>
list_layout index_impl<IdxT>::codes_layout() const noexcept
{
  return codes_layout_;
}

template <typename IdxT>
IdxT index_impl<IdxT>::size() const noexcept
{
  return accum_sorted_sizes_(n_lists());
}

template <typename IdxT>
uint32_t index_impl<IdxT>::dim() const noexcept
{
  return dim_;
}

template <typename IdxT>
uint32_t index_impl<IdxT>::dim_ext() const noexcept
{
  return raft::round_up_safe(dim_ + 1, 8u);
}

template <typename IdxT>
uint32_t index_impl<IdxT>::rot_dim() const noexcept
{
  return pq_len() * pq_dim_;
}

template <typename IdxT>
uint32_t index_impl<IdxT>::pq_bits() const noexcept
{
  return pq_bits_;
}

template <typename IdxT>
uint32_t index_impl<IdxT>::pq_dim() const noexcept
{
  return pq_dim_;
}

template <typename IdxT>
uint32_t index_impl<IdxT>::pq_len() const noexcept
{
  return raft::div_rounding_up_unsafe(dim_, pq_dim_);
}

template <typename IdxT>
uint32_t index_impl<IdxT>::pq_book_size() const noexcept
{
  return 1 << pq_bits_;
}

template <typename IdxT>
uint32_t index_impl<IdxT>::n_lists() const noexcept
{
  return lists_.size();
}

template <typename IdxT>
bool index_impl<IdxT>::conservative_memory_allocation() const noexcept
{
  return conservative_memory_allocation_;
}

template <typename IdxT>
std::optional<bool> index_impl<IdxT>::use_ann_for_extend() const noexcept
{
  return use_ann_for_extend_;
}

template <typename IdxT>
std::vector<std::shared_ptr<list_data_base<IdxT>>>& index_impl<IdxT>::lists() noexcept
{
  return lists_;
}

template <typename IdxT>
const std::vector<std::shared_ptr<list_data_base<IdxT>>>& index_impl<IdxT>::lists() const noexcept
{
  return lists_;
}

template <typename IdxT>
raft::device_vector_view<uint32_t, uint32_t, raft::row_major>
index_impl<IdxT>::list_sizes() noexcept
{
  return list_sizes_.view();
}

template <typename IdxT>
raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> index_impl<IdxT>::list_sizes()
  const noexcept
{
  return list_sizes_.view();
}

template <typename IdxT>
raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> index_impl<IdxT>::data_ptrs() noexcept
{
  return data_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major>
index_impl<IdxT>::data_ptrs() const noexcept
{
  return data_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<IdxT*, uint32_t, raft::row_major> index_impl<IdxT>::inds_ptrs() noexcept
{
  return inds_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> index_impl<IdxT>::inds_ptrs()
  const noexcept
{
  return raft::make_mdspan<const IdxT* const, uint32_t, raft::row_major, false, true>(
    inds_ptrs_.data_handle(), inds_ptrs_.extents());
}

template <typename IdxT>
raft::host_vector_view<IdxT, uint32_t, raft::row_major>
index_impl<IdxT>::accum_sorted_sizes() noexcept
{
  return accum_sorted_sizes_.view();
}

template <typename IdxT>
raft::host_vector_view<const IdxT, uint32_t, raft::row_major> index_impl<IdxT>::accum_sorted_sizes()
  const noexcept
{
  return accum_sorted_sizes_.view();
}

template <typename IdxT>
owning_impl<IdxT>::owning_impl(raft::resources const& handle,
                               cuvs::distance::DistanceType metric,
                               codebook_gen codebook_kind,
                               uint32_t n_lists,
                               uint32_t dim,
                               uint32_t pq_bits,
                               uint32_t pq_dim,
                               bool conservative_memory_allocation,
                               list_layout codes_layout,
                               std::optional<bool> use_ann_for_extend)
  : index_impl<IdxT>(handle,
                     metric,
                     codebook_kind,
                     n_lists,
                     dim,
                     pq_bits,
                     pq_dim,
                     conservative_memory_allocation,
                     codes_layout,
                     use_ann_for_extend),
    pq_centers_{raft::make_device_mdarray<float>(
      handle, index<IdxT>::make_pq_centers_extents(dim, pq_dim, pq_bits, codebook_kind, n_lists))},
    centers_{
      raft::make_device_matrix<float, uint32_t>(handle, n_lists, raft::round_up_safe(dim + 1, 8u))},
    centers_rot_{raft::make_device_matrix<float, uint32_t>(
      handle, n_lists, raft::div_rounding_up_unsafe(dim, pq_dim) * pq_dim)},
    rotation_matrix_{raft::make_device_matrix<float, uint32_t>(
      handle, raft::div_rounding_up_unsafe(dim, pq_dim) * pq_dim, dim)}
{
}

template <typename IdxT>
pq_centers_extents index<IdxT>::make_pq_centers_extents(
  uint32_t dim, uint32_t pq_dim, uint32_t pq_bits, codebook_gen codebook_kind, uint32_t n_lists)
{
  uint32_t pq_len       = raft::div_rounding_up_unsafe(dim, pq_dim);
  uint32_t pq_book_size = 1u << pq_bits;
  switch (codebook_kind) {
    case codebook_gen::PER_SUBSPACE:
      return raft::make_extents<uint32_t>(pq_dim, pq_len, pq_book_size);
    case codebook_gen::PER_CLUSTER:
      return raft::make_extents<uint32_t>(n_lists, pq_len, pq_book_size);
    default: RAFT_FAIL("Unreachable code");
  }
}

template <typename IdxT>
view_impl<IdxT>::view_impl(
  raft::resources const& handle,
  cuvs::distance::DistanceType metric,
  codebook_gen codebook_kind,
  uint32_t n_lists,
  uint32_t dim,
  uint32_t pq_bits,
  uint32_t pq_dim,
  bool conservative_memory_allocation,
  raft::device_mdspan<const float, pq_centers_extents, raft::row_major> pq_centers_view,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_view,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot_view,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix_view,
  list_layout codes_layout,
  std::optional<bool> use_ann_for_extend)
  : index_impl<IdxT>(handle,
                     metric,
                     codebook_kind,
                     n_lists,
                     dim,
                     pq_bits,
                     pq_dim,
                     conservative_memory_allocation,
                     codes_layout,
                     use_ann_for_extend),
    pq_centers_view_(pq_centers_view),
    centers_view_(centers_view),
    centers_rot_view_(centers_rot_view),
    rotation_matrix_view_(rotation_matrix_view)
{
}

template <typename IdxT>
raft::device_mdspan<float, pq_centers_extents, raft::row_major>
owning_impl<IdxT>::pq_centers() noexcept
{
  return pq_centers_.view();
}

template <typename IdxT>
raft::device_mdspan<const float, pq_centers_extents, raft::row_major>
owning_impl<IdxT>::pq_centers() const noexcept
{
  return pq_centers_.view();
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> owning_impl<IdxT>::centers() noexcept
{
  return centers_.view();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> owning_impl<IdxT>::centers()
  const noexcept
{
  return centers_.view();
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> owning_impl<IdxT>::centers_rot() noexcept
{
  return centers_rot_.view();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> owning_impl<IdxT>::centers_rot()
  const noexcept
{
  return centers_rot_.view();
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major>
owning_impl<IdxT>::rotation_matrix() noexcept
{
  return rotation_matrix_.view();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major>
owning_impl<IdxT>::rotation_matrix() const noexcept
{
  return rotation_matrix_.view();
}

template <typename IdxT>
raft::device_mdspan<const float, pq_centers_extents, raft::row_major> view_impl<IdxT>::pq_centers()
  const noexcept
{
  return pq_centers_view_;
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> view_impl<IdxT>::centers()
  const noexcept
{
  return centers_view_;
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> view_impl<IdxT>::centers_rot()
  const noexcept
{
  return centers_rot_view_;
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> view_impl<IdxT>::rotation_matrix()
  const noexcept
{
  return rotation_matrix_view_;
}

index_params index_params::from_dataset(raft::matrix_extent<int64_t> dataset,
                                        cuvs::distance::DistanceType metric)
{
  index_params params;
  params.n_lists =
    dataset.extent(0) < 4 * 2500 ? 4 : static_cast<uint32_t>(std::sqrt(dataset.extent(0)));
  params.n_lists = std::min<uint32_t>(params.n_lists, dataset.extent(0));
  params.pq_dim =
    raft::round_up_safe(static_cast<uint32_t>(dataset.extent(1) / 4), static_cast<uint32_t>(8));
  if (params.pq_dim == 0) params.pq_dim = 8;
  params.pq_bits                  = 8;
  params.kmeans_trainset_fraction = dataset.extent(0) < 10000 ? 1 : 0.1;
  params.metric                   = metric;
  return params;
}

template <typename IdxT>
index<IdxT>::index(std::unique_ptr<index_iface<IdxT>> impl)
  : cuvs::neighbors::index(), impl_(std::move(impl))
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle)
  : index(std::make_unique<owning_impl<IdxT>>(handle,
                                              cuvs::distance::DistanceType::L2Expanded,
                                              codebook_gen::PER_SUBSPACE,
                                              0,
                                              0,
                                              8,
                                              1,
                                              true))
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle,
                   cuvs::distance::DistanceType metric,
                   codebook_gen codebook_kind,
                   uint32_t n_lists,
                   uint32_t dim,
                   uint32_t pq_bits,
                   uint32_t pq_dim,
                   bool conservative_memory_allocation)
  : index(
      std::make_unique<owning_impl<IdxT>>(handle,
                                          metric,
                                          codebook_kind,
                                          n_lists,
                                          dim,
                                          pq_bits,
                                          pq_dim == 0 ? index<IdxT>::calculate_pq_dim(dim) : pq_dim,
                                          conservative_memory_allocation))
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle, const index_params& params, uint32_t dim)
  : index(handle,
          params.metric,
          params.codebook_kind,
          params.n_lists,
          dim,
          params.pq_bits,
          params.pq_dim,
          params.conservative_memory_allocation)
{
}

// Delegation methods - forward to impl accessor methods
template <typename IdxT>
IdxT index<IdxT>::size() const noexcept
{
  return impl_->size();
}

template <typename IdxT>
uint32_t index<IdxT>::dim() const noexcept
{
  return impl_->dim();
}

template <typename IdxT>
uint32_t index<IdxT>::dim_ext() const noexcept
{
  return impl_->dim_ext();
}

template <typename IdxT>
uint32_t index<IdxT>::rot_dim() const noexcept
{
  return impl_->rot_dim();
}

template <typename IdxT>
uint32_t index<IdxT>::pq_bits() const noexcept
{
  return impl_->pq_bits();
}

template <typename IdxT>
uint32_t index<IdxT>::pq_dim() const noexcept
{
  return impl_->pq_dim();
}

template <typename IdxT>
uint32_t index<IdxT>::pq_len() const noexcept
{
  return impl_->pq_len();
}

template <typename IdxT>
uint32_t index<IdxT>::pq_book_size() const noexcept
{
  return impl_->pq_book_size();
}

template <typename IdxT>
cuvs::distance::DistanceType index<IdxT>::metric() const noexcept
{
  return impl_->metric();
}

template <typename IdxT>
codebook_gen index<IdxT>::codebook_kind() const noexcept
{
  return impl_->codebook_kind();
}

template <typename IdxT>
list_layout index<IdxT>::codes_layout() const noexcept
{
  return impl_->codes_layout();
}

template <typename IdxT>
uint32_t index<IdxT>::n_lists() const noexcept
{
  return impl_->n_lists();
}

template <typename IdxT>
bool index<IdxT>::conservative_memory_allocation() const noexcept
{
  return impl_->conservative_memory_allocation();
}

template <typename IdxT>
raft::device_mdspan<const float, pq_centers_extents, raft::row_major> index<IdxT>::pq_centers()
  const noexcept
{
  return impl_->pq_centers();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::centers()
  const noexcept
{
  return impl_->centers();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::centers_rot()
  const noexcept
{
  return impl_->centers_rot();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::rotation_matrix()
  const noexcept
{
  return impl_->rotation_matrix();
}

template <typename IdxT>
std::vector<std::shared_ptr<list_data_base<IdxT>>>& index<IdxT>::lists() noexcept
{
  return impl_->lists();
}

template <typename IdxT>
const std::vector<std::shared_ptr<list_data_base<IdxT>>>& index<IdxT>::lists() const noexcept
{
  return impl_->lists();
}

template <typename IdxT>
raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> index<IdxT>::data_ptrs() noexcept
{
  return impl_->data_ptrs();
}

template <typename IdxT>
raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major> index<IdxT>::data_ptrs()
  const noexcept
{
  return impl_->data_ptrs();
}

template <typename IdxT>
raft::device_vector_view<IdxT*, uint32_t, raft::row_major> index<IdxT>::inds_ptrs() noexcept
{
  return impl_->inds_ptrs();
}

template <typename IdxT>
raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> index<IdxT>::inds_ptrs()
  const noexcept
{
  return impl_->inds_ptrs();
}

template <typename IdxT>
raft::host_vector_view<IdxT, uint32_t, raft::row_major> index<IdxT>::accum_sorted_sizes() noexcept
{
  return impl_->accum_sorted_sizes();
}

template <typename IdxT>
raft::host_vector_view<const IdxT, uint32_t, raft::row_major> index<IdxT>::accum_sorted_sizes()
  const noexcept
{
  return impl_->accum_sorted_sizes();
}

template <typename IdxT>
raft::device_vector_view<uint32_t, uint32_t, raft::row_major> index<IdxT>::list_sizes() noexcept
{
  return impl_->list_sizes();
}

template <typename IdxT>
raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> index<IdxT>::list_sizes()
  const noexcept
{
  return impl_->list_sizes();
}

// centers() and centers_rot() are now pure virtual and implemented in derived classes

template <typename IdxT>
uint32_t index<IdxT>::get_list_size_in_bytes(uint32_t label) const
{
  return impl_->get_list_size_in_bytes(label);
}

template <typename IdxT>
std::optional<bool> index<IdxT>::use_ann_for_extend() const noexcept
{
  return impl_->use_ann_for_extend();
}

template <typename IdxT>
void index_impl<IdxT>::check_consistency()
{
  RAFT_EXPECTS(pq_bits_ >= 4 && pq_bits_ <= 8,
               "`pq_bits` must be within closed range [4,8], but got %u.",
               pq_bits_);
  RAFT_EXPECTS((pq_bits_ * pq_dim_) % 8 == 0,
               "`pq_bits * pq_dim` must be a multiple of 8, but got %u * %u = %u.",
               pq_bits_,
               pq_dim_,
               pq_bits_ * pq_dim_);
}

template <typename IdxT>
uint32_t index<IdxT>::calculate_pq_dim(uint32_t dim)
{
  if (dim >= 128) { dim /= 2; }
  auto r = raft::round_down_safe<uint32_t>(dim, 32);
  if (r > 0) return r;
  r = 1;
  while ((r << 1) <= dim) {
    r = r << 1;
  }
  return r;
}

template <typename IdxT>
uint32_t index_impl<IdxT>::get_list_size_in_bytes(uint32_t label) const
{
  RAFT_EXPECTS(label < lists_.size(),
               "Expected label to be less than number of lists in the index");
  return lists_[label]->data_byte_size();
}

template <typename IdxT>
raft::device_matrix_view<const int8_t, uint32_t, raft::row_major>
index_impl<IdxT>::rotation_matrix_int8(const raft::resources& res) const
{
  if (!rotation_matrix_int8_.has_value()) {
    // Get dimensions without calling virtual function that returns matrix view
    uint32_t rot_dim = this->rot_dim();
    uint32_t dim     = this->dim();
    rotation_matrix_int8_.emplace(
      raft::make_device_mdarray<int8_t, uint32_t>(res, raft::make_extents<uint32_t>(rot_dim, dim)));

    // Use vector views to avoid host_device_accessor issues
    // Calculate size manually to avoid calling view().size()
    size_t matrix_size = static_cast<size_t>(rot_dim) * static_cast<size_t>(dim);
    auto output_vec    = raft::make_device_vector_view<int8_t, size_t>(
      rotation_matrix_int8_->data_handle(), matrix_size);
    auto input_vec = raft::make_device_vector_view<const float, size_t>(
      this->rotation_matrix().data_handle(), matrix_size);

    raft::linalg::map(
      res, output_vec, cuvs::spatial::knn::detail::utils::mapping<int8_t>{}, input_vec);
  }
  // Construct the view directly to avoid copy constructor issues
  return raft::make_device_matrix_view<const int8_t, uint32_t>(rotation_matrix_int8_->data_handle(),
                                                               rotation_matrix_int8_->extent(0),
                                                               rotation_matrix_int8_->extent(1));
}

template <typename IdxT>
raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> index_impl<IdxT>::centers_int8(
  const raft::resources& res) const
{
  if (!centers_int8_.has_value()) {
    uint32_t n_lists      = lists().size();
    uint32_t dim          = this->dim();
    uint32_t dim_ext      = raft::round_up_safe(dim + 1, 8u);
    uint32_t dim_ext_int8 = raft::round_up_safe(dim + 2, 16u);
    centers_int8_.emplace(raft::make_device_matrix<int8_t, uint32_t>(res, n_lists, dim_ext_int8));

    // Get the centers matrix view and immediately extract the raw pointer
    auto centers_view   = this->centers();
    const float* inputs = centers_view.data_handle();
    /* NOTE: maximizing the range and the precision of int8_t GEMM

    int8_t has a very limited range [-128, 127], which is problematic when storing both vectors and
    their squared norms in one place.

    We map all dimensions by multiplying by 128. But that means we need to multiply the squared norm
    component by `128^2`, which we cannot afford, since it most likely overflows.
    So, a naive mapping would be:
    ```
      [c_1 * 128, c_2, * 128, ...., c_(dim-1) * 128,  n2 * 128 * 128, 0 ... 0]
      • [q_1 * 128, q_2 * 128, ..., q_(dim-1)*128, -0.5, 0, ... 0]
    ```

    Which is at first can be improved by moving one 128 to the query side:
    ```
      [c_1 * 128, c_2, * 128, ...., c_(dim-1) * 128,  n2 * 128, 0 ... 0]
      • [q_1 * 128, q_2 * 128, ..., q_(dim-1)*128, -64, 0, ... 0]
    ```

    Yet this still only works for vectors with L2 norms not bigger than one and has a rather awful
    granularity of 64. To improve both the range and the precision, we count the number of available
    slots `m > 2` and decompose the squared norm, such that:
    ```
      0.5 * 128 * n2 = 64 * n2 = 128 * z + (m - 1) * y
    ```
    where `y` maximizes the available range while `z` encodes the rounding error.
    Then we get following dot product during the coarse search:
    ```
      [c_1 * 128, c_2, * 128, ...., c_(dim-1) * 128,  z, y, ... y]
      • [q_1 * 128, q_2 * 128, ..., q_(dim-1)*128, 1 - m,  -128, ... -128]
    ```
    `m` is maximum 16, so we get the coefficient much lower than the naive 64 on the query side; and
    it is limited by the range we can cover (the squared norm must be within `m * 2` before
    normalization).
    */
    // Get raw pointer to avoid capturing matrix view in device lambda
    int8_t* centers_int8_ptr = this->centers_int8_->data_handle();
    // Calculate size manually to avoid calling view().size()
    size_t centers_int8_size = static_cast<size_t>(n_lists) * static_cast<size_t>(dim_ext_int8);

    auto centers_int8_vec_view =
      raft::make_device_vector_view<int8_t, size_t>(centers_int8_ptr, centers_int8_size);

    raft::linalg::map_offset(res,
                             centers_int8_vec_view,
                             [dim, dim_ext, dim_ext_int8, inputs] __device__(size_t ix) -> int8_t {
                               uint32_t col = ix % dim_ext_int8;
                               uint32_t row = ix / dim_ext_int8;
                               if (col < dim) {
                                 return static_cast<int8_t>(fmaxf(
                                   -128.0f, fminf(127.0f, inputs[col + row * dim_ext] * 128.0f)));
                               }
                               auto x = inputs[row * dim_ext + dim];
                               auto c = 64.0f / static_cast<float>(dim_ext_int8 - dim - 1);
                               auto y = fmaxf(-128.0f, fminf(127.0f, x * c));
                               auto z = fmaxf(-128.0f, fminf(127.0f, (y - roundf(y)) * 128.0f));
                               if (col > dim) { return static_cast<int8_t>(roundf(y)); }
                               return static_cast<int8_t>(z);
                             });
  }

  // Construct the view directly to avoid copy constructor issues
  return raft::make_device_matrix_view<const int8_t, uint32_t>(
    centers_int8_->data_handle(), centers_int8_->extent(0), centers_int8_->extent(1));
}

template <typename IdxT>
raft::device_matrix_view<const half, uint32_t, raft::row_major>
index_impl<IdxT>::rotation_matrix_half(const raft::resources& res) const
{
  if (!rotation_matrix_half_.has_value()) {
    // Get dimensions without calling virtual function that returns matrix view
    uint32_t rot_dim = this->rot_dim();
    uint32_t dim     = this->dim();
    rotation_matrix_half_.emplace(
      raft::make_device_mdarray<half, uint32_t>(res, raft::make_extents<uint32_t>(rot_dim, dim)));

    // Use vector views to avoid host_device_accessor issues
    // Calculate size manually to avoid calling view().size()
    size_t matrix_size = static_cast<size_t>(rot_dim) * static_cast<size_t>(dim);
    auto output_vec    = raft::make_device_vector_view<half, size_t>(
      rotation_matrix_half_->data_handle(), matrix_size);
    auto input_vec = raft::make_device_vector_view<const float, size_t>(
      this->rotation_matrix().data_handle(), matrix_size);

    raft::linalg::map(res, output_vec, raft::cast_op<half>{}, input_vec);
  }
  // Construct the view directly to avoid copy constructor issues
  return raft::make_device_matrix_view<const half, uint32_t>(rotation_matrix_half_->data_handle(),
                                                             rotation_matrix_half_->extent(0),
                                                             rotation_matrix_half_->extent(1));
}

template <typename IdxT>
raft::device_matrix_view<const half, uint32_t, raft::row_major> index_impl<IdxT>::centers_half(
  const raft::resources& res) const
{
  if (!centers_half_.has_value()) {
    // Get dimensions without calling virtual function that returns matrix view
    uint32_t n_lists = this->n_lists();
    uint32_t dim_ext = this->dim_ext();
    centers_half_.emplace(raft::make_device_mdarray<half, uint32_t>(
      res, raft::make_extents<uint32_t>(n_lists, dim_ext)));

    // Use vector views to avoid host_device_accessor issues
    // Calculate size manually to avoid calling view().size()
    size_t matrix_size = static_cast<size_t>(n_lists) * static_cast<size_t>(dim_ext);
    auto output_vec =
      raft::make_device_vector_view<half, size_t>(centers_half_->data_handle(), matrix_size);
    auto input_vec = raft::make_device_vector_view<const float, size_t>(
      this->centers().data_handle(), matrix_size);

    raft::linalg::map(res, output_vec, raft::cast_op<half>{}, input_vec);
  }
  // Construct the view directly to avoid copy constructor issues
  return raft::make_device_matrix_view<const half, uint32_t>(
    centers_half_->data_handle(), centers_half_->extent(0), centers_half_->extent(1));
}

template <typename IdxT>
raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> index<IdxT>::rotation_matrix_int8(
  const raft::resources& res) const
{
  return impl_->rotation_matrix_int8(res);
}

template <typename IdxT>
raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> index<IdxT>::centers_int8(
  const raft::resources& res) const
{
  return impl_->centers_int8(res);
}

template <typename IdxT>
raft::device_matrix_view<const half, uint32_t, raft::row_major> index<IdxT>::rotation_matrix_half(
  const raft::resources& res) const
{
  return impl_->rotation_matrix_half(res);
}

template <typename IdxT>
raft::device_matrix_view<const half, uint32_t, raft::row_major> index<IdxT>::centers_half(
  const raft::resources& res) const
{
  return impl_->centers_half(res);
}

namespace helpers {
size_t compressed_dataset_size(raft::resources const& res,
                               raft::matrix_extent<int64_t> dataset,
                               cuvs::neighbors::ivf_pq::index_params params)
{
  auto idx = cuvs::neighbors::ivf_pq::index<int64_t>(res, params, (uint32_t)dataset.extent(1));
  size_t pq_book_size                         = 1 << params.pq_bits;
  constexpr static uint32_t kIndexGroupSize   = 32;
  constexpr static uint32_t kIndexGroupVecLen = 16;

  size_t pq_chunk = (kIndexGroupVecLen * 8) / params.pq_bits;
  size_t pq_centers =
    (params.codebook_kind == codebook_gen::PER_SUBSPACE ? params.pq_dim : params.n_lists) *
    idx.pq_len() * pq_book_size * sizeof(float);
  size_t pq_dataset = raft::ceildiv<size_t>(dataset.extent(0), kIndexGroupSize) * kIndexGroupSize *
                      raft::ceildiv<size_t>(params.pq_dim, pq_chunk) * kIndexGroupVecLen;
  size_t indices         = dataset.extent(0) * sizeof(int64_t);
  size_t rotation_matrix = idx.rot_dim() * idx.dim() * sizeof(float);
  size_t list_offsets    = (params.n_lists + 1) * sizeof(int64_t);
  size_t list_sizes      = params.n_lists * sizeof(int64_t);
  size_t centers         = params.n_lists * idx.dim_ext() * sizeof(float);
  size_t centers_rot     = params.n_lists * idx.rot_dim() * sizeof(float);

  return pq_centers + pq_dataset + indices + rotation_matrix + list_offsets + list_sizes + centers +
         centers_rot;
}
}  // namespace helpers

template class view_impl<int64_t>;
template class owning_impl<int64_t>;
template class index<int64_t>;

}  // namespace cuvs::neighbors::ivf_pq
