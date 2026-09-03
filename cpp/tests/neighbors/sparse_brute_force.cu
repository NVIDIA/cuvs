/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cusparse_v2.h>
#include <gtest/gtest.h>

namespace cuvs {
namespace neighbors {

using namespace raft;
using namespace raft::sparse;

template <typename value_idx, typename value_t>
struct SparseKNNInputs {
  value_idx n_cols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  std::vector<value_t> out_dists_ref_h;
  std::vector<value_idx> out_indices_ref_h;

  int k;

  int batch_size_index = 2;
  int batch_size_query = 2;

  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded;
};

template <typename value_idx, typename value_t>
::std::ostream& operator<<(::std::ostream& os, const SparseKNNInputs<value_idx, value_t>& dims)
{
  return os;
}

template <typename value_idx, typename value_t>
class SparseKNNTest : public ::testing::TestWithParam<SparseKNNInputs<value_idx, value_t>> {
 public:
  SparseKNNTest()
    : params(::testing::TestWithParam<SparseKNNInputs<value_idx, value_t>>::GetParam()),
      indptr(0, resource::get_cuda_stream(handle)),
      indices(0, resource::get_cuda_stream(handle)),
      data(0, resource::get_cuda_stream(handle)),
      out_indices(0, resource::get_cuda_stream(handle)),
      out_dists(0, resource::get_cuda_stream(handle)),
      out_indices_ref(0, resource::get_cuda_stream(handle)),
      out_dists_ref(0, resource::get_cuda_stream(handle))
  {
  }

 protected:
  void SetUp() override
  {
    n_rows = params.indptr_h.size() - 1;
    nnz    = params.indices_h.size();
    k      = params.k;

    make_data();

    auto index_structure =
      raft::make_device_compressed_structure_view<value_idx, value_idx, value_idx>(
        indptr.data(), indices.data(), n_rows, params.n_cols, nnz);
    auto index_csr = raft::make_device_csr_matrix_view<const value_t>(data.data(), index_structure);

    auto index = cuvs::neighbors::brute_force::build(handle, index_csr, params.metric);

    cuvs::neighbors::brute_force::sparse_search_params search_params;
    search_params.batch_size_index = params.batch_size_index;
    search_params.batch_size_query = params.batch_size_query;

    cuvs::neighbors::brute_force::search(
      handle,
      search_params,
      index,
      index_csr,
      raft::make_device_matrix_view<value_idx, int64_t>(out_indices.data(), n_rows, k),
      raft::make_device_matrix_view<value_t, int64_t>(out_dists.data(), n_rows, k));

    RAFT_CUDA_TRY(cudaStreamSynchronize(resource::get_cuda_stream(handle).get()));
  }

  void compare()
  {
    ASSERT_TRUE(devArrMatch(
      out_dists_ref.data(), out_dists.data(), n_rows * k, CompareApprox<value_t>(1e-4)));
    ASSERT_TRUE(
      devArrMatch(out_indices_ref.data(), out_indices.data(), n_rows * k, Compare<value_idx>()));
  }

 protected:
  void make_data()
  {
    std::vector<value_idx> indptr_h  = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h      = params.data_h;

    auto stream = resource::get_cuda_stream(handle);
    indptr.resize(indptr_h.size(), stream);
    indices.resize(indices_h.size(), stream);
    data.resize(data_h.size(), stream);

    update_device(indptr.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(indices.data(), indices_h.data(), indices_h.size(), stream);
    update_device(data.data(), data_h.data(), data_h.size(), stream);

    std::vector<value_t> out_dists_ref_h     = params.out_dists_ref_h;
    std::vector<value_idx> out_indices_ref_h = params.out_indices_ref_h;

    out_indices_ref.resize(out_indices_ref_h.size(), stream);
    out_dists_ref.resize(out_dists_ref_h.size(), stream);

    update_device(
      out_indices_ref.data(), out_indices_ref_h.data(), out_indices_ref_h.size(), stream);
    update_device(out_dists_ref.data(), out_dists_ref_h.data(), out_dists_ref_h.size(), stream);

    out_dists.resize(n_rows * k, stream);
    out_indices.resize(n_rows * k, stream);
  }

  raft::resources handle;

  int n_rows, nnz, k;

  // input data
  rmm::device_uvector<value_idx> indptr, indices;
  rmm::device_uvector<value_t> data;

  // output data
  rmm::device_uvector<value_idx> out_indices;
  rmm::device_uvector<value_t> out_dists;

  rmm::device_uvector<value_idx> out_indices_ref;
  rmm::device_uvector<value_t> out_dists_ref;

  SparseKNNInputs<value_idx, value_t> params;
};

const std::vector<SparseKNNInputs<int, float>> inputs_i32_f = {
  {9,                                                 // ncols
   {0, 2, 4, 6, 8},                                   // indptr
   {0, 4, 0, 3, 0, 2, 0, 8},                          // indices
   {0.0f, 1.0f, 5.0f, 6.0f, 5.0f, 6.0f, 0.0f, 1.0f},  // data
   {0, 1.41421, 0, 7.87401, 0, 7.87401, 0, 1.41421},  // dists
   {0, 3, 1, 0, 2, 0, 3, 0},                          // inds
   2,
   2,
   2,
   cuvs::distance::DistanceType::L2SqrtExpanded}};
typedef SparseKNNTest<int, float> SparseKNNTestF;
TEST_P(SparseKNNTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseKNNTest, SparseKNNTestF, ::testing::ValuesIn(inputs_i32_f));

};  // end namespace neighbors
};  // end namespace cuvs
