/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/preprocessing/spectral_embedding.hpp>

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/laplacian.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/solver/lanczos.cuh>
#include <raft/sparse/solver/lanczos_types.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

namespace cuvs::preprocessing::spectral_embedding::detail {

template <typename DataT, typename OutSparseMatrixType, typename InSparseMatrixViewType>
OutSparseMatrixType create_laplacian(raft::resources const& handle,
                                     params spectral_embedding_config,
                                     InSparseMatrixViewType sparse_matrix_view,
                                     raft::device_vector_view<DataT, int> diagonal)
{
  auto laplacian =
    spectral_embedding_config.norm_laplacian
      ? raft::sparse::linalg::laplacian_normalized(handle, sparse_matrix_view, diagonal)
      : raft::sparse::linalg::compute_graph_laplacian(handle, sparse_matrix_view);

  auto laplacian_elements_view = raft::make_device_vector_view<DataT>(
    laplacian.get_elements().data(), laplacian.structure_view().get_nnz());

  raft::linalg::map(
    handle,
    laplacian_elements_view,
    [] __device__(DataT x) { return -x; },
    raft::make_const_mdspan(laplacian_elements_view));

  return laplacian;
}

template <typename DataT, typename InSparseMatrixViewType>
void compute_eigenpairs(raft::resources const& handle,
                        params spectral_embedding_config,
                        const int n_samples,
                        InSparseMatrixViewType laplacian_view,
                        raft::device_vector_view<DataT, int> diagonal,
                        raft::device_matrix_view<DataT, int, raft::col_major> embedding)
{
  auto config           = raft::sparse::solver::lanczos_solver_config<DataT>();
  config.n_components   = spectral_embedding_config.n_components;
  config.max_iterations = 10 * n_samples;
  RAFT_EXPECTS(n_samples - config.n_components > 0,
               "Please set `ncv` to a value in (0, n_samples)");
  config.ncv = std::min(n_samples - config.n_components, std::max(2 * config.n_components + 1, 20));
  config.tolerance = spectral_embedding_config.tolerance;
  config.which     = raft::sparse::solver::LANCZOS_WHICH::LA;
  config.seed      = spectral_embedding_config.seed;

  auto eigenvalues =
    raft::make_device_vector<DataT, int, raft::col_major>(handle, config.n_components);
  auto eigenvectors =
    raft::make_device_matrix<DataT, int, raft::col_major>(handle, n_samples, config.n_components);

  raft::sparse::solver::lanczos_compute_eigenpairs<int, DataT>(
    handle, config, laplacian_view, std::nullopt, eigenvalues.view(), eigenvectors.view());

  if (spectral_embedding_config.norm_laplacian) {
    raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
      handle,
      raft::make_const_mdspan(eigenvectors.view()),  // input matrix view
      raft::make_const_mdspan(diagonal),             // input vector view
      eigenvectors.view(),                           // output matrix view (in-place)
      [] __device__(DataT elem, DataT diag) { return elem / diag; });
  }

  // Create a sequence of reversed column indices
  config.n_components =
    spectral_embedding_config.drop_first ? config.n_components - 1 : config.n_components;
  auto col_indices = raft::make_device_vector<int>(handle, config.n_components);

  raft::linalg::map_offset(handle,
                           col_indices.view(),
                           [n = config.n_components] __device__(int idx) { return n - 1 - idx; });

  // Create row-major views of the column-major matrices
  // This is just a view re-interpretation, no data movement
  auto eigenvectors_row_view = raft::make_device_matrix_view<DataT, int, raft::row_major>(
    eigenvectors.data_handle(),
    eigenvectors.extent(1),  // Swap dimensions for the view
    eigenvectors.extent(0));

  auto embedding_row_view = raft::make_device_matrix_view<DataT, int, raft::row_major>(
    embedding.data_handle(),
    embedding.extent(1),  // Swap dimensions for the view
    embedding.extent(0));

  raft::matrix::gather<DataT, int, int>(
    handle,
    raft::make_const_mdspan(eigenvectors_row_view),  // Source matrix (as row-major view)
    raft::make_const_mdspan(col_indices.view()),     // Column indices to gather
    embedding_row_view                               // Destination matrix (as row-major view)
  );
}

template <typename DataT, typename NNZType>
void transform(raft::resources const& handle,
               params spectral_embedding_config,
               raft::device_coo_matrix_view<DataT, int, int, NNZType> connectivity_graph,
               raft::device_matrix_view<DataT, int, raft::col_major> embedding)
{
  const int n_samples = connectivity_graph.structure_view().get_n_rows();
  auto diagonal       = raft::make_device_vector<DataT, int>(handle, n_samples);

  auto laplacian = create_laplacian<DataT, raft::device_coo_matrix<DataT, int, int, NNZType>>(
    handle, spectral_embedding_config, connectivity_graph, diagonal.view());
  compute_eigenpairs(
    handle, spectral_embedding_config, n_samples, laplacian.view(), diagonal.view(), embedding);
}

template <typename NNZType>
void create_connectivity_graph(
  raft::resources const& handle,
  cuvs::preprocessing::spectral_embedding::params spectral_embedding_config,
  raft::device_matrix_view<float, int, raft::row_major> dataset,
  raft::device_coo_matrix<float, int, int, NNZType>& connectivity_graph)
{
  const int64_t n_samples  = dataset.extent(0);
  const int64_t n_features = dataset.extent(1);
  const int k_search       = spectral_embedding_config.n_neighbors;
  const NNZType nnz        = static_cast<NNZType>(n_samples) * k_search;

  auto stream = raft::resource::get_cuda_stream(handle);

  auto d_indices   = raft::make_device_matrix<int64_t, int64_t>(handle, n_samples, k_search);
  auto d_distances = raft::make_device_matrix<float, int64_t>(handle, n_samples, k_search);

  cuvs::neighbors::all_neighbors::all_neighbors_params all_neighbors_params{
    .graph_build_params =
      cuvs::neighbors::graph_build_params::brute_force_params{
        .build_params = {{.metric = cuvs::distance::DistanceType::L2SqrtExpanded}}},
    .metric = cuvs::distance::DistanceType::L2SqrtExpanded};
  cuvs::neighbors::all_neighbors::build(handle,
                                        all_neighbors_params,
                                        raft::make_const_mdspan(dataset),
                                        d_indices.view(),
                                        d_distances.view());

  auto knn_rows = raft::make_device_vector<int, NNZType>(handle, nnz);
  auto knn_cols = raft::make_device_vector<int, NNZType>(handle, nnz);

  raft::linalg::map(
    handle,
    knn_cols.view(),
    [] __device__(int64_t x) { return static_cast<int>(x); },
    raft::make_const_mdspan(d_indices.view()));

  raft::linalg::map_offset(handle, knn_rows.view(), [k_search] __device__(NNZType idx) -> int {
    return static_cast<int>(idx / k_search);
  });

  // set all distances to 1.0f (connectivity KNN graph)
  raft::matrix::fill(
    handle, raft::make_device_vector_view<float, NNZType>(d_distances.data_handle(), nnz), 1.0f);

  auto coo_matrix_view = raft::make_device_coo_matrix_view<const float, int, int, NNZType>(
    d_distances.data_handle(),
    raft::make_device_coordinate_structure_view<int, int, NNZType>(
      knn_rows.data_handle(), knn_cols.data_handle(), n_samples, n_samples, nnz));

  auto sym_coo1_matrix =
    raft::make_device_coo_matrix<float, int, int, NNZType>(handle, n_samples, n_samples);
  raft::sparse::linalg::coo_symmetrize<128, float, int, NNZType>(
    handle, coo_matrix_view, sym_coo1_matrix, [] __device__(int row, int col, float a, float b) {
      return 0.5f * (a + b);
    });

  raft::sparse::op::coo_sort<float, int, NNZType>(
    n_samples,
    n_samples,
    sym_coo1_matrix.structure_view().get_nnz(),
    sym_coo1_matrix.structure_view().get_rows().data(),
    sym_coo1_matrix.structure_view().get_cols().data(),
    sym_coo1_matrix.get_elements().data(),
    stream.get());

  raft::sparse::op::coo_remove_scalar<128, float, int, NNZType>(
    handle,
    raft::make_device_coo_matrix_view<const float, int, int, NNZType>(
      sym_coo1_matrix.get_elements().data(), sym_coo1_matrix.structure_view()),
    raft::make_host_scalar<float>(0.0f).view(),
    connectivity_graph);
}

void transform(raft::resources const& handle,
               params spectral_embedding_config,
               raft::device_matrix_view<float, int, raft::row_major> dataset,
               raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  const int n_samples = dataset.extent(0);

  auto sym_coo_matrix =
    raft::make_device_coo_matrix<float, int, int, int64_t>(handle, n_samples, n_samples);
  auto diagonal = raft::make_device_vector<float, int>(handle, n_samples);

  create_connectivity_graph<int64_t>(handle, spectral_embedding_config, dataset, sym_coo_matrix);
  auto laplacian = create_laplacian<float, raft::device_coo_matrix<float, int, int, int64_t>>(
    handle, spectral_embedding_config, sym_coo_matrix.view(), diagonal.view());
  compute_eigenpairs<float>(
    handle, spectral_embedding_config, n_samples, laplacian.view(), diagonal.view(), embedding);
}

}  // namespace cuvs::preprocessing::spectral_embedding::detail
