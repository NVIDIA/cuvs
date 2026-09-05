/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/cluster/spectral.hpp>
#include <cuvs/preprocessing/spectral_embedding.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng_state.hpp>

namespace cuvs::cluster::spectral::detail {

template <typename DataT>
void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_coo_matrix_view<DataT, int, int, int> connectivity_graph,
                 raft::device_vector_view<int, int> labels)
{
  int n_samples = connectivity_graph.structure_view().get_n_rows();
  DataT inertia;
  int n_iter;
  auto embedding_col_major =
    raft::make_device_matrix<DataT, int, raft::col_major>(handle, n_samples, config.n_components);
  auto embedding_row_major =
    raft::make_device_matrix<DataT, int, raft::row_major>(handle, n_samples, config.n_components);
  cuvs::preprocessing::spectral_embedding::params spectral_embedding_config;
  spectral_embedding_config.n_components   = config.n_components;
  spectral_embedding_config.n_neighbors    = config.n_neighbors;
  spectral_embedding_config.norm_laplacian = true;
  spectral_embedding_config.drop_first     = false;
  spectral_embedding_config.seed           = config.rng_state.seed;
  spectral_embedding_config.tolerance      = config.tolerance;

  cuvs::cluster::kmeans::params kmeans_config;
  kmeans_config.n_clusters          = config.n_clusters;
  kmeans_config.rng_state           = config.rng_state;
  kmeans_config.n_init              = config.n_init;
  kmeans_config.oversampling_factor = 0.0;

  cuvs::preprocessing::spectral_embedding::transform(
    handle, spectral_embedding_config, connectivity_graph, embedding_col_major.view());

  raft::linalg::transpose(handle,
                          embedding_col_major.data_handle(),
                          embedding_row_major.data_handle(),
                          n_samples,
                          config.n_components,
                          raft::resource::get_cuda_stream(handle).get());

  cuvs::cluster::kmeans::fit_predict(handle,
                                     kmeans_config,
                                     embedding_row_major.view(),
                                     std::nullopt,
                                     std::nullopt,
                                     labels,
                                     raft::make_host_scalar_view(&inertia),
                                     raft::make_host_scalar_view(&n_iter));
}

void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_matrix_view<float, int, raft::row_major> dataset,
                 raft::device_vector_view<int, int> labels)
{
  int n_samples = dataset.extent(0);

  auto graph = raft::make_device_coo_matrix<float, int, int, int>(handle, n_samples, n_samples);

  cuvs::preprocessing::spectral_embedding::params embed_params;
  embed_params.n_neighbors = config.n_neighbors;

  cuvs::preprocessing::spectral_embedding::helpers::create_connectivity_graph(
    handle, embed_params, dataset, graph);

  fit_predict(handle, config, graph.view(), labels);
}

}  // namespace cuvs::cluster::spectral::detail
