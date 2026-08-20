/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/scann.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>
#include <raft/linalg/map.cuh>

#include "../../../util/kvikio_serialize.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <type_traits>

namespace cuvs::neighbors::experimental::scann::detail {

constexpr int kSerializationVersion = 1;

// Helper for serializing device/host matrix to a given file
template <typename T,
          typename IdxT,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
void serialize_matrix(
  raft::resources const& res,
  std::filesystem::path file_path,
  raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, Accessor> mat)
{
  cuvs::util::kvikio_ofstream mat_of(file_path.string());
  RAFT_EXPECTS(mat_of, "Cannot open file %s", file_path.string().c_str());

  cuvs::util::detail::serialize_mdspan(res, mat_of, mat);

  mat_of.close();
  RAFT_EXPECTS(mat_of, "Error writing output %s", file_path.string().c_str());
}

// ScaNN stores primary and soar cluster assignments interleaved in a single vector.
// When the primary and soar assignment are equal, the soar assignment should be -1
// This is a helper for combining the primary and soar assignments into this format
// and serializing it
template <typename T, typename IdxT>
void save_labels(raft::resources const& res,
                 std::filesystem::path labels_path,
                 const index<T, IdxT>& index_)
{
  auto combined_labels  = raft::make_device_vector<int, IdxT>(res, 2 * index_.labels().extent(0));
  auto labels_view      = index_.labels();
  auto soar_labels_view = index_.soar_labels();

  raft::linalg::map_offset(
    res, combined_labels.view(), [labels_view, soar_labels_view] __device__(size_t i) {
      size_t label_type = i % 2;
      size_t idx        = i / 2;

      int label      = labels_view(idx);
      int soar_label = soar_labels_view(idx);

      if (label == soar_label) { soar_label = -1; }

      if (label_type == 0) { return label; }

      return soar_label;
    });

  cuvs::util::kvikio_ofstream labels_of(labels_path.string());
  RAFT_EXPECTS(labels_of, "Cannot open file %s", labels_path.string().c_str());

  cuvs::util::detail::serialize_device_mdspan(res, labels_of, combined_labels.view());

  labels_of.close();
  RAFT_EXPECTS(labels_of, "Error writing output %s", labels_path.string().c_str());
}

/**
 * Save the ScaNN index into multiple files.
 *
 * Experimental, both the API and the serialization format are subject to change.
 * The format is meant to ease the integration into OSS ScaNN for search.
 * Labels and quantized vectors are stored directly in .npy files, as required by OSS ScaNN.
 * Cluster centers and the pq_codebook are also stored in .npy files, for later
 * conversion into the correct protobuf structs by an external tool. Additional metadata
 * required for this conversion are also serialized.
 *
 * @param[in] res the raft resource handle
 * @param[in] scann_assets_dir:  the directory where ScaNN index assets will be saved
 * @param[in] index_ ScaNN index
 *
 */
template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               const std::string& scann_assets_dir,
               const index<T, IdxT>& index_)
{
  // Metadata
  std::filesystem::path scann_path(scann_assets_dir);

  const auto metadata_path = scann_path / "cuvs_metadata.bin";
  cuvs::util::kvikio_ofstream metadata_of(metadata_path.string());
  RAFT_EXPECTS(metadata_of, "Cannot open file %s", metadata_path.string().c_str());

  raft::serialize_scalar(res, metadata_of, kSerializationVersion);
  raft::serialize_scalar(res, metadata_of, index_.dim());
  raft::serialize_scalar(res, metadata_of, index_.pq_dim());

  metadata_of.close();
  RAFT_EXPECTS(metadata_of, "Error writing output %s", metadata_path.string().c_str());

  // kmeans cluster centers
  serialize_matrix(res, scann_path / "centers.npy", index_.centers());

  // cluster assignments
  save_labels(res, scann_path / "datapoint_to_token.npy", index_);

  // codebook

  serialize_matrix(res, scann_path / "pq_codebook.npy", index_.pq_codebook());

  // quantized residuals
  serialize_matrix(res, scann_path / "hashed_dataset.npy", index_.quantized_residuals());

  // quantized SOAR residuals
  serialize_matrix(res, scann_path / "hashed_dataset_soar.npy", index_.quantized_soar_residuals());

  // bf16 dataset
  if (index_.bf16_dataset().extent(0) > 0) {
    serialize_matrix(res, scann_path / "bf16_dataset.npy", index_.bf16_dataset());
  }
}

}  // namespace cuvs::neighbors::experimental::scann::detail
