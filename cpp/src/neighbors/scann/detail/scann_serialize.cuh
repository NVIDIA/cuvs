/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <type_traits>

namespace cuvs::neighbors::experimental::scann::detail {

constexpr int kSerializationVersion = 1;

namespace {
// Helper for opening an ofstream
std::ofstream open_file(std::string file_name)
{
  std::ofstream file_of(file_name, std::ios::out | std::ios::binary);

  if (!file_of) { RAFT_FAIL("Cannot open file %s", file_name.c_str()); }

  return file_of;
}
}  // namespace

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
  auto mat_of = open_file(file_path);

  raft::serialize_mdspan(res, mat_of, mat);

  mat_of.close();
}

// Helper for serializing device/host vector to a given file
template <typename T,
          typename IdxT,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
void serialize_vector(
  raft::resources const& res,
  std::filesystem::path file_path,
  raft::mdspan<const T, raft::vector_extent<IdxT>, raft::row_major, Accessor> mat)
{
  auto mat_of = open_file(file_path);

  raft::serialize_mdspan(res, mat_of, mat);

  mat_of.close();
}

// ScaNN stores primary and soar cluster assignments interleaved in a single vector.
// When the primary and soar assignment are equal, the soar assignment should be -1
// This is a helper for combining the primary and soar assignments into this format
// and serializing it
template <typename IdxT>
void save_labels(raft::resources const& res,
                 std::filesystem::path labels_path,
                 raft::device_vector_view<const uint32_t, IdxT> labels_view,
                 raft::device_vector_view<const uint32_t, IdxT> soar_labels_view)
{
  auto combined_labels = raft::make_device_vector<int, IdxT>(res, 2 * labels_view.extent(0));
  // auto labels_view      = index_.labels();
  // auto soar_labels_view = index_.soar_labels();

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

  auto labels_of = open_file(labels_path);

  raft::serialize_mdspan(res, labels_of, combined_labels.view());

  labels_of.close();
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

  auto metadata_of = open_file(scann_path / "cuvs_metadata.bin");

  raft::serialize_scalar(res, metadata_of, kSerializationVersion);
  raft::serialize_scalar(res, metadata_of, index_.dim());
  raft::serialize_scalar(res, metadata_of, index_.pq_dim());
  // Record number of tree levels
  raft::serialize_scalar(res, metadata_of, index_.coarse_centers().extent(0) > 0 ? 2 : 1);

  metadata_of.close();

  // kmeans cluster centers
  serialize_matrix(res, scann_path / "centers.npy", index_.centers());

  // cluster assignments
  save_labels(res, scann_path / "datapoint_to_token.npy", index_.labels(), index_.soar_labels());

  // kmeans two-level centers
  if (index_.coarse_centers().extent(0) > 0) {
    serialize_matrix(res, scann_path / "coarse_centers.npy", index_.coarse_centers());

    save_labels(
      res, scann_path / "coarse_labels.npy", index_.coarse_labels(), index_.coarse_soar_labels());
  }
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
