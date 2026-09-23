/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/ivf_rabitq/gpu_index/ivf_gpu.cuh"
#include "../ann_ivf_rabitq.cuh"

#include <fstream>

namespace cuvs::neighbors::ivf_rabitq {

// Check the historical disk layout independently of the save/load round trip:
// matching mistakes in both directions must not make this test pass.
TEST(IvfRabitqSerialization, PreservesLegacyVectorMajorCodes)
{
  raft::resources handle;
  auto stream              = raft::resource::get_cuda_stream(handle);
  constexpr int64_t n_rows = 256;
  constexpr int64_t dim    = 128;
  auto dataset             = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
  raft::random::RngState rng(1234ULL);
  raft::random::uniform(handle, rng, dataset.data_handle(), n_rows * dim, 0.1f, 2.0f);
  index_params params;
  params.n_lists     = 4;
  auto built         = build(handle, params, raft::make_const_mdspan(dataset.view()));
  auto& impl         = built.rabitq_index();
  const size_t words = impl.quantizer().short_code_length();
  ASSERT_GT(words, 1);
  std::vector<detail::IVFGPU::GPUClusterMeta> clusters(params.n_lists);
  raft::copy(clusters.data(), impl.get_cluster_meta().data_handle(), clusters.size(), stream);
  raft::resource::sync_stream(handle);

  // Distinct words expose transposition mistakes even when quantized inputs happen
  // to have identical bit patterns. Include every cluster's independent offset.
  std::vector<uint32_t> memory_codes(n_rows * words);
  std::vector<uint32_t> legacy_codes(n_rows * words);
  bool has_multiple_rows = false;
  for (auto const& cluster : clusters) {
    has_multiple_rows |= cluster.num > 1;
    const size_t offset = cluster.start_index * words;
    for (size_t row = 0; row < cluster.num; ++row) {
      for (size_t word = 0; word < words; ++word) {
        const uint32_t value = static_cast<uint32_t>(offset + row * words + word + 1);
        legacy_codes[offset + row * words + word]       = value;
        memory_codes[offset + word * cluster.num + row] = value;
      }
    }
  }
  ASSERT_TRUE(has_multiple_rows);
  raft::copy(impl.get_short_data_device(), memory_codes.data(), memory_codes.size(), stream);
  raft::resource::sync_stream(handle);
  tmp_index_file file;
  serialize(handle, file.filename, built);

  // Short codes precede factors, long codes, extra factors, and IDs at EOF.
  const size_t trailing_bytes = n_rows * (3 * sizeof(float) + impl.quantizer().long_code_length() +
                                          2 * sizeof(float) + sizeof(uint32_t));
  std::ifstream input(file.filename, std::ios::binary);
  input.seekg(-static_cast<std::streamoff>(trailing_bytes + legacy_codes.size() * sizeof(uint32_t)),
              std::ios::end);
  std::vector<uint32_t> saved_codes(legacy_codes.size());
  input.read(reinterpret_cast<char*>(saved_codes.data()), saved_codes.size() * sizeof(uint32_t));
  ASSERT_TRUE(input.good());
  EXPECT_EQ(saved_codes, legacy_codes);
  input.close();

  // Supply known legacy payload bytes independently of the writer under test.
  std::fstream fixture(file.filename, std::ios::binary | std::ios::in | std::ios::out);
  fixture.seekp(
    -static_cast<std::streamoff>(trailing_bytes + legacy_codes.size() * sizeof(uint32_t)),
    std::ios::end);
  fixture.write(reinterpret_cast<const char*>(legacy_codes.data()),
                legacy_codes.size() * sizeof(uint32_t));
  ASSERT_TRUE(fixture.good());
  fixture.close();

  index<int64_t> loaded(handle);
  deserialize(handle, file.filename, &loaded);
  std::vector<uint32_t> loaded_codes(memory_codes.size());
  raft::copy(loaded_codes.data(),
             loaded.rabitq_index().get_short_data_device(),
             loaded_codes.size(),
             stream);
  raft::resource::sync_stream(handle);
  EXPECT_EQ(loaded_codes, memory_codes);

  // Re-saving must also preserve the historical format rather than transposing twice.
  tmp_index_file resaved;
  serialize(handle, resaved.filename, loaded);
  std::ifstream resaved_input(resaved.filename, std::ios::binary);
  resaved_input.seekg(
    -static_cast<std::streamoff>(trailing_bytes + legacy_codes.size() * sizeof(uint32_t)),
    std::ios::end);
  resaved_input.read(reinterpret_cast<char*>(saved_codes.data()),
                     saved_codes.size() * sizeof(uint32_t));
  ASSERT_TRUE(resaved_input.good());
  EXPECT_EQ(saved_codes, legacy_codes);
}

using f32_f32_i64 = ivf_rabitq_test<float, float, int64_t>;

TEST_BUILD_SEARCH(f32_f32_i64)
TEST_BUILD_HOST_INPUT_SEARCH(f32_f32_i64)
TEST_BUILD_SERIALIZE_SEARCH(f32_f32_i64)
TEST_BUILD_HOST_INPUT_SERIALIZE_SEARCH(f32_f32_i64)
TEST_BUILD_FORCED_STREAMING(f32_f32_i64)
INSTANTIATE(f32_f32_i64,
            defaults() + small_dims() + big_dims() + var_n_probes() + var_k() + var_bits_per_dim() +
              var_search_mode() + var_search_mode_1_bit());

}  // namespace cuvs::neighbors::ivf_rabitq
