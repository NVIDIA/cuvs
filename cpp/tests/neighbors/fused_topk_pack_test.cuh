// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Included inside the shared native test namespace.

template <typename T>
void check_packed_coordinates(size_t rows, int dimensions)
{
  constexpr int alignment = 16 / sizeof(T);
  constexpr size_t guard  = 16;
  const int padded        = (dimensions + alignment - 1) / alignment * alignment;
  SCOPED_TRACE(::testing::Message()
               << "rows=" << rows << " D=" << dimensions << " element bytes=" << sizeof(T));
  // Stored bits include signed zeros, subnormals and extreme finite values.
  // Packing must not perform arithmetic on coordinates, including half values.
  const uint32_t fp32[] = {
    0, 0x80000000, 0x3f800001, 0xbf800001, 0x00800001, 0x00000001, 0x7f7fffff};
  const uint16_t fp16[] = {0, 0x8000, 0x3c01, 0xbc01, 0x0401, 1, 0x7bff};
  std::vector<T> source(rows * dimensions), source_after(source.size());
  for (size_t i = 0; i < source.size(); ++i) {
    if constexpr (sizeof(T) == 4) {
      std::memcpy(static_cast<void*>(&source[i]), &fp32[i % 7], sizeof(T));
    } else {
      std::memcpy(static_cast<void*>(&source[i]), &fp16[i % 7], sizeof(T));
    }
  }
  std::vector<T> expected(rows * padded + 2 * guard), result(expected.size());
  std::memset(static_cast<void*>(expected.data()), 0xa5, expected.size() * sizeof(T));
  std::memset(static_cast<void*>(expected.data() + guard), 0, rows * padded * sizeof(T));
  for (size_t row = 0; row < rows; ++row) {
    std::memcpy(expected.data() + guard + row * padded,
                source.data() + row * dimensions,
                dimensions * sizeof(T));
  }
  rmm::cuda_stream stream;
  rmm::device_uvector<T> input(source.size(), stream.view());
  rmm::device_uvector<T> output(result.size(), stream.view());
  RAFT_CUDA_TRY(cudaMemcpyAsync(input.data(),
                                source.data(),
                                source.size() * sizeof(T),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  RAFT_CUDA_TRY(cudaMemsetAsync(output.data(), 0xa5, output.size() * sizeof(T), stream.value()));
  fused_topk_pack_rows(
    output.data() + guard, input.data(), rows, dimensions, padded, stream.value());
  RAFT_CUDA_TRY(cudaMemcpyAsync(result.data(),
                                output.data(),
                                result.size() * sizeof(T),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  RAFT_CUDA_TRY(cudaMemcpyAsync(source_after.data(),
                                input.data(),
                                source.size() * sizeof(T),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  ASSERT_EQ(std::memcmp(result.data(), expected.data(), result.size() * sizeof(T)), 0)
    << "coordinate bits, zero padding or output redzone changed";
  ASSERT_EQ(std::memcmp(source.data(), source_after.data(), source.size() * sizeof(T)), 0)
    << "input coordinates changed";
}

template <typename T>
void check_packing_widths_and_grid_stride()
{
  for (int d : {7,   8,   9,   15,  16,  17,  31,  32,   33,   63,   64,   65,   127, 128,
                129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049}) {
    for (size_t rows : {1, 7, 8, 9, 259}) {
      check_packed_coordinates<T>(rows, d);
    }
  }
  // Cross the grid cap in each row-width class.
  check_packed_coordinates<T>(8193, 65);
  check_packed_coordinates<T>(32769, 17);
}

TEST(FusedTopkPackingTest, FloatCoordinatesPaddingGuardsAndNondefaultStream)
{
  check_packing_widths_and_grid_stride<float>();
}

TEST(FusedTopkPackingTest, HalfCoordinatesPaddingGuardsAndNondefaultStream)
{
  check_packing_widths_and_grid_stride<half>();
}
