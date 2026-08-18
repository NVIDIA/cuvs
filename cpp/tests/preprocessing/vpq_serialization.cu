/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../neighbors/vpq_utils.cuh"
#include "../test_utils.cuh"

#include <cuvs/preprocessing/quantize/pq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace cuvs::preprocessing::quantize::pq {

using vpq_dataset_t = cuvs::neighbors::device_vpq_dataset<half, int64_t>;

struct VpqSerializationInputs {
  int64_t n_rows;
  int64_t dim;
  uint32_t pq_bits;
  uint32_t pq_dim;
  uint32_t vq_n_centers;  // 0 lets the heuristic choose
  uint64_t seed;
};

std::ostream& operator<<(std::ostream& os, const VpqSerializationInputs& in)
{
  return os << "n_rows:" << in.n_rows << " dim:" << in.dim << " pq_bits:" << in.pq_bits
            << " pq_dim:" << in.pq_dim << " vq_n_centers:" << in.vq_n_centers
            << " seed:" << in.seed;
}

template <typename T, typename IdxT>
auto to_host(const raft::resources& res, raft::device_matrix_view<const T, IdxT> m)
  -> std::vector<T>
{
  std::vector<T> host(static_cast<size_t>(m.extent(0)) * static_cast<size_t>(m.extent(1)));
  raft::copy(host.data(), m.data_handle(), host.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return host;
}

/** Bitwise, not approximate: serialization is expected not to perturb a single bit. */
template <typename T, typename IdxT>
void expect_same_bits(const raft::resources& res,
                      raft::device_matrix_view<const T, IdxT> expected,
                      raft::device_matrix_view<const T, IdxT> actual,
                      const char* what)
{
  ASSERT_EQ(expected.extent(0), actual.extent(0)) << what;
  ASSERT_EQ(expected.extent(1), actual.extent(1)) << what;
  const auto lhs = to_host(res, expected);
  const auto rhs = to_host(res, actual);
  EXPECT_EQ(0, std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T))) << what;
}

class VpqSerializationTest : public ::testing::TestWithParam<VpqSerializationInputs> {
 public:
  VpqSerializationTest()
    : params_(::testing::TestWithParam<VpqSerializationInputs>::GetParam()),
      dataset_(raft::make_device_matrix<float, int64_t>(res_, params_.n_rows, params_.dim))
  {
  }

 protected:
  void SetUp() override
  {
    auto labels = raft::make_device_vector<int64_t, int64_t>(res_, params_.n_rows);
    raft::random::make_blobs<float, int64_t, raft::row_major>(res_,
                                                             dataset_.view(),
                                                             labels.view(),
                                                             5,             // clusters
                                                             std::nullopt,  // random centers
                                                             std::nullopt,  // scalar std
                                                             1.0F,          // cluster std
                                                             true,          // shuffle
                                                             -10.0F,        // center box min
                                                             10.0F,         // center box max
                                                             params_.seed);
    raft::resource::sync_stream(res_);
  }

  auto compress() -> vpq_dataset_t
  {
    cuvs::neighbors::vpq_params vpq;
    vpq.pq_bits      = params_.pq_bits;
    vpq.pq_dim       = params_.pq_dim;
    vpq.vq_n_centers = params_.vq_n_centers;
    // The codebooks only have to be well defined here, not good, so keep training short.
    vpq.kmeans_n_iters = 5;
    return make_vpq_dataset(res_, vpq, raft::make_const_mdspan(dataset_.view()));
  }

  void expect_equivalent(const vpq_dataset_t& expected, const vpq_dataset_t& actual)
  {
    ASSERT_EQ(expected.n_rows(), actual.n_rows());
    ASSERT_EQ(expected.dim(), actual.dim());
    ASSERT_EQ(expected.vq_n_centers(), actual.vq_n_centers());
    ASSERT_EQ(expected.pq_n_centers(), actual.pq_n_centers());
    ASSERT_EQ(expected.pq_len(), actual.pq_len());
    ASSERT_EQ(expected.encoded_row_length(), actual.encoded_row_length());
    ASSERT_EQ(expected.pq_bits(), actual.pq_bits());
    ASSERT_EQ(expected.pq_dim(), actual.pq_dim());

    expect_same_bits(res_,
                     raft::make_const_mdspan(expected.vq_code_book.view()),
                     raft::make_const_mdspan(actual.vq_code_book.view()),
                     "vq_code_book");
    expect_same_bits(res_,
                     raft::make_const_mdspan(expected.pq_code_book.view()),
                     raft::make_const_mdspan(actual.pq_code_book.view()),
                     "pq_code_book");
    expect_same_bits(res_,
                     raft::make_const_mdspan(expected.data.view()),
                     raft::make_const_mdspan(actual.data.view()),
                     "encoded rows");
  }

  /**
   * Decodes both datasets and compares the reconstructions, which checks that a kernel can consume
   * the deserialized extents and strides rather than only that the numbers match.
   */
  void expect_same_decoded(const vpq_dataset_t& expected, const vpq_dataset_t& actual)
  {
    if (expected.pq_bits() != 8) { return; }  // decode_vpq_dataset implements pq_bits == 8 only
    auto stream = raft::resource::get_cuda_stream(res_);
    auto lhs = raft::make_device_matrix<float, int64_t>(res_, expected.n_rows(), expected.dim());
    auto rhs = raft::make_device_matrix<float, int64_t>(res_, actual.n_rows(), actual.dim());
    cuvs::neighbors::decode_vpq_dataset(lhs.view(), expected, stream);
    cuvs::neighbors::decode_vpq_dataset(rhs.view(), actual, stream);
    raft::resource::sync_stream(res_);
    expect_same_bits<float, int64_t>(res_,
                                     raft::make_const_mdspan(lhs.view()),
                                     raft::make_const_mdspan(rhs.view()),
                                     "decoded rows");
  }

  raft::resources res_;
  VpqSerializationInputs params_;
  raft::device_matrix<float, int64_t> dataset_;
};

TEST_P(VpqSerializationTest, RoundTrip)
{
  auto original = compress();

  {
    SCOPED_TRACE("through a stream");
    std::stringstream stream;
    serialize(res_, stream, original);
    std::unique_ptr<vpq_dataset_t> restored;
    deserialize(res_, stream, &restored);
    ASSERT_NE(restored, nullptr);
    expect_equivalent(original, *restored);
    expect_same_decoded(original, *restored);
  }

  {
    SCOPED_TRACE("through a file");
    const std::string path = "cuvs_vpq_serialization_test.bin";
    serialize(res_, path, original);
    std::unique_ptr<vpq_dataset_t> restored;
    deserialize(res_, path, &restored);
    std::remove(path.c_str());
    ASSERT_NE(restored, nullptr);
    expect_equivalent(original, *restored);
  }
}

// Named for this suite rather than `inputs`: product_quantization.cu declares a variable of that
// name in this same namespace, which would collide under a unity build.
const std::vector<VpqSerializationInputs> vpq_serialization_inputs = {
  // pq_len = dim / pq_dim of 2, 4 and 8: the three values CAGRA-Q accepts.
  {1000, 64, 8, 32, 0, 42ULL},
  {1000, 128, 8, 32, 0, 42ULL},
  {1000, 256, 8, 32, 0, 42ULL},
  // An explicit VQ codebook size rather than the heuristic.
  {2000, 128, 8, 64, 64, 42ULL},
  // pq_bits below 8 packs several codes per byte, so encoded_row_length stops being pq_dim.
  {500, 96, 6, 24, 0, 42ULL},
  {500, 32, 4, 16, 0, 42ULL},
};

INSTANTIATE_TEST_CASE_P(VpqSerializationTests,
                        VpqSerializationTest,
                        ::testing::ValuesIn(vpq_serialization_inputs));

/** Writes the preamble that `serialize` emits, so only the field under test differs. */
static void write_preamble(const raft::resources& res, std::ostream& os, int version)
{
  std::string dtype_string = raft::numpy_serializer::get_numpy_dtype<half>().to_string();
  dtype_string.resize(4);
  os << dtype_string;
  raft::serialize_scalar(res, os, version);
}

TEST(VpqSerialization, RejectsEmptyStream)
{
  raft::resources res;
  std::stringstream stream;
  std::unique_ptr<vpq_dataset_t> restored;
  EXPECT_THROW(deserialize(res, stream, &restored), raft::exception);
}

TEST(VpqSerialization, RejectsForeignDtypePrefix)
{
  raft::resources res;
  std::stringstream stream;
  std::string dtype_string = raft::numpy_serializer::get_numpy_dtype<float>().to_string();
  dtype_string.resize(4);
  stream << dtype_string;
  raft::serialize_scalar(res, stream, vpq_serialization_version);

  std::unique_ptr<vpq_dataset_t> restored;
  EXPECT_THROW(deserialize(res, stream, &restored), raft::exception);
}

TEST(VpqSerialization, RejectsFutureVersion)
{
  raft::resources res;
  std::stringstream stream;
  write_preamble(res, stream, vpq_serialization_version + 1);

  std::unique_ptr<vpq_dataset_t> restored;
  EXPECT_THROW(deserialize(res, stream, &restored), raft::exception);
}

TEST(VpqSerialization, RejectsTruncatedPayload)
{
  raft::resources res;
  std::stringstream stream;
  write_preamble(res, stream, vpq_serialization_version);
  // A correct preamble followed by nothing: the payload reader must fail rather than return a
  // dataset built from whatever the scalars happened to deserialize to.
  std::unique_ptr<vpq_dataset_t> restored;
  EXPECT_THROW(deserialize(res, stream, &restored), raft::exception);
}

TEST(VpqSerialization, RejectsNullOutParameter)
{
  raft::resources res;
  std::stringstream stream;
  write_preamble(res, stream, vpq_serialization_version);
  EXPECT_THROW(deserialize(res, stream, nullptr), raft::exception);
}

}  // namespace cuvs::preprocessing::quantize::pq
