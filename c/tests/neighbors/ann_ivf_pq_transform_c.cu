/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/cluster/assignment.h>
#include <cuvs/neighbors/ivf_pq.h>

#include <gtest/gtest.h>

#include <vector>

namespace {

DLManagedTensor make_tensor(void *data, DLDataTypeCode code, uint8_t bits,
                            int64_t *shape, int ndim) {
  DLManagedTensor t{};
  t.dl_tensor.data = data;
  t.dl_tensor.device.device_type = kDLCUDA;
  t.dl_tensor.ndim = ndim;
  t.dl_tensor.dtype.code = code;
  t.dl_tensor.dtype.bits = bits;
  t.dl_tensor.dtype.lanes = 1;
  t.dl_tensor.shape = shape;
  t.dl_tensor.strides = nullptr;
  return t;
}

class IvfPqTransformAssignmentC : public ::testing::Test {
protected:
  static constexpr int64_t kTrain = 20000;
  static constexpr int64_t kRows = 3000;
  static constexpr int64_t kDim = 64;
  static constexpr uint32_t kLists = 256;

  void SetUp() override {
    ASSERT_EQ(cuvsResourcesCreate(&res_), CUVS_SUCCESS);
    stream_ = raft::resource::get_cuda_stream(handle_).get();
    train_ = std::make_unique<raft::device_matrix<float, int64_t>>(
        raft::make_device_matrix<float, int64_t>(handle_, kTrain, kDim));
    data_ = std::make_unique<raft::device_matrix<float, int64_t>>(
        raft::make_device_matrix<float, int64_t>(handle_, kRows, kDim));
    raft::random::RngState rng(7ULL);
    raft::random::normal(handle_, rng, train_->data_handle(), kTrain * kDim,
                         0.f, 1.f);
    raft::random::normal(handle_, rng, data_->data_handle(), kRows * kDim, 0.f,
                         1.f);

    int64_t train_shape[2] = {kTrain, kDim};
    auto train_t =
        make_tensor(train_->data_handle(), kDLFloat, 32, train_shape, 2);
    ASSERT_EQ(cuvsIvfPqIndexCreate(&index_), CUVS_SUCCESS);
    cuvsIvfPqIndexParams_t ip;
    ASSERT_EQ(cuvsIvfPqIndexParamsCreate(&ip), CUVS_SUCCESS);
    ip->n_lists = kLists;
    ip->pq_dim = 32;
    ip->kmeans_n_iters = 5;
    ip->add_data_on_build = false;
    ASSERT_EQ(cuvsIvfPqBuild(res_, ip, &train_t, index_), CUVS_SUCCESS);
    cuvsIvfPqIndexParamsDestroy(ip);
    codes_cols_ = 32; // pq_dim * 8 bits / 8
  }

  void TearDown() override {
    cuvsIvfPqIndexDestroy(index_);
    cuvsResourcesDestroy(res_);
  }

  // Transform the data; `assignment` NULL uses cuvsIvfPqTransform.
  cuvsError_t run(cuvsAssignmentParams_t assignment,
                  std::vector<uint32_t> &labels_out,
                  std::vector<uint8_t> *codes_out = nullptr) {
    auto labels = raft::make_device_vector<uint32_t, int64_t>(handle_, kRows);
    auto codes =
        raft::make_device_matrix<uint8_t, int64_t>(handle_, kRows, codes_cols_);
    int64_t data_shape[2] = {kRows, kDim};
    int64_t labels_shape[1] = {kRows};
    int64_t codes_shape[2] = {kRows, codes_cols_};
    auto data_t =
        make_tensor(data_->data_handle(), kDLFloat, 32, data_shape, 2);
    auto labels_t =
        make_tensor(labels.data_handle(), kDLUInt, 32, labels_shape, 1);
    auto codes_t = make_tensor(codes.data_handle(), kDLUInt, 8, codes_shape, 2);
    cuvsError_t err =
        assignment == nullptr
            ? cuvsIvfPqTransform(res_, index_, &data_t, &labels_t, &codes_t)
            : cuvsIvfPqTransformWithAssignment(res_, index_, &data_t, &labels_t,
                                               &codes_t, assignment);
    if (err != CUVS_SUCCESS) {
      return err;
    }
    labels_out.resize(kRows);
    raft::update_host(labels_out.data(), labels.data_handle(), kRows, stream_);
    if (codes_out != nullptr) {
      codes_out->resize(kRows * codes_cols_);
      raft::update_host(codes_out->data(), codes.data_handle(),
                        codes_out->size(), stream_);
    }
    raft::resource::sync_stream(handle_);
    return CUVS_SUCCESS;
  }

  raft::handle_t handle_;
  cudaStream_t stream_;
  cuvsResources_t res_;
  cuvsIvfPqIndex_t index_;
  std::unique_ptr<raft::device_matrix<float, int64_t>> train_, data_;
  int64_t codes_cols_ = 0;
};

TEST_F(IvfPqTransformAssignmentC, NullIsDefault) {
  std::vector<uint32_t> a, b;
  ASSERT_EQ(run(nullptr, a), CUVS_SUCCESS);
  // The IVF backend with every list probed is exhaustive: nearly all labels
  // agree.
  cuvsAssignmentParams_t ap;
  ASSERT_EQ(cuvsAssignmentParamsCreate(&ap), CUVS_SUCCESS);
  EXPECT_EQ(ap->kind, CUVS_ASSIGNMENT_IVF);
  ap->ivf.n_probes = kLists;
  ASSERT_EQ(run(ap, b), CUVS_SUCCESS);
  int64_t same = 0;
  for (int64_t i = 0; i < kRows; i++) {
    same += a[i] == b[i];
  }
  EXPECT_GE(same, kRows * 99 / 100);
  cuvsAssignmentParamsDestroy(ap);
}

TEST_F(IvfPqTransformAssignmentC, PrebuiltIndexIsReused) {
  cuvsAssignmentParams_t build_params;
  ASSERT_EQ(cuvsAssignmentParamsCreate(&build_params), CUVS_SUCCESS);
  EXPECT_EQ(build_params->ivf.kmeans_n_iters, 10u);
  cuvsAssignmentIndex_t aidx;
  ASSERT_EQ(cuvsAssignmentIndexCreate(&aidx), CUVS_SUCCESS);
  ASSERT_EQ(cuvsIvfPqBuildAssignmentIndex(res_, index_, build_params, aidx),
            CUVS_SUCCESS);
  EXPECT_EQ(aidx->kind, CUVS_ASSIGNMENT_IVF);

  cuvsAssignmentParams_t per_call, prebuilt;
  ASSERT_EQ(cuvsAssignmentParamsCreate(&per_call), CUVS_SUCCESS);
  ASSERT_EQ(cuvsAssignmentParamsCreate(&prebuilt), CUVS_SUCCESS);
  per_call->ivf.n_probes = 4;
  prebuilt->ivf.n_probes = 4;
  prebuilt->index = aidx;

  std::vector<uint32_t> a, b, c;
  std::vector<uint8_t> codes_b, codes_c;
  ASSERT_EQ(run(per_call, a), CUVS_SUCCESS);
  ASSERT_EQ(run(prebuilt, b, &codes_b), CUVS_SUCCESS);
  ASSERT_EQ(run(prebuilt, c, &codes_c), CUVS_SUCCESS);
  EXPECT_EQ(a, b);
  EXPECT_EQ(b, c);
  EXPECT_EQ(codes_b, codes_c);

  cuvsAssignmentParamsDestroy(per_call);
  cuvsAssignmentParamsDestroy(prebuilt);
  cuvsAssignmentParamsDestroy(build_params);
  cuvsAssignmentIndexDestroy(aidx);
}

TEST_F(IvfPqTransformAssignmentC, UnbuiltOrMismatchedIndexFails) {
  cuvsAssignmentParams_t ap;
  ASSERT_EQ(cuvsAssignmentParamsCreate(&ap), CUVS_SUCCESS);
  cuvsAssignmentIndex_t aidx;
  ASSERT_EQ(cuvsAssignmentIndexCreate(&aidx), CUVS_SUCCESS);
  ap->index = aidx; // created but never built
  std::vector<uint32_t> labels;
  EXPECT_NE(run(ap, labels), CUVS_SUCCESS);

  // Built from the wrong number of centroids.
  int64_t shape[2] = {kLists / 2, kDim};
  auto centroids = make_tensor(train_->data_handle(), kDLFloat, 32, shape, 2);
  ASSERT_EQ(cuvsAssignmentIndexBuild(res_, ap, &centroids, L2Expanded, aidx),
            CUVS_SUCCESS);
  EXPECT_NE(run(ap, labels), CUVS_SUCCESS);

  cuvsAssignmentParamsDestroy(ap);
  cuvsAssignmentIndexDestroy(aidx);
}

} // namespace
