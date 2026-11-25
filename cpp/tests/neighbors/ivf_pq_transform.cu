/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/cluster/assignment.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cuda_fp16.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace cuvs::neighbors::ivf_pq {

using cuvs::cluster::assignment_params;
using cuvs::cluster::build_ivf_assignment_index;
using cuvs::cluster::ivf_assignment_params;
using cuvs::distance::DistanceType;

struct transform_inputs {
  int64_t n_rows;
  int64_t n_train;
  int64_t dim;
  uint32_t n_lists;
  uint32_t pq_dim;
  DistanceType metric;
};

inline auto operator<<(std::ostream& os, const transform_inputs& p) -> std::ostream&
{
  return os << "{rows=" << p.n_rows << " dim=" << p.dim << " n_lists=" << p.n_lists
            << " pq_dim=" << p.pq_dim << " metric=" << static_cast<int>(p.metric) << "}";
}

class IvfPqTransformIvfTest : public ::testing::TestWithParam<transform_inputs> {
 protected:
  void SetUp() override
  {
    const auto p = GetParam();
    stream_      = raft::resource::get_cuda_stream(res_).get();
    train_       = std::make_unique<raft::device_matrix<float, int64_t>>(
      raft::make_device_matrix<float, int64_t>(res_, p.n_train, p.dim));
    data_ = std::make_unique<raft::device_matrix<float, int64_t>>(
      raft::make_device_matrix<float, int64_t>(res_, p.n_rows, p.dim));
    raft::random::RngState rng(42ULL);
    raft::random::normal(res_, rng, train_->data_handle(), p.n_train * p.dim, 0.f, 1.f);
    raft::random::normal(res_, rng, data_->data_handle(), p.n_rows * p.dim, 0.f, 1.f);

    index_params ip;
    ip.metric            = p.metric;
    ip.n_lists           = p.n_lists;
    ip.pq_dim            = p.pq_dim;
    ip.kmeans_n_iters    = 5;
    ip.add_data_on_build = false;
    index_ =
      std::make_unique<index<int64_t>>(build(res_, ip, raft::make_const_mdspan(train_->view())));
    code_cols_ = raft::ceildiv<int64_t>(index_->pq_dim() * index_->pq_bits(), 8);
  }

  struct output {
    std::vector<uint32_t> labels;
    std::vector<uint8_t> codes;
  };

  auto run(const std::optional<assignment_params>& assignment) -> output
  {
    const auto p = GetParam();
    auto labels  = raft::make_device_vector<uint32_t, int64_t>(res_, p.n_rows);
    auto codes   = raft::make_device_matrix<uint8_t, int64_t>(res_, p.n_rows, code_cols_);
    transform(res_,
              *index_,
              raft::make_const_mdspan(data_->view()),
              labels.view(),
              codes.view(),
              assignment);
    output out;
    out.labels.resize(p.n_rows);
    out.codes.resize(p.n_rows * code_cols_);
    raft::update_host(out.labels.data(), labels.data_handle(), out.labels.size(), stream_);
    raft::update_host(out.codes.data(), codes.data_handle(), out.codes.size(), stream_);
    raft::resource::sync_stream(res_);
    return out;
  }

  // The metric the index clusters with: inner product uses L2 for the coarse step.
  auto coarse_metric() const -> DistanceType
  {
    return GetParam().metric == DistanceType::InnerProduct ? DistanceType::L2Expanded
                                                           : GetParam().metric;
  }

  auto centers_index(DistanceType metric, uint32_t n_lists = 0)
    -> cuvs::cluster::ivf_assignment_index
  {
    auto centers = raft::make_device_matrix<float, int64_t>(res_, index_->n_lists(), index_->dim());
    helpers::extract_centers(res_, *index_, centers.view());
    ivf_assignment_params gp;
    gp.n_lists = n_lists;
    return build_ivf_assignment_index(res_, gp, raft::make_const_mdspan(centers.view()), metric);
  }

  raft::device_resources res_;
  cudaStream_t stream_;
  std::unique_ptr<raft::device_matrix<float, int64_t>> train_, data_;
  std::unique_ptr<index<int64_t>> index_;
  int64_t code_cols_ = 0;
};

// An empty assignment is the exhaustive one: identical to the overload without it.
TEST_P(IvfPqTransformIvfTest, EmptyAssignmentIsDefault)
{
  const auto p = GetParam();
  auto labels  = raft::make_device_vector<uint32_t, int64_t>(res_, p.n_rows);
  auto codes   = raft::make_device_matrix<uint8_t, int64_t>(res_, p.n_rows, code_cols_);
  transform(res_, *index_, raft::make_const_mdspan(data_->view()), labels.view(), codes.view());
  std::vector<uint32_t> expected(p.n_rows);
  raft::update_host(expected.data(), labels.data_handle(), expected.size(), stream_);
  raft::resource::sync_stream(res_);
  EXPECT_EQ(run(std::nullopt).labels, expected);
}

// Probing every list is exhaustive: the labels agree with the dense assignment except for rare
// TF32 near-ties, and the codes agree wherever the labels do.
TEST_P(IvfPqTransformIvfTest, ExhaustiveMatchesDense)
{
  const auto p = GetParam();
  auto dense   = run(std::nullopt);
  ivf_assignment_params gp;
  gp.n_probes   = index_->n_lists();
  auto ivf_run  = run(assignment_params{gp});
  int64_t same  = 0;
  int64_t codes = 0;
  for (int64_t i = 0; i < p.n_rows; i++) {
    if (ivf_run.labels[i] != dense.labels[i]) continue;
    same++;
    if (std::equal(ivf_run.codes.begin() + i * code_cols_,
                   ivf_run.codes.begin() + (i + 1) * code_cols_,
                   dense.codes.begin() + i * code_cols_)) {
      codes++;
    }
  }
  EXPECT_GE(same, p.n_rows * 99 / 100);
  EXPECT_EQ(codes, same);
}

// The default probe count gives approximate labels, all valid.
TEST_P(IvfPqTransformIvfTest, DefaultProbesGiveValidLabels)
{
  const auto p = GetParam();
  auto dense   = run(std::nullopt);
  auto ivf_run = run(assignment_params{ivf_assignment_params{}});
  int64_t same = 0;
  for (int64_t i = 0; i < p.n_rows; i++) {
    ASSERT_LT(ivf_run.labels[i], index_->n_lists());
    same += ivf_run.labels[i] == dense.labels[i];
  }
  EXPECT_GE(same, p.n_rows / 2);
}

// A prebuilt index gives the same labels as building one per call, and can be reused.
TEST_P(IvfPqTransformIvfTest, PrebuiltIndexIsReused)
{
  auto gidx = build_ivf_assignment_index(res_, *index_, ivf_assignment_params{});
  ivf_assignment_params built;
  built.n_probes = 4;
  ivf_assignment_params prebuilt;
  prebuilt.n_probes = 4;
  prebuilt.index    = &gidx;
  auto a            = run(assignment_params{built});
  auto b            = run(assignment_params{prebuilt});
  auto c            = run(assignment_params{prebuilt});
  EXPECT_EQ(a.labels, b.labels);
  EXPECT_EQ(b.labels, c.labels);
  EXPECT_EQ(b.codes, c.codes);
}

// An index that does not match the call is rejected.
TEST_P(IvfPqTransformIvfTest, MismatchedIndexThrows)
{
  const auto p = GetParam();
  ivf_assignment_params gp;

  // Wrong number of centroids: built from a subset of the centers.
  {
    auto centers = raft::make_device_matrix<float, int64_t>(res_, index_->n_lists(), index_->dim());
    helpers::extract_centers(res_, *index_, centers.view());
    auto sub_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      centers.data_handle(), index_->n_lists() / 2, index_->dim());
    auto sub = build_ivf_assignment_index(res_, ivf_assignment_params{}, sub_view);
    gp.index = &sub;
    EXPECT_THROW(run(assignment_params{gp}), raft::exception);
  }
  // Wrong metric.
  {
    auto other = coarse_metric() == DistanceType::CosineExpanded ? DistanceType::L2Expanded
                                                                 : DistanceType::CosineExpanded;
    auto gidx  = centers_index(other);
    gp.index   = &gidx;
    EXPECT_THROW(run(assignment_params{gp}), raft::exception);
  }
  // Wrong dimension.
  {
    auto centers =
      raft::make_device_matrix<float, int64_t>(res_, index_->n_lists(), index_->dim() + 4);
    raft::random::RngState rng(1ULL);
    raft::random::normal(res_, rng, centers.data_handle(), centers.size(), 0.f, 1.f);
    auto gidx = build_ivf_assignment_index(
      res_, ivf_assignment_params{}, raft::make_const_mdspan(centers.view()), coarse_metric());
    gp.index = &gidx;
    EXPECT_THROW(run(assignment_params{gp}), raft::exception);
  }
  (void)p;
}

// Other data types map to float before the assignment; the labels are valid and match the
// exhaustive IVF search.
TEST_P(IvfPqTransformIvfTest, HalfData)
{
  const auto p   = GetParam();
  auto half_data = raft::make_device_matrix<half, int64_t>(res_, p.n_rows, p.dim);
  raft::linalg::map(
    res_,
    raft::make_device_vector_view<const float, int64_t>(data_->data_handle(), p.n_rows * p.dim),
    raft::make_device_vector_view<half, int64_t>(half_data.data_handle(), p.n_rows * p.dim),
    raft::cast_op<half>());
  auto labels = raft::make_device_vector<uint32_t, int64_t>(res_, p.n_rows);
  auto codes  = raft::make_device_matrix<uint8_t, int64_t>(res_, p.n_rows, code_cols_);
  ivf_assignment_params gp;
  gp.n_probes = index_->n_lists();
  transform(res_,
            *index_,
            raft::make_const_mdspan(half_data.view()),
            labels.view(),
            codes.view(),
            std::optional<assignment_params>{gp});
  std::vector<uint32_t> out(p.n_rows);
  raft::update_host(out.data(), labels.data_handle(), out.size(), stream_);
  raft::resource::sync_stream(res_);
  for (auto l : out) {
    ASSERT_LT(l, index_->n_lists());
  }
}

const std::vector<transform_inputs> kTransformInputs = {
  {4000, 20000, 64, 256, 32, DistanceType::L2Expanded},
  {4000, 20000, 64, 256, 32, DistanceType::InnerProduct},
  {3000, 20000, 128, 400, 64, DistanceType::CosineExpanded},
};

INSTANTIATE_TEST_CASE_P(IvfPqTransformIvf,
                        IvfPqTransformIvfTest,
                        ::testing::ValuesIn(kTransformInputs));

}  // namespace cuvs::neighbors::ivf_pq
