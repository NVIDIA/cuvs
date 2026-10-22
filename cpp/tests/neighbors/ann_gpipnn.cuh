/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "cagra_padded_build_helpers.cuh"
#include "naive_knn.cuh"

#include "../../src/neighbors/detail/gpipnn/gpipnn_build.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/gpipnn.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <vector>

namespace cuvs::neighbors::experimental::gpipnn {

struct AnnGpipnnInputs {
  int n_rows;
  int dim;
  int graph_degree;
  int max_candidates;
  int min_leaf_size;
  int max_leaf_size;
  int n_queries;
  int k;
  int itopk_size;
  double min_recall;
  /**
   * Multiplies the generated data, so that a case can be placed at a chosen dynamic range.
   *
   * Squared distances grow as the square of this, which is what exercises the scoring path's
   * finite score range: at 1.0 the generated data sits far below it, while real datasets like
   * SIFT do not.
   */
  double data_scale;
};

inline auto operator<<(std::ostream& os, const AnnGpipnnInputs& p) -> std::ostream&
{
  os << "n_rows=" << p.n_rows << ", dim=" << p.dim << ", degree=" << p.graph_degree
     << ", l_max=" << p.max_candidates << ", leaf=[" << p.min_leaf_size << "," << p.max_leaf_size
     << "], itopk=" << p.itopk_size << ", data_scale=" << p.data_scale;
  return os;
}

/** Builds the fp16 scoring operand and its fp32 row norms, exactly as the builder does. */
struct scoring_operand {
  rmm::device_uvector<half> data;
  rmm::device_uvector<float> norms;
  int64_t n_rows;
  int64_t dim;

  scoring_operand(raft::resources const& res, const float* dataset, int64_t n, int64_t d)
    : data(n * d, raft::resource::get_cuda_stream(res)),
      norms(n, raft::resource::get_cuda_stream(res)),
      n_rows(n),
      dim(d)
  {
    auto stream           = raft::resource::get_cuda_stream(res).get();
    const int64_t threads = 256;
    detail::cast_to_half_kernel<<<(n * d + threads - 1) / threads, threads, 0, stream>>>(
      dataset, data.data(), n * d);
    detail::row_norms_kernel<<<n, 256, 0, stream>>>(data.data(), n, d, norms.data());
    raft::resource::sync_stream(res);
  }

  [[nodiscard]] auto view() const
  {
    return raft::make_device_matrix_view<const half, int64_t, raft::row_major>(
      data.data(), n_rows, dim);
  }
  [[nodiscard]] auto norms_view() const
  {
    return raft::make_device_vector_view<const float, int64_t>(norms.data(), n_rows);
  }
};

template <typename DistanceT, typename DataT, typename IdxT>
class AnnGpipnnTest : public ::testing::TestWithParam<AnnGpipnnInputs> {
 public:
  AnnGpipnnTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnGpipnnInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  auto make_params() const -> index_params
  {
    index_params p;
    p.graph_degree   = ps.graph_degree;
    p.max_candidates = ps.max_candidates;
    p.min_leaf_size  = ps.min_leaf_size;
    p.max_leaf_size  = ps.max_leaf_size;
    return p;
  }

  /** Every slot must hold an in-range id that is not the row itself. */
  void checkGraphStructure(const std::vector<IdxT>& graph)
  {
    for (int64_t i = 0; i < ps.n_rows; i++) {
      for (int64_t j = 0; j < ps.graph_degree; j++) {
        const IdxT v = graph[i * ps.graph_degree + j];
        ASSERT_LT(v, static_cast<IdxT>(ps.n_rows)) << "row " << i << " slot " << j;
        ASSERT_NE(v, static_cast<IdxT>(i)) << "self loop at row " << i << " slot " << j;
      }
    }
  }

  /** Ball carving must cover every point and respect the leaf size bound. */
  void testCarving()
  {
    scoring_operand op(handle_, database.data(), ps.n_rows, ps.dim);
    detail::carve_stats stats;
    auto leaves = detail::ball_carve(handle_, op.view(), op.norms_view(), make_params(), stats);
    ASSERT_GT(leaves.count(), 0) << "carving produced no leaves at all";

    std::cout << "  carve: rounds " << stats.rounds_run << ", chopped " << stats.chopped_runs
              << ", merged " << stats.merged_runs << ", still oversized " << stats.oversized_left
              << ", leaves " << leaves.count() << " (mean size "
              << static_cast<double>(leaves.total()) / static_cast<double>(leaves.count()) << "), "
              << static_cast<double>(leaves.total()) / static_cast<double>(ps.n_rows)
              << " memberships/point" << std::endl;

    for (int64_t i = 0; i < leaves.count(); i++) {
      ASSERT_GT(leaves.size_of(i), 0) << "empty leaf " << i;
      ASSERT_LE(leaves.size_of(i), ps.max_leaf_size) << "oversized leaf " << i;
    }

    std::vector<detail::id_t> members(leaves.total());
    raft::update_host(members.data(), leaves.members.data(), leaves.total(), stream_);
    raft::resource::sync_stream(handle_);

    std::vector<uint8_t> covered(ps.n_rows, 0);
    for (auto m : members) {
      ASSERT_LT(m, static_cast<detail::id_t>(ps.n_rows));
      covered[m] = 1;
    }
    ASSERT_EQ(std::count(covered.begin(), covered.end(), uint8_t{1}), ps.n_rows)
      << "some points belong to no leaf";
  }

  /** HashPrune rows must be ascending by distance and free of duplicate targets. */
  void testAccumulator()
  {
    auto params = make_params();
    scoring_operand op(handle_, database.data(), ps.n_rows, ps.dim);
    detail::carve_stats stats;
    auto leaves  = detail::ball_carve(handle_, op.view(), op.norms_view(), params, stats);
    auto records = detail::leaf_pick(handle_, op.view(), op.norms_view(), leaves, 2);
    auto acc     = detail::hash_prune(handle_, op.view(), records, params);

    const int64_t l_max = acc.l_max;
    std::vector<detail::id_t> ids(ps.n_rows * l_max);
    std::vector<float> dists(ps.n_rows * l_max);
    std::vector<uint32_t> counts(ps.n_rows);
    raft::update_host(ids.data(), acc.ids.data(), ids.size(), stream_);
    raft::update_host(dists.data(), acc.dists.data(), dists.size(), stream_);
    raft::update_host(counts.data(), acc.counts.data(), counts.size(), stream_);
    raft::resource::sync_stream(handle_);

    int64_t nonempty = 0;
    int64_t total    = 0;
    for (int64_t p = 0; p < ps.n_rows; p++) {
      const auto c = static_cast<int64_t>(counts[p]);
      ASSERT_LE(c, l_max);
      total += c;
      if (c > 0) { nonempty++; }
      std::set<detail::id_t> seen;
      for (int64_t j = 0; j < c; j++) {
        const detail::id_t t = ids[p * l_max + j];
        ASSERT_LT(t, static_cast<detail::id_t>(ps.n_rows));
        ASSERT_NE(t, static_cast<detail::id_t>(p)) << "self edge in accumulator row " << p;
        ASSERT_TRUE(seen.insert(t).second) << "duplicate target in accumulator row " << p;
        if (j > 0) {
          ASSERT_LE(dists[p * l_max + j - 1], dists[p * l_max + j])
            << "accumulator row " << p << " not ascending";
        }
      }
    }
    int64_t short_rows = 0;
    for (int64_t p = 0; p < ps.n_rows; p++) {
      if (static_cast<int64_t>(counts[p]) < ps.graph_degree) { short_rows++; }
    }
    std::cout << "  accumulator: mean fill "
              << static_cast<double>(total) / static_cast<double>(ps.n_rows) << "/" << l_max
              << ", rows below graph_degree(" << ps.graph_degree << "): " << short_rows << "/"
              << ps.n_rows << ", records " << records.n_valid << ", leaves " << leaves.count()
              << " (mean size "
              << static_cast<double>(leaves.total()) / static_cast<double>(leaves.count()) << ")"
              << std::endl;

    ASSERT_EQ(nonempty, ps.n_rows) << "every row should collect candidates";
    // Guards against a pipeline that technically runs but starves the prune step, which would
    // otherwise show up only as a soft recall drop. The bound is a fraction of graph_degree, not
    // of l_max: the reservoir capacity is deliberately larger than the output degree, so the
    // depth that matters is the one relative to the slots the prune has to fill.
    const double mean_fill = static_cast<double>(total) / static_cast<double>(ps.n_rows);
    ASSERT_GE(mean_fill, ps.graph_degree / 4.0)
      << "accumulator rows only " << mean_fill << " deep on average, for a graph_degree of "
      << ps.graph_degree << " (capacity " << l_max << ")";
  }

  /**
   * Guards the fp16 scoring choice: selection is approximate, but a leaf's chosen nearest
   * neighbour must still be nearly as close as the true nearest under fp32.
   */
  void testFp16Precision()
  {
    // A dataset that fits in one leaf makes the leaf join an exhaustive 2-NN, so the fp32
    // reference is unambiguous.
    auto params          = make_params();
    params.max_leaf_size = ps.n_rows;
    params.min_leaf_size = ps.n_rows;

    scoring_operand op(handle_, database.data(), ps.n_rows, ps.dim);
    detail::carve_stats stats;
    auto leaves = detail::ball_carve(handle_, op.view(), op.norms_view(), params, stats);
    ASSERT_EQ(leaves.count(), 1);
    auto records = detail::leaf_pick(handle_, op.view(), op.norms_view(), leaves, 2);

    std::vector<uint64_t> keys(records.keys.size());
    std::vector<detail::id_t> targets(records.targets.size());
    raft::update_host(keys.data(), records.keys.data(), keys.size(), stream_);
    raft::update_host(targets.data(), records.targets.data(), targets.size(), stream_);
    std::vector<float> host_data(static_cast<size_t>(ps.n_rows) * ps.dim);
    raft::update_host(host_data.data(), database.data(), host_data.size(), stream_);
    raft::resource::sync_stream(handle_);

    auto sq_dist = [&](int64_t a, int64_t b) {
      double acc = 0.0;
      for (int64_t i = 0; i < ps.dim; i++) {
        const double d = static_cast<double>(host_data[a * ps.dim + i]) -
                         static_cast<double>(host_data[b * ps.dim + i]);
        acc += d * d;
      }
      return acc;
    };

    // Best forward pick per owner, taken from the records the leaf phase emitted.
    std::vector<double> best(ps.n_rows, std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < keys.size(); i++) {
      if (keys[i] == detail::kInvalidKey) { continue; }
      const auto owner = detail::key_owner(keys[i]);
      best[owner]      = std::min(best[owner], sq_dist(owner, targets[i]));
    }

    int64_t checked = 0;
    for (int64_t p = 0; p < ps.n_rows; p++) {
      if (!std::isfinite(best[p])) { continue; }
      double truth = std::numeric_limits<double>::infinity();
      for (int64_t q = 0; q < ps.n_rows; q++) {
        if (q != p) { truth = std::min(truth, sq_dist(p, q)); }
      }
      // fp16 operands cost a few significant bits, not the ranking.
      ASSERT_LE(best[p], truth * 1.05 + 1e-4) << "fp16 selection far off at row " << p;
      checked++;
    }
    ASSERT_GT(checked, ps.n_rows / 2);
  }

  /** A half-precision caller must reach the same code path with no fp32 copy. */
  void testHalfInput()
  {
    auto params = make_params();
    rmm::device_uvector<half> half_data(static_cast<size_t>(ps.n_rows) * ps.dim, stream_);
    const int64_t threads = 256;
    const int64_t n       = static_cast<int64_t>(ps.n_rows) * ps.dim;
    detail::cast_to_half_kernel<<<(n + threads - 1) / threads, threads, 0, stream_.get()>>>(
      database.data(), half_data.data(), n);

    auto graph = raft::make_device_matrix<IdxT, int64_t>(handle_, ps.n_rows, ps.graph_degree);
    build_graph(
      handle_,
      params,
      raft::make_device_matrix_view<const half, int64_t>(half_data.data(), ps.n_rows, ps.dim),
      graph.view());

    std::vector<IdxT> h_graph(static_cast<size_t>(ps.n_rows) * ps.graph_degree);
    raft::update_host(h_graph.data(), graph.data_handle(), h_graph.size(), stream_);
    raft::resource::sync_stream(handle_);
    checkGraphStructure(h_graph);
  }

  /**
   * The narrow integer entry points, against the float one, on data both hold exactly.
   *
   * On integer-valued input these are not an approximation of the float path but the same
   * computation: half represents every integer up to 2048, so `float -> half` and `uint8 -> half`
   * of one integer give the same bits, and every kernel downstream sees an identical operand. The
   * graphs are therefore required to be *bit-identical*, which is a far sharper assertion than a
   * recall bound -- one differing edge is a defect rather than noise.
   *
   * Its own data rather than `database`, which is drawn from a normal and so is not
   * integer-valued. uint8 is checked on [0, 255] and int8 on the same values shifted into
   * [-128, 127]. The exactness argument also needs each row's squared norm under 2^24, which
   * holds here with room to spare; the per-type dimension bounds are on the public overloads.
   */
  void testIntegerInput()
  {
    const auto n = static_cast<size_t>(ps.n_rows) * ps.dim;
    std::vector<float> h_f(n), h_f_shift(n);
    std::vector<uint8_t> h_u8(n);
    std::vector<int8_t> h_i8(n);

    raft::random::RngState r(9876ULL);
    rmm::device_uvector<uint32_t> d_raw(n, stream_);
    raft::random::uniformInt<uint32_t>(handle_, r, d_raw.data(), n, 0u, 256u);
    std::vector<uint32_t> h_raw(n);
    raft::update_host(h_raw.data(), d_raw.data(), n, stream_);
    raft::resource::sync_stream(handle_);

    for (size_t i = 0; i < n; i++) {
      const int v  = static_cast<int>(h_raw[i] % 256u);
      h_f[i]       = static_cast<float>(v);
      h_u8[i]      = static_cast<uint8_t>(v);
      h_f_shift[i] = static_cast<float>(v - 128);
      h_i8[i]      = static_cast<int8_t>(v - 128);
    }

    auto build = [&](const auto& host) {
      using T = std::decay_t<decltype(host[0])>;
      rmm::device_uvector<T> d(host.size(), stream_);
      raft::update_device(d.data(), host.data(), host.size(), stream_);
      auto graph = raft::make_device_matrix<IdxT, int64_t>(handle_, ps.n_rows, ps.graph_degree);
      build_graph(handle_,
                  make_params(),
                  raft::make_device_matrix_view<const T, int64_t>(d.data(), ps.n_rows, ps.dim),
                  graph.view());
      std::vector<IdxT> out(static_cast<size_t>(ps.n_rows) * ps.graph_degree);
      raft::update_host(out.data(), graph.data_handle(), out.size(), stream_);
      raft::resource::sync_stream(handle_);
      return out;
    };

    const auto g_f = build(h_f);
    const auto g_u = build(h_u8);
    ASSERT_EQ(g_f, g_u) << "uint8 graph differs from the float graph on integer-valued data";

    const auto g_fs = build(h_f_shift);
    const auto g_i  = build(h_i8);
    ASSERT_EQ(g_fs, g_i) << "int8 graph differs from the float graph on integer-valued data";

    checkGraphStructure(g_u);
    checkGraphStructure(g_i);
  }

  /**
   * Cosine must be invariant to per-row scaling, and L2 must not be.
   *
   * Invariance under positive per-row scaling is the defining property of cosine, and it is the
   * property the implementation could plausibly get wrong, since cosine is served by normalizing
   * rows during the conversion to the fp16 scoring operand rather than by any change to the
   * scoring kernels. If that normalization is right the graph cannot move when the input rows are
   * rescaled; if it is missing, mis-indexed, or applied with the wrong norm, the graph moves.
   *
   * The scale factors are exact powers of two, which makes the assertion bit-identity rather than
   * a tolerance. Multiplying a row by `2^e` is exact in fp32 and scales the squared norm by
   * exactly `4^e`, and `rsqrtf` of a value whose exponent moved by an even amount returns the
   * same mantissa with the exponent moved back, so the recovered scale is exactly `2^-e` times
   * the original and each stored half is bit-for-bit what it was. An earlier version of this test
   * normalized on the host and compared against an L2 build, which does not work: the host cannot
   * reproduce the device reduction order or `rsqrtf` bit-exactly, and the fp32 scale then differs
   * in its last bit often enough to flip real half roundings.
   *
   * The companion L2 assertion is what keeps this from being vacuous. L2 is not scale invariant,
   * so if rescaling moved the L2 graph then it genuinely changed the data, and cosine's agreement
   * is a property of cosine rather than of a rescaling that did nothing.
   */
  void testCosineScaleInvariance()
  {
    const auto n = static_cast<size_t>(ps.n_rows) * ps.dim;
    std::vector<DataT> h_raw(n);
    raft::update_host(h_raw.data(), database.data(), n, stream_);
    raft::resource::sync_stream(handle_);

    // A row of zeros has no direction. It is included so the two builds have to agree on whatever
    // convention the normalization applies to it, rather than merely not crashing.
    for (int64_t j = 0; j < ps.dim; j++) {
      h_raw[j] = static_cast<DataT>(0);
    }

    std::vector<DataT> h_scaled(h_raw);
    for (int64_t i = 0; i < ps.n_rows; i++) {
      // Exponents small enough that no row over- or underflows; the pattern varies per row so a
      // single global scale cannot pass the test by accident.
      const float factor = std::ldexp(1.0f, static_cast<int>(i % 5) - 2);
      for (int64_t j = 0; j < ps.dim; j++) {
        h_scaled[i * ps.dim + j] =
          static_cast<DataT>(static_cast<float>(h_raw[i * ps.dim + j]) * factor);
      }
    }

    auto build = [&](const std::vector<DataT>& host, cuvs::distance::DistanceType metric) {
      rmm::device_uvector<DataT> d(host.size(), stream_);
      raft::update_device(d.data(), host.data(), host.size(), stream_);
      auto params   = make_params();
      params.metric = metric;
      auto graph    = raft::make_device_matrix<IdxT, int64_t>(handle_, ps.n_rows, ps.graph_degree);
      build_graph(handle_,
                  params,
                  raft::make_device_matrix_view<const DataT, int64_t>(d.data(), ps.n_rows, ps.dim),
                  graph.view());
      std::vector<IdxT> out(static_cast<size_t>(ps.n_rows) * ps.graph_degree);
      raft::update_host(out.data(), graph.data_handle(), out.size(), stream_);
      raft::resource::sync_stream(handle_);
      return out;
    };

    const auto cos_raw    = build(h_raw, cuvs::distance::DistanceType::CosineExpanded);
    const auto cos_scaled = build(h_scaled, cuvs::distance::DistanceType::CosineExpanded);
    checkGraphStructure(cos_raw);
    ASSERT_EQ(cos_raw, cos_scaled) << "cosine graph moved when rows were rescaled";

    const auto l2_raw    = build(h_raw, cuvs::distance::DistanceType::L2Expanded);
    const auto l2_scaled = build(h_scaled, cuvs::distance::DistanceType::L2Expanded);
    ASSERT_NE(l2_raw, l2_scaled) << "rescaling did not move the L2 graph either, so the cosine "
                                    "agreement above is vacuous";
    ASSERT_NE(cos_raw, l2_raw) << "cosine and L2 produced the same graph on unscaled data";
  }

  /** Metrics other than L2Expanded and CosineExpanded must be refused, not silently run as L2. */
  void testUnsupportedMetricRejected()
  {
    auto params        = make_params();
    params.metric      = cuvs::distance::DistanceType::InnerProduct;
    auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
      static_cast<const DataT*>(database.data()), ps.n_rows, ps.dim);
    auto graph = raft::make_device_matrix<IdxT, int64_t>(handle_, ps.n_rows, ps.graph_degree);
    ASSERT_THROW(build_graph(handle_, params, database_view, graph.view()), raft::exception);
  }

  /**
   * Every fanout and `leaf_degree` becomes the `k` of a `segmented_topk` call, which selects at
   * most `kMaxSelectK`. Checks that the bound is refused above the limit and honoured at it --
   * the boundary case is the one no benchmark configuration reaches.
   */
  void testSelectKBoundEnforced()
  {
    constexpr uint32_t kMax = static_cast<uint32_t>(detail::kMaxSelectK);
    auto database_view      = raft::make_device_matrix_view<const DataT, int64_t>(
      static_cast<const DataT*>(database.data()), ps.n_rows, ps.dim);
    auto graph = raft::make_device_matrix<IdxT, int64_t>(handle_, ps.n_rows, ps.graph_degree);

    for (int which = 0; which < 3; which++) {
      auto params = make_params();
      switch (which) {
        case 0: params.fanout_level0 = kMax + 1; break;
        case 1: params.fanout_level1 = kMax + 1; break;
        default: params.leaf_degree = kMax + 1; break;
      }
      ASSERT_THROW(build_graph(handle_, params, database_view, graph.view()), raft::exception)
        << "parameter " << which << " accepted a value above kMaxSelectK";
    }

    auto params           = make_params();
    params.fanout_level0  = kMax;
    params.fanout_level1  = kMax;
    params.leaf_degree    = kMax;
    params.max_leaf_size  = std::max<uint32_t>(params.max_leaf_size, 2048);
    params.max_candidates = std::max<uint32_t>(params.max_candidates, params.graph_degree);
    build_graph(handle_, params, database_view, graph.view());
    std::vector<IdxT> h_graph(static_cast<size_t>(ps.n_rows) * ps.graph_degree);
    raft::update_host(h_graph.data(), graph.data_handle(), h_graph.size(), stream_);
    raft::resource::sync_stream(handle_);
    checkGraphStructure(h_graph);
  }

  /**
   * Reports the distance-rank distribution of the graph's edges: for each edge (p, q), how many
   * points are strictly closer to p than q is. A graph with only short-range edges has every
   * rank bounded by roughly the reservoir depth; long-range edges show up as high ranks.
   */
  void testEdgeLengths()
  {
    auto params        = make_params();
    auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
      static_cast<const DataT*>(database.data()), ps.n_rows, ps.dim);
    auto graph = raft::make_device_matrix<IdxT, int64_t>(handle_, ps.n_rows, ps.graph_degree);
    build_graph(handle_, params, database_view, graph.view());

    std::vector<IdxT> h_graph(static_cast<size_t>(ps.n_rows) * ps.graph_degree);
    std::vector<DataT> host_data(static_cast<size_t>(ps.n_rows) * ps.dim);
    raft::update_host(h_graph.data(), graph.data_handle(), h_graph.size(), stream_);
    raft::update_host(host_data.data(), database.data(), host_data.size(), stream_);
    raft::resource::sync_stream(handle_);

    const int64_t stride = std::max<int64_t>(1, ps.n_rows / 200);
    std::vector<int64_t> ranks;
    std::vector<float> all(ps.n_rows);
    std::vector<int64_t> order(ps.n_rows);
    int64_t found  = 0;
    int64_t wanted = 0;
    for (int64_t p = 0; p < ps.n_rows; p += stride) {
      for (int64_t x = 0; x < ps.n_rows; x++) {
        float acc = 0.0f;
        for (int64_t i = 0; i < ps.dim; i++) {
          const float d = host_data[p * ps.dim + i] - host_data[x * ps.dim + i];
          acc += d * d;
        }
        all[x] = acc;
      }
      auto sorted = all;
      std::sort(sorted.begin(), sorted.end());
      for (int64_t j = 0; j < ps.graph_degree; j++) {
        const auto q  = h_graph[p * ps.graph_degree + j];
        const auto it = std::lower_bound(sorted.begin(), sorted.end(), all[q]);
        ranks.push_back(static_cast<int64_t>(it - sorted.begin()));
      }

      // Graph recall: what fraction of p's true k nearest neighbours does its adjacency hold?
      std::iota(order.begin(), order.end(), int64_t{0});
      std::partial_sort(
        order.begin(), order.begin() + ps.k + 1, order.end(), [&](int64_t a, int64_t b) {
          return all[a] < all[b];
        });
      const auto* row = &h_graph[p * ps.graph_degree];
      for (int64_t j = 0; j <= ps.k; j++) {
        if (order[j] == p) { continue; }
        wanted++;
        if (std::find(row, row + ps.graph_degree, static_cast<IdxT>(order[j])) !=
            row + ps.graph_degree) {
          found++;
        }
      }
    }
    std::cout << "  graph recall@" << ps.k << " = "
              << (static_cast<double>(found) / static_cast<double>(wanted)) << std::endl;

    std::sort(ranks.begin(), ranks.end());
    auto pct = [&](double f) { return ranks[static_cast<size_t>(f * (ranks.size() - 1))]; };
    const auto far100 =
      std::count_if(ranks.begin(), ranks.end(), [](int64_t r) { return r > 100; });
    const auto far1000 =
      std::count_if(ranks.begin(), ranks.end(), [](int64_t r) { return r > 1000; });
    std::cout << "  edge ranks over " << ranks.size() << " edges: p50=" << pct(0.5)
              << " p90=" << pct(0.9) << " p99=" << pct(0.99) << " max=" << ranks.back()
              << "  >100: " << (100.0 * far100 / ranks.size()) << "%"
              << "  >1000: " << (100.0 * far1000 / ranks.size()) << "%" << std::endl;

    // The typical edge must be a genuine near neighbour. Slots the prune step cannot fill are
    // ring-padded and land at arbitrary ranks, so only the median is bounded here.
    EXPECT_LT(pct(0.5), 1000) << "graph is not dominated by short-range edges";
  }

  /**
   * The build must be a pure function of the dataset, the parameters and the seed.
   *
   * Reproducibility is not a nicety here: without it no recall figure can be compared against
   * another, and a regression cannot be told apart from run noise. It is also the only cheap
   * check that catches a whole class of defect -- a shared-memory race, an unsynchronized read,
   * an order-dependent reduction -- from the outside, since all of those show up as a graph that
   * moves between runs while every input stays put.
   *
   * The comparison is bitwise over the whole adjacency rather than on recall, which is far more
   * sensitive: a race that perturbs a few thousand rows can leave aggregate recall inside its
   * own noise band and pass a recall gate unnoticed.
   *
   * The third build changes only the seed, and asserts the graph *does* move. Without it this
   * test would be satisfied by a builder that had accidentally stopped randomizing at all.
   */
  void testDeterminism()
  {
    auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
      static_cast<const DataT*>(database.data()), ps.n_rows, ps.dim);
    const int64_t degree = ps.graph_degree;

    auto build_once = [&](uint64_t seed) {
      auto p = make_params();
      p.seed = seed;
      auto g = raft::make_device_matrix<IdxT, int64_t>(handle_, ps.n_rows, degree);
      build_graph(handle_, p, database_view, g.view());
      std::vector<IdxT> h(static_cast<size_t>(ps.n_rows) * degree);
      raft::update_host(h.data(), g.data_handle(), h.size(), stream_);
      raft::resource::sync_stream(handle_);
      return h;
    };

    const uint64_t seed = make_params().seed;
    const auto first    = build_once(seed);
    const auto second   = build_once(seed);

    int64_t differing_rows = 0;
    int64_t first_row      = -1;
    int64_t first_slot     = -1;
    for (int64_t p = 0; p < ps.n_rows; p++) {
      for (int64_t j = 0; j < degree; j++) {
        if (first[p * degree + j] != second[p * degree + j]) {
          if (first_row < 0) {
            first_row  = p;
            first_slot = j;
          }
          differing_rows++;
          break;
        }
      }
    }
    ASSERT_EQ(differing_rows, 0) << "two builds at seed " << seed << " disagree on "
                                 << differing_rows << " of " << ps.n_rows
                                 << " rows; earliest at row " << first_row << " slot " << first_slot
                                 << " ("
                                 << (first_row < 0 ? 0 : first[first_row * degree + first_slot])
                                 << " vs "
                                 << (first_row < 0 ? 0 : second[first_row * degree + first_slot])
                                 << ")";

    const auto other = build_once(seed + 1);
    ASSERT_NE(first, other) << "changing the seed left the graph identical, so the build is no "
                               "longer randomized by it";
  }

  /** Builds the graph, then measures recall by searching it with CAGRA. */
  void testGpipnn(cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
  {
    auto params        = make_params();
    params.metric      = metric;
    auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
      static_cast<const DataT*>(database.data()), ps.n_rows, ps.dim);

    auto graph = raft::make_device_matrix<IdxT, int64_t>(handle_, ps.n_rows, ps.graph_degree);
    build_graph(handle_, params, database_view, graph.view());

    std::vector<IdxT> h_graph(static_cast<size_t>(ps.n_rows) * ps.graph_degree);
    raft::update_host(h_graph.data(), graph.data_handle(), h_graph.size(), stream_);
    raft::resource::sync_stream(handle_);
    checkGraphStructure(h_graph);

    const size_t queries_size = static_cast<size_t>(ps.n_queries) * ps.k;
    std::vector<IdxT> indices_gpipnn(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_gpipnn(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      cuvs::neighbors::naive_knn<DistanceT, DataT, IdxT>(handle_,
                                                         distances_naive_dev.data(),
                                                         indices_naive_dev.data(),
                                                         search_queries.data(),
                                                         database.data(),
                                                         ps.n_queries,
                                                         ps.n_rows,
                                                         ps.dim,
                                                         ps.k,
                                                         metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    cuvs::neighbors::test::padded_device_matrix_for_cagra<DataT> cagra_base(handle_, database_view);
    auto cagra_index = cagra::device_padded_index<DataT, IdxT>(
      handle_, metric, cagra_base.view, raft::make_const_mdspan(graph.view()));

    cagra::search_params search_params;
    search_params.algo       = cagra::search_algo::AUTO;
    search_params.team_size  = 0;
    search_params.itopk_size = ps.itopk_size;

    rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);
    cagra::search(
      handle_,
      search_params,
      cagra_index,
      raft::make_device_matrix_view<const DataT, int64_t>(
        search_queries.data(), ps.n_queries, ps.dim),
      raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k),
      raft::make_device_matrix_view<DistanceT, int64_t>(distances_dev.data(), ps.n_queries, ps.k));
    raft::update_host(distances_gpipnn.data(), distances_dev.data(), queries_size, stream_);
    raft::update_host(indices_gpipnn.data(), indices_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);

    EXPECT_TRUE(eval_neighbours(indices_naive,
                                indices_gpipnn,
                                distances_naive,
                                distances_gpipnn,
                                ps.n_queries,
                                ps.k,
                                0.003,
                                ps.min_recall));
  }

  void SetUp() override
  {
    database.resize(static_cast<size_t>(ps.n_rows) * ps.dim, stream_);
    search_queries.resize(static_cast<size_t>(ps.n_queries) * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    // Scaling the distribution rather than the samples keeps the default case (scale 1) drawing
    // exactly the data it always did.
    const auto mean   = static_cast<DataT>(0.1 * ps.data_scale);
    const auto stddev = static_cast<DataT>(2.0 * ps.data_scale);
    raft::random::normal(handle_, r, database.data(), ps.n_rows * ps.dim, mean, stddev);
    raft::random::normal(handle_, r, search_queries.data(), ps.n_queries * ps.dim, mean, stddev);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

  raft::resources handle_;
  cuda::stream_ref stream_;
  AnnGpipnnInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

inline auto generate_inputs() -> std::vector<AnnGpipnnInputs>
{
  return raft::util::itertools::product<AnnGpipnnInputs>({4000},      // n_rows
                                                         {32, 128},   // dim
                                                         {64},        // graph_degree
                                                         {128},       // max_candidates
                                                         {32},        // min_leaf_size
                                                         {128, 512},  // max_leaf_size
                                                         {100},       // n_queries
                                                         {10},        // k
                                                         {64},        // itopk_size
                                                         {0.9},       // min_recall
                                                         {1.0});      // data_scale
}

const std::vector<AnnGpipnnInputs> inputs = generate_inputs();

}  // namespace cuvs::neighbors::experimental::gpipnn
