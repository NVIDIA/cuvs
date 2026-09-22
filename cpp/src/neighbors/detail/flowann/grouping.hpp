/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

struct grouping_plan {
  std::uint16_t n_bits;
  std::uint32_t n_groups;
  std::uint64_t capacity;
  std::uint64_t lower;
  std::uint64_t upper;
};

/** Plan once, independently of the graph memory budget. This arithmetic also accepts dataset
 * sizes beyond the current uint32 graph implementation, for capacity planning only.
 */
inline auto plan_grouping(std::uint64_t n_rows,
                          std::uint16_t bits   = 0,
                          std::uint32_t groups = 0,
                          double tolerance     = 0.10) -> grouping_plan
{
  RAFT_EXPECTS(n_rows > 0, "FlowANN cannot group an empty dataset");
  RAFT_EXPECTS(tolerance > 0 && tolerance < 1, "Grouping tolerance must be in (0, 1)");
  if (bits == 0) {
    std::uint16_t required = 0;
    for (auto remaining = n_rows - 1; remaining != 0; remaining >>= 1) {
      ++required;
    }
    bits = std::min<std::uint16_t>(24, std::max<std::uint16_t>(4, (required + 3) / 4 * 4));
  }
  RAFT_EXPECTS(bits <= 32, "FlowANN grouping n_bits must be in [1, 32]");
  auto const capacity = std::uint64_t{1} << bits;
  // Round away sub-ULP arithmetic noise before ceil at exact capacity boundaries.
  auto ceil_count = [](double value) {
    auto nearest = std::round(value);
    if (std::abs(value - nearest) <= 4 * std::numeric_limits<double>::epsilon() * value) {
      value = nearest;
    }
    return std::ceil(value);
  };
  if (groups == 0) {
    auto const required =
      n_rows <= capacity ? 1.0
                         : ceil_count(static_cast<double>(n_rows) * (1.0 + tolerance) / capacity);
    RAFT_EXPECTS(required <= std::numeric_limits<std::uint32_t>::max(), "Too many groups");
    groups = static_cast<std::uint32_t>(required);
  }
  RAFT_EXPECTS(groups <= n_rows && groups * capacity >= n_rows,
               "Group count cannot satisfy local-ID capacity");
  auto const average = static_cast<double>(n_rows) / groups;
  auto const lower   = std::max<std::uint64_t>(1, std::floor(average * (1.0 - tolerance)));
  auto const upper   = std::min<std::uint64_t>(capacity, std::ceil(average * (1.0 + tolerance)));
  RAFT_EXPECTS(lower * groups <= n_rows && upper * groups >= n_rows,
               "Infeasible grouping size bounds");
  return {bits, groups, capacity, lower, upper};
}

/** Each entry is the number of final leaves beneath a child. Intermediate capacities are
 * additive: child_leaves * leaf_capacity (and likewise for the balance bounds).
 */
inline auto grouping_children(std::uint32_t leaves) -> std::vector<std::uint32_t>
{
  RAFT_EXPECTS(leaves > 1, "A leaf has no children");
  auto const branches = leaves <= 16 ? leaves : std::min<std::uint32_t>(16, (leaves + 15) / 16);
  auto result         = std::vector<std::uint32_t>(branches, leaves / branches);
  for (std::uint32_t i = 0; i < leaves % branches; ++i) {
    ++result[i];
  }
  return result;
}

/** Reassign nearest-centroid labels using distance-increase priority. This is a bounded greedy
 * repair, not optimal transport. Costs are row-major, with smaller values always preferable.
 * Every move preserves donor lower bounds and receiver upper bounds. Overflow is removed first,
 * then underfull groups are filled; selection is repeated when a destination/source fills up.
 */
inline auto rebalance_group_distances(float const* costs,
                                      std::vector<std::uint8_t>& labels,
                                      std::vector<std::uint64_t> const& lower,
                                      std::vector<std::uint64_t> const& upper)
  -> std::vector<std::uint64_t>
{
  auto const k = lower.size();
  RAFT_EXPECTS(k > 0 && k <= 16 && upper.size() == k, "Invalid grouping branches");
  std::vector<std::uint64_t> sizes(k);
  for (auto label : labels) {
    RAFT_EXPECTS(label < k, "Invalid group label");
    ++sizes[label];
  }
  RAFT_EXPECTS(std::accumulate(lower.begin(), lower.end(), std::uint64_t{0}) <= labels.size() &&
                 std::accumulate(upper.begin(), upper.end(), std::uint64_t{0}) >= labels.size(),
               "Infeasible subtree capacities");
  for (std::size_t g = 0; g < k; ++g) {
    RAFT_EXPECTS(lower[g] <= upper[g], "Invalid subtree bounds");
  }
  struct move {
    // collect() writes every slot. Avoid zero-filling large temporary arrays before that write.
    move() noexcept {}
    move(float cost, std::uint32_t id, std::uint8_t destination)
      : penalty(cost), row(id), target(destination)
    {
    }
    float penalty;
    std::uint32_t row;
    std::uint8_t target;
  };
  auto less = [](move const& a, move const& b) {
    return a.penalty < b.penalty || (a.penalty == b.penalty && a.row < b.row);
  };
  // Compact eligible rows in parallel, retaining original row order for deterministic ties.
  auto collect = [](std::size_t rows, auto eligible, auto make_move) {
    constexpr std::size_t block_rows = 65536;
    auto const blocks                = (rows + block_rows - 1) / block_rows;
    std::vector<std::size_t> offsets(blocks + 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (rows > block_rows)
#endif
    for (std::size_t block = 0; block < blocks; ++block) {
      std::size_t count = 0;
      for (auto row = block * block_rows; row < std::min(rows, (block + 1) * block_rows); ++row) {
        count += eligible(row);
      }
      offsets[block + 1] = count;
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    std::vector<move> result(offsets.back());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (rows > block_rows)
#endif
    for (std::size_t block = 0; block < blocks; ++block) {
      auto position = offsets[block];
      for (auto row = block * block_rows; row < std::min(rows, (block + 1) * block_rows); ++row) {
        if (eligible(row)) { result[position++] = make_move(row); }
      }
    }
    return result;
  };
  std::vector<move> candidates;
  auto apply = [&](std::uint64_t requested, bool overflow) {
    auto count = std::min<std::size_t>(requested, candidates.size());
    RAFT_EXPECTS(count > 0, "No feasible capacity repair");
    std::vector<move> selected;
#ifdef _OPENMP
    if (count < candidates.size() && candidates.size() >= (1u << 20)) {
      // Exact radix selection parallelizes the large partition operation. The row ID
      // completes a unique key, so equal distances preserve the same priority as `less`.
      auto key = [](move const& value) {
        auto penalty = value.penalty == 0 ? 0.0f : value.penalty;
        std::uint32_t bits;
        std::memcpy(&bits, &penalty, sizeof(bits));
        bits ^= (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
        return (std::uint64_t{bits} << 32) | value.row;
      };
      std::uint64_t prefix = 0, mask = 0;
      auto rank = count - 1;
      for (int shift = 56; shift >= 0; shift -= 8) {
        std::uint64_t histogram[256]{};
#pragma omp parallel for schedule(static) reduction(+ : histogram[ : 256])
        for (std::size_t i = 0; i < candidates.size(); ++i) {
          auto const value = key(candidates[i]);
          if ((value & mask) == prefix) { ++histogram[(value >> shift) & 255]; }
        }
        std::size_t bucket = 0;
        while (bucket < 255 && rank >= histogram[bucket]) {
          rank -= histogram[bucket++];
        }
        prefix |= std::uint64_t{bucket} << shift;
        mask |= std::uint64_t{255} << shift;
        if (histogram[bucket] == 1) { break; }
      }
      auto threshold = prefix | ~mask;
      selected       = collect(
        candidates.size(),
        [&](auto i) { return key(candidates[i]) <= threshold; },
        [&](auto i) { return candidates[i]; });
      RAFT_EXPECTS(selected.size() == count, "Invalid capacity-repair selection");
    } else
#endif
    {
      if (count < candidates.size()) {
        std::nth_element(candidates.begin(), candidates.begin() + count, candidates.end(), less);
      }
      selected.assign(candidates.begin(), candidates.begin() + count);
    }
    // All moves share a donor (overflow) or receiver (underflow). Only the opposite
    // endpoint can saturate. Select that endpoint's cheapest feasible subset, without
    // sorting millions of candidates that can all be moved independently.
    std::vector<std::vector<move>> buckets(k);
    for (std::size_t i = 0; i < count; ++i) {
      auto const& candidate = selected[i];
      auto const group      = overflow ? candidate.target : labels[candidate.row];
      buckets[group].push_back(candidate);
    }
    std::uint64_t moved = 0;
    for (std::size_t group = 0; group < k; ++group) {
      auto& bucket         = buckets[group];
      auto const available = overflow ? upper[group] - std::min(sizes[group], upper[group])
                                      : sizes[group] - std::min(sizes[group], lower[group]);
      auto const take      = std::min<std::size_t>(available, bucket.size());
      if (take == 0) { continue; }
      if (take < bucket.size()) {
        std::nth_element(bucket.begin(), bucket.begin() + take, bucket.end(), less);
      }
      for (std::size_t i = 0; i < take; ++i) {
        auto const& candidate = bucket[i];
        auto& source          = labels[candidate.row];
        --sizes[source];
        source = candidate.target;
        ++sizes[source];
      }
      moved += take;
    }
    RAFT_EXPECTS(moved > 0, "Capacity repair made no progress");
  };
  for (std::size_t donor = 0; donor < k; ++donor) {
    while (sizes[donor] > upper[donor]) {
      candidates = collect(
        labels.size(),
        [&](auto row) { return labels[row] == donor; },
        [&](auto row) {
          auto target = k;
          for (std::size_t g = 0; g < k; ++g) {
            if (g != donor && sizes[g] < upper[g] &&
                (target == k || costs[row * k + g] < costs[row * k + target])) {
              target = g;
            }
          }
          // Feasible aggregate bounds guarantee at least one receiver.
          return move{costs[row * k + target] - costs[row * k + donor],
                      static_cast<std::uint32_t>(row),
                      static_cast<std::uint8_t>(target)};
        });
      apply(sizes[donor] - upper[donor], true);
    }
  }
  for (std::size_t receiver = 0; receiver < k; ++receiver) {
    while (sizes[receiver] < lower[receiver]) {
      candidates = collect(
        labels.size(),
        [&](auto row) {
          auto const donor = labels[row];
          return donor != receiver && sizes[donor] > lower[donor];
        },
        [&](auto row) {
          return move{costs[row * k + receiver] - costs[row * k + labels[row]],
                      static_cast<std::uint32_t>(row),
                      static_cast<std::uint8_t>(receiver)};
        });
      // Restrict each donor to its available surplus.
      apply(lower[receiver] - sizes[receiver], false);
    }
  }
  for (std::size_t g = 0; g < k; ++g) {
    RAFT_EXPECTS(sizes[g] >= lower[g] && sizes[g] <= upper[g], "Subtree capacity violated");
  }
  return sizes;
}

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
