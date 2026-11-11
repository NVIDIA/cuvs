/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace cuvs::cluster::kmeans::detail {

// Recipe and centroid/mass algebra follow faiss contrib/e_means/base.py at
// f34c0b881d9dd5d3a248abea827505f4d700bc02 (MIT, Meta Platforms, Inc. and affiliates).
struct emeans_schedule {
  int64_t batch_size;
  double alpha0, unused_threshold;
  std::vector<int64_t> epoch_rows;
  uint64_t steps = 0;

  double alpha(uint64_t step) const
  {
    constexpr double pi = 3.14159265358979323846;
    return alpha0 * (.55 + .45 * std::cos(pi * (double(step) / steps)));
  }
};

inline emeans_schedule resolve_emeans(const emeans_params& p, int64_t n, int64_t d, int64_t k)
{
  RAFT_EXPECTS(n > 0 && d > 0 && k > 0 && k <= n, "Invalid e-means extents");
  RAFT_EXPECTS(std::isfinite(p.epochs) && p.epochs > 0 && p.epochs <= 1000000 &&
                 p.batch_size >= 0 && std::isfinite(p.alpha0) && p.alpha0 >= 0 && p.alpha0 <= 1 &&
                 std::isfinite(p.unused_threshold) &&
                 (p.unused_threshold == -1 || p.unused_threshold >= 0),
               "Invalid e-means parameters");
  auto round_even = [](double x) {
    auto lo = int64_t(std::floor(x));
    return lo + (x - lo > .5 || (x - lo == .5 && (lo & 1)));
  };
  auto power_of_two = [](double x) {
    x       = std::max(1., x);
    auto lo = std::exp2(std::floor(std::log2(x)));
    return x / lo < 2 * lo / x ? lo : 2 * lo;
  };
  auto batch = p.batch_size;
  if (batch == 0) {
    double c        = d <= 16 ? .00439175 : .035134;
    double proposed = power_of_two(c * std::pow(double(k), .134749) * std::pow(double(n), .876473) *
                                   std::pow(p.epochs, .813344));
    double cap      = d <= 16 ? double(n / 2) : power_of_two(n / 2.);
    auto requested  = std::max(1., std::min(proposed, cap));
    batch           = requested >= double(n) ? n : int64_t(requested);
  }
  batch = std::min(batch, n);
  const auto alpha =
    p.alpha0 == 0
      ? std::clamp(.8244 * std::pow(double(k), -.4617) * std::pow(double(batch), .9003) *
                     std::pow(double(n), -.4498) * std::pow(p.epochs, -.1057),
                   1e-4,
                   1.)
      : p.alpha0;
  emeans_schedule result{
    batch, alpha, p.unused_threshold < 0 ? (d <= 2 ? 0. : .1) : p.unused_threshold, {}};
  const auto whole     = int64_t(std::floor(p.epochs));
  const auto remainder = round_even((p.epochs - whole) * n);
  result.epoch_rows.assign(std::max(int64_t{0}, whole - 1), n);
  if (remainder) { result.epoch_rows.push_back(remainder); }
  if (whole) { result.epoch_rows.push_back(n); }
  RAFT_EXPECTS(!result.epoch_rows.empty(), "e-means epoch budget contains no rows");
  for (auto rows : result.epoch_rows) {
    auto steps = uint64_t(rows / batch + (rows % batch != 0));
    RAFT_EXPECTS(steps <= std::numeric_limits<uint64_t>::max() - result.steps,
                 "e-means update count overflow");
    result.steps += steps;
  }
  return result;
}

template <typename T, typename IdxT>
RAFT_KERNEL emeans_update_kernel(T* centers,
                                 T* masses,
                                 const T* sums,
                                 const T* counts,
                                 IdxT dim,
                                 double alpha,
                                 T* closing_sums,
                                 uint64_t* closing_counts)
{
  const auto cluster = IdxT(blockIdx.x);
  const T count = counts[cluster], incoming = T(alpha) * count;
  const T retained =
    alpha == 1 ? (count > 0 ? T{0} : masses[cluster]) : T(1 - alpha) * masses[cluster];
  const T next = count > 0 ? retained + incoming : retained;
  __syncthreads();  // All threads must read the old mass before thread zero replaces it.
  for (IdxT j = threadIdx.x; j < dim; j += blockDim.x) {
    auto i = int64_t(cluster) * dim + j;
    if (count > 0) {
      T mean = sums[i] / count, weight = incoming / next;
      centers[i] = weight == T{1} ? mean : centers[i] + weight * (mean - centers[i]);
    }
    if (closing_sums) { closing_sums[i] += sums[i]; }
  }
  if (threadIdx.x == 0) {
    masses[cluster] = next;
    if (closing_counts) { closing_counts[cluster] += static_cast<uint64_t>(count); }
  }
}

template <typename T>
struct emeans_unused {
  const T *mass, *total;
  T threshold;
  int k;
  __device__ bool operator()(int i) const { return mass[i] < (*total / k) * threshold; }
};

template <typename T>
RAFT_KERNEL emeans_donor_keys(
  const T* mass, emeans_unused<T> unused, double* keys, int* ids, int k, uint64_t seed)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= k) { return; }
  // SplitMix64 counter keys give reproducible, weighted sampling without replacement.
  uint64_t x = seed + uint64_t(i) + 0x9e3779b97f4a7c15ULL;
  x          = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x          = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= x >> 31;
  double u = ((x >> 12) + .5) * 0x1p-52;
  keys[i]  = !unused(i) && mass[i] > 0 ? log(double(mass[i])) - log(-log(u)) : -INFINITY;
  ids[i]   = i;
}

template <typename T, typename IdxT>
RAFT_KERNEL emeans_split(T* centers,
                         T* mass,
                         const int* donors,
                         const int* targets,
                         const int* unused_count,
                         int k,
                         IdxT dim)
{
  int slot = blockIdx.x;
  if (slot >= min(*unused_count, k - *unused_count)) { return; }
  int from = donors[slot], to = targets[slot];
  __shared__ int distinct;
  if (threadIdx.x == 0) { distinct = 0; }
  __syncthreads();
  for (IdxT j = threadIdx.x; j < dim; j += blockDim.x) {
    T original = centers[int64_t(from) * dim + j];
    T delta    = j % 2 ? T(-1e-4) : T(1e-4);
    T donor = original * (T{1} - delta), child = original * (T{1} + delta);
    if (donor != child) { atomicExch(&distinct, 1); }
    centers[int64_t(from) * dim + j] = donor;
    centers[int64_t(to) * dim + j]   = child;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // A zero donor otherwise produces identical children and can never revive the target.
    if (!distinct) {
      auto i = int64_t(to) * dim;
      if constexpr (std::is_same_v<T, float>) {
        centers[i] = nextafterf(centers[i], INFINITY);
      } else {
        centers[i] = nextafter(centers[i], INFINITY);
      }
    }
    mass[from] = mass[to] = mass[from] * T{.5};
  }
}

/** EMA state uses only the caller's existing centroid arithmetic type. No assignment kernels. */
template <typename T, typename IdxT>
class emeans_state {
 public:
  emeans_state(raft::resources const& handle, emeans_schedule schedule, IdxT k, IdxT dim)
    : schedule_(std::move(schedule)),
      mass_(raft::make_device_vector<T, IdxT>(handle, k)),
      closing_sums_(raft::make_device_matrix<T, IdxT>(handle, k, dim)),
      closing_counts_(raft::make_device_vector<uint64_t, IdxT>(handle, k)),
      keys_(schedule_.unused_threshold ? k : 0, raft::resource::get_cuda_stream(handle)),
      sorted_keys_(keys_.size(), raft::resource::get_cuda_stream(handle)),
      ids_(keys_.size(), raft::resource::get_cuda_stream(handle)),
      sorted_ids_(keys_.size(), raft::resource::get_cuda_stream(handle)),
      unused_ids_(keys_.size(), raft::resource::get_cuda_stream(handle)),
      unused_count_(1, raft::resource::get_cuda_stream(handle)),
      total_(raft::make_device_scalar<T>(handle)),
      scratch_(0, raft::resource::get_cuda_stream(handle))
  {
    raft::matrix::fill(handle, mass_.view(), T(0x1p-16));
    raft::matrix::fill(handle, closing_sums_.view(), T{0});
    raft::matrix::fill(handle, closing_counts_.view(), uint64_t{0});
  }

  const emeans_schedule& schedule() const { return schedule_; }

  void update(raft::resources const& handle,
              raft::device_matrix_view<T, IdxT> centers,
              const T* sums,
              const T* counts,
              bool closing,
              uint64_t seed)
  {
    raft::launch_kernel(handle,
                        unsigned(centers.extent(0)),
                        256,
                        emeans_update_kernel<T, IdxT>,
                        centers.data_handle(),
                        mass_.data_handle(),
                        sums,
                        counts,
                        centers.extent(1),
                        schedule_.alpha(step_),
                        closing ? closing_sums_.data_handle() : nullptr,
                        closing ? closing_counts_.data_handle() : nullptr);
    if (++step_ % 5 == 0 && schedule_.unused_threshold > 0) { revive(handle, centers, seed); }
  }

  void finalize(raft::resources const& handle, raft::device_matrix_view<T, IdxT> centers)
  {
    auto sums   = closing_sums_.data_handle();
    auto counts = closing_counts_.data_handle();
    auto dim    = centers.extent(1);
    raft::linalg::map_offset(handle,
                             raft::make_device_vector_view(centers.data_handle(), centers.size()),
                             [=] __device__(auto i) {
                               return counts[i / dim] > 0 ? sums[i] / counts[i / dim]
                                                          : centers.data_handle()[i];
                             });
  }

 private:
  void revive(raft::resources const& handle,
              raft::device_matrix_view<T, IdxT> centers,
              uint64_t seed)
  {
    int k       = int(centers.extent(0));
    auto stream = raft::resource::get_cuda_stream(handle).get();
    auto ids    = thrust::make_counting_iterator(0);
    emeans_unused<T> unused{
      mass_.data_handle(), total_.data_handle(), T(schedule_.unused_threshold), k};
    auto flags = thrust::make_transform_iterator(ids, unused);
    size_t a{}, b{}, c{};
    RAFT_CUDA_TRY(
      cub::DeviceReduce::Sum(nullptr, a, mass_.data_handle(), total_.data_handle(), k, stream));
    RAFT_CUDA_TRY(cub::DeviceSelect::Flagged(
      nullptr, b, ids, flags, unused_ids_.data(), unused_count_.data(), k, stream));
    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                                            c,
                                                            keys_.data(),
                                                            sorted_keys_.data(),
                                                            ids_.data(),
                                                            sorted_ids_.data(),
                                                            k,
                                                            0,
                                                            64,
                                                            stream));
    if (scratch_.size() < std::max({a, b, c})) { scratch_.resize(std::max({a, b, c}), stream); }
    a = b = c = scratch_.size();
    RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
      scratch_.data(), a, mass_.data_handle(), total_.data_handle(), k, stream));
    RAFT_CUDA_TRY(cub::DeviceSelect::Flagged(
      scratch_.data(), b, ids, flags, unused_ids_.data(), unused_count_.data(), k, stream));
    raft::launch_kernel(handle,
                        unsigned((int64_t(k) + 255) / 256),
                        256,
                        emeans_donor_keys<T>,
                        mass_.data_handle(),
                        unused,
                        keys_.data(),
                        ids_.data(),
                        k,
                        seed ^ step_);
    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairsDescending(scratch_.data(),
                                                            c,
                                                            keys_.data(),
                                                            sorted_keys_.data(),
                                                            ids_.data(),
                                                            sorted_ids_.data(),
                                                            k,
                                                            0,
                                                            64,
                                                            stream));
    raft::launch_kernel(handle,
                        unsigned(k),
                        256,
                        emeans_split<T, IdxT>,
                        centers.data_handle(),
                        mass_.data_handle(),
                        sorted_ids_.data(),
                        unused_ids_.data(),
                        unused_count_.data(),
                        k,
                        centers.extent(1));
  }

  emeans_schedule schedule_;
  uint64_t step_ = 0;
  raft::device_vector<T, IdxT> mass_;
  raft::device_matrix<T, IdxT> closing_sums_;
  raft::device_vector<uint64_t, IdxT> closing_counts_;
  rmm::device_uvector<double> keys_, sorted_keys_;
  rmm::device_uvector<int> ids_, sorted_ids_, unused_ids_, unused_count_;
  raft::device_scalar<T> total_;
  rmm::device_uvector<char> scratch_;
};

/** Physical chunks only accumulate; updates happen exactly once per complete logical batch. */
template <typename T, typename IdxT, typename Loader, typename Accumulate>
void train_emeans(raft::resources const& handle,
                  emeans_state<T, IdxT>& state,
                  raft::device_matrix_view<T, IdxT> centers,
                  raft::device_matrix_view<T, IdxT> sums,
                  raft::device_vector_view<T, IdxT> counts,
                  Loader& loader,
                  uint64_t seed,
                  size_t recycle_offset,
                  Accumulate&& accumulate)
{
  const auto& schedule = state.schedule();
  for (size_t epoch = 0; epoch < schedule.epoch_rows.size(); ++epoch) {
    raft::matrix::fill(handle, sums, T{0});
    raft::matrix::fill(handle, counts, T{0});
    int64_t pending = 0, consumed = 0;
    loader.start();
    for (size_t pos = 0; pos < loader.num_batches(); ++pos) {
      auto batch = loader.acquire(pos);
      loader.prefetch((pos + 1) % loader.num_batches());
      for (int64_t offset = 0;
           offset < int64_t(batch.size()) && consumed < schedule.epoch_rows[epoch];) {
        auto rows = std::min({int64_t(batch.size()) - offset,
                              schedule.batch_size - pending,
                              schedule.epoch_rows[epoch] - consumed});
        accumulate(
          batch.data() + offset * centers.extent(1), IdxT(rows), IdxT(batch.offset() + offset));
        pending += rows;
        consumed += rows;
        offset += rows;
        if (pending == schedule.batch_size || consumed == schedule.epoch_rows[epoch]) {
          state.update(handle,
                       centers,
                       sums.data_handle(),
                       counts.data_handle(),
                       epoch + 1 == schedule.epoch_rows.size(),
                       seed);
          raft::matrix::fill(handle, sums, T{0});
          raft::matrix::fill(handle, counts, T{0});
          pending = 0;
        }
      }
      loader.recycle(batch, (pos + recycle_offset) % loader.num_batches());
    }
  }
  state.finalize(handle, centers);
}

}  // namespace cuvs::cluster::kmeans::detail
