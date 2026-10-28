/*
 * SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 */

#pragma once

#include <cub/device/device_histogram.cuh>
#include <cuda/iterator>
#include <distance/top_1_nn.cuh>

namespace cuvs::detail::histogram_launch {

template <typename Policy, int Channels, typename Counter, typename Offset>
using init_pointer_t = void (*)(cuda::std::array<int, Channels>,
                                cuda::std::array<Counter*, Channels>,
                                cub::GridQueue<int>);

template <typename Policy, int Channels, typename Counter, typename Offset>
CUB_RUNTIME_FUNCTION init_pointer_t<Policy, Channels, Counter, Offset> init();

template <typename Policy,
          int Bins,
          int Channels,
          int ActiveChannels,
          typename Samples,
          typename Counter,
          typename PrivatizedDecode,
          typename OutputDecode,
          typename Offset>
using sweep_pointer_t = void (*)(Samples,
                                 cuda::std::array<int, ActiveChannels>,
                                 cuda::std::array<int, ActiveChannels>,
                                 cuda::std::array<Counter*, ActiveChannels>,
                                 cuda::std::array<Counter*, ActiveChannels>,
                                 cuda::std::array<OutputDecode, ActiveChannels>,
                                 cuda::std::array<PrivatizedDecode, ActiveChannels>,
                                 Offset,
                                 Offset,
                                 Offset,
                                 int,
                                 cub::GridQueue<int>);

template <typename Policy,
          int Bins,
          int Channels,
          int ActiveChannels,
          typename Samples,
          typename Counter,
          typename PrivatizedDecode,
          typename OutputDecode,
          typename Offset>
CUB_RUNTIME_FUNCTION sweep_pointer_t<Policy,
                                     Bins,
                                     Channels,
                                     ActiveChannels,
                                     Samples,
                                     Counter,
                                     PrivatizedDecode,
                                     OutputDecode,
                                     Offset>
sweep();

// Adapted from CUB DeviceHistogramKernelSource; kernel implementations remain in CUB.
template <int NumChannels,
          int NumActiveChannels,
          typename SampleIteratorT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          typename SampleT>
struct source {
  using TransformsT = cub::detail::histogram::Transforms<LevelT, OffsetT, SampleT>;

  template <typename PolicyT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramInitKernel()
  {
    return init<PolicyT, NumActiveChannels, CounterT, OffsetT>();
  }

  /// Returns the default histogram sweep kernel that receives pre-initialized decode operators from
  /// the host.
  template <typename PolicyT,
            int PrivatizedSmemBins,
            typename PrivatizedDecodeOpT,
            typename OutputDecodeOpT>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepKernel()
  {
    if constexpr (::cuda::std::is_same_v<SampleIteratorT, unsigned int const*> &&
                  ::cuda::std::is_same_v<LevelT, int> && ::cuda::std::is_same_v<OffsetT, int> &&
                  (::cuda::std::is_same_v<CounterT, unsigned int> ||
                   ::cuda::std::is_same_v<CounterT, unsigned long long>)) {
      return sweep<PolicyT,
                   PrivatizedSmemBins,
                   NumChannels,
                   NumActiveChannels,
                   SampleIteratorT,
                   CounterT,
                   PrivatizedDecodeOpT,
                   OutputDecodeOpT,
                   OffsetT>();
    } else {
      return &cub::detail::histogram::DeviceHistogramSweepKernel<PolicyT,
                                                                 PrivatizedSmemBins,
                                                                 NumChannels,
                                                                 NumActiveChannels,
                                                                 SampleIteratorT,
                                                                 CounterT,
                                                                 PrivatizedDecodeOpT,
                                                                 OutputDecodeOpT,
                                                                 OffsetT>;
    }
  }

  /// Returns the device-init histogram sweep kernel that initializes decode operators from level
  /// arrays in the kernel.
  template <typename PolicyT,
            int PrivatizedSmemBins,
            typename FirstLevelArrayT,
            typename SecondLevelArrayT,
            bool IsEven,
            bool IsByteSample>
  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION static constexpr auto HistogramSweepKernelDeviceInit()
  {
    // For DispatchEven, we use the scale transform to convert samples to
    // privatized bins and pass-thru transform to convert privatized bins to
    // output bins, vice verse for byte samples.

    // For DispatchRange, we use the search transform to convert samples to
    // privatized bins and scale transform to convert privatized bins to output bins,
    // vice verse for byte samples.

    using DecodeOpT =
      ::cuda::std::conditional_t<IsEven,
                                 typename TransformsT::ScaleTransform,
                                 typename TransformsT::template SearchTransform<const LevelT*>>;

    using PrivatizedDecodeOpT =
      ::cuda::std::conditional_t<IsByteSample, typename TransformsT::PassThruTransform, DecodeOpT>;
    using OutputDecodeOpT =
      ::cuda::std::conditional_t<IsByteSample, DecodeOpT, typename TransformsT::PassThruTransform>;

    return &cub::detail::histogram::DeviceHistogramSweepDeviceInitKernel<PolicyT,
                                                                         PrivatizedSmemBins,
                                                                         NumChannels,
                                                                         NumActiveChannels,
                                                                         SampleIteratorT,
                                                                         CounterT,
                                                                         FirstLevelArrayT,
                                                                         SecondLevelArrayT,
                                                                         PrivatizedDecodeOpT,
                                                                         OutputDecodeOpT,
                                                                         OffsetT,
                                                                         IsEven>;
  }

  CUB_RUNTIME_FUNCTION static constexpr size_t CounterSize() { return sizeof(CounterT); }

  template <typename NumBinsT, typename UpperLevelArrayT, typename LowerLevelArrayT>
  CUB_RUNTIME_FUNCTION static constexpr bool MayOverflow(
    [[maybe_unused]] NumBinsT num_bins,
    [[maybe_unused]] const UpperLevelArrayT& upper_level,
    [[maybe_unused]] const LowerLevelArrayT& lower_level,
    [[maybe_unused]] int channel)
  {
    using CommonT = typename TransformsT::ScaleTransform::CommonT;

    if constexpr (::cuda::std::is_integral_v<CommonT>) {
      using IntArithmeticT = typename TransformsT::ScaleTransform::IntArithmeticT;
      // The unary plus promotes plain char to int, which cuda::std::cmp_greater requires
      if (::cuda::std::cmp_greater(num_bins, +::cuda::std::numeric_limits<CommonT>::max())) {
        return true;
      }
      const auto upper_level_cast = static_cast<IntArithmeticT>(upper_level[channel]);
      const auto lower_level_cast = static_cast<IntArithmeticT>(lower_level[channel]);
      const auto range = static_cast<IntArithmeticT>(upper_level_cast - lower_level_cast);
      return range > (::cuda::std::numeric_limits<IntArithmeticT>::max() /
                      static_cast<IntArithmeticT>(num_bins));
    } else {
      return false;
    }
  }
};

}  // namespace cuvs::detail::histogram_launch

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<
  1,
  1,
  ::cuda::transform_iterator<
    cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<int, double>*>::key_op,
    raft::KeyValuePair<int, double>*>,
  double,
  int,
  int,
  int>
  : cuvs::detail::histogram_launch::source<
      1,
      1,
      ::cuda::transform_iterator<
        cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<int, double>*>::key_op,
        raft::KeyValuePair<int, double>*>,
      double,
      int,
      int,
      int> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<
  1,
  1,
  ::cuda::transform_iterator<
    cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<int, float>*>::key_op,
    raft::KeyValuePair<int, float>*>,
  float,
  int,
  int,
  int>
  : cuvs::detail::histogram_launch::source<
      1,
      1,
      ::cuda::transform_iterator<
        cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<int, float>*>::key_op,
        raft::KeyValuePair<int, float>*>,
      float,
      int,
      int,
      int> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<
  1,
  1,
  ::cuda::transform_iterator<
    cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, double>*>::key_op,
    raft::KeyValuePair<long, double>*>,
  double,
  long,
  int,
  long>
  : cuvs::detail::histogram_launch::source<
      1,
      1,
      ::cuda::transform_iterator<
        cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, double>*>::key_op,
        raft::KeyValuePair<long, double>*>,
      double,
      long,
      int,
      long> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<
  1,
  1,
  ::cuda::transform_iterator<
    cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, double>*>::key_op,
    raft::KeyValuePair<long, double>*>,
  double,
  long,
  long,
  long>
  : cuvs::detail::histogram_launch::source<
      1,
      1,
      ::cuda::transform_iterator<
        cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, double>*>::key_op,
        raft::KeyValuePair<long, double>*>,
      double,
      long,
      long,
      long> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<
  1,
  1,
  ::cuda::transform_iterator<
    cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, float>*>::key_op,
    raft::KeyValuePair<long, float>*>,
  float,
  long,
  int,
  long>
  : cuvs::detail::histogram_launch::source<
      1,
      1,
      ::cuda::transform_iterator<
        cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, float>*>::key_op,
        raft::KeyValuePair<long, float>*>,
      float,
      long,
      int,
      long> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<
  1,
  1,
  ::cuda::transform_iterator<
    cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, float>*>::key_op,
    raft::KeyValuePair<long, float>*>,
  float,
  long,
  long,
  long>
  : cuvs::detail::histogram_launch::source<
      1,
      1,
      ::cuda::transform_iterator<
        cuvs::distance::detail::Top1nnResultTraits<raft::KeyValuePair<long, float>*>::key_op,
        raft::KeyValuePair<long, float>*>,
      float,
      long,
      long,
      long> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1, 1, int const*, double, int, int, int>
  : cuvs::detail::histogram_launch::source<1, 1, int const*, double, int, int, int> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1, 1, int const*, float, int, int, int>
  : cuvs::detail::histogram_launch::source<1, 1, int const*, float, int, int, int> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1, 1, int const*, int, int, int, int>
  : cuvs::detail::histogram_launch::source<1, 1, int const*, int, int, int, int> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1, 1, int*, double, int, int, int>
  : cuvs::detail::histogram_launch::source<1, 1, int*, double, int, int, int> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1, 1, int*, float, int, int, int>
  : cuvs::detail::histogram_launch::source<1, 1, int*, float, int, int, int> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1, 1, int*, int, int, int, int>
  : cuvs::detail::histogram_launch::source<1, 1, int*, int, int, int, int> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1, 1, long*, double, long, int, long>
  : cuvs::detail::histogram_launch::source<1, 1, long*, double, long, int, long> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1, 1, long*, double, long, long, long>
  : cuvs::detail::histogram_launch::source<1, 1, long*, double, long, long, long> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1, 1, long*, float, long, int, long>
  : cuvs::detail::histogram_launch::source<1, 1, long*, float, long, int, long> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1, 1, long*, float, long, long, long>
  : cuvs::detail::histogram_launch::source<1, 1, long*, float, long, long, long> {};

template <>
struct cub::detail::histogram::
  DeviceHistogramKernelSource<1, 1, unsigned int const*, unsigned int, int, int, unsigned int>
  : cuvs::detail::histogram_launch::
      source<1, 1, unsigned int const*, unsigned int, int, int, unsigned int> {};

template <>
struct cub::detail::histogram::
  DeviceHistogramKernelSource<1, 1, unsigned int const*, unsigned int, int, long, unsigned int>
  : cuvs::detail::histogram_launch::
      source<1, 1, unsigned int const*, unsigned int, int, long, unsigned int> {};

template <>
struct cub::detail::histogram::
  DeviceHistogramKernelSource<1, 1, unsigned int const*, unsigned long long, int, int, unsigned int>
  : cuvs::detail::histogram_launch::
      source<1, 1, unsigned int const*, unsigned long long, int, int, unsigned int> {};

template <>
struct cub::detail::histogram::
  DeviceHistogramKernelSource<1, 1, unsigned int const*, unsigned int, long, int, unsigned int>
  : cuvs::detail::histogram_launch::
      source<1, 1, unsigned int const*, unsigned int, long, int, unsigned int> {};

template <>
struct cub::detail::histogram::
  DeviceHistogramKernelSource<1, 1, unsigned int const*, unsigned int, long, long, unsigned int>
  : cuvs::detail::histogram_launch::
      source<1, 1, unsigned int const*, unsigned int, long, long, unsigned int> {};

template <>
struct cub::detail::histogram::DeviceHistogramKernelSource<1,
                                                           1,
                                                           unsigned int const*,
                                                           unsigned long long,
                                                           long,
                                                           int,
                                                           unsigned int>
  : cuvs::detail::histogram_launch::
      source<1, 1, unsigned int const*, unsigned long long, long, int, unsigned int> {};
