/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cub_select_launchers.hpp"
#include <cub/device/device_select.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cuda/iterator>

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::If<cub::ArgIndexInputIterator<double*, long, double>,
                      raft::KeyValuePair<long, double>*,
                      int*,
                      cuvs::cluster::kmeans::detail::SamplingOp<double, int>,
                      cudaStream_t,
                      0>(void* storage,
                         size_t& bytes,
                         cub::ArgIndexInputIterator<double*, long, double> input,
                         raft::KeyValuePair<long, double>* output,
                         int* selected,
                         cuda::std::int64_t items,
                         cuvs::cluster::kmeans::detail::SamplingOp<double, int> predicate,
                         const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::If<cub::ArgIndexInputIterator<double*, long, double>,
                      raft::KeyValuePair<long, double>*,
                      long*,
                      cuvs::cluster::kmeans::detail::SamplingOp<double, long>,
                      cudaStream_t,
                      0>(void* storage,
                         size_t& bytes,
                         cub::ArgIndexInputIterator<double*, long, double> input,
                         raft::KeyValuePair<long, double>* output,
                         long* selected,
                         cuda::std::int64_t items,
                         cuvs::cluster::kmeans::detail::SamplingOp<double, long> predicate,
                         const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::If<cub::ArgIndexInputIterator<float*, long, float>,
                      raft::KeyValuePair<long, float>*,
                      int*,
                      cuvs::cluster::kmeans::detail::SamplingOp<float, int>,
                      cudaStream_t,
                      0>(void* storage,
                         size_t& bytes,
                         cub::ArgIndexInputIterator<float*, long, float> input,
                         raft::KeyValuePair<long, float>* output,
                         int* selected,
                         cuda::std::int64_t items,
                         cuvs::cluster::kmeans::detail::SamplingOp<float, int> predicate,
                         const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::If<cub::ArgIndexInputIterator<float*, long, float>,
                      raft::KeyValuePair<long, float>*,
                      long*,
                      cuvs::cluster::kmeans::detail::SamplingOp<float, long>,
                      cudaStream_t,
                      0>(void* storage,
                         size_t& bytes,
                         cub::ArgIndexInputIterator<float*, long, float> input,
                         raft::KeyValuePair<long, float>* output,
                         long* selected,
                         cuda::std::int64_t items,
                         cuvs::cluster::kmeans::detail::SamplingOp<float, long> predicate,
                         const cudaStream_t& environment);
