/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/device_udf.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include <raft/core/error.hpp>

namespace cuvs::jit {

enum class device_udf_payload_kind { ltoir, cuda_source };

struct udf_capture {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  int32_t device_id = 0;
  std::uintptr_t pointer = 0;
  bool readonly = true;
};

struct device_udf {
  std::string abi;
  device_udf_payload_kind payload_kind = device_udf_payload_kind::ltoir;
  std::vector<uint8_t> payload;
  std::string symbol_name;
  std::string cache_key;
  std::vector<udf_capture> captures;
};

using ltoir_udf = device_udf;

inline device_udf_payload_kind payload_kind_from_c(cuvsDeviceUDFPayloadKind kind)
{
  switch (kind) {
    case CUVS_DEVICE_UDF_PAYLOAD_LTOIR: return device_udf_payload_kind::ltoir;
    case CUVS_DEVICE_UDF_PAYLOAD_CUDA_SOURCE: return device_udf_payload_kind::cuda_source;
  }
  RAFT_FAIL("Unsupported cuVS device UDF payload kind: %d", static_cast<int>(kind));
  return device_udf_payload_kind::ltoir;
}

inline device_udf make_device_udf(cuvsDeviceUDF const& desc)
{
  RAFT_EXPECTS(desc.abi != nullptr, "device UDF abi must not be null");
  RAFT_EXPECTS(desc.payload != nullptr, "device UDF payload must not be null");
  RAFT_EXPECTS(desc.payload_size > 0, "device UDF payload_size must be non-zero");
  RAFT_EXPECTS(desc.symbol_name != nullptr, "device UDF symbol_name must not be null");
  RAFT_EXPECTS(desc.cache_key != nullptr, "device UDF cache_key must not be null");
  RAFT_EXPECTS(desc.flags == 0, "device UDF flags must be zero");
  RAFT_EXPECTS(desc.n_captures == 0 || desc.captures != nullptr,
               "device UDF captures must not be null when n_captures is non-zero");

  auto const* payload_begin = static_cast<std::uint8_t const*>(desc.payload);
  auto out                 = device_udf{.abi          = std::string{desc.abi},
                                        .payload_kind = payload_kind_from_c(desc.payload_kind),
                                        .payload      = std::vector<std::uint8_t>{
                                          payload_begin, payload_begin + desc.payload_size},
                                        .symbol_name  = std::string{desc.symbol_name},
                                        .cache_key    = std::string{desc.cache_key}};

  out.captures.reserve(desc.n_captures);
  for (size_t i = 0; i < desc.n_captures; ++i) {
    auto const& capture = desc.captures[i];
    RAFT_EXPECTS(capture.name != nullptr, "device UDF capture name must not be null");
    RAFT_EXPECTS(capture.dtype != nullptr, "device UDF capture dtype must not be null");
    RAFT_EXPECTS(capture.ndim >= 0, "device UDF capture ndim must be non-negative");
    RAFT_EXPECTS(capture.ndim == 0 || capture.shape != nullptr,
                 "device UDF capture shape must not be null when ndim is non-zero");
    RAFT_EXPECTS(capture.pointer != 0, "device UDF capture pointer must not be zero");
    RAFT_EXPECTS((capture.flags & ~CUVS_UDF_CAPTURE_READONLY) == 0,
                 "device UDF capture has unsupported flags");

    auto next = udf_capture{.name      = std::string{capture.name},
                            .dtype     = std::string{capture.dtype},
                            .device_id = capture.device_id,
                            .pointer   = capture.pointer,
                            .readonly  = (capture.flags & CUVS_UDF_CAPTURE_READONLY) != 0};

    auto const ndim = static_cast<size_t>(capture.ndim);
    if (ndim > 0) { next.shape.assign(capture.shape, capture.shape + ndim); }
    if (capture.strides != nullptr) {
      next.strides.assign(capture.strides, capture.strides + ndim);
    }
    out.captures.push_back(std::move(next));
  }

  return out;
}

}  // namespace cuvs::jit
