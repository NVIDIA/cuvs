/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/core/export.hpp>

#include <raft/core/error.hpp>
#include <raft/core/numpy_serializer.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace util {
namespace detail {

/**
 * @brief Return the NumPy dtype descriptor for a cuVS scalar type.
 *
 * @tparam T Scalar type to describe.
 *
 * @return A NumPy dtype descriptor string, for example `"<f2"` for `half`. `half` is handled
 * explicitly to preserve the standard NumPy half descriptor; all other types are delegated to
 * RAFT's public NumPy serializer.
 *
 * @note This helper performs no I/O and has no side effects.
 */
template <typename T>
inline auto numpy_dtype_string() -> std::string
{
  if constexpr (std::is_same_v<T, half>) {
    return "<f2";
  } else {
    return raft::numpy_serializer::get_numpy_dtype<T>().to_string();
  }
}

/**
 * @brief Build a complete NumPy v1 `.npy` header for a known dtype descriptor and shape.
 *
 * @param dtype NumPy dtype descriptor to serialize into the header dictionary, such as `"<f2"` or
 * `"<f4"`.
 * @param shape Row-major array extents to serialize in NumPy shape syntax.
 *
 * @return Serialized NumPy v1 header bytes, including magic string, version, little-endian header
 * length, padded dictionary payload, and trailing newline.
 *
 * @throws raft::exception if the generated header does not fit in the NumPy v1 16-bit header length
 * field.
 *
 * @note This helper does not write to disk. Callers append the returned bytes before array payload
 * data.
 */
inline auto make_numpy_header_from_dtype(const std::string& dtype, const std::vector<size_t>& shape)
  -> std::string
{
  std::stringstream dict;
  dict << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) { dict << ", "; }
    dict << shape[i];
  }
  if (shape.size() == 1) { dict << ","; }
  dict << "), }";

  std::string header = dict.str();
  constexpr size_t preamble_size =
    6 + 2 + sizeof(uint16_t);  // magic string, version, v1 header length
  const size_t remainder = (preamble_size + header.size() + 1) % 16;
  if (remainder != 0) { header.append(16 - remainder, ' '); }
  header.push_back('\n');

  RAFT_EXPECTS(header.size() <= std::numeric_limits<uint16_t>::max(),
               "NumPy v1 header is too large: %zu bytes",
               header.size());

  const auto header_len = static_cast<uint16_t>(header.size());
  std::string result;
  result.reserve(preamble_size + header.size());
  result.append("\x93NUMPY", 6);
  result.push_back(1);
  result.push_back(0);
  result.push_back(static_cast<char>(header_len & 0xff));
  result.push_back(static_cast<char>((header_len >> 8) & 0xff));
  result.append(header);
  return result;
}

/**
 * @brief Build a complete NumPy v1 `.npy` header for a cuVS scalar type and shape.
 *
 * @tparam T Scalar type of the serialized array.
 *
 * @param shape Row-major array extents to serialize in NumPy shape syntax.
 *
 * @return Serialized NumPy v1 header bytes, including magic string, version, little-endian header
 * length, padded dictionary payload, and trailing newline.
 *
 * @throws raft::exception when the generated half-precision header is too large for NumPy v1.
 * Other scalar types use RAFT's public header writer and propagate any exceptions from that path.
 *
 * @note `half` uses cuVS's explicit `"<f2"` dtype mapping; other scalar types preserve RAFT's
 * existing header serialization behavior. This helper performs no file I/O.
 */
template <typename T>
inline auto make_numpy_header_string(const std::vector<size_t>& shape) -> std::string
{
  if constexpr (std::is_same_v<T, half>) {
    return make_numpy_header_from_dtype(numpy_dtype_string<T>(), shape);
  } else {
    const auto dtype                              = raft::numpy_serializer::get_numpy_dtype<T>();
    const bool fortran_order                      = false;
    const raft::numpy_serializer::header_t header = {dtype, fortran_order, shape};

    std::stringstream ss;
    raft::numpy_serializer::write_header(ss, header);
    return ss.str();
  }
}

}  // namespace detail
}  // namespace util
}  // namespace CUVS_EXPORT cuvs
