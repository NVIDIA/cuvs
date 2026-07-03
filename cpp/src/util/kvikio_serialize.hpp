/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/util/file_io.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cstddef>
#include <ostream>
#include <type_traits>
#include <vector>

namespace cuvs::util::detail {

inline constexpr size_t kDeviceSerializationBatchBytes = size_t{64} << 20;

template <typename T>
raft::numpy_serializer::header_t get_numpy_header(const std::vector<size_t>& shape,
                                                  bool fortran_order = false)
{
  const auto dtype = raft::numpy_serializer::get_numpy_dtype<T>();
  return raft::numpy_serializer::header_t{dtype, fortran_order, shape};
}

template <typename T>
void write_numpy_header(std::ostream& os, const std::vector<size_t>& shape)
{
  raft::numpy_serializer::write_header(os, get_numpy_header<T>(shape));
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void serialize_device_mdspan(
  const raft::resources& res,
  kvikio_ofstream& os,
  const raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::layout_c_contiguous> ||
                  std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>,
                "The serializer only supports row-major and column-major layouts");

  using obj_t = raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  std::vector<size_t> shape;
  shape.reserve(obj.rank());
  for (typename obj_t::rank_type i = 0; i < obj.rank(); ++i) {
    shape.push_back(static_cast<size_t>(obj.extent(i)));
  }

  const bool fortran_order = std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>;
  raft::numpy_serializer::write_header(os, get_numpy_header<ElementType>(shape, fortran_order));

  raft::resource::sync_stream(res);
  os.write_device(obj.data_handle(), obj.size() * sizeof(ElementType));
  RAFT_EXPECTS(os.good(), "Error writing content of device mdspan");
}

}  // namespace cuvs::util::detail
