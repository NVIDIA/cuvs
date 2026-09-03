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
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

namespace cuvs::util::detail {

inline constexpr size_t kDeviceSerializationBatchBytes = size_t{64} << 20;

inline std::istream& input_stream(std::istream& is) { return is; }

inline std::istream& input_stream(kvikio_file_reader& reader) { return reader.stream(); }

template <typename T>
raft::numpy_serializer::header_t get_numpy_header(const std::vector<size_t>& shape,
                                                  bool fortran_order = false)
{
  const auto dtype = raft::numpy_serializer::get_numpy_dtype<std::remove_cv_t<T>>();
  return raft::numpy_serializer::header_t{dtype, fortran_order, shape};
}

template <typename T>
void write_numpy_header(std::ostream& os, const std::vector<size_t>& shape)
{
  raft::numpy_serializer::write_header(os, get_numpy_header<T>(shape));
}

template <typename T>
void validate_numpy_header(kvikio_file_reader& reader,
                           const std::vector<size_t>& shape,
                           bool fortran_order = false)
{
  const auto expected = get_numpy_header<T>(shape, fortran_order);
  const auto actual   = raft::numpy_serializer::read_header(reader.stream());
  RAFT_EXPECTS(actual.dtype.to_string() == expected.dtype.to_string(),
               "Unexpected numpy dtype: expected %s, found %s",
               expected.dtype.to_string().c_str(),
               actual.dtype.to_string().c_str());
  RAFT_EXPECTS(actual.fortran_order == expected.fortran_order, "Unexpected numpy memory order");
  RAFT_EXPECTS(actual.shape == expected.shape, "Unexpected numpy array shape");
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

template <typename MdspanT>
void serialize_mdspan(const raft::resources& res, std::ostream& os, const MdspanT& obj)
{
  if constexpr (raft::is_device_mdspan_v<MdspanT>) {
    if (auto* kvikio_stream = dynamic_cast<kvikio_ofstream*>(&os); kvikio_stream != nullptr) {
      return serialize_device_mdspan(res, *kvikio_stream, obj);
    }
  }
  raft::serialize_mdspan(res, os, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void serialize_mdspan(
  const raft::resources& res,
  kvikio_ofstream& os,
  const raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  serialize_device_mdspan(res, os, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void deserialize_device_mdspan(
  const raft::resources& res,
  kvikio_file_reader& reader,
  raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> obj)
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
  validate_numpy_header<ElementType>(reader, shape, fortran_order);

  // KvikIO operates outside the RAFT stream. Ensure stream-ordered allocation and initialization
  // of the destination have completed before handing its pointer to GDS.
  raft::resource::sync_stream(res);
  reader.read_device(obj.data_handle(), obj.size() * sizeof(ElementType));
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void deserialize_mdspan(const raft::resources& res,
                        std::istream& is,
                        raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> obj)
{
  raft::deserialize_mdspan(res, is, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void deserialize_mdspan(const raft::resources& res,
                        kvikio_file_reader& reader,
                        raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> obj)
{
  deserialize_device_mdspan(res, reader, obj);
}

}  // namespace cuvs::util::detail
