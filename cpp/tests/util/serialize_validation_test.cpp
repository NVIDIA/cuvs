/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/util/serialize_validation.hpp"

#include <gtest/gtest.h>

namespace cuvs::util {

TEST(SerializeValidation, AcceptsLegacyAndStandardHalfPrefixes)
{
  // Literal bytes from the old and current writers, independent of the dtype helper.
  constexpr char legacy[]   = {'<', 'e', '2', '\0'};
  constexpr char standard[] = {'<', 'f', '2', '\0'};
  EXPECT_TRUE(validate_serialized_dtype<half>(legacy, sizeof(legacy)));
  EXPECT_TRUE(validate_serialized_dtype<half>(standard, sizeof(standard)));
  EXPECT_EQ(detail::numpy_dtype_string<half>(), "<f2");
}

TEST(SerializeValidation, RejectsMalformedHalfPrefixes)
{
  EXPECT_FALSE(validate_serialized_dtype<half>(nullptr, 4));
  EXPECT_FALSE(validate_serialized_dtype<half>("<e2", 3));
  EXPECT_FALSE(validate_serialized_dtype<half>("<f2", 3));
  EXPECT_FALSE(validate_serialized_dtype<half>("<e2x", 4));
  EXPECT_FALSE(validate_serialized_dtype<half>("<f2x", 4));
  EXPECT_FALSE(validate_serialized_dtype<half>(">e2", 4));
  EXPECT_FALSE(validate_serialized_dtype<half>(">f2", 4));
  EXPECT_FALSE(validate_serialized_dtype<half>("<f4", 4));
}

TEST(SerializeValidation, PreservesOtherScalarPrefixes)
{
  EXPECT_TRUE(validate_serialized_dtype<float>("<f4", 4));
  EXPECT_TRUE(validate_serialized_dtype<double>("<f8", 4));
  EXPECT_FALSE(validate_serialized_dtype<float>("<e2", 4));
  EXPECT_FALSE(validate_serialized_dtype<float>("<f2", 4));
  EXPECT_FALSE(validate_serialized_dtype<double>("<e2", 4));
  EXPECT_FALSE(validate_serialized_dtype<double>("<f2", 4));
}

}  // namespace cuvs::util
