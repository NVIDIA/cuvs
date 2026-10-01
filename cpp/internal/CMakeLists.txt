# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(BUILD_TESTS
   OR BUILD_MICRO_BENCH
   OR BUILD_CUVS_BENCH
)
  add_library(cuvs_internal INTERFACE)
  target_include_directories(
    cuvs_internal INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  )
  target_compile_features(cuvs_internal INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)
endif()
