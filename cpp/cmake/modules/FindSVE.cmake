# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
# FindSVE.cmake This module finds the SVE (Scalable Vector Extension) support in the compiler.
# =============================================================================

INCLUDE(CheckCXXSourceRuns)

set(SVE_CODE
    "
  #include <arm_sve.h>

  int main()
  {
    svfloat32_t a = svdup_f32(0);
    return 0;
  }
"
)

# Check for SVE support
message("Checking for SVE support")
set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
set(CMAKE_REQUIRED_FLAGS "-march=armv9-a+sve")
check_cxx_source_runs("${SVE_CODE}" CXX_HAS_SVE)
set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

if(CXX_HAS_SVE)
  set(CXX_SVE_FOUND
      TRUE
      CACHE BOOL "SVE support found"
  )
  set(CXX_SVE_FLAGS
      "-march=armv9-a+sve"
      CACHE STRING "Flags for SVE support"
  )
else()
  set(CXX_SVE_FOUND
      FALSE
      CACHE BOOL "SVE support not found"
  )
  set(CXX_SVE_FLAGS
      ""
      CACHE STRING "Flags for SVE support"
  )
endif()

mark_as_advanced(CXX_SVE_FOUND CXX_SVE_FLAGS)
