# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

function(cuvs_find_build_python output_var)
  # cuTile is a build dependency. In conda builds, it is installed in BUILD_PREFIX while CMake's
  # default search can resolve the host interpreter from PREFIX instead. Use the build prefix so
  # configure-time matrix expansion and build-time kernel exports see the cuTile package.
  if(DEFINED ENV{BUILD_PREFIX})
    set(Python_ROOT "$ENV{BUILD_PREFIX}")
  endif()
  find_package(Python REQUIRED COMPONENTS Interpreter)
  set(${output_var}
      "${Python_EXECUTABLE}"
      PARENT_SCOPE
  )
endfunction()

function(compute_matrix_product output_var)
  set(options)
  set(one_value MATRIX_JSON_FILE MATRIX_JSON_STRING)
  set(multi_value)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  cuvs_find_build_python(_matrix_python_executable)

  if(_JIT_LTO_MATRIX_JSON_FILE)
    execute_process(
      COMMAND
        "${_matrix_python_executable}"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/compute_matrix_product.py"
        "${_JIT_LTO_MATRIX_JSON_FILE}"
      OUTPUT_VARIABLE output COMMAND_ERROR_IS_FATAL ANY
    )
  else()
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E echo "${_JIT_LTO_MATRIX_JSON_STRING}"
      COMMAND "${_matrix_python_executable}"
              "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/compute_matrix_product.py" -
      OUTPUT_VARIABLE output COMMAND_ERROR_IS_FATAL ANY
    )
  endif()

  set(${output_var}
      "${output}"
      PARENT_SCOPE
  )
endfunction()

function(populate_matrix_variables matrix_json_entry)
  string(JSON len LENGTH "${matrix_json_entry}")
  if(len EQUAL 0)
    return()
  endif()
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON key MEMBER "${matrix_json_entry}" "${i}")
    string(JSON value GET "${matrix_json_entry}" "${key}")
    set(${key}
        "${value}"
        PARENT_SCOPE
    )
  endforeach()
endfunction()
