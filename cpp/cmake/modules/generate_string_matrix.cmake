# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/compute_matrix_product.cmake)

function(process_string_matrix_entry string_var)
  set(options)
  set(one_value ITEM_FORMAT GLUE MATRIX_JSON_ENTRY)
  set(multi_value)

  cmake_parse_arguments(_GSM "${options}" "${one_value}" "${multi_value}" ${ARGN})

  populate_matrix_variables("${_GSM_MATRIX_JSON_ENTRY}")
  # It's impossible to pass a semicolon inside `ITEM_FORMAT` due to how CMake divides up arguments,
  # so provide a constant placeholder to insert it into code
  set(semicolon ";")
  string(CONFIGURE "${_GSM_ITEM_FORMAT}" item @ONLY)

  string(APPEND ${string_var} "${_GSM_GLUE}${item}")
  set(${string_var}
      "${${string_var}}"
      PARENT_SCOPE
  )
endfunction()

function(generate_string_matrix string_var)
  set(options)
  set(one_value ITEM_FORMAT GLUE MATRIX_JSON_FILE MATRIX_JSON_STRING)
  set(multi_value)

  cmake_parse_arguments(_GSM "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(_GSM_MATRIX_JSON_FILE)
    set_property(
      DIRECTORY
      PROPERTY CMAKE_CONFIGURE_DEPENDS "${_GSM_MATRIX_JSON_FILE}"
      APPEND
    )
    compute_matrix_product(matrix_product MATRIX_JSON_FILE "${_GSM_MATRIX_JSON_FILE}")
  else()
    compute_matrix_product(matrix_product MATRIX_JSON_STRING "${_GSM_MATRIX_JSON_STRING}")
  endif()

  string(JSON len LENGTH "${matrix_product}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  set(glue)
  foreach(i RANGE "${last}")
    string(JSON matrix_json_entry GET "${matrix_product}" "${i}")
    process_string_matrix_entry(
      "${string_var}" ITEM_FORMAT "${_GSM_ITEM_FORMAT}" GLUE "${glue}" MATRIX_JSON_ENTRY
      "${matrix_json_entry}"
    )
    set(glue "${_GSM_GLUE}")
  endforeach()

  set(${string_var}
      "${${string_var}}"
      PARENT_SCOPE
  )
endfunction()
