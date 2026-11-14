# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

include("${rapids-cmake-dir}/json/array_items.cmake")
include("${rapids-cmake-dir}/json/parse.cmake")
include(${CMAKE_CURRENT_LIST_DIR}/generate_inst_matrix.cmake)

# This function generates every instantiation file described by the `inst_matrix` array of each JSON
# file by calling `generate_inst_matrix` for each of them. The generated files are appended to
# `source_list_var`.
#
# Each entry of `inst_matrix` has the following members, all of which are required:
#
# * `matrix`, `input` and `output_file_format` map to the `MATRIX_JSON_FILE`, `INPUT_FILE` and
#   `OUTPUT_FILE_FORMAT` arguments.
#
# `matrix` and `input` are relative to the directory of the JSON file. The `@name@` placeholders of
# `output_file_format` are replaced by `generate_inst_matrix`, from the matrix and from the variables
# visible to the caller, which is used to refer to `@CMAKE_CURRENT_BINARY_DIR@`.
function(generate_inst_matrix_collection source_list_var)
  set(options)
  set(one_value)
  set(multi_value JSON_FILES)

  cmake_parse_arguments(_INST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  foreach(json_file IN LISTS _INST_JSON_FILES)
    rapids_json_array_items(
      inst_matrix FILE "${json_file}" KEY inst_matrix BASE_DIR_VARIABLE inst_matrix_dir
    )

    foreach(inst_matrix_entry IN LISTS inst_matrix)
      rapids_json_parse(
        _inst_entry
        STRING "${inst_matrix_entry}"
        BASE_DIR "${inst_matrix_dir}"
        VALUES output_file_format
        PATHS matrix input
      )

      foreach(member IN ITEMS matrix input output_file_format)
        if(NOT DEFINED _inst_entry_${member})
          message(FATAL_ERROR "'${json_file}' has an entry without the required '${member}' member")
        endif()
      endforeach()

      generate_inst_matrix(
        ${source_list_var}
        MATRIX_JSON_FILE "${_inst_entry_matrix}"
        INPUT_FILE "${_inst_entry_input}"
        OUTPUT_FILE_FORMAT "${_inst_entry_output_file_format}"
      )
    endforeach()
  endforeach()

  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()
