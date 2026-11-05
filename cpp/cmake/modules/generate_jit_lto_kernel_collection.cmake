# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

include("${rapids-cmake-dir}/json/array_items.cmake")
include("${rapids-cmake-dir}/json/parse.cmake")

# This function expands every JIT LTO kernel described by the `kernels` array of each JSON file
# into kernel and fatbin source files by calling `generate_jit_lto_kernels` for each of them.
#
# Each entry of `kernels` has the following members:
#
# * `name_format`, `matrix`, `kernel_input`, `fragment_tag_format` and `output_dir` map to the
#   `NAME_FORMAT`, `MATRIX_JSON_FILE`, `KERNEL_INPUT_FILE`, `FRAGMENT_TAG_FORMAT` and
#   `OUTPUT_DIRECTORY` arguments.
# * `fragment_tag_headers` maps to `FRAGMENT_TAG_HEADER_FILES`.
# * `link_libraries` maps to `KERNEL_LINK_LIBRARIES`, and overrides the libraries provided to this
#   function.
# * `extra_compile_options` maps to `KERNEL_EXTRA_COMPILE_OPTIONS`.
#
# `matrix` and `kernel_input` are relative to the directory of the JSON file. `@name@` placeholders
# are replaced using the variables visible to the caller, which is used to refer to
# `@CMAKE_CURRENT_BINARY_DIR@` in `output_dir` and the namespace variables used by the fragment
# tags. `name_format` and `fragment_tag_format` are replaced by `generate_jit_lto_kernels`, along
# with the placeholders of the matrix. `output_dir` is replaced here, so it can't use the matrix
# placeholders.
function(generate_jit_lto_kernel_collection source_list_var)
  set(options)
  set(one_value)
  set(multi_value JSON_FILES KERNEL_LINK_LIBRARIES)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  foreach(json_file IN LISTS _JIT_LTO_JSON_FILES)
    rapids_json_array_items(jit_lto_kernels FILE "${json_file}" KEY kernels)

    foreach(jit_lto_kernel IN LISTS jit_lto_kernels)
      rapids_json_parse(
        _jit_lto_kernel
        STRING "${jit_lto_kernel}"
        BASE_DIR "${jit_lto_kernels_BASE_DIR}"
        VALUES name_format fragment_tag_format output_dir
        LISTS fragment_tag_headers link_libraries extra_compile_options
        PATHS matrix kernel_input
        REQUIRED name_format matrix kernel_input fragment_tag_format output_dir
      )

      set(link_libraries ${_JIT_LTO_KERNEL_LINK_LIBRARIES})
      if(DEFINED _jit_lto_kernel_link_libraries)
        set(link_libraries ${_jit_lto_kernel_link_libraries})
      endif()

      string(CONFIGURE "${_jit_lto_kernel_output_dir}" _jit_lto_kernel_output_dir @ONLY)

      set(extra_args)
      if(_jit_lto_kernel_extra_compile_options)
        list(APPEND extra_args KERNEL_EXTRA_COMPILE_OPTIONS ${_jit_lto_kernel_extra_compile_options})
      endif()

      generate_jit_lto_kernels(
        ${source_list_var}
        NAME_FORMAT "${_jit_lto_kernel_name_format}"
        MATRIX_JSON_FILE "${_jit_lto_kernel_matrix}"
        KERNEL_INPUT_FILE "${_jit_lto_kernel_kernel_input}"
        FRAGMENT_TAG_FORMAT "${_jit_lto_kernel_fragment_tag_format}"
        FRAGMENT_TAG_HEADER_FILES ${_jit_lto_kernel_fragment_tag_headers}
        OUTPUT_DIRECTORY "${_jit_lto_kernel_output_dir}"
        KERNEL_LINK_LIBRARIES ${link_libraries}
        ${extra_args}
      )
    endforeach()
  endforeach()

  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()
