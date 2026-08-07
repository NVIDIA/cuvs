# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/compute_matrix_product.cmake)

function(generate_cutile_kernels_stub)
  set(CUVS_CUTILE_ENABLED
      0
      PARENT_SCOPE
  )
endfunction()

function(_cutile_fragment_tag_header_files output_var)
  set(${output_var} "")
  foreach(_header IN LISTS ARGN)
    if(NOT _header MATCHES "^(\".*\"|<.*>)$")
      set(_header "\"${_header}\"")
    endif()
    string(APPEND ${output_var} "#include ${_header}\n")
  endforeach()
  set(${output_var}
      "${${output_var}}"
      PARENT_SCOPE
  )
endfunction()

function(_cutile_kernels_setup)
  set(options)
  set(one_value MATRIX_JSON_FILE OUTPUT_DIRECTORY)
  set(multi_value)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_package(CUDAToolkit REQUIRED)

  if(CUDAToolkit_VERSION VERSION_LESS 13.0)
    message(
      STATUS
        "cuTile embedded kernels require CUDA 13.0+; skipping cuTile generation (found ${CUDAToolkit_VERSION})."
    )
    set(_CUTILE_SETUP_OK
        FALSE
        PARENT_SCOPE
    )
    return()
  endif()

  find_program(
    CUTILE_BIN2C
    NAMES bin2c
    PATHS ${CUDAToolkit_BIN_DIR} REQUIRED
  )

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import cuda.tile"
    RESULT_VARIABLE _cutile_import_result
    OUTPUT_QUIET ERROR_QUIET
  )
  if(NOT _cutile_import_result EQUAL 0)
    message(
      FATAL_ERROR
        "cuda.tile (cuTile Python) is required to build cuTile embedded kernels. "
        "Install it in the active Python environment, e.g. pip install cuda-tile[tileiras]."
    )
  endif()

  set_property(
    DIRECTORY
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${_CUTILE_MATRIX_JSON_FILE}"
    APPEND
  )

  file(MAKE_DIRECTORY "${_CUTILE_OUTPUT_DIRECTORY}")

  set(Python3_EXECUTABLE
      "${Python3_EXECUTABLE}"
      PARENT_SCOPE
  )
  set(CUTILE_BIN2C
      "${CUTILE_BIN2C}"
      PARENT_SCOPE
  )
  set(_CUTILE_SETUP_OK
      TRUE
      PARENT_SCOPE
  )
endfunction()

macro(_cutile_append_matrix_tile_aliases entry abi_abbrev tile_m tile_n tile_k)
  string(JSON _cutile_export_len LENGTH "${entry}" "_export")
  set(_cutile_export_idx 0)
  while(_cutile_export_idx LESS _cutile_export_len)
    string(JSON _cutile_export_entry GET "${entry}" "_export" "${_cutile_export_idx}")
    string(JSON _cutile_register GET "${_cutile_export_entry}" "register")
    if(_cutile_register STREQUAL "cubin")
      string(JSON _cutile_arch_tag GET "${_cutile_export_entry}" "arch_tag")
      set(_cutile_alias_suffix "${_cutile_arch_tag}_${abi_abbrev}")
    elseif(_cutile_register STREQUAL "tileir")
      set(_cutile_alias_suffix "tileir_${abi_abbrev}")
    else()
      message(FATAL_ERROR "Unknown cuTile register kind '${_cutile_register}'")
    endif()

    set(_cutile_tile_value "${tile_m},${tile_n},${tile_k}")
    if(DEFINED _tile_alias_value_${_cutile_alias_suffix})
      if(NOT "${_tile_alias_value_${_cutile_alias_suffix}}" STREQUAL "${_cutile_tile_value}")
        message(FATAL_ERROR "Conflicting cuTile tile geometry for ${_cutile_alias_suffix}: "
                            "${_tile_alias_value_${_cutile_alias_suffix}} vs ${_cutile_tile_value}"
        )
      endif()
    else()
      set(_tile_alias_value_${_cutile_alias_suffix} "${_cutile_tile_value}")
      string(
        APPEND
        _tile_aliases
        "using fused_1nn_matrix_tile_${_cutile_alias_suffix} = cutile_tile_config<${tile_m}, ${tile_n}, ${tile_k}>;\n"
      )
    endif()
    math(EXPR _cutile_export_idx "${_cutile_export_idx} + 1")
  endwhile()
endmacro()

function(_cutile_generate_matrix_tiles_header header_path matrix_json_file)
  file(READ "${matrix_json_file}" _matrix_json)
  set(_tile_aliases "")
  string(JSON _entry_len LENGTH "${_matrix_json}")
  set(_entry_idx 0)
  while(_entry_idx LESS _entry_len)
    string(JSON _entry GET "${_matrix_json}" "${_entry_idx}")
    string(JSON _entry_tile ERROR_VARIABLE _entry_tile_error GET "${_entry}" "_tile" 0)
    if(NOT _entry_tile_error)
      string(JSON _default_tile_m GET "${_entry_tile}" "tile_m")
      string(JSON _default_tile_n GET "${_entry_tile}" "tile_n")
      string(JSON _default_tile_k GET "${_entry_tile}" "tile_k")
    endif()

    string(JSON _abi_len LENGTH "${_entry}" "_abi")
    set(_abi_idx 0)
    while(_abi_idx LESS _abi_len)
      string(JSON _abi_entry GET "${_entry}" "_abi" "${_abi_idx}")
      string(JSON _abi_abbrev GET "${_abi_entry}" "abi_abbrev")
      string(JSON _tile_m ERROR_VARIABLE _tile_m_error GET "${_abi_entry}" "tile_m")
      string(JSON _tile_n ERROR_VARIABLE _tile_n_error GET "${_abi_entry}" "tile_n")
      string(JSON _tile_k ERROR_VARIABLE _tile_k_error GET "${_abi_entry}" "tile_k")
      if(_tile_m_error
         OR _tile_n_error
         OR _tile_k_error
      )
        if(_entry_tile_error)
          message(FATAL_ERROR "Missing cuTile geometry for ABI ${_abi_abbrev}")
        endif()
        set(_tile_m "${_default_tile_m}")
        set(_tile_n "${_default_tile_n}")
        set(_tile_k "${_default_tile_k}")
      endif()

      _cutile_append_matrix_tile_aliases(
        "${_entry}" "${_abi_abbrev}" "${_tile_m}" "${_tile_n}" "${_tile_k}"
      )
      math(EXPR _abi_idx "${_abi_idx} + 1")
    endwhile()
    math(EXPR _entry_idx "${_entry_idx} + 1")
  endwhile()
  file(
    WRITE "${header_path}"
    "/*
 * Generated from ${matrix_json_file} by generate_cutile_kernels.cmake — do not edit.
 */
#pragma once

#include <cuvs/detail/jit_lto/fused_distance_nn/fused_1nn_fragments.hpp>

namespace cuvs::distance::detail {

${_tile_aliases}

}  // namespace cuvs::distance::detail
"
  )
endfunction()

function(process_cutile_matrix_entry source_list_var)
  set(options)
  set(one_value KERNEL_DIR KERNEL_BASENAME KERNEL_PYTHON EXPORT_SCRIPT OUTPUT_DIRECTORY
                FRAGMENT_TAG_FORMAT_CUBIN FRAGMENT_TAG_FORMAT_TILEIR MATRIX_JSON_ENTRY
  )
  set(multi_value FRAGMENT_TAG_HEADER_FILES)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  populate_matrix_variables("${_CUTILE_MATRIX_JSON_ENTRY}")

  if(register STREQUAL "cubin")
    string(CONFIGURE "${_CUTILE_FRAGMENT_TAG_FORMAT_CUBIN}" fragment_tag @ONLY)
    set(bin2c_symbol embedded_cubin)
    set(fragment_entry_type "StaticCubinFragmentEntry<fragment_tag>")
  elseif(register STREQUAL "tileir")
    string(CONFIGURE "${_CUTILE_FRAGMENT_TAG_FORMAT_TILEIR}" fragment_tag @ONLY)
    set(bin2c_symbol embedded_tileir)
    set(fragment_entry_type "StaticTileIrBytecodeFragmentEntry<fragment_tag>")
  else()
    message(FATAL_ERROR "Unknown cuTile register kind '${register}'")
  endif()

  _cutile_fragment_tag_header_files(fragment_tag_header_files ${_CUTILE_FRAGMENT_TAG_HEADER_FILES})

  string(CONFIGURE "${artifact_basename}" _artifact_basename @ONLY)
  set(_artifact_stem "${_CUTILE_KERNEL_BASENAME}_${_artifact_basename}")
  set(_artifact_file "${_CUTILE_OUTPUT_DIRECTORY}/${_artifact_stem}.${artifact_ext}")
  set(_embedded_header "${_CUTILE_OUTPUT_DIRECTORY}/${_artifact_stem}_${register}.h")
  set(_fragment_cpp "${_CUTILE_OUTPUT_DIRECTORY}/${_artifact_stem}_${register}.cpp")
  set(embedded_header_file "${_artifact_stem}_${register}.h")

  set(_python_args
      --format
      "${output_format}"
      --data-type
      "${data_type}"
      --metric
      "${metric}"
      --index-type
      "${index_type}"
      --tile-m
      "${tile_m}"
      --tile-n
      "${tile_n}"
      --tile-k
      "${tile_k}"
      --gpu-code
      "${gpu_code}"
  )
  if(DEFINED bytecode_version AND NOT "${bytecode_version}" STREQUAL "")
    list(APPEND _python_args --bytecode-version "${bytecode_version}")
  endif()
  if(DEFINED matrix_layout AND NOT "${matrix_layout}" STREQUAL "")
    list(APPEND _python_args --matrix-layout "${matrix_layout}")
  endif()

  set(_export_python_executable "${Python3_EXECUTABLE}")
  if(DEFINED python_executable AND NOT "${python_executable}" STREQUAL "")
    string(CONFIGURE "${python_executable}" _export_python_executable @ONLY)
  endif()

  if(DEFINED prebuilt_artifact AND NOT "${prebuilt_artifact}" STREQUAL "")
    string(CONFIGURE "${prebuilt_artifact}" _prebuilt_artifact @ONLY)
    if(NOT IS_ABSOLUTE "${_prebuilt_artifact}")
      set(_prebuilt_artifact "${_CUTILE_KERNEL_DIR}/${_prebuilt_artifact}")
    endif()
    add_custom_command(
      OUTPUT "${_artifact_file}"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_prebuilt_artifact}" "${_artifact_file}"
      DEPENDS "${_prebuilt_artifact}"
      COMMENT "Copying prebuilt cuTile ${_CUTILE_KERNEL_BASENAME} ${output_format} ${data_type}"
      VERBATIM
    )
  else()
    add_custom_command(
      OUTPUT "${_artifact_file}"
      COMMAND "${_export_python_executable}" "${_CUTILE_KERNEL_DIR}/${_CUTILE_EXPORT_SCRIPT}"
              "${_artifact_file}" ${_python_args}
      WORKING_DIRECTORY "${_CUTILE_KERNEL_DIR}"
      DEPENDS "${_CUTILE_KERNEL_DIR}/${_CUTILE_EXPORT_SCRIPT}"
              "${_CUTILE_KERNEL_DIR}/${_CUTILE_KERNEL_PYTHON}"
      COMMENT "Exporting cuTile ${_CUTILE_KERNEL_BASENAME} ${output_format} ${data_type}"
      VERBATIM
    )
  endif()

  add_custom_command(
    OUTPUT "${_embedded_header}"
    COMMAND "${CUTILE_BIN2C}" --const --name ${bin2c_symbol} --static "${_artifact_file}" >
            "${_embedded_header}"
    DEPENDS "${_artifact_file}"
    VERBATIM
  )

  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/register_cutile_fragment.cpp.in" "${_fragment_cpp}" @ONLY
  )
  list(APPEND ${source_list_var} "${_embedded_header}" "${_fragment_cpp}")
  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()

function(generate_cutile_kernels source_list_var)
  set(options)
  set(one_value KERNEL_DIR KERNEL_BASENAME KERNEL_PYTHON EXPORT_SCRIPT OUTPUT_DIRECTORY
                MATRIX_JSON_FILE FRAGMENT_TAG_FORMAT_CUBIN FRAGMENT_TAG_FORMAT_TILEIR
  )
  set(multi_value FRAGMENT_TAG_HEADER_FILES)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT _CUTILE_KERNEL_BASENAME)
    message(FATAL_ERROR "generate_cutile_kernels: KERNEL_BASENAME is required")
  endif()
  if(NOT _CUTILE_KERNEL_PYTHON)
    set(_CUTILE_KERNEL_PYTHON "fused_1nn_kernel.py")
  endif()

  _cutile_kernels_setup(
    MATRIX_JSON_FILE "${_CUTILE_MATRIX_JSON_FILE}" OUTPUT_DIRECTORY "${_CUTILE_OUTPUT_DIRECTORY}"
  )
  if(NOT _CUTILE_SETUP_OK)
    generate_cutile_kernels_stub()
    set(${source_list_var}
        ""
        PARENT_SCOPE
    )
    return()
  endif()

  compute_matrix_product(matrix_product MATRIX_JSON_FILE "${_CUTILE_MATRIX_JSON_FILE}")

  set(_matrix_tiles_header "${_CUTILE_OUTPUT_DIRECTORY}/fused_1nn_cutile_tiles.hpp")
  _cutile_generate_matrix_tiles_header("${_matrix_tiles_header}" "${_CUTILE_MATRIX_JSON_FILE}")

  string(JSON len LENGTH "${matrix_product}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON matrix_json_entry GET "${matrix_product}" "${i}")
    process_cutile_matrix_entry(
      "${source_list_var}"
      KERNEL_DIR
      "${_CUTILE_KERNEL_DIR}"
      KERNEL_BASENAME
      "${_CUTILE_KERNEL_BASENAME}"
      KERNEL_PYTHON
      "${_CUTILE_KERNEL_PYTHON}"
      EXPORT_SCRIPT
      "${_CUTILE_EXPORT_SCRIPT}"
      OUTPUT_DIRECTORY
      "${_CUTILE_OUTPUT_DIRECTORY}"
      FRAGMENT_TAG_FORMAT_CUBIN
      "${_CUTILE_FRAGMENT_TAG_FORMAT_CUBIN}"
      FRAGMENT_TAG_FORMAT_TILEIR
      "${_CUTILE_FRAGMENT_TAG_FORMAT_TILEIR}"
      FRAGMENT_TAG_HEADER_FILES
      ${_CUTILE_FRAGMENT_TAG_HEADER_FILES}
      MATRIX_JSON_ENTRY
      "${matrix_json_entry}"
    )
  endforeach()

  set(CUVS_CUTILE_ENABLED
      1
      PARENT_SCOPE
  )
  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()
