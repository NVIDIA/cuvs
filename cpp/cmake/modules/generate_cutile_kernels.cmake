# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/compute_matrix_product.cmake)

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

# Probes the toolchain for cuTile support and records the result in the CUVS_CUTILE_* global
# properties. Missing prerequisites disable cuTile instead of failing configuration.
function(_cutile_detect)
  set_property(GLOBAL PROPERTY CUVS_CUTILE_ENABLED 0)

  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_VERSION VERSION_LESS 13.0)
    message(STATUS "cuTile disabled: requires CUDA 13.0+ (found ${CUDAToolkit_VERSION}).")
    return()
  endif()

  find_program(
    CUVS_CUTILE_BIN2C
    NAMES bin2c
    PATHS ${CUDAToolkit_BIN_DIR}
    NO_DEFAULT_PATH
  )
  if(NOT CUVS_CUTILE_BIN2C)
    message(STATUS "cuTile disabled: bin2c not found in ${CUDAToolkit_BIN_DIR}.")
    return()
  endif()

  cuvs_find_build_python(python)
  execute_process(
    COMMAND "${python}" -c "import cuda.tile"
    RESULT_VARIABLE import_result
    ERROR_VARIABLE import_error
    OUTPUT_QUIET ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT import_result EQUAL 0)
    message(
      STATUS "cuTile disabled: cuda.tile (cuTile Python) is not importable by ${python}. "
             "Install cutile-python and cuda-tileiras (conda), or cuda-tile[tileiras] (pip).\n"
             "Import error: ${import_error}"
    )
    return()
  endif()

  message(STATUS "Using cuTile Python: ${python}")
  set_property(GLOBAL PROPERTY CUVS_CUTILE_ENABLED 1)
  set_property(GLOBAL PROPERTY CUVS_CUTILE_PYTHON "${python}")
  set_property(GLOBAL PROPERTY CUVS_CUTILE_BIN2C "${CUVS_CUTILE_BIN2C}")
endfunction()

# Returns whether cuTile can be built, and the python interpreter and bin2c to build it with, in the
# named output variables. The toolchain is only probed on the first call.
function(cuvs_cutile_setup enabled_var python_var bin2c_var)
  get_property(
    probed GLOBAL
    PROPERTY CUVS_CUTILE_ENABLED
    SET
  )
  if(NOT probed)
    _cutile_detect()
  endif()

  get_property(enabled GLOBAL PROPERTY CUVS_CUTILE_ENABLED)
  get_property(python GLOBAL PROPERTY CUVS_CUTILE_PYTHON)
  get_property(bin2c GLOBAL PROPERTY CUVS_CUTILE_BIN2C)
  set(${enabled_var}
      "${enabled}"
      PARENT_SCOPE
  )
  set(${python_var}
      "${python}"
      PARENT_SCOPE
  )
  set(${bin2c_var}
      "${bin2c}"
      PARENT_SCOPE
  )
endfunction()

function(_cutile_make_python_args output_var)
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
  if(DEFINED matrix_layout AND NOT "${matrix_layout}" STREQUAL "")
    list(APPEND _python_args --matrix-layout "${matrix_layout}")
  endif()
  if(DEFINED occupancy AND NOT "${occupancy}" STREQUAL "")
    list(APPEND _python_args --occupancy "${occupancy}")
  endif()
  set(${output_var}
      "${_python_args}"
      PARENT_SCOPE
  )
endfunction()

function(process_cutile_matrix_entry source_list_var)
  set(options)
  set(one_value KERNEL_DIR KERNEL_BASENAME KERNEL_PYTHON EXPORT_SCRIPT OUTPUT_DIRECTORY
                FRAGMENT_TAG_FORMAT_CUBIN MATRIX_JSON_ENTRY PYTHON BIN2C
  )
  set(multi_value FRAGMENT_TAG_HEADER_FILES)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  populate_matrix_variables("${_CUTILE_MATRIX_JSON_ENTRY}")

  if(NOT register STREQUAL "cubin")
    message(FATAL_ERROR "Unknown cuTile register kind '${register}'")
  endif()
  string(CONFIGURE "${_CUTILE_FRAGMENT_TAG_FORMAT_CUBIN}" fragment_tag @ONLY)
  set(bin2c_symbol embedded_cubin)
  set(fragment_entry_type "cuvs::detail::jit_lto::StaticCubinFragmentEntry<fragment_tag>")

  _cutile_fragment_tag_header_files(fragment_tag_header_files ${_CUTILE_FRAGMENT_TAG_HEADER_FILES})

  string(CONFIGURE "${artifact_basename}" _artifact_basename @ONLY)
  set(_artifact_stem "${_CUTILE_KERNEL_BASENAME}_${_artifact_basename}")
  set(_artifact_file "${_CUTILE_OUTPUT_DIRECTORY}/${_artifact_stem}.${artifact_ext}")
  set(_embedded_header "${_CUTILE_OUTPUT_DIRECTORY}/${_artifact_stem}_${register}.h")
  set(_fragment_cpp "${_CUTILE_OUTPUT_DIRECTORY}/${_artifact_stem}_${register}.cpp")
  set(embedded_header_file "${_artifact_stem}_${register}.h")

  _cutile_make_python_args(_python_args)

  set(_export_python_executable "${_CUTILE_PYTHON}")
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
    COMMAND "${_CUTILE_BIN2C}" --const --name ${bin2c_symbol} --static "${_artifact_file}" >
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
                MATRIX_JSON_FILE FRAGMENT_TAG_FORMAT_CUBIN PYTHON BIN2C
  )
  set(multi_value FRAGMENT_TAG_HEADER_FILES)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT _CUTILE_KERNEL_BASENAME)
    message(FATAL_ERROR "generate_cutile_kernels: KERNEL_BASENAME is required")
  endif()
  if(NOT _CUTILE_KERNEL_PYTHON)
    message(FATAL_ERROR "generate_cutile_kernels: KERNEL_PYTHON is required")
  endif()

  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${_CUTILE_MATRIX_JSON_FILE}"
  )
  file(MAKE_DIRECTORY "${_CUTILE_OUTPUT_DIRECTORY}")

  compute_matrix_product(matrix_product MATRIX_JSON_FILE "${_CUTILE_MATRIX_JSON_FILE}")

  string(JSON len LENGTH "${matrix_product}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON matrix_json_entry GET "${matrix_product}" "${i}")
    process_cutile_matrix_entry(
      "${source_list_var}"
      KERNEL_DIR "${_CUTILE_KERNEL_DIR}"
      KERNEL_BASENAME "${_CUTILE_KERNEL_BASENAME}"
      KERNEL_PYTHON "${_CUTILE_KERNEL_PYTHON}"
      EXPORT_SCRIPT "${_CUTILE_EXPORT_SCRIPT}"
      OUTPUT_DIRECTORY "${_CUTILE_OUTPUT_DIRECTORY}"
      FRAGMENT_TAG_FORMAT_CUBIN "${_CUTILE_FRAGMENT_TAG_FORMAT_CUBIN}"
      FRAGMENT_TAG_HEADER_FILES ${_CUTILE_FRAGMENT_TAG_HEADER_FILES}
      MATRIX_JSON_ENTRY "${matrix_json_entry}" PYTHON "${_CUTILE_PYTHON}" BIN2C "${_CUTILE_BIN2C}"
    )
  endforeach()

  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()
