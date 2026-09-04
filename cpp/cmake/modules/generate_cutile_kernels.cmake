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

function(_cutile_kernels_setup)
  set(options)
  set(one_value MATRIX_JSON_FILE OUTPUT_DIRECTORY)
  set(multi_value)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

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

  cuvs_find_build_python(Python3_EXECUTABLE)

  find_program(
    CUTILE_BIN2C
    NAMES bin2c
    PATHS ${CUDAToolkit_BIN_DIR} REQUIRED
  )

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import cuda.tile"
    RESULT_VARIABLE _cutile_import_result
    ERROR_VARIABLE _cutile_import_error
    OUTPUT_QUIET ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _cutile_import_result EQUAL 0)
    message(
      FATAL_ERROR
        "cuda.tile (cuTile Python) is required to build cuTile embedded kernels. "
        "Install cutile-python and cuda-tileiras (conda), or cuda-tile[tileiras] (pip).\n"
        "Interpreter: ${Python3_EXECUTABLE}\n"
        "Import error: ${_cutile_import_error}"
    )
  endif()
  message(STATUS "Using cuTile Python: ${Python3_EXECUTABLE}")

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
  if(DEFINED bytecode_version AND NOT "${bytecode_version}" STREQUAL "")
    list(APPEND _python_args --bytecode-version "${bytecode_version}")
  endif()
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
                FRAGMENT_TAG_FORMAT_CUBIN FRAGMENT_TAG_FORMAT_TILEIR MATRIX_JSON_ENTRY
  )
  set(multi_value FRAGMENT_TAG_HEADER_FILES)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT Python3_EXECUTABLE)
    cuvs_find_build_python(Python3_EXECUTABLE)
  endif()

  populate_matrix_variables("${_CUTILE_MATRIX_JSON_ENTRY}")

  if(register STREQUAL "cubin")
    string(CONFIGURE "${_CUTILE_FRAGMENT_TAG_FORMAT_CUBIN}" fragment_tag @ONLY)
    set(bin2c_symbol embedded_cubin)
    set(fragment_entry_type "cuvs::detail::jit_lto::StaticCubinFragmentEntry<fragment_tag>")
  elseif(register STREQUAL "tileir")
    string(CONFIGURE "${_CUTILE_FRAGMENT_TAG_FORMAT_TILEIR}" fragment_tag @ONLY)
    set(bin2c_symbol embedded_tileir)
    set(fragment_entry_type
        "cuvs::detail::jit_lto::StaticTileIrBytecodeFragmentEntry<fragment_tag>"
    )
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

  _cutile_make_python_args(_python_args)

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
    message(FATAL_ERROR "generate_cutile_kernels: KERNEL_PYTHON is required")
  endif()

  _cutile_kernels_setup(
    MATRIX_JSON_FILE "${_CUTILE_MATRIX_JSON_FILE}" OUTPUT_DIRECTORY "${_CUTILE_OUTPUT_DIRECTORY}"
  )
  if(NOT _CUTILE_SETUP_OK)
    # This function's parent is cpp/CMakeLists.txt. Propagate the disabled feature state there so
    # the compile definition cannot retain a stale value from a previous generator invocation.
    set(CUVS_CUTILE_ENABLED
        0
        PARENT_SCOPE
    )
    set(${source_list_var}
        ""
        PARENT_SCOPE
    )
    return()
  endif()

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
      FRAGMENT_TAG_FORMAT_TILEIR "${_CUTILE_FRAGMENT_TAG_FORMAT_TILEIR}"
      FRAGMENT_TAG_HEADER_FILES ${_CUTILE_FRAGMENT_TAG_HEADER_FILES}
      MATRIX_JSON_ENTRY "${matrix_json_entry}"
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
