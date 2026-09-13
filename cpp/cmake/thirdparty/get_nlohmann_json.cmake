#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_nlohmann_json)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/rapids_cpm_project_package_info.cmake")
    cuvs_cpm_project_package_info(nlohmann_json FIND_VAR find_args CPM_VAR cpm_args)

    rapids_cpm_find(nlohmann_json ${PKG_VERSION} ${find_args}
            GLOBAL_TARGETS      nlohmann_json::nlohmann_json
            CPM_ARGS ${cpm_args}
            EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
            )

    if(nlohmann_json_ADDED)
        message(VERBOSE "cuVS: Using nlohmann_json located in ${nlohmann_json_SOURCE_DIR}")
    else()
        message(VERBOSE "cuVS: Using nlohmann_json located in ${nlohmann_json_DIR}")
    endif()

endfunction()

find_and_configure_nlohmann_json(VERSION  3.12.0
        FORK             nlohmann
        PINNED_TAG       v3.12.0
        EXCLUDE_FROM_ALL ON
        )
