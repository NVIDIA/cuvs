#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_glog)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/rapids_cpm_project_package_info.cmake")
    cuvs_cpm_project_package_info(glog FIND_VAR find_args CPM_VAR cpm_args)

    rapids_cpm_find(glog ${PKG_VERSION} ${find_args}
            GLOBAL_TARGETS      glog::glog
            CPM_ARGS ${cpm_args}
            EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
            )

    if(glog_ADDED)
        message(VERBOSE "cuVS: Using glog located in ${glog_SOURCE_DIR}")
    else()
        message(VERBOSE "cuVS: Using glog located in ${glog_DIR}")
    endif()

endfunction()

find_and_configure_glog(VERSION 0.6.0
        FORK             google
        PINNED_TAG       v0.6.0
        EXCLUDE_FROM_ALL ON
        )
