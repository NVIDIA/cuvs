# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Use RAPIDS_VERSION_MAJOR_MINOR from cmake/rapids_config.cmake
set(CUVS_VERSION "${RAPIDS_VERSION_MAJOR_MINOR}")
set(CUVS_FORK "rapidsai")
set(CUVS_PINNED_TAG "${rapids-cmake-branch}")

function(find_and_configure_cuvs)
    set(oneValueArgs VERSION FORK PINNED_TAG ENABLE_NVTX BUILD_CUVS_C_LIBRARY)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )


    set(CUVS_COMPONENTS "")
    if(PKG_BUILD_CUVS_C_LIBRARY)
        string(APPEND CUVS_COMPONENTS " c_api")
    endif()
    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../../cpp/cmake/thirdparty/rapids_cpm_project_package_info.cmake")
    cuvs_cpm_project_package_info(cuvs FIND_VAR find_args CPM_VAR cpm_args)
    #-----------------------------------------------------
    # Invoke CPM find_package()
    #-----------------------------------------------------
    rapids_cpm_find(cuvs ${PKG_VERSION} ${find_args}
            GLOBAL_TARGETS      cuvs::cuvs
            BUILD_EXPORT_SET    cuvs-examples-exports
            INSTALL_EXPORT_SET  cuvs-examples-exports
            COMPONENTS ${CUVS_COMPONENTS}
            CPM_ARGS ${cpm_args}
            OPTIONS
            "BUILD_C_LIBRARY ${PKG_BUILD_CUVS_C_LIBRARY}"
            "BUILD_TESTS OFF"
            "CUVS_NVTX ${PKG_ENABLE_NVTX}"
            )
endfunction()

# Change pinned tag here to test a commit in CI
# To use a different CUVS locally, set the CMake variable
# CPM_cuvs_SOURCE=/path/to/local/cuvs
find_and_configure_cuvs(VERSION  ${CUVS_VERSION}.00
        FORK                     ${CUVS_FORK}
        PINNED_TAG               ${CUVS_PINNED_TAG}
        ENABLE_NVTX              OFF
        BUILD_CUVS_C_LIBRARY     ${BUILD_CUVS_C_LIBRARY}
)
