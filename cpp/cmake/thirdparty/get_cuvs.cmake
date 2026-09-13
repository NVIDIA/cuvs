# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Use RAPIDS_VERSION_MAJOR_MINOR from cmake/rapids_config.cmake
set(CUVS_VERSION "${RAPIDS_VERSION_MAJOR_MINOR}")
set(CUVS_FORK "rapidsai")
set(CUVS_PINNED_TAG "${rapids-cmake-checkout-tag}")

function(find_and_configure_cuvs)
    set(oneValueArgs VERSION FORK PINNED_TAG ENABLE_NVTX CLONE_ON_PIN BUILD_CPU_ONLY BUILD_SHARED_LIBS)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message(STATUS "cuVS: pinned tag found: ${PKG_PINNED_TAG}. Cloning cuVS locally.")
        set(CPM_DOWNLOAD_cuvs ON)
    endif()

    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/rapids_cpm_project_package_info.cmake")
    cuvs_cpm_project_package_info(cuvs FIND_VAR find_args CPM_VAR cpm_args)

    #-----------------------------------------------------
    # Invoke CPM find_package()
    #-----------------------------------------------------
    rapids_cpm_find(cuvs ${PKG_VERSION} ${find_args}
            GLOBAL_TARGETS      cuvs::cuvs
            BUILD_EXPORT_SET    cuvs-bench-exports
            INSTALL_EXPORT_SET  cuvs-bench-exports
            COMPONENTS          cuvs
            CPM_ARGS ${cpm_args}
              OPTIONS
              "BUILD_SHARED_LIBS ${PKG_BUILD_SHARED_LIBS}"
              "BUILD_CPU_ONLY ${PKG_BUILD_CPU_ONLY}"
              "BUILD_TESTS OFF"
              "BUILD_CAGRA_HNSWLIB OFF"
              "CUVS_CLONE_ON_PIN ${PKG_CLONE_ON_PIN}"
            )
endfunction()


# Change pinned tag here to test a commit in CI
# To use a different cuVS locally, set the CMake variable
# CPM_cuvs_SOURCE=/path/to/local/cuvs
find_and_configure_cuvs(VERSION  ${CUVS_VERSION}.00
        FORK                     ${CUVS_FORK}
        PINNED_TAG               ${CUVS_PINNED_TAG}
        ENABLE_NVTX              OFF
        # When PINNED_TAG above doesn't match the default rapids branch,
        # force local cuvs clone in build directory
        # even if it's already installed.
        CLONE_ON_PIN     ${CUVS_CLONE_ON_PIN}
        BUILD_CPU_ONLY ${BUILD_CPU_ONLY}
        BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}
)
