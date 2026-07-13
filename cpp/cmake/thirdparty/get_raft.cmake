# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Use RAPIDS_VERSION_MAJOR_MINOR from rapids_config.cmake
set(RAFT_VERSION "${RAPIDS_VERSION_MAJOR_MINOR}")
set(RAFT_FORK "rapidsai")
set(RAFT_PINNED_TAG "${rapids-cmake-checkout-tag}")
set(LIBRAFT_LOGGING_LEVEL "INFO"
    CACHE STRING "Choose the RAFT logging level compiled into cuVS."
)

function(find_and_configure_raft)
    set(oneValueArgs VERSION FORK PINNED_TAG BUILD_STATIC_DEPS ENABLE_NVTX ENABLE_MNMG_DEPENDENCIES CLONE_ON_PIN)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    # Set BUILD_SHARED_LIBS whenever building static dependencies
    if(PKG_BUILD_STATIC_DEPS)
        set(BUILD_SHARED_LIBS OFF)
    endif()

    # Determine whether to clone raft locally
    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message(STATUS "cuVS: RAFT pinned tag found: ${PKG_PINNED_TAG}. Cloning raft locally.")
        set(CPM_DOWNLOAD_raft ON)
    elseif(PKG_BUILD_STATIC_DEPS AND (NOT CPM_raft_SOURCE))
        message(STATUS "cuVS: Cloning raft locally to build static libraries.")
        set(CPM_DOWNLOAD_raft ON)
    endif()

    set(RAFT_COMPONENTS "")

    if(PKG_ENABLE_MNMG_DEPENDENCIES)
        string(APPEND RAFT_COMPONENTS " distributed")
    endif()

    #-----------------------------------------------------
    # Invoke CPM find_package()
    #-----------------------------------------------------
    rapids_cpm_find(raft ${PKG_VERSION}
            GLOBAL_TARGETS      raft::raft
            BUILD_EXPORT_SET    cuvs-exports
            INSTALL_EXPORT_SET  cuvs-exports
            COMPONENTS          ${RAFT_COMPONENTS}
            CPM_ARGS
              EXCLUDE_FROM_ALL TRUE
              GIT_REPOSITORY        https://github.com/${PKG_FORK}/raft.git
              GIT_TAG               ${PKG_PINNED_TAG}
              SOURCE_SUBDIR         cpp
              OPTIONS
              "BUILD_TESTS OFF"
              "BUILD_PRIMS_BENCH OFF"
              "RAFT_NVTX ${PKG_ENABLE_NVTX}"
              "RAFT_COMPILE_LIBRARY OFF"
            )
endfunction()

# RAFT target is shipped together with the RAFT_LOG_ACTIVE_LEVEL macro. Simply redefining the macro
# would yield a redefine error. We instead query and modify the definition list of the RAFT target manually.
function(overwrite_raft_log_active_level)
    set(oneValueArgs LEVEL)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN})

    get_target_property(_tgt raft::raft ALIASED_TARGET)
    if(NOT _tgt)
        set(_tgt raft::raft)
    endif()
    get_target_property(_defs ${_tgt} INTERFACE_COMPILE_DEFINITIONS)
    if(_defs)
        # Get definitions without previous logging level
        list(FILTER _defs EXCLUDE REGEX "^RAFT_LOG_ACTIVE_LEVEL=")
        # Append the new logging level
        list(APPEND _defs "RAFT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${PKG_LEVEL}")
        # Set the modified definitions back to the raft target
        set_target_properties(${_tgt} PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${_defs}")
        message(STATUS "cuVS: set RAFT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${PKG_LEVEL}")
    endif()
endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION  ${RAFT_VERSION}.00
        FORK                     ${RAFT_FORK}
        PINNED_TAG               ${RAFT_PINNED_TAG}
        ENABLE_MNMG_DEPENDENCIES OFF
        ENABLE_NVTX              OFF
        BUILD_STATIC_DEPS ${CUVS_STATIC_RAPIDS_LIBRARIES}
        # When PINNED_TAG above doesn't match the default rapids branch,
        # force local raft clone in build directory
        # even if it's already installed.
        CLONE_ON_PIN     ${CUVS_RAFT_CLONE_ON_PIN}

)

if(NOT LIBRAFT_LOGGING_LEVEL MATCHES "^(TRACE|DEBUG|INFO|WARN|ERROR|CRITICAL|OFF)$")
    message(FATAL_ERROR "cuVS: invalid LIBRAFT_LOGGING_LEVEL '${LIBRAFT_LOGGING_LEVEL}'. Valid are: TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, OFF")
endif()
overwrite_raft_log_active_level(LEVEL ${LIBRAFT_LOGGING_LEVEL})
