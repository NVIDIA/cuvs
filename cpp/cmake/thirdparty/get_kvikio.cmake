# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Use RAPIDS_VERSION_MAJOR_MINOR from rapids_config.cmake
set(KVIKIO_VERSION "${RAPIDS_VERSION_MAJOR_MINOR}")
set(KVIKIO_FORK "rapidsai")
set(KVIKIO_PINNED_TAG "${rapids-cmake-checkout-tag}")

function(find_and_configure_kvikio)
  set(oneValueArgs VERSION FORK PINNED_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # ----------------------------------------------------------------------------
  # KvikIO provides the GPUDirect Storage (cuFile) and POSIX file I/O backends used by the CAGRA
  # ACE disk-mode build. It is a private, build-time dependency: it is linked into libcuvs via
  # $<BUILD_LOCAL_INTERFACE:kvikio::kvikio> and therefore is NOT part of cuvs' exported interface
  # (no export set / find_dependency is generated). At runtime the conda 'libkvikio' package
  # provides the shared library.
  # ----------------------------------------------------------------------------
  rapids_cpm_find(
    kvikio ${PKG_VERSION}
    GLOBAL_TARGETS kvikio::kvikio
    CPM_ARGS
    EXCLUDE_FROM_ALL TRUE
    GIT_REPOSITORY https://github.com/${PKG_FORK}/kvikio.git
    GIT_TAG ${PKG_PINNED_TAG}
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cpp
    # Force KvikIO checkout to get public headers.
    PATCH_COMMAND ""
    OPTIONS "KvikIO_BUILD_EXAMPLES OFF" "KvikIO_REMOTE_SUPPORT OFF"
  )
endfunction()

# Change pinned tag here to test a commit in CI.
# To use a different KvikIO locally, set the CMake variable CPM_kvikio_SOURCE=/path/to/local/kvikio
find_and_configure_kvikio(
  VERSION ${KVIKIO_VERSION}.00 FORK ${KVIKIO_FORK} PINNED_TAG ${KVIKIO_PINNED_TAG}
)
