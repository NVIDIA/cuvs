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
  # ACE disk-mode build. cuVS consumes it privately, while installed cuVS targets retain it as a
  # link-only dependency. At runtime package managers provide both the shared library and its CMake
  # package configuration.
  # ----------------------------------------------------------------------------
  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/rapids_cpm_project_package_info.cmake")
  cuvs_cpm_project_package_info(kvikio FIND_VAR find_args CPM_VAR cpm_args)
  rapids_cpm_find(
    kvikio ${PKG_VERSION} ${find_args}
    GLOBAL_TARGETS kvikio::kvikio
    BUILD_EXPORT_SET cuvs-exports
    INSTALL_EXPORT_SET cuvs-exports
    CPM_ARGS ${cpm_args}
    EXCLUDE_FROM_ALL TRUE
    OPTIONS
      "KvikIO_BUILD_BENCHMARKS OFF"
      "KvikIO_BUILD_EXAMPLES OFF"
      "KvikIO_BUILD_NSYS_PLUGIN OFF"
      "KvikIO_REMOTE_SUPPORT OFF"
  )
endfunction()

# Change pinned tag here to test a commit in CI.
# To use a different KvikIO locally, set the CMake variable CPM_kvikio_SOURCE=/path/to/local/kvikio
find_and_configure_kvikio(
  VERSION ${KVIKIO_VERSION}.00 FORK ${KVIKIO_FORK} PINNED_TAG ${KVIKIO_PINNED_TAG}
)
