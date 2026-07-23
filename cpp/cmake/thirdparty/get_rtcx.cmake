#==============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================-

# This function finds rtcx
function(find_and_configure_rtcx VERSION)

  # Ensure rtcx installs its targets unconditionally. In shared builds the static library is
  # absorbed into libcuvs.so, but CMake still requires the target to be in an export set for
  # install(EXPORT) validation. In static builds consumers need to link librtcx.a directly.
  set(RTCX_INSTALL ON)

  include("${rapids-cmake-dir}/cmake/default_install_component.cmake")
  rapids_cmake_default_install_component(DEFAULT_USE_PROJECT_NAME)

  rapids_cpm_find(
    rtcx ${VERSION}
    GLOBAL_TARGETS rtcx::rtcx
    BUILD_EXPORT_SET    cuvs-static-exports
    INSTALL_EXPORT_SET  cuvs-static-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/arhag23/librtcx.git
    GIT_TAG 2a3e06f93d4123f3cd0546dfd11e40348f0d23a8
    GIT_SHALLOW FALSE
  )

  # When CPM fetches from source (add_subdirectory), embed.cmake is not auto-included. Include it
  # explicitly so add_embed/embed_includes/embed functions are available.
  if(rtcx_ADDED OR DEFINED CPM_rtcx_SOURCE)
    include("${rtcx_SOURCE_DIR}/generate_jit_lto_kernels.cmake")
  endif()
endfunction()

set(RTCX_MIN_VERSION_cuvs "0.1")
find_and_configure_rtcx(${RTCX_MIN_VERSION_cuvs})
