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

  rapids_cpm_find(
    rtcx ${VERSION}
    GLOBAL_TARGETS rtcx::rtcx
    BUILD_EXPORT_SET    cuvs-exports
    INSTALL_EXPORT_SET  cuvs-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/arhag23/librtcx.git
    GIT_TAG 9100b080dd3182614c1f85590b7c637bfdaa6483
    GIT_SHALLOW FALSE
    EXCLUDE_FROM_ALL TRUE
  )

  # When CPM fetches from source (add_subdirectory), embed.cmake is not auto-included. Include it
  # explicitly so add_embed/embed_includes/embed functions are available.
  if(rtcx_ADDED OR DEFINED CPM_rtcx_SOURCE)
    include("${rtcx_SOURCE_DIR}/generate_jit_lto_kernels.cmake")
  endif()
endfunction()

set(RTCX_MIN_VERSION_cuvs "0.1")
find_and_configure_rtcx(${RTCX_MIN_VERSION_cuvs})
