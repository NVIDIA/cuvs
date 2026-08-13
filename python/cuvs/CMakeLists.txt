# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

# We always need CUDA for cuvs because the cuvs dependency brings in a header-only cuco dependency
# that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(cuvs_py)

project(
  cuvs_py
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

# ##################################################################################################
# * Process User Options  ------------------------------------------------------

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cython-core)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()

# --- cuVS ---#
find_package(cuvs "${RAPIDS_VERSION}" REQUIRED COMPONENTS cuvs_cpp_headers cuvs cuvs_shared c_api)

# --- dlpack ---#
include(../../cpp/cmake/thirdparty/get_dlpack.cmake)

# ensure Cython targets can find dlpack headers (these do not come installed with with cuVS)
target_include_directories(cuvs::cuvs INTERFACE "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")

# ##################################################################################################
# * Build Cython artifacts -----------------------------------------------------

rapids_cython_init()

add_subdirectory(cuvs/common)
add_subdirectory(cuvs/cluster)
add_subdirectory(cuvs/distance)
add_subdirectory(cuvs/neighbors)
add_subdirectory(cuvs/preprocessing)
