# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Set the list of Cython files to build
set(cython_sources cydlpack.pyx exceptions.pyx resources.pyx mg_resources.pyx
                   device_tensor_view.pyx dataset.pyx
)
set(linked_libraries cuvs::cuvs cuvs::c_api)

# Build all of the Cython targets
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX common_
)
