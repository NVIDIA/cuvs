# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#
add_subdirectory(brute_force)
add_subdirectory(cagra)
add_subdirectory(hnsw)
add_subdirectory(ivf_flat)
add_subdirectory(ivf_pq)
add_subdirectory(ivf_sq)
add_subdirectory(filters)
add_subdirectory(nn_descent)
add_subdirectory(tiered_index)
add_subdirectory(mg)
add_subdirectory(all_neighbors)
add_subdirectory(vamana)

# Set the list of Cython files to build
set(cython_sources refine.pyx)
set(linked_libraries cuvs::cuvs cuvs::c_api)

# Build all of the Cython targets
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX neighbors_refine_
)
