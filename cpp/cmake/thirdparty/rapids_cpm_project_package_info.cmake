# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

set_property(
  GLOBAL PROPERTY cuvs_cpm_project_versions_file
                  "${CMAKE_CURRENT_LIST_DIR}/rapids-cpm-versions.json"
)

# Resolve project-owned defaults through rapids-cmake so a parent build can
# replace a package source with RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE.
macro(cuvs_cpm_project_package_info)
  get_property(_cuvs_cpm_project_versions_file GLOBAL PROPERTY cuvs_cpm_project_versions_file)
  include("${rapids-cmake-dir}/cpm/package_override.cmake")
  rapids_cpm_package_override("${_cuvs_cpm_project_versions_file}")
  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(${ARGV})
endmacro()
