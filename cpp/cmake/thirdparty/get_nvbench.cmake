# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds or builds NVBench.
function(find_and_configure_nvbench)
  include(${rapids-cmake-dir}/cpm/nvbench.cmake)
  rapids_cpm_nvbench(BUILD_STATIC)
endfunction()

find_and_configure_nvbench()
