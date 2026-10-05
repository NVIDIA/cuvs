#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Report the peak RMM device memory usage of each libcuvs gtest as CSV.
# Run from the build directory (the one containing gtests/). Set GTEST_CUVS_RMM_MODE
# (async or cuda, default async) to run the tests on a different RMM memory resource.

export GTEST_CUVS_MEMORY_PEAK=1
export GTEST_BRIEF=1
for gt in gtests/*_TEST ; do
  test_name=$(basename "${gt}")
  echo -n "$test_name"
  # dependent on the string output from cpp/tests/test_main.cpp
  bytes=$(${gt} 2>/dev/null | grep Peak | cut -d' ' -f4)
  echo ",${bytes}"
done
unset GTEST_BRIEF
unset GTEST_CUVS_MEMORY_PEAK
