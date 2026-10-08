/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <omp.h>

#include <raft/core/logger.hpp>

#include "omp_wrapper.hpp"

namespace cuvs::core::omp {

constexpr bool is_omp_enabled()
{
#if defined(_OPENMP)
  return true;
#else
  return false;
#endif
}

int get_max_threads()
{
  if constexpr (is_omp_enabled()) { return omp_get_max_threads(); }
  return 1;
}

int get_num_procs()
{
  if constexpr (is_omp_enabled()) { return omp_get_num_procs(); }
  return 1;
}

int get_num_threads()
{
  if constexpr (is_omp_enabled()) { return omp_get_num_threads(); }
  return 1;
}

int get_thread_num()
{
  if constexpr (is_omp_enabled()) { return omp_get_thread_num(); }
  return 0;
}

int get_nested()
{
  if constexpr (is_omp_enabled()) { return omp_get_nested(); }
  return 0;
}

void set_nested(int v)
{
  (void)v;
  if constexpr (is_omp_enabled()) { omp_set_nested(v); }
}

void set_num_threads(int v)
{
  (void)v;
  if constexpr (is_omp_enabled()) { omp_set_num_threads(v); }
}

void check_threads(const int requirements)
{
  const int max_threads = get_max_threads();
  if (max_threads < requirements) {
    RAFT_LOG_WARN(
      "Insufficient OpenMP threads: only %d available but %d required for optimal parallel "
      "execution across GPUs. Please increase OMP_NUM_THREADS to at least %d to avoid potential "
      "performance degradation or synchronization issues.",
      max_threads,
      requirements,
      requirements);
  }
}

}  // namespace cuvs::core::omp
