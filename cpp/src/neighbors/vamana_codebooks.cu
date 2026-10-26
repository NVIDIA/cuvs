/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vamana.cuh"

namespace cuvs::neighbors::vamana {

#define CUVS_INST_VAMANA_CODEBOOKS(T)                                               \
  auto deserialize_codebooks(const std::string& codebook_prefix, const int dim)     \
    -> cuvs::neighbors::vamana::codebook_params<T>                                  \
  {                                                                                 \
    return cuvs::neighbors::vamana::deserialize_codebooks<T>(codebook_prefix, dim); \
  }

CUVS_INST_VAMANA_CODEBOOKS(float);

#undef CUVS_INST_VAMANA_CODEBOOKS

}  // namespace cuvs::neighbors::vamana
