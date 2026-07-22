/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// This is a list of required header files that cuvs_c rust/go/java
// will bind to.

#pragma once

#include <cuvs/core/c_config.h>
#include <cuvs/core/export.h>
#include <cuvs/core/device_udf.h>
#include <cuvs/core/c_api.h>

#include <cuvs/cluster/kmeans.h>

#include <cuvs/distance/distance.h>
#include <cuvs/distance/pairwise_distance.h>

#include <cuvs/neighbors/all_neighbors.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/ivf_flat.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/ivf_sq.h>
#include <cuvs/neighbors/nn_descent.h>
#include <cuvs/neighbors/refine.h>
#include <cuvs/neighbors/tiered_index.h>
#include <cuvs/neighbors/vamana.h>

#ifdef CUVS_BUILD_CAGRA_HNSWLIB
  #include <cuvs/neighbors/hnsw.h>
#endif

#ifdef CUVS_BUILD_MG_ALGOS
  #include <cuvs/cluster/mg_kmeans.h>
  #include <cuvs/neighbors/mg_cagra.h>
  #include <cuvs/neighbors/mg_common.h>
  #include <cuvs/neighbors/mg_ivf_flat.h>
  #include <cuvs/neighbors/mg_ivf_pq.h>
#endif

#include <cuvs/preprocessing/pca.h>
#include <cuvs/preprocessing/quantize/binary.h>
#include <cuvs/preprocessing/quantize/pq.h>
#include <cuvs/preprocessing/quantize/scalar.h>
