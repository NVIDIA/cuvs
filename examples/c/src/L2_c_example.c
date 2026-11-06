/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include <cuvs/distance/pairwise_distance.h>
#include <cuvs/neighbors/cagra.h>

#include <dlpack/dlpack.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

#define DIM    4
#define N_ROWS 1

float PointA[N_ROWS][DIM] = {{1.0, 2.0, 3.0, 4.0}};
float PointB[N_ROWS][DIM] = {{2.0, 3.0, 4.0, 5.0}};

cuvsResources_t res;

void outputVector(float* Vec)
{
  printf("Vector is ");
  for (int i = 0; i < DIM; ++i) {
    printf(" %f", Vec[i]);
  }
  printf("\n");
}

/**
 * @brief Initialize Tensor.
 *
 * @param[in] x_d Pointer to a vector
 * @param[in] x_shape[] Two-dimensional array, which stores the number of rows
 * and columns of vectors.
 * @param[out] x_tensor Stores the initialized DLManagedTensor.
 */
void tensor_initialize(float* x_d, int64_t x_shape[2], DLManagedTensor* x_tensor)
{
  x_tensor->dl_tensor.data               = x_d;
  x_tensor->dl_tensor.device.device_type = kDLCUDA;
  x_tensor->dl_tensor.ndim               = 2;
  x_tensor->dl_tensor.dtype.code         = kDLFloat;
  x_tensor->dl_tensor.dtype.bits         = 32;
  x_tensor->dl_tensor.dtype.lanes        = 1;
  x_tensor->dl_tensor.shape              = x_shape;
  x_tensor->dl_tensor.strides            = NULL;
}

/**
 * @brief Calculate the euclidean distance between two arrays.
 *
 * @param[in] n_cols array length，also the dimension of the vector
 * @param[in] x[] Pointer to a vector
 * @param[in] y[] Pointer to another vector
 * @param[out] ret will store the result about the euclidean distance
 */
void l2_distance_calc(int64_t n_cols, float x[], float y[], float* ret)
{
  float *x_d, *y_d;
  float* distance_d;
  CHECK_CUVS(cuvsRMMAlloc(res, (void**)&x_d, sizeof(float) * N_ROWS * n_cols));
  CHECK_CUVS(cuvsRMMAlloc(res, (void**)&y_d, sizeof(float) * N_ROWS * n_cols));
  CHECK_CUVS(cuvsRMMAlloc(res, (void**)&distance_d, sizeof(float) * N_ROWS * N_ROWS));

  // Use DLPack to represent x[] and y[] as tensors
  CHECK_CUDA(cudaMemcpy(x_d, x, sizeof(float) * N_ROWS * n_cols, cudaMemcpyDefault));
  CHECK_CUDA(cudaMemcpy(y_d, y, sizeof(float) * N_ROWS * n_cols, cudaMemcpyDefault));

  DLManagedTensor x_tensor = {0};
  int64_t x_shape[2]       = {N_ROWS, n_cols};
  tensor_initialize(x_d, x_shape, &x_tensor);

  DLManagedTensor y_tensor = {0};
  int64_t y_shape[2]       = {N_ROWS, n_cols};
  tensor_initialize(y_d, y_shape, &y_tensor);

  DLManagedTensor dist_tensor = {0};
  int64_t distances_shape[2]  = {N_ROWS, N_ROWS};
  tensor_initialize(distance_d, distances_shape, &dist_tensor);

  // metric_arg default value is 2.0,used for Minkowski distance
  CHECK_CUVS(cuvsPairwiseDistance(res, &x_tensor, &y_tensor, &dist_tensor, L2SqrtUnexpanded, 2.0));

  CHECK_CUDA(cudaMemcpy(ret, distance_d, sizeof(float) * N_ROWS * N_ROWS, cudaMemcpyDefault));

  CHECK_CUVS(cuvsRMMFree(res, distance_d, sizeof(float) * N_ROWS * N_ROWS));
  CHECK_CUVS(cuvsRMMFree(res, x_d, sizeof(float) * N_ROWS * n_cols));
  CHECK_CUVS(cuvsRMMFree(res, y_d, sizeof(float) * N_ROWS * n_cols));
}

int euclidean_distance_calculation_example()
{
  // Create a cuvsResources_t object
  cuvsResourcesCreate(&res);

  outputVector((float*)PointA);
  outputVector((float*)PointB);

  float ret;

  l2_distance_calc(DIM, (float*)PointA, (float*)PointB, &ret);
  printf("L2 distance is %f.\n", ret);

  CHECK_CUVS(cuvsResourcesDestroy(res));

  return 0;
}

int main()
{
  euclidean_distance_calculation_example();
  return 0;
}
