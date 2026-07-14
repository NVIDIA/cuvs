/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.spi;

import com.nvidia.cuvs.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.time.Duration;
import java.util.logging.Level;

/**
 * A provider that unconditionally throws UnsupportedOperationException.
 */
final class UnsupportedProvider implements CuVSProvider {

  private final String reasons;

  public UnsupportedProvider(String reasons) {
    this.reasons = reasons;
  }

  @Override
  public CuVSResources newCuVSResources(Path tempDirectory) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSResources newCuVSResources(
      Path tempDirectory,
      Path memoryTrackingCsvPath,
      Duration memoryTrackingSampleInterval) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public HnswIndex hnswIndexFromCagra(HnswIndexParams hnswParams, CagraIndex cagraIndex)
      throws Throwable {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public HnswIndex hnswIndexBuild(
      CuVSResources resources, HnswIndexParams hnswParams, CuVSMatrix dataset) throws Throwable {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] indexes) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long dimensions, CuVSMatrix.DataType dataType) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long columns, int rowStride, int columnStride, CuVSMatrix.DataType dataType) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources cuVSResources, long size, long dimensions, CuVSMatrix.DataType dataType) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public GPUInfoProvider gpuInfoProvider() {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CagraIndexParams cagraIndexParamsFromHnswParams(
      long rows,
      long dim,
      int m,
      int efConstruction,
      CagraIndexParams.HnswHeuristicType heuristic,
      CagraIndexParams.CuvsDistanceType metric) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public void setLogLevel(Level level) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public Level getLogLevel() {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public void enableRMMPooledMemory(int initialPoolSizePercent, int maxPoolSizePercent) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public void enableRMMManagedPooledMemory(int initialPoolSizePercent, int maxPoolSizePercent) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public void resetRMMPooledMemory() {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources cuVSResources,
      long size,
      long dimensions,
      int rowStride,
      int columnStride,
      CuVSMatrix.DataType dataType) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public MethodHandle newNativeMatrixBuilder() {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public MethodHandle newNativeMatrixBuilderWithStrides() {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(float[][] vectors) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(int[][] vectors) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(byte[][] vectors) {
    throw new UnsupportedOperationException(reasons);
  }
}
