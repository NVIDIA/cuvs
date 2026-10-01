/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.foreign.*;
import java.util.Locale;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class CuVSMatrixIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(CuVSMatrixIT.class);

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
  }

  private static final float DELTA = 1e-9f;

  private static final byte[][] byteData = {
    {1, 2, 3},
    {0, 2, 3},
    {4, 1, 3},
    {3, 0, 2},
    {0, 4, 2}
  };

  @FunctionalInterface
  interface TestFunction {
    void apply() throws Throwable;
  }

  private static void withPooledMemory(TestFunction testFunction) throws Throwable {
    try {
      CuVSProvider.provider().enableRMMManagedPooledMemory(10, 60);
      testFunction.apply();
    } finally {
      CuVSProvider.provider().resetRMMPooledMemory();
    }
  }

  private void testByteDatasetRowGetAccess(CuVSMatrix dataset) {
    for (int n = 0; n < dataset.size(); ++n) {
      var row = dataset.getRow(n);
      assertEquals(dataset.columns(), row.size());
      for (int i = 0; i < dataset.columns(); ++i) {
        assertEquals(byteData[n][i], row.getAsByte(i));
      }
    }
  }

  @Test
  public void testByteHostDatasetRowGetAccess() {
    try (var matrix = CuVSMatrix.ofArray(byteData)) {
      testByteDatasetRowGetAccess(matrix);
    }
  }

  @Test
  public void testPooledByteDeviceDatasetRowGetAccess() throws Throwable {
    withPooledMemory(this::testByteDeviceDatasetRowGetAccess);
  }

  @Test
  public void testByteDeviceDatasetRowGetAccess() throws Throwable {
    try (var resources = CheckedCuVSResources.create()) {
      var builder =
          CuVSMatrix.deviceBuilder(
              resources, byteData.length, byteData[0].length, CuVSMatrix.DataType.BYTE);
      for (int i = 0; i < byteData.length; i++) {
        builder.addVector(byteData[i]);
      }
      try (var matrix = builder.build()) {
        testByteDatasetRowGetAccess(matrix);
      }
    }
  }

  private void testByteDatasetRowCopy(CuVSMatrix dataset) {
    for (int n = 0; n < dataset.size(); ++n) {
      var row = dataset.getRow(n);
      assertEquals(dataset.columns(), row.size());

      var rowCopy = new byte[(int) row.size()];
      row.toArray(rowCopy);
      assertArrayEquals(byteData[n], rowCopy);
    }
  }

  @Test
  public void testByteHostDatasetRowCopy() {
    try (var dataset = CuVSMatrix.ofArray(byteData)) {
      testByteDatasetRowCopy(dataset);
    }
  }

  @Test
  public void testPooledByteDeviceDatasetRowCopy() throws Throwable {
    withPooledMemory(this::testByteDeviceDatasetRowCopy);
  }

  @Test
  public void testByteDeviceDatasetRowCopy() throws Throwable {
    try (var resources = CheckedCuVSResources.create()) {
      var builder =
          CuVSMatrix.deviceBuilder(
              resources, byteData.length, byteData[0].length, CuVSMatrix.DataType.BYTE);
      for (int i = 0; i < byteData.length; i++) {
        builder.addVector(byteData[i]);
      }
      try (var matrix = builder.build()) {
        testByteDatasetRowCopy(matrix);
      }
    }
  }

  private void testByteDatasetCopy(CuVSMatrix dataset) {
    var dataCopy = new byte[(int) dataset.size()][(int) dataset.columns()];
    dataset.toArray(dataCopy);
    for (int n = 0; n < dataset.size(); ++n) {
      for (int i = 0; i < dataset.columns(); ++i) {
        assertEquals(byteData[n][i], dataCopy[n][i]);
      }
    }
  }

  @Test
  public void testByteHostDatasetCopy() {
    try (var dataset = CuVSMatrix.ofArray(byteData)) {
      testByteDatasetCopy(dataset);
    }
  }

  @Test
  public void testPooledByteDeviceDatasetCopy() throws Throwable {
    withPooledMemory(this::testByteDeviceDatasetCopy);
  }

  @Test
  public void testByteDeviceDatasetCopy() throws Throwable {
    try (var resources = CheckedCuVSResources.create()) {
      var builder =
          CuVSMatrix.deviceBuilder(
              resources, byteData.length, byteData[0].length, CuVSMatrix.DataType.BYTE);
      for (int i = 0; i < byteData.length; i++) {
        builder.addVector(byteData[i]);
      }
      try (var matrix = builder.build()) {
        testByteDatasetCopy(matrix);
      }
    }
  }

  @Test
  public void testIntDatasetRowGetAccess() {
    var intData = createIntMatrix();
    try (var dataset = CuVSMatrix.ofArray(intData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(intData[n][i], row.getAsInt(i));
        }
      }
    }
  }

  @Test
  public void testIntDatasetRowCopy() {
    var intData = createIntMatrix();
    try (var dataset = CuVSMatrix.ofArray(intData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());

        var rowCopy = new int[(int) row.size()];
        row.toArray(rowCopy);
        assertArrayEquals(intData[n], rowCopy);
      }
    }
  }

  @Test
  public void testIntDatasetCopy() {
    var intData = createIntMatrix();
    try (var dataset = CuVSMatrix.ofArray(intData)) {
      var intDataCopy = new int[(int) dataset.size()][(int) dataset.columns()];
      dataset.toArray(intDataCopy);
      assertSame2dArray(dataset.size(), dataset.columns(), intData, intDataCopy);
    }
  }

  @Test
  public void testFloatDatasetRowGetAccess() {
    var floatData = createFloatMatrix();
    try (var dataset = CuVSMatrix.ofArray(floatData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(floatData[n][i], row.getAsFloat(i), DELTA);
        }
      }
    }
  }

  @Test
  public void testFloatDatasetRowCopy() {
    var floatData = createFloatMatrix();
    try (var dataset = CuVSMatrix.ofArray(floatData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());

        var rowCopy = new float[(int) row.size()];
        row.toArray(rowCopy);
        assertArrayEquals(floatData[n], rowCopy, DELTA);
      }
    }
  }

  @Test
  public void testFloatDatasetCopy() {
    var floatData = createFloatMatrix();
    try (var dataset = CuVSMatrix.ofArray(floatData)) {
      var dataCopy = new float[(int) dataset.size()][(int) dataset.columns()];
      dataset.toArray(dataCopy);
      assertSame2dArray(dataset.size(), dataset.columns(), floatData, dataCopy);
    }
  }

  static void assertSame2dArray(long rows, long cols, float[][] array1, float[][] array2) {
    assertEquals(rows, array1.length);
    assertEquals(cols, array1[0].length);
    assertEquals(rows, array2.length);
    assertEquals(cols, array2[0].length);

    for (int n = 0; n < rows; ++n) {
      for (int i = 0; i < cols; ++i) {
        assertEquals(array1[n][i], array2[n][i], DELTA);
      }
    }
  }

  static void assertSame2dArray(long rows, long cols, int[][] array1, int[][] array2) {
    assertEquals(rows, array1.length);
    assertEquals(cols, array1[0].length);
    assertEquals(rows, array2.length);
    assertEquals(cols, array2[0].length);

    for (int n = 0; n < rows; ++n) {
      for (int i = 0; i < cols; ++i) {
        assertEquals(array1[n][i], array2[n][i]);
      }
    }
  }

  private void testFloatDatasetBuilder(int rows, int cols, CuVSMatrix.Builder<?> builder) {

    float[][] data = new float[rows][cols];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        data[r][c] = randomFloat();
      }
    }

    for (int r = 0; r < rows; ++r) {
      builder.addVector(data[r]);
    }

    float[][] roundTripData = new float[rows][cols];

    try (var dataset = builder.build()) {
      dataset.toArray(roundTripData);

      assertSame2dArray(dataset.size(), dataset.columns(), data, roundTripData);
    }
  }

  @Test
  public void testFloatDatasetHostBuilder() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);
    testFloatDatasetBuilder(
        rows, cols, CuVSMatrix.hostBuilder(rows, cols, CuVSMatrix.DataType.FLOAT));
  }

  @Test
  public void testFloatDatasetDeviceBuilder() throws Throwable {
    int rows = randomIntBetween(1, 1024);
    int cols = randomIntBetween(1, 2048);
    try (var resources = CheckedCuVSResources.create()) {
      testFloatDatasetBuilder(
          rows, cols, CuVSMatrix.deviceBuilder(resources, rows, cols, CuVSMatrix.DataType.FLOAT));
    }
  }

  private void testIntDatasetBuilder(int rows, int cols, CuVSMatrix.Builder<?> builder) {

    var data = new int[rows][cols];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        data[r][c] = randomInt();
      }
    }

    for (int r = 0; r < rows; ++r) {
      builder.addVector(data[r]);
    }

    var roundTripData = new int[rows][cols];

    try (var dataset = builder.build()) {
      dataset.toArray(roundTripData);
      assertSame2dArray(dataset.size(), dataset.columns(), data, roundTripData);
    }
  }

  @Test
  public void testIntDatasetHostBuilder() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);
    testIntDatasetBuilder(rows, cols, CuVSMatrix.hostBuilder(rows, cols, CuVSMatrix.DataType.INT));
  }

  @Test
  public void testIntDatasetDeviceBuilder() throws Throwable {
    int rows = randomIntBetween(1, 1024);
    int cols = randomIntBetween(1, 2048);
    try (var resources = CheckedCuVSResources.create()) {
      testIntDatasetBuilder(
          rows, cols, CuVSMatrix.deviceBuilder(resources, rows, cols, CuVSMatrix.DataType.INT));
    }
  }

  private void testByteDatasetBuilder(int rows, int cols, CuVSMatrix.Builder<?> builder) {

    var data = new byte[rows][cols];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        data[r][c] = randomByte();
      }
    }

    for (int r = 0; r < rows; ++r) {
      builder.addVector(data[r]);
    }

    var roundTripData = new byte[rows][cols];

    try (var dataset = builder.build()) {
      dataset.toArray(roundTripData);

      for (int n = 0; n < dataset.size(); ++n) {
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(data[n][i], roundTripData[n][i]);
        }
      }
    }
  }

  @Test
  public void testByteDatasetHostBuilder() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);
    testByteDatasetBuilder(
        rows, cols, CuVSMatrix.hostBuilder(rows, cols, CuVSMatrix.DataType.BYTE));
  }

  @Test
  public void testByteDatasetDeviceBuilder() throws Throwable {
    int rows = randomIntBetween(1, 1024);
    int cols = randomIntBetween(1, 2048);
    try (var resources = CheckedCuVSResources.create()) {
      testByteDatasetBuilder(
          rows, cols, CuVSMatrix.deviceBuilder(resources, rows, cols, CuVSMatrix.DataType.BYTE));
    }
  }

  @Test
  public void testDeviceToHost() throws Throwable {

    final int size = 16 * 1024;
    final int columns = 2048;
    final float[][] data = createFloatMatrix(size, columns);

    try (var resources = CuVSResources.create()) {

      var builder = CuVSMatrix.deviceBuilder(resources, size, columns, CuVSMatrix.DataType.FLOAT);
      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder.addVector(array);
      }

      try (var deviceMatrix = builder.build();
          var hostMatrix = deviceMatrix.toHost()) {
        checkSameData(deviceMatrix, hostMatrix, data, size, columns);
      }
    }
  }

  @Test
  public void testPooledDeviceToHost() throws Throwable {
    withPooledMemory(this::testDeviceToHost);
  }

  @Test
  public void testHostToDevice() throws Throwable {

    final int size = 16 * 1024;
    final int columns = 2048;
    final float[][] data = createFloatMatrix(size, columns);

    try (var resources = CuVSResources.create()) {

      try (var hostMatrix = CuVSMatrix.ofArray(data);
          var deviceMatrix = hostMatrix.toDevice(resources)) {
        checkSameData(hostMatrix, deviceMatrix, data, size, columns);
      }
    }
  }

  @Test
  public void testPooledHostToDevice() throws Throwable {
    withPooledMemory(this::testHostToDevice);
  }

  @Test
  public void testHostToHostReturnsWeakReferenceSameData() {

    final int size = 16 * 1024;
    final int columns = 2048;
    final float[][] data = createFloatMatrix(size, columns);

    try (var hostMatrix = CuVSMatrix.ofArray(data);
        var hostMatrix2 = hostMatrix.toHost()) {
      checkSameData(hostMatrix, hostMatrix2, data, size, columns);
    }
  }

  @Test
  public void testDeviceToDeviceReturnsWeakReferenceSameData() throws Throwable {

    final int size = 16 * 1024;
    final int columns = 2048;
    final float[][] data = createFloatMatrix(size, columns);

    try (var resources = CuVSResources.create()) {

      var builder = CuVSMatrix.deviceBuilder(resources, size, columns, CuVSMatrix.DataType.FLOAT);
      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder.addVector(array);
      }

      try (var deviceMatrix = builder.build();
          var deviceMatrix2 = deviceMatrix.toDevice(resources)) {
        checkSameData(deviceMatrix, deviceMatrix2, data, size, columns);
      }
    }
  }

  @Test
  public void testHostToHostWithDifferentStrides() {

    int size = randomIntBetween(1, 32 * 1024);
    int columns = randomIntBetween(16, 2048);
    int rowStride1 = randomIntBetween(columns, columns * 2);
    int rowStride2 = randomIntBetween(columns, columns * 2);
    final float[][] data = createFloatMatrix(size, columns);

    var builder1 = CuVSMatrix.hostBuilder(size, columns, rowStride1, -1, CuVSMatrix.DataType.FLOAT);
    for (int i = 0; i < size; ++i) {
      var array = data[i];
      builder1.addVector(array);
    }

    var builder2 = CuVSMatrix.hostBuilder(size, columns, rowStride2, -1, CuVSMatrix.DataType.FLOAT);

    try (var matrix1 = builder1.build();
        var matrix2 = builder2.build()) {

      matrix1.toHost(matrix2);
      checkSameData(matrix1, matrix2, data, size, columns);
    }
  }

  @Test
  public void testHostBuilderWithDifferentStrides() {
    int size = randomIntBetween(1, 32 * 1024);
    int columns = randomIntBetween(16, 2048);
    int rowStride1 = randomIntBetween(columns, columns * 2);
    int rowStride2 = randomIntBetween(columns, columns * 2);
    final float[][] data = createFloatMatrix(size, columns);

    var builder1 = CuVSMatrix.hostBuilder(size, columns, rowStride1, -1, CuVSMatrix.DataType.FLOAT);
    for (int i = 0; i < size; ++i) {
      var array = data[i];
      builder1.addVector(array);
    }

    var builder2 = CuVSMatrix.hostBuilder(size, columns, rowStride2, -1, CuVSMatrix.DataType.FLOAT);
    for (int i = 0; i < size; ++i) {
      var array = data[i];
      builder2.addVector(array);
    }

    try (var matrix1 = builder1.build();
        var matrix2 = builder2.build()) {
      checkSameData(matrix1, matrix2, data, size, columns);
    }
  }

  @Test
  public void testDeviceToHostWithDifferentStrides() throws Throwable {

    int size = randomIntBetween(1, 32 * 1024);
    int columns = randomIntBetween(16, 2048);
    int rowStride1 = randomIntBetween(columns, columns * 2);
    int rowStride2 = randomIntBetween(columns, columns * 2);
    final float[][] data = createFloatMatrix(size, columns);

    try (var resources = CuVSResources.create()) {
      var builder1 =
          CuVSMatrix.deviceBuilder(
              resources, size, columns, rowStride1, -1, CuVSMatrix.DataType.FLOAT);
      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder1.addVector(array);
      }

      var builder2 =
          CuVSMatrix.hostBuilder(size, columns, rowStride2, -1, CuVSMatrix.DataType.FLOAT);

      try (var matrix1 = builder1.build();
          var matrix2 = builder2.build()) {

        matrix1.toHost(matrix2);
        checkSameData(matrix1, matrix2, data, size, columns);
      }
    }
  }

  @Test
  public void testDeviceToDeviceWithDifferentStrides() throws Throwable {

    int size = randomIntBetween(1, 32 * 1024);
    int columns = randomIntBetween(16, 2048);
    int rowStride1 = randomIntBetween(columns, columns * 2);
    int rowStride2 = randomIntBetween(columns, columns * 2);
    final float[][] data = createFloatMatrix(size, columns);

    try (var resources = CuVSResources.create()) {
      var builder1 =
          CuVSMatrix.deviceBuilder(
              resources, size, columns, rowStride1, -1, CuVSMatrix.DataType.FLOAT);
      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder1.addVector(array);
      }

      var builder2 =
          CuVSMatrix.deviceBuilder(
              resources, size, columns, rowStride2, -1, CuVSMatrix.DataType.FLOAT);

      try (var matrix1 = builder1.build();
          var matrix2 = builder2.build()) {

        matrix1.toDevice(matrix2, resources);
        checkSameData(matrix1, matrix2, data, size, columns);
      }
    }
  }

  private static void checkSameData(
      CuVSMatrix matrix1, CuVSMatrix matrix2, float[][] data, int size, int columns) {
    assertEquals(data.length, matrix1.size());
    assertEquals(data[0].length, matrix1.columns());

    assertEquals(matrix1.size(), matrix2.size());
    assertEquals(matrix1.columns(), matrix2.columns());

    var roundTripData1 = new float[size][columns];
    var roundTripData2 = new float[size][columns];

    matrix1.toArray(roundTripData1);
    matrix2.toArray(roundTripData2);

    for (int n = 0; n < matrix2.size(); ++n) {
      for (int i = 0; i < matrix2.columns(); ++i) {
        var diff1 = Math.abs(data[n][i] - roundTripData1[n][i]);
        if (diff1 > DELTA) {
          throw new AssertionError(
              String.format(
                  Locale.ROOT,
                  "Matrix1 mismatch. Expected:<%f> but was:<%f> at row [%d]of[%d], col [%d]of[%d]",
                  data[n][i],
                  roundTripData2[n][i],
                  n,
                  size,
                  i,
                  columns));
        }
        var diff2 = Math.abs(data[n][i] - roundTripData2[n][i]);
        if (diff2 > DELTA) {
          throw new AssertionError(
              String.format(
                  Locale.ROOT,
                  "Matrix2 mismatch. Expected:<%f> but was:<%f> at row [%d]of[%d], col [%d]of[%d]",
                  data[n][i],
                  roundTripData2[n][i],
                  n,
                  size,
                  i,
                  columns));
        }
      }
    }
  }

  @Test
  public void testDeviceBuilderWithDifferentStrides() throws Throwable {

    int size = randomIntBetween(1, 32 * 1024);
    int columns = randomIntBetween(16, 2048);
    int rowStride1 = randomIntBetween(columns, columns * 2);
    int rowStride2 = randomIntBetween(columns, columns * 2);
    final float[][] data = createFloatMatrix(size, columns);

    try (var resources = CuVSResources.create()) {
      var builder1 =
          CuVSMatrix.deviceBuilder(
              resources, size, columns, rowStride1, -1, CuVSMatrix.DataType.FLOAT);
      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder1.addVector(array);
      }

      var builder2 =
          CuVSMatrix.deviceBuilder(
              resources, size, columns, rowStride2, -1, CuVSMatrix.DataType.FLOAT);
      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder2.addVector(array);
      }

      try (var matrix1 = builder1.build();
          var matrix2 = builder2.build()) {
        checkSameData(matrix1, matrix2, data, size, columns);
      }
    }
  }

  @Test
  public void testHostMatrixFromNativeDataset() {
    int size = randomIntBetween(1, 32 * 1024);
    int columns = randomIntBetween(16, 2048);
    final float[][] data = createFloatMatrix(size, columns);

    ValueLayout.OfFloat C_FLOAT =
        (ValueLayout.OfFloat) Linker.nativeLinker().canonicalLayouts().get("float");

    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout((long) size * columns, C_FLOAT);

    try (Arena arena = Arena.ofShared()) {
      MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
      for (int r = 0; r < size; r++) {
        MemorySegment.copy(
            data[r], 0, dataMemorySegment, C_FLOAT, (r * columns * C_FLOAT.byteSize()), columns);
      }

      try (var nativeDataset =
          DatasetHelper.fromMemorySegment(
              dataMemorySegment, size, columns, CuVSMatrix.DataType.FLOAT)) {

        var roundTripData = new float[size][columns];
        nativeDataset.toArray(roundTripData);

        for (int n = 0; n < nativeDataset.size(); ++n) {
          for (int i = 0; i < nativeDataset.columns(); ++i) {
            assertEquals(data[n][i], roundTripData[n][i], DELTA);
          }
        }
      }
    }
  }

  @Test
  public void testHostMatrixFromNativeDatasetWithStride() {
    int size = randomIntBetween(1, 32 * 1024);
    int columns = randomIntBetween(16, 2048);
    int rowStride = randomIntBetween(columns, columns * 2);
    final float[][] data = createFloatMatrix(size, columns);

    ValueLayout.OfFloat C_FLOAT =
        (ValueLayout.OfFloat) Linker.nativeLinker().canonicalLayouts().get("float");

    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout((long) size * rowStride, C_FLOAT);

    try (Arena arena = Arena.ofShared()) {
      MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
      for (int r = 0; r < size; r++) {
        MemorySegment.copy(
            data[r], 0, dataMemorySegment, C_FLOAT, (r * rowStride * C_FLOAT.byteSize()), columns);
      }

      try (var nativeDataset =
          DatasetHelper.fromMemorySegment(
              dataMemorySegment, size, columns, rowStride, -1, CuVSMatrix.DataType.FLOAT)) {

        var roundTripData = new float[size][columns];
        nativeDataset.toArray(roundTripData);

        for (int n = 0; n < nativeDataset.size(); ++n) {
          for (int i = 0; i < nativeDataset.columns(); ++i) {
            assertEquals(data[n][i], roundTripData[n][i], DELTA);
          }
        }
      }
    }
  }
}
