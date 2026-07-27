/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.panama.headers_h.*;
import static com.nvidia.cuvs.internal.panama.headers_h_1.C_INT;

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.DelegatingScopedAccess;
import com.nvidia.cuvs.internal.common.PinnedMemoryBuffer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;

/**
 * Used for allocating resources for cuVS
 *
 * @since 25.02
 */
public class CuVSResourcesImpl implements CuVSResources {

  private final Path tempDirectory;
  private final long resourceHandle;
  private final ScopedAccess access;
  private final int deviceId;

  private final PinnedMemoryBuffer hostBuffer = new PinnedMemoryBuffer();

  /**
   * Constructor that allocates the resources needed for cuVS
   *
   */
  public CuVSResourcesImpl(Path tempDirectory) {
    this.tempDirectory = tempDirectory;
    try (var localArena = Arena.ofConfined()) {
      var resourcesMemorySegment = localArena.allocate(cuvsResources_t);
      checkCuVSError(cuvsResourcesCreate(resourcesMemorySegment), "cuvsResourcesCreate");
      this.resourceHandle = resourcesMemorySegment.get(cuvsResources_t, 0);
      var deviceIdPtr = localArena.allocate(C_INT);
      checkCuVSError(cuvsDeviceIdGet(resourceHandle, deviceIdPtr), "cuvsDeviceIdGet");
      this.deviceId = deviceIdPtr.get(C_INT, 0);
      this.access = new ScopedAccessWithHostBuffer(resourceHandle, hostBuffer.address());
    }
  }

  /**
   * Constructor that allocates a tracking resources handle. All memory
   * allocations made through this handle are written as CSV samples to
   * {@code memoryTrackingCsvPath} from a background thread, restoring the
   * global memory resources on {@link #close()}.
   *
   * <p>Note: the ~8MB pinned host buffer backing this handle is allocated via a
   * raw {@code cudaMallocHost} (see {@code PinnedMemoryBuffer.createPinnedBuffer()})
   * outside the tracking infrastructure, so it is not reflected in the CSV
   * samples.
   *
   * @param tempDirectory                the temporary directory to use for
   *                                     intermediate operations
   * @param memoryTrackingCsvPath        path to the output CSV file
   *                                     (created/truncated)
   * @param memoryTrackingSampleInterval minimum interval between successive
   *                                     CSV samples
   */
  public CuVSResourcesImpl(
      Path tempDirectory,
      Path memoryTrackingCsvPath,
      Duration memoryTrackingSampleInterval) {
    this.tempDirectory = tempDirectory;
    try (var localArena = Arena.ofConfined()) {
      var resourcesMemorySegment = localArena.allocate(cuvsResources_t);
      byte[] pathBytes =
          memoryTrackingCsvPath.toString().getBytes(StandardCharsets.UTF_8);
      var pathSegment = localArena.allocate(pathBytes.length + 1L);
      MemorySegment.copy(
          pathBytes, 0, pathSegment, ValueLayout.JAVA_BYTE, 0, pathBytes.length);
      pathSegment.set(ValueLayout.JAVA_BYTE, pathBytes.length, (byte) 0);
      long sampleIntervalMs = memoryTrackingSampleInterval.toMillis();
      checkCuVSError(
          cuvsResourcesCreateWithMemoryTracking(
              resourcesMemorySegment, pathSegment, sampleIntervalMs),
          "cuvsResourcesCreateWithMemoryTracking");
      this.resourceHandle = resourcesMemorySegment.get(cuvsResources_t, 0);
      var deviceIdPtr = localArena.allocate(C_INT);
      checkCuVSError(cuvsDeviceIdGet(resourceHandle, deviceIdPtr), "cuvsDeviceIdGet");
      this.deviceId = deviceIdPtr.get(C_INT, 0);
      this.access = new ScopedAccessWithHostBuffer(resourceHandle, hostBuffer.address());
    }
  }

  @Override
  public ScopedAccess access() {
    return this.access;
  }

  @Override
  public int deviceId() {
    return this.deviceId;
  }

  @Override
  public void close() {
    synchronized (this) {
      int returnValue = cuvsResourcesDestroy(resourceHandle);
      checkCuVSError(returnValue, "cuvsResourcesDestroy");
      hostBuffer.close();
    }
  }

  @Override
  public Path tempDirectory() {
    return tempDirectory;
  }

  public static MemorySegment getHostBuffer(ScopedAccess access) {

    while (access instanceof DelegatingScopedAccess delegatingScopedAccess) {
      access = delegatingScopedAccess.inner();
    }

    if (access instanceof ScopedAccessWithHostBuffer withHostBuffer) {
      return withHostBuffer.hostBuffer();
    }

    throw new IllegalArgumentException("Unsupported access type: " + access.getClass().getName());
  }
}
