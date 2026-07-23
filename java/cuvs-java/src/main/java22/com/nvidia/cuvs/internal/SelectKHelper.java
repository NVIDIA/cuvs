/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCUDA;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLFloat;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLInt;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

/**
 * Panama FFI binding for {@code cuvsSelectK}.
 *
 * <p>Selects the k smallest float values from a flat device array of n candidates, writing output
 * distances and their flat-array positions (int64) into caller-supplied device buffers.
 */
public class SelectKHelper {

  private static final MethodHandle cuvsSelectK$mh;

  static {
    var linker = Linker.nativeLinker();
    SymbolLookup lookup =
        SymbolLookup.libraryLookup(System.mapLibraryName("cuvs_c"), Arena.ofAuto())
            .or(SymbolLookup.loaderLookup())
            .or(linker.defaultLookup());

    cuvsSelectK$mh =
        linker.downcallHandle(
            lookup
                .find("cuvsSelectK")
                .orElseThrow(() -> new UnsatisfiedLinkError("cuvsSelectK not found in libcuvs_c")),
            FunctionDescriptor.of(
                C_INT, // return: cuvsError_t
                C_LONG, // cuvsResources_t res
                C_POINTER, // DLManagedTensor* in_val
                C_POINTER, // DLManagedTensor* out_val
                C_POINTER // DLManagedTensor* out_idx
                ));
  }

  private SelectKHelper() {}

  /**
   * Selects the {@code k} smallest distances from a flat device array of {@code n} candidates.
   *
   * <p>Output positions in {@code outIdxDP} are int64 column indices into [0, n). The caller
   * recovers per-segment identity as {@code segment = position / segmentK}.
   *
   * @param cuvsRes    cuvsResources_t handle (raw long)
   * @param inValDP    device pointer to float[n] input distances
   * @param n          number of input candidates
   * @param outValDP   device pointer to float[k] output distances
   * @param outIdxDP   device pointer to int64[k] output positions
   * @param k          number of results to select
   */
  public static void selectK(
      long cuvsRes,
      MemorySegment inValDP,
      long n,
      MemorySegment outValDP,
      MemorySegment outIdxDP,
      long k) {
    try (var arena = Arena.ofConfined()) {
      long[] inShape = {1, n};
      long[] outShape = {1, k};

      MemorySegment inValTensor = prepareTensor(arena, inValDP, inShape, kDLFloat(), 32, kDLCUDA());
      MemorySegment outValTensor =
          prepareTensor(arena, outValDP, outShape, kDLFloat(), 32, kDLCUDA());
      MemorySegment outIdxTensor =
          prepareTensor(arena, outIdxDP, outShape, kDLInt(), 64, kDLCUDA());

      int rc = (int) cuvsSelectK$mh.invokeExact(cuvsRes, inValTensor, outValTensor, outIdxTensor);
      checkCuVSError(rc, "cuvsSelectK");
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new RuntimeException("cuvsSelectK failed", t);
    }
  }
}
