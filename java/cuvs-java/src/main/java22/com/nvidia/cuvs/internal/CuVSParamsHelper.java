/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.internal.common.CloseableHandle;
import com.nvidia.cuvs.internal.panama.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * A set of helpers to create various types of native CuVS "params" data structures.
 * The CuVS API encapsulates and hides these data structures behind opaque handles
 * (e.g. {@code cuvsCagraIndexParams_t}, and provides functions to create (allocate)
 * and destroy (free) them.
 * The helpers wrap the Create calls and return a {@link CloseableHandle} to hold the
 * opaque handle (to access the params data structure) and pair them with their respective
 * Destroy calls, so that the params native resources will be cleared when
 * {@link AutoCloseable#close()} is called.
 */
public final class CuVSParamsHelper {

  private CuVSParamsHelper() {}

  /**
   * Allocates and populates a {@code cuvsCagraSearchParams} struct into {@code arena}.
   * The returned segment is valid for the lifetime of {@code arena}.
   */
  public static MemorySegment buildCagraSearchParams(Arena arena, CagraSearchParams params) {
    MemorySegment seg = cuvsCagraSearchParams.allocate(arena);
    cuvsCagraSearchParams.max_queries(seg, params.getMaxQueries());
    cuvsCagraSearchParams.itopk_size(seg, params.getITopKSize());
    cuvsCagraSearchParams.max_iterations(seg, params.getMaxIterations());
    cuvsCagraSearchParams.algo(seg, params.getCagraSearchAlgo().value);
    cuvsCagraSearchParams.team_size(seg, params.getTeamSize());
    cuvsCagraSearchParams.search_width(seg, params.getSearchWidth());
    cuvsCagraSearchParams.min_iterations(seg, params.getMinIterations());
    cuvsCagraSearchParams.thread_block_size(seg, params.getThreadBlockSize());
    cuvsCagraSearchParams.hashmap_mode(seg, params.getHashMapMode().value);
    cuvsCagraSearchParams.hashmap_max_fill_rate(seg, params.getHashMapMaxFillRate());
    cuvsCagraSearchParams.num_random_samplings(seg, params.getNumRandomSamplings());
    cuvsCagraSearchParams.rand_xor_mask(seg, params.getRandXORMask());
    return seg;
  }

  public static CloseableHandle createCagraIndexParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsCagraIndexParams_t);
      checkCuVSError(cuvsCagraIndexParamsCreate(paramsPtrPtr), "cuvsCagraIndexParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsCagraIndexParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsCagraIndexParamsDestroy(paramsPtr), "cuvsCagraIndexParamsDestroy");
        }
      };
    }
  }

  static CloseableHandle createCagraCompressionParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsCagraCompressionParams_t);
      checkCuVSError(
          cuvsCagraCompressionParamsCreate(paramsPtrPtr), "cuvsCagraCompressionParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsCagraCompressionParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(
              cuvsCagraCompressionParamsDestroy(paramsPtr), "cuvsCagraCompressionParamsDestroy");
        }
      };
    }
  }

  public static CloseableHandle createIvfPqIndexParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsIvfPqIndexParams_t);
      checkCuVSError(cuvsIvfPqIndexParamsCreate(paramsPtrPtr), "cuvsIvfPqIndexParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsIvfPqIndexParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsIvfPqIndexParamsDestroy(paramsPtr), "cuvsIvfPqIndexParamsDestroy");
        }
      };
    }
  }

  public static CloseableHandle createIvfPqSearchParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsIvfPqSearchParams_t);
      checkCuVSError(cuvsIvfPqSearchParamsCreate(paramsPtrPtr), "cuvsIvfPqSearchParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsIvfPqSearchParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsIvfPqSearchParamsDestroy(paramsPtr), "cuvsIvfPqSearchParamsDestroy");
        }
      };
    }
  }

  public static CloseableHandle createAceParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsAceParams_t);
      checkCuVSError(cuvsAceParamsCreate(paramsPtrPtr), "cuvsAceParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsAceParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsAceParamsDestroy(paramsPtr), "cuvsAceParamsDestroy");
        }
      };
    }
  }

  static CloseableHandle createHnswIndexParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsHnswIndexParams_t);
      checkCuVSError(cuvsHnswIndexParamsCreate(paramsPtrPtr), "cuvsHnswIndexParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsHnswIndexParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsHnswIndexParamsDestroy(paramsPtr), "cuvsHnswIndexParamsDestroy");
        }
      };
    }
  }

  static CloseableHandle createHnswAceParamsNative() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsHnswAceParams_t);
      checkCuVSError(cuvsHnswAceParamsCreate(paramsPtrPtr), "cuvsHnswAceParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsHnswAceParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsHnswAceParamsDestroy(paramsPtr), "cuvsHnswAceParamsDestroy");
        }
      };
    }
  }

  static CloseableHandle createTieredIndexParams() {
    try (var localArena = Arena.ofConfined()) {
      var paramsPtrPtr = localArena.allocate(cuvsTieredIndexParams_t);
      checkCuVSError(cuvsTieredIndexParamsCreate(paramsPtrPtr), "cuvsTieredIndexParamsCreate");
      var paramsPtr = paramsPtrPtr.get(cuvsTieredIndexParams_t, 0L);
      return new CloseableHandle() {
        @Override
        public MemorySegment handle() {
          return paramsPtr;
        }

        @Override
        public void close() {
          checkCuVSError(cuvsTieredIndexParamsDestroy(paramsPtr), "cuvsTieredIndexParamsDestroy");
        }
      };
    }
  }
}
