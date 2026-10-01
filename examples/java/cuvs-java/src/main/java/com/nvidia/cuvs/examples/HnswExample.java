/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.examples;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CuVSIvfPqIndexParams;
import com.nvidia.cuvs.CuVSIvfPqParams;
import com.nvidia.cuvs.CuVSIvfPqSearchParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.HnswIndexParams;
import com.nvidia.cuvs.HnswIndexParams.HnswHierarchy;
import com.nvidia.cuvs.HnswQuery;
import com.nvidia.cuvs.HnswSearchParams;
import com.nvidia.cuvs.SearchResults;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.UUID;
import java.util.logging.Logger;

public class HnswExample {

  private static final Logger log = Logger.getLogger(HnswExample.class.getName());

  public static void main(String[] args) throws Throwable {

    // Sample data and query
    float[][] vectors = {
      {0.74021935f, 0.9209938f},
      {0.03902049f, 0.9689629f},
      {0.92514056f, 0.4463501f},
      {0.6673192f, 0.10993068f}
    };

    float[][] queries = {
      {0.48216683f, 0.0428398f},
      {0.5084142f, 0.6545497f},
      {0.51260436f, 0.2643005f},
      {0.05198065f, 0.5789965f}
    };

    try (CuVSResources resources = CuVSResources.create()) {

      // Configure IVF_PQ index parameters (optional - default values set when not defined)
      CuVSIvfPqIndexParams cuVSIvfPqIndexParams = new CuVSIvfPqIndexParams.Builder().build();

      // Configure IVF_PQ search parameters (optional - default values set when not defined)
      CuVSIvfPqSearchParams cuVSIvfPqSearchParams = new CuVSIvfPqSearchParams.Builder().build();

      // Configure IVF_PQ search parameters (used when build algo is IVF_PQ, optional otherwise)
      CuVSIvfPqParams cuVSIvfPqParams =
          new CuVSIvfPqParams.Builder()
              .withCuVSIvfPqIndexParams(cuVSIvfPqIndexParams)
              .withCuVSIvfPqSearchParams(cuVSIvfPqSearchParams)
              .build();

      // Configure index parameters
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.IVF_PQ)
              .withMetric(CuvsDistanceType.L2Expanded)
              .withCuVSIvfPqParams(cuVSIvfPqParams)
              .build();

      // Create the index with the dataset
      CagraIndex index =
          CagraIndex.newBuilder(resources)
              .withDataset(vectors)
              .withIndexParams(indexParams)
              .build();

      // Saving the HNSW index on to the disk.
      String hnswIndexFileName = UUID.randomUUID().toString() + ".hnsw";
      index.serializeToHNSW(new FileOutputStream(hnswIndexFileName));

      // Use NONE hierarchy since serializeToHNSW creates a base-layer-only index
      HnswIndexParams hnswIndexParams =
          new HnswIndexParams.Builder()
              .withVectorDimension(2)
              .withHierarchy(HnswHierarchy.NONE)
              .build();
      InputStream inputStreamHNSW = new FileInputStream(hnswIndexFileName);
      File hnswIndexFile = new File(hnswIndexFileName);

      HnswIndex hnswIndex =
          HnswIndex.newBuilder(resources)
              .from(inputStreamHNSW)
              .withIndexParams(hnswIndexParams)
              .build();

      HnswSearchParams hnswSearchParams = new HnswSearchParams.Builder().build();

      HnswQuery hnswQuery =
          new HnswQuery.Builder(resources)
              .withQueryVectors(queries)
              .withSearchParams(hnswSearchParams)
              .withTopK(3)
              .build();

      SearchResults results = hnswIndex.search(hnswQuery);

      // Check results
      log.info(results.getResults().toString());

      // Cleanup
      if (hnswIndexFile.exists()) {
        hnswIndexFile.delete();
      }
      index.close();
      hnswIndex.close();
    }
  }
}
