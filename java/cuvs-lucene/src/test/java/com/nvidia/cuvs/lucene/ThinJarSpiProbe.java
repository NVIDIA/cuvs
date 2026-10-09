/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Set;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;

/** Fresh-JVM entry point used by {@link ThinJarContentsIT}. */
public final class ThinJarSpiProbe {

  private static final Set<String> CODECS =
      Set.of(
          "CuVS2510GPUSearchCodec",
          "Lucene101AcceleratedHNSWCodec",
          "Lucene101AcceleratedHNSWBinaryQuantizedCodec",
          "Lucene101AcceleratedHNSWScalarQuantizedCodec");
  private static final Set<String> FORMATS =
      Set.of(
          "CuVS2510GPUVectorsFormat",
          "Lucene99HnswVectorsFormat",
          "Lucene99HnswScalarQuantizedVectorsFormat",
          "Lucene99AcceleratedHNSWVectorsFormat",
          "Lucene99AcceleratedHNSWBinaryQuantizedVectorsFormat",
          "Lucene99AcceleratedHNSWScalarQuantizedVectorsFormat");

  private ThinJarSpiProbe() {}

  public static void main(String[] arguments) {
    for (String name : CODECS) {
      if (!Codec.availableCodecs().contains(name) || !Codec.forName(name).getName().equals(name)) {
        throw new AssertionError("Codec SPI did not resolve " + name);
      }
    }
    for (String name : FORMATS) {
      if (!KnnVectorsFormat.availableKnnVectorsFormats().contains(name)
          || !KnnVectorsFormat.forName(name).getName().equals(name)) {
        throw new AssertionError("KnnVectorsFormat SPI did not resolve " + name);
      }
    }
  }
}
