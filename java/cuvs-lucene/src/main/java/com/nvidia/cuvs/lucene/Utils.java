/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.util.List;

public class Utils {

  static void handleThrowable(Throwable t) throws IOException {
    switch (t) {
      case IOException ioe -> throw ioe;
      case Error error -> throw error;
      case RuntimeException re -> throw re;
      case null, default -> throw new RuntimeException("UNEXPECTED: exception type", t);
    }
  }

  /**
   * A method to build a {@link CuVSMatrix} from a list of float vectors.
   *
   * Note: This could be a memory-intensive operation and should therefore be avoided.
   * Consider using this {@link CuVSMatrix.Builder} instead for copying the vectors without loading them in heap.
   *
   * @param data The float vectors
   * @param dimensions The number float elements in each vector
   * @return an instance of {@link CuVSMatrix}
   */
  static CuVSMatrix createFloatMatrix(List<float[]> data, int dimensions) {
    // Convert List<float[]> to float[][] for the ofArray method
    float[][] vectors = data.toArray(new float[0][]);
    return CuVSMatrix.ofArray(vectors);
  }
}
