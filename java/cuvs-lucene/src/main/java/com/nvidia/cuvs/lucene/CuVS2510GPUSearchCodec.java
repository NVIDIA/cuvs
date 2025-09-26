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

import com.nvidia.cuvs.LibraryException;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;

/** CuVS based codec for GPU based vector search
 *
 * @apiNote cuVS serialization formats are in experimental phase and hence backward compatibility cannot be guaranteed.
 *
 * */
public class CuVS2510GPUSearchCodec extends FilterCodec {

  private static final Logger log = Logger.getLogger(CuVS2510GPUSearchCodec.class.getName());
  private static final String NAME = "CuVS2510GPUSearchCodec";

  private static final int DEFAULT_CUVS_WRITER_THREADS = 1;
  private static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  private static final int DEFAULT_GRAPH_DEGREE = 64;
  private static final int DEFAULT_HNSW_LAYERS = 1;
  private static final IndexType DEFAULT_INDEX_TYPE = IndexType.CAGRA;

  private KnnVectorsFormat format;

  public CuVS2510GPUSearchCodec() {
    this(NAME, new Lucene101Codec());
  }

  public CuVS2510GPUSearchCodec(String name, Codec delegate) {
    super(name, delegate);
    try {
      format =
          new CuVS2510GPUVectorsFormat(
              DEFAULT_CUVS_WRITER_THREADS,
              DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
              DEFAULT_GRAPH_DEGREE,
              DEFAULT_HNSW_LAYERS,
              DEFAULT_INDEX_TYPE);
      setKnnFormat(format);
    } catch (LibraryException ex) {
      log.severe("Couldn't load native library, possible classloader issue. " + ex.getMessage());
    }
  }

  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return format;
  }

  public void setKnnFormat(KnnVectorsFormat format) {
    this.format = format;
  }
}
