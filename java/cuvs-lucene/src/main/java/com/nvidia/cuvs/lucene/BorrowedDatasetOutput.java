/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.Closeable;
import java.io.IOException;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;

/** Persistence strategy for a bulk build whose vectors live in a borrowed native dataset. */
interface BorrowedDatasetOutput extends Closeable {

  void writeField(
      FieldInfo field,
      ExternalFloat32Dataset dataset,
      int maxDoc,
      DocsWithFieldSet docsWithField,
      AcceleratedHnswGraphOutput graphOutput)
      throws IOException;

  void finish() throws IOException;
}
