/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static java.lang.reflect.Modifier.isPublic;

import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.Arrays;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

/** Keeps external-FBIN build controls on the owned bulk API rather than the generic codec API. */
public class TestBulkExternalFbinApiSurface extends LuceneTestCase {

  @Test
  public void testGenericAcceleratedParamsDoNotExposeBulkStorageControls() {
    for (Method method : AcceleratedHNSWParams.Builder.class.getMethods()) {
      String name = method.getName().toLowerCase();
      assertFalse("generic params must not expose external FBIN: " + method, name.contains("fbin"));
      assertFalse(
          "generic params must not expose external datasets: " + method, name.contains("external"));
      assertFalse(
          "generic params must not expose an exact bulk count: " + method,
          name.contains("numinputvectors"));
    }
  }

  @Test
  public void testExternalBuildIsOnlyExposedByBulkWriter() throws Exception {
    Method external =
        CagraHnswBulkIndexWriter.class.getMethod(
            "indexImmutableFbin",
            ExternalFbinFileRegistry.Registration.class,
            CagraHnswBulkIndexWriter.Config.class,
            ExternalFbinOptions.class);
    Method mapped =
        CagraHnswBulkIndexWriter.class.getMethod(
            "indexMappedFbin", Path.class, CagraHnswBulkIndexWriter.Config.class);

    assertTrue(isPublic(external.getModifiers()));
    assertTrue(isPublic(mapped.getModifiers()));
    assertFalse(
        Arrays.stream(Lucene101AcceleratedHNSWCodec.class.getConstructors())
            .flatMap(constructor -> Arrays.stream(constructor.getParameterTypes()))
            .anyMatch(
                type ->
                    type == ExternalFbinOptions.class
                        || type == ImmutableExternalFbinDataset.class
                        || type == ExternalFloat32Dataset.class));
  }

  @Test
  public void testBorrowedDatasetImplementationTypesAreNotPublicApi() {
    assertFalse(isPublic(ExternalFloat32Dataset.class.getModifiers()));
    assertFalse(isPublic(ImmutableExternalFbinDataset.class.getModifiers()));
    assertFalse(isPublic(ExternalFbinReference.class.getModifiers()));
    assertThrows(
        NoSuchMethodException.class,
        () ->
            ExternalFbinFileRegistry.Registration.class.getMethod(
                "reference", int.class, int.class));
    assertThrows(
        NoSuchMethodException.class,
        () -> ExternalFbinFileRegistry.Registration.class.getMethod("map", int.class, int.class));
  }

  @Test
  public void testExternalOptionsValidateHeadStart() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new ExternalFbinOptions(ExternalFbinBuildValidation.VERIFY_SHA256, -1L));
    assertThrows(NullPointerException.class, () -> new ExternalFbinOptions(null, 0L));
  }
}
