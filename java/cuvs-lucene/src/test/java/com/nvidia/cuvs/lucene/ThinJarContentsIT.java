/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.junit.Test;

/**
 * Verifies selected ownership boundaries of the standard thin JAR after Maven has packaged it.
 *
 * <p>The JAR must contain the exact Lucene SPI registrations and provider classes without bundling
 * consumer-supplied Lucene or base cuvs-java dependencies, or test-only PyLucene support.
 */
public class ThinJarContentsIT {

  private static final String THIN_JAR_PROPERTY = "cuvs.lucene.thinJar";
  private static final String CODEC_SERVICE = "META-INF/services/org.apache.lucene.codecs.Codec";
  private static final String FORMAT_SERVICE =
      "META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat";
  private static final Map<String, Set<String>> EXPECTED_SERVICES =
      Map.of(
          CODEC_SERVICE,
          Set.of(
              "com.nvidia.cuvs.lucene.CuVS2510GPUSearchCodec",
              "com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec",
              "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedCodec",
              "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWScalarQuantizedCodec"),
          FORMAT_SERVICE,
          Set.of(
              "com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat",
              "com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat",
              "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat",
              "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWScalarQuantizedVectorsFormat"));

  @Test
  public void testPackagedThinJarHasExactServicesWithoutDependenciesOrPyLuceneTestSupport()
      throws Exception {
    Path thinJar = requireConfiguredThinJar();

    try (JarFile jar = new JarFile(thinJar.toFile())) {
      Set<String> entries = readFileEntries(jar);

      assertExactLuceneServices(jar, entries);
      assertNoBundledLuceneOrBaseCuvsJavaClasses(entries);
      assertNoPyLuceneTestSupport(entries);
    }
  }

  @Test
  public void testPackagedThinJarProvidersResolveThroughLuceneSpiInFreshJvm() throws Exception {
    Path thinJar = requireConfiguredThinJar();
    String packagedClasspath = requirePackagedTestClasspath(thinJar);

    SPIColdStartProbe.assertSucceeds(SPIColdStartProbe.CODEC_SPI_DISCOVERY_MODE, packagedClasspath);
    SPIColdStartProbe.assertSucceeds(
        SPIColdStartProbe.VECTOR_FORMAT_SPI_DISCOVERY_MODE, packagedClasspath);
  }

  private static Path requireConfiguredThinJar() {
    String configuredJar = System.getProperty(THIN_JAR_PROPERTY);
    assertNotNull("Missing system property " + THIN_JAR_PROPERTY, configuredJar);
    Path thinJar = Path.of(configuredJar);
    assertTrue("Thin JAR does not exist: " + thinJar, Files.isRegularFile(thinJar));
    return thinJar;
  }

  private static Set<String> readFileEntries(JarFile jar) {
    return jar.stream()
        .filter(entry -> !entry.isDirectory())
        .map(JarEntry::getName)
        .collect(Collectors.toUnmodifiableSet());
  }

  private static String requirePackagedTestClasspath(Path thinJar) {
    Path mainClasses = thinJar.getParent().resolve("classes").toAbsolutePath().normalize();
    Path packagedJar = thinJar.toAbsolutePath().normalize();
    String currentClasspath =
        System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
    List<Path> classpathEntries =
        Pattern.compile(Pattern.quote(File.pathSeparator))
            .splitAsStream(currentClasspath)
            .map(Path::of)
            .map(Path::toAbsolutePath)
            .map(Path::normalize)
            .toList();

    assertTrue(
        "Failsafe classpath does not contain the packaged thin JAR: " + packagedJar,
        classpathEntries.contains(packagedJar));
    assertFalse(
        "Failsafe classpath contains compiled main classes instead of only the packaged JAR",
        classpathEntries.contains(mainClasses));
    return currentClasspath;
  }

  private static void assertExactLuceneServices(JarFile jar, Set<String> entries)
      throws IOException {
    Set<String> serviceDescriptors =
        entries.stream()
            .filter(name -> name.startsWith("META-INF/services/org.apache.lucene."))
            .collect(Collectors.toUnmodifiableSet());
    assertEquals(
        "Unexpected Lucene service descriptors", EXPECTED_SERVICES.keySet(), serviceDescriptors);

    for (Map.Entry<String, Set<String>> expectedService : EXPECTED_SERVICES.entrySet()) {
      List<String> providers = readProviders(jar, expectedService.getKey());
      assertEquals(
          "Duplicate providers in " + expectedService.getKey(),
          Set.copyOf(providers).size(),
          providers.size());
      assertEquals(
          "Unexpected providers in " + expectedService.getKey(),
          expectedService.getValue(),
          Set.copyOf(providers));
      for (String provider : providers) {
        assertTrue(
            "Missing provider class " + provider,
            entries.contains(provider.replace('.', '/') + ".class"));
      }
    }
  }

  private static void assertNoBundledLuceneOrBaseCuvsJavaClasses(Set<String> entries) {
    for (String entry : entries) {
      assertFalse(
          "Thin JAR bundles a Lucene class: " + entry,
          entry.endsWith(".class")
              && (entry.startsWith("org/apache/lucene/") || entry.contains("/org/apache/lucene/")));
      assertFalse(
          "Thin JAR bundles a base cuvs-java class: " + entry,
          entry.startsWith("com/nvidia/cuvs/")
              && entry.endsWith(".class")
              && !entry.startsWith("com/nvidia/cuvs/lucene/"));
      assertFalse(
          "Thin JAR bundles a multi-release cuvs-java payload: " + entry,
          entry.startsWith("META-INF/versions/") && entry.contains("/com/nvidia/cuvs/"));
    }
  }

  private static void assertNoPyLuceneTestSupport(Set<String> entries) {
    for (String entry : entries) {
      assertFalse(
          "Thin JAR contains PyLucene test support: " + entry,
          entry.contains("PyLuceneTestSupport"));
    }
  }

  private static List<String> readProviders(JarFile jar, String descriptor) throws IOException {
    JarEntry entry = jar.getJarEntry(descriptor);
    assertNotNull("Missing service descriptor " + descriptor, entry);
    try (BufferedReader reader =
        new BufferedReader(
            new InputStreamReader(jar.getInputStream(entry), StandardCharsets.UTF_8))) {
      return reader
          .lines()
          .map(ThinJarContentsIT::stripComment)
          .filter(line -> !line.isEmpty())
          .toList();
    }
  }

  private static String stripComment(String line) {
    int commentStart = line.indexOf('#');
    return (commentStart < 0 ? line : line.substring(0, commentStart)).trim();
  }
}
