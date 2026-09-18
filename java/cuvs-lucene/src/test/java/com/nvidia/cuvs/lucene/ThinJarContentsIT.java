/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.junit.Test;

/** Verifies the SPI and ownership boundaries of the packaged thin JAR. */
public class ThinJarContentsIT {

  private static final String MAVEN_METADATA_DIRECTORY =
      "META-INF/maven/com.nvidia.cuvs.lucene/cuvs-lucene/";
  private static final String CODEC_SERVICE = "META-INF/services/org.apache.lucene.codecs.Codec";
  private static final String FORMAT_SERVICE =
      "META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat";
  private static final Set<String> EXPECTED_RESOURCES =
      Set.of(
          "META-INF/MANIFEST.MF",
          MAVEN_METADATA_DIRECTORY + "pom.properties",
          MAVEN_METADATA_DIRECTORY + "pom.xml",
          CODEC_SERVICE,
          FORMAT_SERVICE,
          "logging.properties");
  private static final long PROBE_TIMEOUT_SECONDS = 30;
  private static final long TERMINATION_TIMEOUT_SECONDS = 5;
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
  public void testThinJarContainsOnlyOwnedProductionClassesAndExactServices() throws Exception {
    Path thinJar = configuredThinJar();
    try (JarFile jar = new JarFile(thinJar.toFile())) {
      Set<String> entries =
          jar.stream()
              .filter(entry -> !entry.isDirectory())
              .map(JarEntry::getName)
              .collect(Collectors.toUnmodifiableSet());

      Set<String> resources =
          entries.stream()
              .filter(entry -> !entry.endsWith(".class"))
              .collect(Collectors.toUnmodifiableSet());
      assertEquals("Unexpected thin-JAR resources", EXPECTED_RESOURCES, resources);
      assertExactServices(jar, entries);
      for (String entry : entries) {
        if (entry.endsWith(".class")) {
          assertTrue(
              "Thin JAR contains a dependency class: " + entry,
              entry.startsWith("com/nvidia/cuvs/lucene/"));
          assertFalse("Thin JAR contains a test class: " + entry, isTestClass(entry));
        }
      }
    }
  }

  @Test
  public void testThinJarProvidersResolveThroughLuceneSpiInFreshJvm() throws Exception {
    Path thinJar = configuredThinJar();
    Path outputFile = Files.createTempFile("cuvs-lucene-thin-jar-spi-", ".log");
    Process process = null;
    try {
      process =
          new ProcessBuilder(
                  javaExecutable(),
                  "-cp",
                  packagedTestClasspath(thinJar),
                  ThinJarSpiProbe.class.getName())
              .redirectErrorStream(true)
              .redirectOutput(outputFile.toFile())
              .start();

      boolean completed = process.waitFor(PROBE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
      if (!completed) {
        process.destroyForcibly();
        boolean terminated = process.waitFor(TERMINATION_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        throw new AssertionError(
            "Timed out waiting for thin-JAR SPI probe; terminated="
                + terminated
                + "; output:\n"
                + Files.readString(outputFile, StandardCharsets.UTF_8));
      }

      String output = Files.readString(outputFile, StandardCharsets.UTF_8);
      assertEquals("Fresh-JVM SPI probe failed:\n" + output, 0, process.exitValue());
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw interrupted;
    } finally {
      if (process != null && process.isAlive()) {
        process.destroyForcibly();
      }
      Files.deleteIfExists(outputFile);
    }
  }

  private static Path configuredThinJar() {
    String configured = System.getProperty("cuvs.lucene.thinJar");
    assertNotNull("Missing cuvs.lucene.thinJar", configured);
    Path thinJar = Path.of(configured).toAbsolutePath().normalize();
    assertTrue("Thin JAR does not exist: " + thinJar, Files.isRegularFile(thinJar));
    return thinJar;
  }

  private static void assertExactServices(JarFile jar, Set<String> entries) throws IOException {
    Set<String> descriptors =
        entries.stream()
            .filter(name -> name.startsWith("META-INF/services/org.apache.lucene."))
            .collect(Collectors.toUnmodifiableSet());
    assertEquals(EXPECTED_SERVICES.keySet(), descriptors);
    for (Map.Entry<String, Set<String>> service : EXPECTED_SERVICES.entrySet()) {
      JarEntry descriptor = jar.getJarEntry(service.getKey());
      assertNotNull("Missing service descriptor " + service.getKey(), descriptor);
      List<String> providers =
          new String(jar.getInputStream(descriptor).readAllBytes(), StandardCharsets.UTF_8)
              .lines()
              .map(ThinJarContentsIT::stripComment)
              .filter(line -> !line.isEmpty())
              .toList();
      assertEquals(
          "Duplicate providers in " + service.getKey(),
          providers.size(),
          Set.copyOf(providers).size());
      assertEquals(
          "Unexpected providers in " + service.getKey(), service.getValue(), Set.copyOf(providers));
      for (String provider : providers) {
        assertTrue(
            "Missing provider class " + provider,
            entries.contains(provider.replace('.', '/') + ".class"));
      }
    }
  }

  private static boolean isTestClass(String entry) {
    String name = entry.substring(entry.lastIndexOf('/') + 1);
    return name.startsWith("Test")
        || name.contains("Test$")
        || name.endsWith("IT.class")
        || name.contains("IT$")
        || name.contains("Probe");
  }

  private static String packagedTestClasspath(Path thinJar) throws URISyntaxException {
    Path mainClasses = thinJar.getParent().resolve("classes").toAbsolutePath().normalize();
    String current =
        System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
    List<String> entries = new ArrayList<>();
    entries.add(thinJar.toString());
    Pattern.compile(Pattern.quote(File.pathSeparator))
        .splitAsStream(current)
        .map(Path::of)
        .map(Path::toAbsolutePath)
        .map(Path::normalize)
        .filter(path -> !path.equals(mainClasses))
        .filter(path -> !path.equals(thinJar))
        .map(Path::toString)
        .forEach(entries::add);
    assertFalse(
        "Packaged probe classpath includes target/classes",
        entries.contains(mainClasses.toString()));
    Path testClasses =
        Path.of(ThinJarSpiProbe.class.getProtectionDomain().getCodeSource().getLocation().toURI())
            .toAbsolutePath()
            .normalize();
    assertTrue(
        "Packaged probe classpath is missing test classes",
        entries.contains(testClasses.toString()));
    return String.join(File.pathSeparator, entries);
  }

  private static String javaExecutable() {
    return Path.of(System.getProperty("java.home"), "bin", "java").toString();
  }

  private static String stripComment(String line) {
    int comment = line.indexOf('#');
    return (comment < 0 ? line : line.substring(0, comment)).trim();
  }
}
