/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.Lucene101ConfiguredHNSWCodec.BEAM_WIDTH_PROPERTY;
import static com.nvidia.cuvs.lucene.Lucene101ConfiguredHNSWCodec.MAX_CONN_PROPERTY;
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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.regex.Pattern;
import org.apache.lucene.codecs.Codec;
import org.junit.Test;

/** Verifies the configured codec from the packaged thin JAR in a fresh JVM. */
public class Lucene101ConfiguredHNSWCodecIT {

  private static final String THIN_JAR_PROPERTY = "cuvs.lucene.thinJar";
  private static final String CODEC_SERVICE = "META-INF/services/org.apache.lucene.codecs.Codec";
  private static final String CONFIGURED_CODEC_CLASS =
      "com.nvidia.cuvs.lucene.Lucene101ConfiguredHNSWCodec";
  private static final String CONFIGURED_CODEC_ENTRY =
      "com/nvidia/cuvs/lucene/Lucene101ConfiguredHNSWCodec.class";
  private static final int PROBE_MAX_CONN = 19;
  private static final int PROBE_BEAM_WIDTH = 73;
  private static final long PROBE_TIMEOUT_SECONDS = 30;
  private static final long TERMINATION_TIMEOUT_SECONDS = 5;

  @Test
  public void testThinJarContainsConfiguredCodecWithoutRegisteringItAsSpiProvider()
      throws Exception {
    Path thinJar = requireThinJar();

    try (JarFile jar = new JarFile(thinJar.toFile())) {
      assertNotNull(
          "thin JAR is missing " + CONFIGURED_CODEC_ENTRY, jar.getJarEntry(CONFIGURED_CODEC_ENTRY));
      assertFalse(
          "configured codec must not be registered as a Lucene SPI provider",
          readProviders(jar, CODEC_SERVICE).contains(CONFIGURED_CODEC_CLASS));
    }
  }

  @Test
  public void testConfiguredCodecLoadsFromThinJarInFreshJvm() throws Exception {
    Path thinJar = requireThinJar();
    String packagedClasspath = buildPackagedClasspath(thinJar);
    String javaExecutable =
        Path.of(System.getProperty("java.home"), "bin", "java").toAbsolutePath().toString();
    Path outputFile = Files.createTempFile("cuvs-lucene-configured-codec-", ".log");
    Process process = null;
    try {
      process =
          new ProcessBuilder(
                  javaExecutable,
                  "--add-modules=jdk.incubator.vector",
                  "--enable-native-access=ALL-UNNAMED",
                  "-D" + THIN_JAR_PROPERTY + "=" + thinJar,
                  "-D" + MAX_CONN_PROPERTY + "=" + PROBE_MAX_CONN,
                  "-D" + BEAM_WIDTH_PROPERTY + "=" + PROBE_BEAM_WIDTH,
                  "-cp",
                  packagedClasspath,
                  Lucene101ConfiguredHNSWCodecIT.class.getName(),
                  "probe")
              .redirectErrorStream(true)
              .redirectOutput(outputFile.toFile())
              .start();

      boolean completed = process.waitFor(PROBE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
      if (!completed) {
        process.destroyForcibly();
        boolean terminated = process.waitFor(TERMINATION_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        throw new AssertionError(
            "Timed out waiting for configured codec probe; terminated="
                + terminated
                + "; output:\n"
                + Files.readString(outputFile, StandardCharsets.UTF_8));
      }

      String output = Files.readString(outputFile, StandardCharsets.UTF_8);
      if (process.exitValue() != 0) {
        throw new AssertionError(
            "Configured codec probe exited " + process.exitValue() + "; output:\n" + output);
      }
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

  /** Fresh-JVM entry point used by {@link #testConfiguredCodecLoadsFromThinJarInFreshJvm()}. */
  public static void main(String[] args) throws Exception {
    if (args.length != 1 || !"probe".equals(args[0])) {
      throw new IllegalArgumentException("Expected the configured codec probe mode");
    }

    Object reflectedCodec = instantiateThroughClassNewInstance();
    if (!(reflectedCodec instanceof Lucene101ConfiguredHNSWCodec codec)) {
      throw new AssertionError("Configured class is not a Lucene codec: " + reflectedCodec);
    }
    assertLoadedFromConfiguredThinJar(codec);
    assertProbeState(codec, PROBE_MAX_CONN, PROBE_BEAM_WIDTH);

    System.setProperty(MAX_CONN_PROPERTY, "31");
    System.setProperty(BEAM_WIDTH_PROPERTY, "101");
    assertProbeState(codec, PROBE_MAX_CONN, PROBE_BEAM_WIDTH);

    codec.configuredParameters().getMergeExec().shutdownNow();
  }

  @SuppressWarnings("deprecation")
  private static Object instantiateThroughClassNewInstance()
      throws ClassNotFoundException, InstantiationException, IllegalAccessException {
    // Downstream PyLucene uses this entry point because JCC does not expose
    // Constructor.newInstance().
    return Class.forName(CONFIGURED_CODEC_CLASS).newInstance();
  }

  private static void assertLoadedFromConfiguredThinJar(Codec codec) throws Exception {
    Path expectedJar = Path.of(System.getProperty(THIN_JAR_PROPERTY)).toAbsolutePath().normalize();
    Path loadedFrom =
        Path.of(codec.getClass().getProtectionDomain().getCodeSource().getLocation().toURI())
            .toAbsolutePath()
            .normalize();
    if (!expectedJar.equals(loadedFrom)) {
      throw new AssertionError(
          "Configured codec loaded from " + loadedFrom + " instead of thin JAR " + expectedJar);
    }
  }

  private static void assertProbeState(
      Lucene101ConfiguredHNSWCodec codec, int expectedMaxConn, int expectedBeamWidth) {
    if (!"Lucene101AcceleratedHNSWCodec".equals(codec.getName())) {
      throw new AssertionError("Unexpected codec name: " + codec.getName());
    }
    if (codec.knnVectorsFormat() == null
        || !"Lucene99AcceleratedHNSWVectorsFormat".equals(codec.knnVectorsFormat().getName())) {
      throw new AssertionError("Unexpected vector format: " + codec.knnVectorsFormat());
    }
    if (codec.configuredParameters().getMaxConn() != expectedMaxConn) {
      throw new AssertionError(
          "Expected maxConn="
              + expectedMaxConn
              + ", got "
              + codec.configuredParameters().getMaxConn());
    }
    if (codec.configuredParameters().getBeamWidth() != expectedBeamWidth) {
      throw new AssertionError(
          "Expected beamWidth="
              + expectedBeamWidth
              + ", got "
              + codec.configuredParameters().getBeamWidth());
    }
  }

  private static Path requireThinJar() {
    String configuredJar = System.getProperty(THIN_JAR_PROPERTY);
    assertNotNull("Missing system property " + THIN_JAR_PROPERTY, configuredJar);
    Path thinJar = Path.of(configuredJar).toAbsolutePath().normalize();
    assertTrue("Thin JAR does not exist: " + thinJar, Files.isRegularFile(thinJar));
    return thinJar;
  }

  private static String buildPackagedClasspath(Path thinJar) {
    Path compiledMainClasses = thinJar.getParent().resolve("classes").toAbsolutePath().normalize();
    String testClasspath =
        System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
    List<String> packagedEntries = new ArrayList<>();
    packagedEntries.add(thinJar.toString());
    Pattern.compile(Pattern.quote(File.pathSeparator))
        .splitAsStream(testClasspath)
        .map(Path::of)
        .map(Path::toAbsolutePath)
        .map(Path::normalize)
        .filter(entry -> !entry.equals(compiledMainClasses))
        .filter(entry -> !entry.equals(thinJar))
        .map(Path::toString)
        .forEach(packagedEntries::add);
    return String.join(File.pathSeparator, packagedEntries);
  }

  private static List<String> readProviders(JarFile jar, String descriptor) throws IOException {
    JarEntry entry = jar.getJarEntry(descriptor);
    assertNotNull("thin JAR is missing service descriptor " + descriptor, entry);
    try (BufferedReader reader =
        new BufferedReader(
            new InputStreamReader(jar.getInputStream(entry), StandardCharsets.UTF_8))) {
      return reader
          .lines()
          .map(Lucene101ConfiguredHNSWCodecIT::stripComment)
          .filter(line -> !line.isEmpty())
          .toList();
    }
  }

  private static String stripComment(String line) {
    int commentStart = line.indexOf('#');
    return (commentStart < 0 ? line : line.substring(0, commentStart)).trim();
  }
}
