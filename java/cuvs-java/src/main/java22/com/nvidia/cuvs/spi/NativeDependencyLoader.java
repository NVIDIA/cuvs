/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.spi;

import static com.nvidia.cuvs.internal.common.NativeLibraryUtils.JVM_LoadLibrary$mh;

import java.io.*;
import java.lang.foreign.Arena;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.jar.JarFile;
import java.util.jar.Manifest;

/**
 * A class that loads native dependencies if they are available in the jar.
 */
class NativeDependencyLoader {

  interface NativeDependencyLoaderStrategy {
    void loadLibraries() throws ProviderInitializationException;
  }

  private static final NativeDependencyLoaderStrategy LOADER_STRATEGY = createLoaderStrategy();

  private static NativeDependencyLoaderStrategy createLoaderStrategy() {
    if (jarHasNativeDependencies()) {
      return new EmbeddedNativeDependencyLoaderStrategy();
    } else {
      return new SystemNativeDependencyLoaderStrategy();
    }
  }

  private static boolean jarHasNativeDependencies() {
    try (var jarFile =
        new JarFile(
            JDKProvider.class.getProtectionDomain().getCodeSource().getLocation().getPath())) {
      Manifest manifest = jarFile.getManifest();
      // TODO: use this variable to add a check on the installed CUDA version
      // (which will be system-loaded in any case, even with the fat-jar)
      var embeddedLibrariesCudaVersion =
          manifest.getMainAttributes().getValue("Embedded-Libraries-Cuda-Version");
      return embeddedLibrariesCudaVersion != null;
    } catch (IOException e) {
      return false;
    }
  }

  private static boolean loaded = false;

  static void loadLibraries() throws ProviderInitializationException {
    if (!loaded) {
      try {
        preloadCudaRuntime();
        LOADER_STRATEGY.loadLibraries();
      } finally {
        loaded = true;
      }
    }
  }

  /**
   * Best-effort preload of the CUDA runtime (libcudart) into the process before loading
   * {@code cuvs_c}.
   *
   * <p>Neither the embedded nor the system loader strategy bundles cudart (it is always
   * system-loaded, even with the fat jar), so it must be resolvable when {@code cuvs_c} is loaded.
   * In some deployments (e.g. cuvs-java embedded in an application that manages the classpath, such
   * as Solr) the CUDA runtime is present but not resolved at {@code cuvs_c} load time, which makes
   * the first native downcall fail with an unresolved-symbol {@link UnsatisfiedLinkError}.
   * Preloading cudart here places its soname in the process so the subsequent {@code cuvs_c} load
   * can satisfy its dependency.
   *
   * <p>This is intentionally best-effort: any failure is swallowed. If cudart is genuinely required
   * but missing, the {@code cuvs_c} load below fails and is surfaced as a
   * {@link ProviderInitializationException}, degrading to {@code UnsupportedProvider} — we must not
   * pre-empt that graceful path by throwing here. Reasons a preload may fail while the system is
   * still usable: cudart is exposed only as a versioned soname ({@code libcudart.so.N}) with no
   * unversioned {@code libcudart.so} devel symlink for {@link System#loadLibrary} to find, or it
   * was already loaded by another classloader.
   */
  private static void preloadCudaRuntime() {
    try {
      System.loadLibrary("cudart");
    } catch (UnsatisfiedLinkError e) {
      // Best-effort: cudart is either genuinely missing (the cuvs_c load below will report it) or
      // already loaded / only present as a versioned soname. Either way, continue.
    }
  }

  private static class EmbeddedNativeDependencyLoaderStrategy
      implements NativeDependencyLoaderStrategy {

    private static final String OS = System.getProperty("os.name");
    private static final String ARCH = System.getProperty("os.arch");
    private static final ClassLoader CLASS_LOADER = JDKProvider.class.getClassLoader();

    private static final String[] FILES_TO_LOAD = {
      "rapids_logger", "rmm", "cuvs", "cuvs_c",
    };

    @Override
    public void loadLibraries() throws ProviderInitializationException {
      for (String file : FILES_TO_LOAD) {
        // Uncomment the following line to trace the loading of native dependencies.
        // System.out.println("Loading native dependency: " + file);
        try {
          System.load(createFile(file).getAbsolutePath());
        } catch (Throwable t) {
          throw new ProviderInitializationException(
              "Failed to load native dependency: "
                  + System.mapLibraryName(file)
                  + ".so: "
                  + t.getMessage(),
              t);
        }
      }
    }

    /**
     * Extract the contents of a library resource into a temporary file
     */
    private static File createFile(String baseName) throws IOException {
      String path =
          EmbeddedNativeDependencyLoaderStrategy.ARCH
              + "/"
              + EmbeddedNativeDependencyLoaderStrategy.OS
              + "/"
              + System.mapLibraryName(baseName);
      File loc;
      URL resource = CLASS_LOADER.getResource(path);
      if (resource == null) {
        throw new FileNotFoundException("Could not locate native dependency " + path);
      }
      try (InputStream in = resource.openStream()) {
        loc = File.createTempFile(baseName, ".so");
        loc.deleteOnExit();

        Files.copy(in, loc.toPath(), StandardCopyOption.REPLACE_EXISTING);
      }
      return loc;
    }
  }

  private static class SystemNativeDependencyLoaderStrategy
      implements NativeDependencyLoaderStrategy {

    @Override
    public void loadLibraries() throws ProviderInitializationException {
      // Try load libcuvs using directly JVM_LoadLibrary with the correct flags for in-depth failure
      // diagnosis.
      //
      // jextract loads the dynamic libraries it references with SymbolLookup.libraryLookup; this
      // uses
      // RawNativeLibraries::load
      // https://github.com/openjdk/jdk/blob/master/src/java.base/share/native/libjava/RawNativeLibraries.c#L58
      // RawNativeLibraries::load in turn calls JVM_LoadLibrary. Unfortunately, it calls it with a
      // JNI_FALSE parameter for throwException, which means that the detailed error messages are
      // not surfaced.
      //
      // Here we invoke it with throwException true, so in case of error we can see what's broken
      String cuvsLibraryName = System.mapLibraryName("cuvs_c");

      final Object lib;
      try (var localArena = Arena.ofConfined()) {
        var name = localArena.allocateFrom(cuvsLibraryName);
        lib = JVM_LoadLibrary$mh.invoke(name, true);
      } catch (Throwable ex) {
        if (ex instanceof UnsatisfiedLinkError ulex) {
          throw new ProviderInitializationException(ulex.getMessage(), ulex);
        } else {
          throw new ProviderInitializationException("Error while loading " + cuvsLibraryName, ex);
        }
      }
      if (lib == null) {
        throw new ProviderInitializationException("Unspecified failure loading " + cuvsLibraryName);
      }
    }
  }
}
