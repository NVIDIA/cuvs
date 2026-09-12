/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * JVM/classloader-local allowlist that resolves content-addressed external FBIN references.
 *
 * <p>Lucene recreates codecs through no-argument SPI constructors, so writer-time configuration is
 * not available when a process later opens an index. Every indexing or search application must
 * register the immutable local file for each referenced content ID through the same defining
 * cuvs-lucene classloader before opening the index. Registrations are reference counted; at least
 * one matching registration must remain alive for every dependent reader's complete lifetime.
 *
 * <p>Registration validates the path and content-ID syntax but deliberately does not hash the file.
 * Use {@link ExternalFbinBuildValidation#VERIFY_SHA256} while building, or run Lucene integrity
 * checks, unless the registered storage already enforces the supplied content identity.
 */
public final class ExternalFbinFileRegistry {

  private static final Object LOCK = new Object();
  private static final Map<String, Entry> ENTRIES = new HashMap<>();

  private ExternalFbinFileRegistry() {}

  /**
   * Registers an allowlisted local file for a complete-file SHA-256 content identity. The path is
   * resolved once to its absolute real path, which is returned by {@link Registration#path()}.
   */
  public static Registration register(Path path, String sha256Hex) throws IOException {
    Objects.requireNonNull(path, "path");
    byte[] digest = ExternalFbinReference.parseSha256(sha256Hex);
    String contentId = ExternalFbinReference.contentId(digest);
    Path realPath = path.toRealPath();
    if (!Files.isRegularFile(realPath) || !Files.isReadable(realPath)) {
      throw new IOException("External FBIN must be a readable regular file: " + realPath);
    }

    synchronized (LOCK) {
      Entry existing = ENTRIES.get(contentId);
      if (existing == null) {
        ENTRIES.put(contentId, new Entry(realPath, 1));
      } else if (existing.path.equals(realPath)) {
        existing.references++;
      } else {
        throw new IllegalStateException(
            "Content ID "
                + contentId
                + " is already registered to "
                + existing.path
                + "; refusing ambiguous replacement with "
                + realPath);
      }
    }
    return new Registration(realPath, contentId);
  }

  static Path resolve(ExternalFbinReference reference) throws IOException {
    synchronized (LOCK) {
      Entry entry = ENTRIES.get(reference.contentId());
      if (entry == null) {
        throw new IOException(
            "No allowlisted external FBIN is registered in this cuvs-lucene classloader for "
                + reference.contentId()
                + ". Register it through the same defining classloader before opening the index.");
      }
      return entry.path;
    }
  }

  static void clearForTests() {
    synchronized (LOCK) {
      ENTRIES.clear();
    }
  }

  private static final class Entry {
    private final Path path;
    private int references;

    private Entry(Path path, int references) {
      this.path = path;
      this.references = references;
    }
  }

  /**
   * A scoped registry lease. Closing the final lease removes the path from this classloader's
   * allowlist. The registry does not own the file, and every reader that depends on its content ID
   * must be closed before the final matching lease is closed.
   */
  public static final class Registration implements Closeable {

    private final Path path;
    private final String contentId;
    private boolean closed;

    private Registration(Path path, String contentId) {
      this.path = path;
      this.contentId = contentId;
    }

    /** Creates a path-independent reference to a contiguous row range in the registered FBIN. */
    synchronized ExternalFbinReference reference(int firstRow, int rowCount) throws IOException {
      ensureOpen();
      return ExternalFbinReference.fromFile(path, contentId, firstRow, rowCount);
    }

    /**
     * Maps a registered row range and couples its native build dataset to the persisted reference.
     */
    synchronized ImmutableExternalFbinDataset map(int firstRow, int rowCount) throws IOException {
      ensureOpen();
      return ImmutableExternalFbinDataset.map(
          ExternalFbinReference.fromFile(path, contentId, firstRow, rowCount));
    }

    private void ensureOpen() {
      if (closed) {
        throw new IllegalStateException("External FBIN registration is closed");
      }
    }

    public String contentId() {
      return contentId;
    }

    /** Returns the absolute real path captured when this registration was created. */
    public Path path() {
      return path;
    }

    @Override
    public void close() {
      synchronized (this) {
        if (closed) {
          return;
        }
        closed = true;
      }
      synchronized (LOCK) {
        Entry entry = ENTRIES.get(contentId);
        if (entry != null && entry.path.equals(path)) {
          entry.references--;
          if (entry.references == 0) {
            ENTRIES.remove(contentId);
          }
        }
      }
    }
  }
}
