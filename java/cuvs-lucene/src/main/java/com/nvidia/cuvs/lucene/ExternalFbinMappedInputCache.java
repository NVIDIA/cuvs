/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOUtils;

/** Shares the one owning full-file FBIN mapping among every live reader for the same artifact. */
final class ExternalFbinMappedInputCache {

  private static final Object LOCK = new Object();
  private static final Map<Key, Owner> OWNERS = new HashMap<>();

  private ExternalFbinMappedInputCache() {}

  static Lease acquire(
      Path realPath, ExternalFbinReference reference, long maxMmapChunkSize, IOContext context)
      throws IOException {
    Key key = new Key(realPath, reference.contentId(), reference.fileLength(), maxMmapChunkSize);
    synchronized (LOCK) {
      Owner owner = OWNERS.get(key);
      if (owner == null) {
        owner = openOwner(realPath, maxMmapChunkSize, context);
        OWNERS.put(key, owner);
      }
      IndexInput sourceView = owner.rootInput.clone();
      owner.leases++;
      return new Lease(key, owner, sourceView);
    }
  }

  private static Owner openOwner(Path realPath, long maxMmapChunkSize, IOContext context)
      throws IOException {
    Path parent = realPath.getParent();
    if (parent == null) {
      throw new IOException("External FBIN path has no parent directory: " + realPath);
    }
    MMapDirectory directory = null;
    IndexInput rootInput = null;
    boolean success = false;
    try {
      directory = new MMapDirectory(parent, maxMmapChunkSize);
      rootInput = directory.openInput(realPath.getFileName().toString(), context);
      Owner owner = new Owner(directory, rootInput);
      success = true;
      return owner;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(rootInput, directory);
      }
    }
  }

  static int activeOwnerCountForTests(
      Path realPath, ExternalFbinReference reference, long maxMmapChunkSize) throws IOException {
    synchronized (LOCK) {
      return OWNERS.containsKey(key(realPath, reference, maxMmapChunkSize)) ? 1 : 0;
    }
  }

  static int activeLeaseCountForTests(
      Path realPath, ExternalFbinReference reference, long maxMmapChunkSize) throws IOException {
    synchronized (LOCK) {
      Owner owner = OWNERS.get(key(realPath, reference, maxMmapChunkSize));
      return owner == null ? 0 : owner.leases;
    }
  }

  private static Key key(Path path, ExternalFbinReference reference, long maxMmapChunkSize)
      throws IOException {
    return new Key(
        path.toRealPath(), reference.contentId(), reference.fileLength(), maxMmapChunkSize);
  }

  private static final class Owner {
    private final MMapDirectory directory;
    private final IndexInput rootInput;
    private int leases;

    private Owner(MMapDirectory directory, IndexInput rootInput) {
      this.directory = directory;
      this.rootInput = rootInput;
    }
  }

  /**
   * A reader lease over a non-owning full-file clone.
   *
   * <p>Lucene's multi-chunk mmap clone shares its segment array with the owning input, so the clone
   * must never be closed directly. Closing this lease releases only the shared owner reference;
   * the final lease closes the sole owning input and its directory.
   */
  static final class Lease implements Closeable {
    private final Key key;
    private final Owner owner;
    private IndexInput sourceView;
    private boolean closed;

    private Lease(Key key, Owner owner, IndexInput sourceView) {
      this.key = key;
      this.owner = owner;
      this.sourceView = sourceView;
    }

    IndexInput sourceInput() {
      synchronized (LOCK) {
        if (closed) {
          throw new IllegalStateException("External FBIN mapping lease is closed");
        }
        return sourceView;
      }
    }

    @Override
    public void close() throws IOException {
      synchronized (LOCK) {
        if (closed) {
          return;
        }
        closed = true;
        sourceView = null;
        Owner current = OWNERS.get(key);
        if (current != owner || owner.leases <= 0) {
          throw new IllegalStateException("External FBIN mapping cache reference count is corrupt");
        }
        owner.leases--;
        if (owner.leases == 0) {
          OWNERS.remove(key);
          IOUtils.close(owner.rootInput, owner.directory);
        }
      }
    }
  }

  private record Key(
      Path realPath, String contentId, long expectedFileLength, long maxMmapChunkSize) {}
}
