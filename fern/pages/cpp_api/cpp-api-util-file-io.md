---
slug: api-reference/cpp-api-util-file-io
---

# File Io

_Source header: `cuvs/util/file_io.hpp`_

## Types

<a id="util-fd-streambuf"></a>
### util::fd_streambuf

Streambuf that reads from a POSIX file descriptor

```cpp
class fd_streambuf;
```

<a id="util-fd-istream"></a>
### util::fd_istream

Istream that reads from a POSIX file descriptor

```cpp
class fd_istream;
```

<a id="util-file-descriptor"></a>
### util::file_descriptor

RAII wrapper for POSIX file descriptors

Manages file descriptor lifecycle with automatic cleanup. Used to own the lifetime of disk-backed ACE artifacts and to parse their numpy headers; the bulk data transfers go through kvikio (see :read_large_file / ::write_large_file). Non-copyable, move-only.

```cpp
class file_descriptor;
```

<a id="util-kvikio-file-reader"></a>
### util::kvikio_file_reader

Sequential file reader supporting mixed stream and direct-to-device reads.

Small metadata can be consumed through stream(), while read_device() transfers the next bytes through KvikIO into device memory. Both operations advance one logical file position. KvikIO uses GPUDirect Storage when available and falls back to its compatible I/O path otherwise. Non-copyable, non-movable.

```cpp
class kvikio_file_reader;
```

<a id="util-kvikio-ofstream"></a>
### util::kvikio_ofstream

Sequential std::ostream backed by kvikio.

Ordinary stream output is staged into a large host buffer and written to disk through kvikio, which bypasses the page cache via O_DIRECT when supported (and falls back to buffered POSIX writes otherwise). Device buffers passed to write_device() use GPUDirect Storage when available. This can be passed to APIs accepting a std::ostream& for sequential output (e.g. the hnswlib serializer). It supports querying the current output position, but not random-access seeking. std::fstream-specific APIs such as is_open() are not part of its interface. Non-copyable, non-movable.

```cpp
class kvikio_ofstream;
```

<a id="util-buffered-ofstream"></a>
### util::buffered_ofstream

Buffered output stream wrapper

Wraps an std::ostream with a buffer to improve write performance by reducing the number of system calls. Automatically flushes on destruction. Used by the hnswlib serializer. Non-copyable, non-movable.

```cpp
class buffered_ofstream;
```
