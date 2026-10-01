---
slug: api-reference/cpp-api-common-types-errors-and-logging
---

# Errors and Logging

NVIDIA cuVS public headers also use public RAFT error and logging facilities in inline code paths.

<a id="raft-exception"></a>
### raft::exception

_Source header: `raft/core/error.hpp`_

Exception base used by some NVIDIA cuVS utility errors.

```cpp
class exception;
```

<a id="raft-expects"></a>
### RAFT_EXPECTS

_Source header: `raft/core/error.hpp`_

Validates a condition and raises a RAFT exception if it is false.

```cpp
#define RAFT_EXPECTS(condition, reason, ...)
```

<a id="raft-fail"></a>
### RAFT_FAIL

_Source header: `raft/core/error.hpp`_

Raises a RAFT exception with a formatted message.

```cpp
#define RAFT_FAIL(reason, ...)
```

<a id="raft-cuda-try"></a>
### RAFT_CUDA_TRY

_Source header: `raft/core/error.hpp`_

Checks a CUDA runtime call and raises a RAFT exception on failure.

```cpp
#define RAFT_CUDA_TRY(call)
```

<a id="raft-log-debug"></a>
### RAFT_LOG_DEBUG

_Source header: `raft/core/logger.hpp`_

Emits a debug log message through RAFT logging.

```cpp
#define RAFT_LOG_DEBUG(message, ...)
```
