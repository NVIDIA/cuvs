---
slug: api-reference/cpp-api-util-host-memory
---

# Host Memory

_Source header: `cuvs/util/host_memory.hpp`_

## Types

<a id="util-host-memory-info"></a>
### util::host_memory_info

Snapshot of host and cgroup memory available to the current process.

```cpp
struct host_memory_info {
  size_t system_available;
  size_t available;
  std::optional<size_t> cgroup_limit;
  std::optional<size_t> cgroup_current;
  std::optional<size_t> cgroup_reclaimable_file;
  std::optional<size_t> cgroup_working_set;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `system_available` | `size_t` | Host-wide MemAvailable from /proc/meminfo. |
| `available` | `size_t` | Effective memory available after applying any cgroup limit. |
| `cgroup_limit` | `std::optional<size_t>` | Hard limit of the most constrained cgroup ancestor, when finite. |
| `cgroup_current` | `std::optional<size_t>` | Current usage charged to the most constrained cgroup ancestor. |
| `cgroup_reclaimable_file` | `std::optional<size_t>` | Clean file cache treated as reclaimable for capacity planning. |
| `cgroup_working_set` | `std::optional<size_t>` | Current cgroup usage after excluding reclaimable file cache. |
