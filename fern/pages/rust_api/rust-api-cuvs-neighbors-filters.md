---
slug: api-reference/rust-api-cuvs-neighbors-filters
---

# Neighbors Filters Module

_Rust module: `cuvs::neighbors::filters`_

_Source: `rust/cuvs/src/neighbors/filters.rs`_

Shared filter payloads for nearest-neighbor search APIs.

## FilterError

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FilterError {
    /* variants omitted */
}
```

Error returned when constructing an invalid filter payload.

_Source: `rust/cuvs/src/neighbors/filters.rs:15`_

## Bitset

```rust
pub enum Bitset {
    /* variants omitted */
}
```

Marker for a row-level bitset filter.

_Source: `rust/cuvs/src/neighbors/filters.rs:34`_

## Bitmap

```rust
pub enum Bitmap {
    /* variants omitted */
}
```

Marker for a per-query bitmap filter.

_Source: `rust/cuvs/src/neighbors/filters.rs:37`_

## SearchFilter

```rust
#[non_exhaustive]
pub enum SearchFilter<'a> {
    /* variants omitted */
}
```

Shared filter options for nearest-neighbor search.

**Methods**

| Name | Source |
| --- | --- |
| `bitset` | `rust/cuvs/src/neighbors/filters.rs:76` |
| `bitmap` | `rust/cuvs/src/neighbors/filters.rs:84` |

### bitset

```rust
pub fn bitset<T>(filter_words: &'a T) -> Result<Self, FilterError>
where
T: AsDlTensor + ?Sized,
```

Creates a row-level bitset filter borrowing `filter_words`.

_Source: `rust/cuvs/src/neighbors/filters.rs:76`_

### bitmap

```rust
pub fn bitmap<T>(filter_words: &'a T) -> Result<Self, FilterError>
where
T: AsDlTensor + ?Sized,
```

Creates a per-query bitmap filter borrowing `filter_words`.

_Source: `rust/cuvs/src/neighbors/filters.rs:84`_

_Source: `rust/cuvs/src/neighbors/filters.rs:41`_

## Filter

```rust
pub struct Filter<'a, K> {
    /* private fields */
}
```

Packed filter words used to include or exclude rows during search.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/filters.rs:56` |
| `new` | `rust/cuvs/src/neighbors/filters.rs:66` |

### new

```rust
pub fn new<T>(filter_words: &'a T) -> Result<Self, FilterError>
where
T: AsDlTensor + ?Sized,
```

Creates a row-level bitset borrowing `filter_words`.

_Source: `rust/cuvs/src/neighbors/filters.rs:56`_

### new

```rust
pub fn new<T>(filter_words: &'a T) -> Result<Self, FilterError>
where
T: AsDlTensor + ?Sized,
```

Creates a per-query bitmap borrowing `filter_words`.

_Source: `rust/cuvs/src/neighbors/filters.rs:66`_

_Source: `rust/cuvs/src/neighbors/filters.rs:49`_
