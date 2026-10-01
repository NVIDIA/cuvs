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

## Sealed

```rust
pub trait Sealed {
    /* required methods omitted */
}
```

_Source: `rust/cuvs/src/neighbors/filters.rs:40`_

## FilterKind

```rust
pub trait FilterKind: sealed::Sealed {
    /* required methods omitted */
}
```

Kind of packed filter payload.

This trait is sealed; only [`Bitset`] and [`Bitmap`] implement it.

_Source: `rust/cuvs/src/neighbors/filters.rs:46`_

## Filter

```rust
pub struct Filter<'a, K: FilterKind> {
    /* private fields */
}
```

Packed filter words used to include or exclude rows during search.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/filters.rs:71` |

### new

```rust
pub fn new<T>(filter_words: &'a T) -> Result<Self, FilterError>
where
T: AsDlTensor + ?Sized,
```

Creates a packed filter borrowing `filter_words`.

_Source: `rust/cuvs/src/neighbors/filters.rs:71`_

_Source: `rust/cuvs/src/neighbors/filters.rs:64`_
