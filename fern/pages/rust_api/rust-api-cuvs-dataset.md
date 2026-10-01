---
slug: api-reference/rust-api-cuvs-dataset
---

# Dataset Module

_Rust module: `cuvs::dataset`_

_Source: `rust/cuvs/src/dataset.rs`_

## DatasetKind

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DatasetKind {
    /* variants omitted */
}
```

Host/device residency and row layout of a [`DatasetView`].

_Source: `rust/cuvs/src/dataset.rs:19`_

## Sealed

```rust
pub trait Sealed {
    /* required methods omitted */
}
```

_Source: `rust/cuvs/src/dataset.rs:56`_

## CuvsDataset

```rust
pub trait CuvsDataset: private::Sealed {
    /* required methods omitted */
}
```

A Rust wrapper accepted by native cuVS dataset operations.

This trait is sealed; dataset handles can only be created by this crate.

_Source: `rust/cuvs/src/dataset.rs:64`_

## DatasetView

```rust
#[derive(Debug)]
pub struct DatasetView<'a> {
    /* private fields */
}
```

A non-owning CAGRA dataset view.

The view records the storage's residency and layout while borrowing its
backing tensor for `'a`. Constructing a view allocates only native metadata;
it never copies vector storage.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/dataset.rs:123` |

### new

```rust
pub fn new<T>(res: &Resources, dataset: &'a T) -> Result<Self>
where
T: AsDlTensor + ?Sized,
```

Borrow a tensor as the host/device and padded/standard view matching its
DLPack shape/strides (CAGRA row-width rule).

_Source: `rust/cuvs/src/dataset.rs:123`_

_Source: `rust/cuvs/src/dataset.rs:115`_

## PaddedDataset

```rust
#[derive(Debug)]
pub struct PaddedDataset {
    /* private fields */
}
```

Storage owned by the caller, padded to CAGRA's required row width.

Construction performs an explicit allocation and copy. Memory residency is
inferred from the source tensor; use [`DatasetView::new`] when its existing
layout is already suitable.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/dataset.rs:172` |

### new

```rust
pub fn new<T>(res: &Resources, dataset: &T) -> Result<Self>
where
T: AsDlTensor + ?Sized,
```

Copy a tensor into freshly allocated, CAGRA-padded storage.

_Source: `rust/cuvs/src/dataset.rs:172`_

_Source: `rust/cuvs/src/dataset.rs:166`_

## Dataset

```rust
#[derive(Debug)]
pub struct Dataset {
    /* private fields */
}
```

Owning dataset storage returned by CAGRA deserialization.

The allocation preserves the serialized host/device residency and
standard/padded row layout. CAGRA keeps only a non-owning view, so this
owner must remain alive while the deserialized index uses it.

_Source: `rust/cuvs/src/dataset.rs:220`_
