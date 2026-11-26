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

_Source: `rust/cuvs/src/dataset.rs:68`_

## CuvsDataset

```rust
pub trait CuvsDataset: private::Sealed {
    /* required methods omitted */
}
```

A Rust wrapper accepted by native cuVS dataset operations.

This trait is sealed; dataset handles can only be created by this crate.

_Source: `rust/cuvs/src/dataset.rs:76`_

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
| `new` | `rust/cuvs/src/dataset.rs:135` |

### new

```rust
pub fn new<T>(res: &Resources, dataset: &'a T) -> Result<Self>
where
T: AsDlTensor + ?Sized,
```

Borrow a tensor as the host/device and padded/standard view matching its
DLPack shape/strides (CAGRA row-width rule).

_Source: `rust/cuvs/src/dataset.rs:135`_

_Source: `rust/cuvs/src/dataset.rs:127`_

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
| `new` | `rust/cuvs/src/dataset.rs:184` |

### new

```rust
pub fn new<T>(res: &Resources, dataset: &T) -> Result<Self>
where
T: AsDlTensor + ?Sized,
```

Copy a tensor into freshly allocated, CAGRA-padded storage.

_Source: `rust/cuvs/src/dataset.rs:184`_

_Source: `rust/cuvs/src/dataset.rs:178`_

## PqDataset

```rust
#[derive(Debug)]
pub struct PqDataset {
    /* private fields */
}
```

Owning device PQ dataset for CAGRA-Q search.

Prefer [`crate::neighbors::cagra::make_pq_dataset`] which accepts
[`crate::neighbors::cagra::ProductQuantizerParams`]. Keep this owner alive while
any index uses it.

_Source: `rust/cuvs/src/dataset.rs:232`_

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

_Source: `rust/cuvs/src/dataset.rs:283`_
