---
slug: api-reference/cpp-api-core-dataset
---

# Dataset

_Source header: `cuvs/core/dataset.hpp`_

## Types

<a id="core-dataset-view"></a>
### core::dataset_view

Non-owning dataset view: holds only the view-shaped payload. Deliberately not derived from

`dataset` -- a view type holds "all view state" with no inheritance and no shared ownership tying it to the owning type. Reuses the same `get_n_rows`/`get_dim` spec functions as `dataset`, fed the view payload instead of the owning one.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset_view;
```

<a id="core-dataset"></a>
### core::dataset

Owning dataset: value-held payload (no shared_ptr -- exclusive ownership). Every member is a

one-line forward to `spec_type::get_*` or to the payload; all per-kind state and logic lives in the spec's `data_type`, never inside this struct.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset;
```

<a id="core-is-padded-dataset"></a>
### core::is_padded_dataset

Owning-side kind traits (true for both `dataset&lt;...&gt;` and `dataset_view&lt;...&gt;` of that kind).

```cpp
template <typename DatasetT>
struct is_padded_dataset;
```

<a id="core-is-dataset-view"></a>
### core::is_dataset_view

True for any `dataset_view&lt;...&gt;` specialization. Evaluates to `false` (never a hard error) for

everything else, e.g. a plain mdspan passed to a deprecated `build(matrix_view)` overload.

```cpp
template <typename V>
struct is_dataset_view;
```

<a id="core-dataset-view-has-spec"></a>
### core::dataset_view_has_spec

True when `V` is a `dataset_view` whose spec satisfies the predicate `SpecPred&lt;SpecT&gt;::value`.

This is how a kind that lives outside this header classifies its own views.

```cpp
template <typename V, template <typename> typename SpecPred>
struct dataset_view_has_spec;
```

<a id="core-dataset-view-is-device-accessible"></a>
### core::dataset_view_is_device_accessible

True when the dataset view accessor is device-accessible.

```cpp
template <typename V>
struct dataset_view_is_device_accessible;
```

<a id="core-with-accessor"></a>
### core::with_accessor

Generic accessor retargeting while preserving the spec kind and value/index types:

`dataset&lt;T, IdxT, SpecT&lt;..., OldAccessor&gt;&gt;      -&gt; dataset&lt;T, IdxT, SpecT&lt;..., NewAccessor&gt;&gt;` `dataset_view&lt;T, IdxT, SpecT&lt;..., OldAccessor&gt;&gt; -&gt; dataset_view&lt;T, IdxT, SpecT&lt;..., NewAccessor&gt;&gt;` Every spec provides `rebind_accessor&lt;NewAccessor&gt;` for this, so this header does not need to know about any particular kind.

```cpp
template <typename DatasetLikeT, typename NewAccessor>
struct with_accessor;
```

<a id="core-to-device-accessor"></a>
### core::to_device_accessor

Map any host accessor to its device counterpart (same payload policy).

```cpp
template <typename Accessor>
struct to_device_accessor;
```

<a id="core-device-counterpart"></a>
### core::device_counterpart

Maps a host dataset view type to its device-resident counterpart.

```cpp
template <typename HostViewT>
struct device_counterpart;
```
