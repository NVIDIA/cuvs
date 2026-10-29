/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::marker::PhantomData;
use std::path::Path;

use super::{CagraError, CompressionParams, GraphStorage, IndexParams, SearchParams};
use crate::dataset::private::Sealed as _;
use crate::dataset::{CuvsDataset, Dataset, DatasetKind, DatasetView};
use crate::dlpack::{AsDlTensor, AsDlTensorMut, DLTensorView, DLTensorViewMut};
use crate::error::check_cuvs;
use crate::ffi_utils::{init_handle, path_to_cstring, report_drop_failure};
use crate::neighbors::filters::{Bitset, Filter, with_filter};
use crate::resources::Resources;

type Result<T> = std::result::Result<T, CagraError>;

/// Sole RAII owner of a native CAGRA index handle.
#[derive(Debug)]
struct IndexHandle {
    raw: ffi::cuvsCagraIndex_t,
}

impl IndexHandle {
    fn new() -> Result<Self> {
        let raw = unsafe { init_handle(|out| ffi::cuvsCagraIndexCreate(out))? };
        Ok(Self { raw })
    }

    fn raw(&self) -> ffi::cuvsCagraIndex_t {
        self.raw
    }
}

impl Drop for IndexHandle {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraIndexDestroy(self.raw) }) {
            report_drop_failure("CAGRA index", &e);
        }
    }
}

/// A CAGRA approximate nearest neighbor index borrowing caller-owned dataset storage.
#[derive(Debug)]
pub struct Index<'d> {
    handle: IndexHandle,
    _dataset: PhantomData<&'d ()>,
}

/// A deserialized CAGRA index and the optional dataset storage it views.
///
/// A file serialized without vectors yields `dataset == None` and must have
/// matching storage attached before search. Field order is significant: the
/// native index is destroyed before its dataset owner. Tiered indexes own
/// vectors internally and return no separate dataset owner.
#[derive(Debug)]
pub struct DeserializedIndex<D> {
    handle: IndexHandle,
    dataset: Option<D>,
}

impl<'d> Index<'d> {
    /// Builds a CAGRA index over `dataset` for efficient search.
    ///
    /// `dataset` is a row-major matrix on the host or device implementing
    /// [`AsDlTensor`]. The C++ index keeps a non-owning
    /// view of it, so the returned [`Index`] borrows `dataset` for `'d` and
    /// cannot outlive it.
    pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &'d T) -> Result<Index<'d>>
    where
        T: AsDlTensor + ?Sized,
    {
        let view = DatasetView::new(res, dataset)?;
        let handle = Self::build_handle(res, params, view.raw_dataset_handle())?;
        Ok(Index { handle, _dataset: PhantomData })
    }

    /// Build from an owning dataset or non-owning dataset view.
    pub fn build_from_dataset<'a, D>(
        res: &Resources,
        params: &IndexParams,
        dataset: &'a D,
    ) -> Result<Index<'a>>
    where
        D: CuvsDataset + ?Sized,
    {
        let handle = Self::build_handle(res, params, dataset.raw_dataset_handle())?;
        Ok(Index { handle, _dataset: PhantomData })
    }

    /// Build an owning tiered dense index from a device-padded dataset.
    ///
    /// The input may be released after this call because tiered storage owns
    /// its reordered vectors. `params` must select [`GraphStorage::Tiered`].
    pub fn build_tiered<D>(
        res: &Resources,
        params: &IndexParams,
        dataset: &D,
    ) -> Result<Index<'static>>
    where
        D: CuvsDataset + ?Sized,
    {
        if unsafe { (*params.handle()).graph_storage }
            != ffi::cuvsCagraGraphStorage::CUVS_CAGRA_GRAPH_TIERED
        {
            return Err(CagraError::Validation(
                "build_tiered requires GraphStorage::Tiered".into(),
            ));
        }
        let handle = Self::build_handle(res, params, dataset.raw_dataset_handle())?;
        Ok(Index { handle, _dataset: PhantomData })
    }

    /// Build a tiered VPQ-F16 index from full-precision host standard vectors.
    ///
    /// The native index owns its reordered compressed dataset. The input may be
    /// released after this call. Enable FlowANN building in the native library
    /// and select [`GraphStorage::Tiered`] in `params`.
    pub fn build_compressed<D>(
        res: &Resources,
        params: &IndexParams,
        compression: &CompressionParams,
        dataset: &D,
    ) -> Result<Index<'static>>
    where
        D: CuvsDataset + ?Sized,
    {
        let handle = IndexHandle::new()?;
        check_cuvs(unsafe {
            ffi::cuvsCagraBuildCompressed(
                res.handle(),
                params.handle(),
                compression.handle(),
                dataset.raw_dataset_handle(),
                handle.raw(),
            )
        })?;
        Ok(Index { handle, _dataset: PhantomData })
    }

    /// Return the index's native graph storage mode.
    pub fn graph_storage(&self) -> Result<GraphStorage> {
        graph_storage_impl(&self.handle)
    }

    fn build_handle(
        res: &Resources,
        params: &IndexParams,
        dataset: ffi::cuvsDataset_t,
    ) -> Result<IndexHandle> {
        let handle = IndexHandle::new()?;
        check_cuvs(unsafe {
            ffi::cuvsCagraBuild(res.handle(), params.handle(), dataset, handle.raw())
        })?;
        Ok(handle)
    }

    /// Attach a device-padded dataset and return a search-ready index borrowing it.
    pub fn update_dataset<'a, D>(self, res: &Resources, dataset: &'a D) -> Result<Index<'a>>
    where
        D: CuvsDataset + ?Sized,
    {
        let kind = dataset.dataset_kind()?;
        if kind != DatasetKind::DevicePadded {
            return Err(CagraError::Validation(format!(
                "CAGRA dataset update requires a device-padded view, got {:?}",
                kind
            )));
        }
        check_cuvs(unsafe {
            ffi::cuvsCagraUpdateDataset(
                res.handle(),
                dataset.raw_dataset_handle(),
                self.handle.raw(),
            )
        })?;
        let Self { handle, _dataset: _ } = self;
        Ok(Index { handle, _dataset: PhantomData })
    }

    /// Searches the index for the `k` nearest neighbors of each query.
    ///
    /// `queries`, `neighbors`, and `distances` must reside in device memory and
    /// implement [`AsDlTensor`] /
    /// [`AsDlTensorMut`]. `neighbors` (shape
    /// `n_queries × k`) receives the neighbor indices and `distances` their
    /// distances; both are written in place.
    pub fn search<Q, N, D>(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &Q,
        neighbors: &mut N,
        distances: &mut D,
    ) -> Result<()>
    where
        Q: AsDlTensor + ?Sized,
        N: AsDlTensorMut + ?Sized,
        D: AsDlTensorMut + ?Sized,
    {
        let queries = queries.as_dl_tensor()?;
        let mut neighbors = neighbors.as_dl_tensor_mut()?;
        let mut distances = distances.as_dl_tensor_mut()?;
        search_impl(&self.handle, res, params, &queries, &mut neighbors, &mut distances, None)
    }

    /// Searches the index with a row-level bitset filter.
    pub fn search_filtered<Q, N, D>(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &Q,
        neighbors: &mut N,
        distances: &mut D,
        filter: &Filter<'_, Bitset>,
    ) -> Result<()>
    where
        Q: AsDlTensor + ?Sized,
        N: AsDlTensorMut + ?Sized,
        D: AsDlTensorMut + ?Sized,
    {
        let queries = queries.as_dl_tensor()?;
        let mut neighbors = neighbors.as_dl_tensor_mut()?;
        let mut distances = distances.as_dl_tensor_mut()?;
        search_impl(
            &self.handle,
            res,
            params,
            &queries,
            &mut neighbors,
            &mut distances,
            Some(filter),
        )
    }

    /// Save the CAGRA index to file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file path for saving the index
    /// * `include_dataset` - Whether to write out the dataset to the file
    ///
    /// Deserialize a graph-only file with [`Index::deserialize_graph`], or
    /// recreate the serialized dataset's residency and layout with
    /// [`Index::deserialize_graph_and_dataset`].
    pub fn serialize<P: AsRef<Path>>(
        &self,
        res: &Resources,
        filename: P,
        include_dataset: bool,
    ) -> Result<()> {
        serialize_impl(&self.handle, res, filename.as_ref(), include_dataset)
    }

    /// Save the CAGRA index to file in hnswlib format.
    ///
    /// NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
    /// as the serialization format is not compatible with the original hnswlib.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file path for saving the index
    pub fn serialize_to_hnswlib<P: AsRef<Path>>(&self, res: &Resources, filename: P) -> Result<()> {
        serialize_to_hnswlib_impl(&self.handle, res, filename.as_ref())
    }

    /// Load only the graph for ordinary device storage.
    /// Tiered files always load the complete owning graph and dataset.
    pub fn deserialize_graph<P: AsRef<Path>>(
        res: &Resources,
        filename: P,
    ) -> Result<DeserializedIndex<Dataset>> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        let handle = IndexHandle::new()?;
        check_cuvs(unsafe {
            ffi::cuvsCagraDeserializeGraph(res.handle(), c_filename.as_ptr(), handle.raw())
        })?;
        Ok(DeserializedIndex { handle, dataset: None })
    }

    /// Load the graph and recreate its serialized dataset allocation.
    pub fn deserialize_graph_and_dataset<P: AsRef<Path>>(
        res: &Resources,
        filename: P,
    ) -> Result<DeserializedIndex<Dataset>> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        let handle = IndexHandle::new()?;
        let mut out: ffi::cuvsDataset_t = std::ptr::null_mut();
        check_cuvs(unsafe {
            ffi::cuvsCagraDeserializeGraphAndDataset(
                res.handle(),
                c_filename.as_ptr(),
                handle.raw(),
                &mut out,
            )
        })?;
        let dataset = if out.is_null() {
            if graph_storage_impl(&handle)? != GraphStorage::Tiered {
                return Err(CagraError::Validation(
                    "native load returned no dataset for a device graph".into(),
                ));
            }
            None
        } else {
            Some(Dataset::from_raw(out)?)
        };
        Ok(DeserializedIndex { handle, dataset })
    }
}

impl<D> DeserializedIndex<D> {
    /// Return the index's native graph storage mode.
    pub fn graph_storage(&self) -> Result<GraphStorage> {
        graph_storage_impl(&self.handle)
    }

    /// Borrow the dataset owner when the serialized file included vectors.
    pub fn dataset(&self) -> Option<&D> {
        self.dataset.as_ref()
    }

    /// Whether a separate dataset owner was returned by deserialization.
    /// Tiered indexes keep vector storage inside the native index instead.
    pub fn has_dataset(&self) -> bool {
        self.dataset.is_some()
    }

    /// Save this index to file.
    pub fn serialize<P: AsRef<Path>>(
        &self,
        res: &Resources,
        filename: P,
        include_dataset: bool,
    ) -> Result<()> {
        serialize_impl(&self.handle, res, filename.as_ref(), include_dataset)
    }

    /// Save this index to file in the cuVS hnswlib format.
    pub fn serialize_to_hnswlib<P: AsRef<Path>>(&self, res: &Resources, filename: P) -> Result<()> {
        serialize_to_hnswlib_impl(&self.handle, res, filename.as_ref())
    }

    /// Replace the deserialized storage with a caller-owned device-padded view.
    pub fn update_dataset<'a, T>(self, res: &Resources, dataset: &'a T) -> Result<Index<'a>>
    where
        T: CuvsDataset + ?Sized,
    {
        let kind = dataset.dataset_kind()?;
        if kind != DatasetKind::DevicePadded {
            return Err(CagraError::Validation(format!(
                "CAGRA dataset update requires a device-padded view, got {:?}",
                kind
            )));
        }
        check_cuvs(unsafe {
            ffi::cuvsCagraUpdateDataset(
                res.handle(),
                dataset.raw_dataset_handle(),
                self.handle.raw(),
            )
        })?;
        let Self { handle, .. } = self;
        Ok(Index { handle, _dataset: PhantomData })
    }
}

impl DeserializedIndex<Dataset> {
    /// Search an index whose deserialized owner is device-padded.
    pub fn search<Q, N, D>(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &Q,
        neighbors: &mut N,
        distances: &mut D,
    ) -> Result<()>
    where
        Q: AsDlTensor + ?Sized,
        N: AsDlTensorMut + ?Sized,
        D: AsDlTensorMut + ?Sized,
    {
        self.require_dataset()?;
        let queries = queries.as_dl_tensor()?;
        let mut neighbors = neighbors.as_dl_tensor_mut()?;
        let mut distances = distances.as_dl_tensor_mut()?;
        search_impl(&self.handle, res, params, &queries, &mut neighbors, &mut distances, None)
    }

    /// Search a padded deserialized index with a row-level bitset filter.
    pub fn search_filtered<Q, N, D>(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &Q,
        neighbors: &mut N,
        distances: &mut D,
        filter: &Filter<'_, Bitset>,
    ) -> Result<()>
    where
        Q: AsDlTensor + ?Sized,
        N: AsDlTensorMut + ?Sized,
        D: AsDlTensorMut + ?Sized,
    {
        self.require_dataset()?;
        let queries = queries.as_dl_tensor()?;
        let mut neighbors = neighbors.as_dl_tensor_mut()?;
        let mut distances = distances.as_dl_tensor_mut()?;
        search_impl(
            &self.handle,
            res,
            params,
            &queries,
            &mut neighbors,
            &mut distances,
            Some(filter),
        )
    }

    fn require_dataset(&self) -> Result<()> {
        if self.graph_storage()? == GraphStorage::Tiered {
            return Ok(());
        }
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(CagraError::Validation(
                "cannot search a graph-only index without an attached dataset".to_string(),
            ));
        };
        match dataset.dataset_kind()? {
            DatasetKind::DevicePadded => Ok(()),
            kind => Err(CagraError::Validation(format!(
                "cannot search a deserialized {kind:?} index; attach a device-padded dataset"
            ))),
        }
    }
}

fn graph_storage_impl(handle: &IndexHandle) -> Result<GraphStorage> {
    let mut storage = ffi::cuvsCagraGraphStorage::CUVS_CAGRA_GRAPH_DEVICE;
    check_cuvs(unsafe { ffi::cuvsCagraIndexGetGraphStorage(handle.raw(), &mut storage) })?;
    Ok(storage.into())
}

fn search_impl(
    handle: &IndexHandle,
    res: &Resources,
    params: &SearchParams,
    queries: &DLTensorView<'_>,
    neighbors: &mut DLTensorViewMut<'_>,
    distances: &mut DLTensorViewMut<'_>,
    filter: Option<&Filter<'_, Bitset>>,
) -> Result<()> {
    with_filter(filter, |prefilter| {
        check_cuvs(unsafe {
            ffi::cuvsCagraSearch(
                res.handle(),
                params.handle(),
                handle.raw(),
                queries.to_c().as_mut_ptr(),
                neighbors.to_c().as_mut_ptr(),
                distances.to_c().as_mut_ptr(),
                prefilter,
            )
        })?;
        Ok(())
    })
}

fn serialize_impl(
    handle: &IndexHandle,
    res: &Resources,
    filename: &Path,
    include_dataset: bool,
) -> Result<()> {
    let filename = path_to_cstring(filename)?;
    check_cuvs(unsafe {
        if include_dataset {
            ffi::cuvsCagraSerializeGraphAndDataset(res.handle(), filename.as_ptr(), handle.raw())
        } else {
            ffi::cuvsCagraSerializeGraph(res.handle(), filename.as_ptr(), handle.raw())
        }
    })
    .map_err(CagraError::from)
}

fn serialize_to_hnswlib_impl(handle: &IndexHandle, res: &Resources, filename: &Path) -> Result<()> {
    let filename = path_to_cstring(filename)?;
    check_cuvs(unsafe {
        ffi::cuvsCagraSerializeToHnswlib(res.handle(), filename.as_ptr(), handle.raw())
    })
    .map_err(CagraError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::PaddedDataset;
    use crate::neighbors::filters::{Bitset, Filter};
    use crate::test_utils::DeviceTensor;
    use ndarray::s;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    const N_DATAPOINTS: usize = 256;
    const N_FEATURES: usize = 16;

    fn tiered_dense_fixture() -> ndarray::Array2<f32> {
        ndarray::Array2::from_shape_fn((1024, 32), |(r, c)| {
            let value = (((r * 17 + c * 31) % 1009) as f32) / 1009.0;
            // The modulo pattern repeats after 1009 rows. Distinguish those rows so
            // the self-neighbor assertion cannot reject an equally close duplicate.
            if c == 0 { value + r as f32 / 1024.0 } else { value }
        })
    }

    #[test]
    fn tiered_dense_fixture_has_unique_query_rows() {
        let dataset = tiered_dense_fixture();
        for q in 0..8 {
            let query = dataset.row(q);
            let matches = dataset.outer_iter().filter(|row| row.iter().eq(query.iter())).count();
            assert_eq!(matches, 1, "query row {q} must have a unique exact nearest neighbor");
        }
    }

    /// Search the first `n_queries` rows of `dataset` against `index` and
    /// assert each query finds itself as the top-1 neighbor. CAGRA search
    /// requires queries and outputs to live in device memory.
    fn search_and_verify_self_neighbors(
        res: &Resources,
        index: &Index<'_>,
        dataset: &ndarray::Array2<f32>,
        n_queries: usize,
        k: usize,
    ) {
        let queries = dataset.slice(s![0..n_queries, ..]);
        let queries = DeviceTensor::from_host(res, &queries.to_owned()).unwrap();

        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<u32>::zeros(res, &[n_queries, k]).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let mut distances = DeviceTensor::<f32>::zeros(res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::builder().build().unwrap();
        index
            .search(res, &search_params, &queries, &mut neighbors, &mut distances)
            .expect("search failed");

        distances.copy_to_host(res, &mut distances_host).unwrap();
        neighbors.copy_to_host(res, &mut neighbors_host).unwrap();

        for i in 0..n_queries {
            assert_eq!(
                neighbors_host[[i, 0]],
                i as u32,
                "query {i} should be its own nearest neighbor"
            );
        }
    }

    fn test_cagra(build_params: IndexParams) {
        let res = Resources::new().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to build cagra index");
        search_and_verify_self_neighbors(&res, &index, &dataset, 4, 10);
    }

    // FlowANN tests require a native library configured with SEARCH and BUILD.
    #[test]
    #[ignore = "requires native FlowANN SEARCH and BUILD support"]
    fn tiered_dense_build_search_and_round_trip() {
        use super::super::{GraphStorage, TieredGraphParams, TieredSearchParams};
        let res = Resources::new().unwrap();
        let dataset = tiered_dense_fixture();
        let device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let params = IndexParams::builder()
            .nn_descent_with_iterations(10)
            .intermediate_graph_degree(32)
            .graph_degree(16)
            .graph_storage(GraphStorage::Tiered)
            .tiered(TieredGraphParams {
                device_graph_budget_bytes: 8192,
                validate: true,
                ..Default::default()
            })
            .build()
            .unwrap();
        let view = DatasetView::new(&res, &device).unwrap();
        let index = Index::build_tiered(&res, &params, &view).unwrap();
        drop(view);
        drop(device);
        assert_eq!(index.graph_storage().unwrap(), GraphStorage::Tiered);
        let search = SearchParams::builder()
            .algo(super::super::SearchAlgo::MultiCta)
            .itopk_size(64)
            .max_iterations(32)
            .tiered(TieredSearchParams { keep_pollers_running: true, ..Default::default() })
            .build()
            .unwrap();
        let queries =
            DeviceTensor::from_host(&res, &dataset.slice(s![0..8, ..]).to_owned()).unwrap();
        let mut neighbors = DeviceTensor::<u32>::zeros(&res, &[8, 1]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[8, 1]).unwrap();
        for _ in 0..2 {
            index.search(&res, &search, &queries, &mut neighbors, &mut distances).unwrap();
        }
        let dir = tempfile::tempdir().unwrap();
        let filename = dir.path().join("tiered.index");
        index.serialize(&res, &filename, true).unwrap();
        assert!(index.serialize(&res, dir.path().join("graph-only.index"), false).is_err());
        drop(index);
        let loaded = Index::deserialize_graph_and_dataset(&res, &filename).unwrap();
        assert_eq!(loaded.graph_storage().unwrap(), GraphStorage::Tiered);
        assert!(!loaded.has_dataset());
        loaded.search(&res, &search, &queries, &mut neighbors, &mut distances).unwrap();
        let mut found = ndarray::Array2::<u32>::zeros((8, 1));
        neighbors.copy_to_host(&res, &mut found).unwrap();
        assert!(found.iter().all(|&id| id < 1024));
        assert!(found.column(0).iter().enumerate().filter(|&(i, id)| *id == i as u32).count() >= 6);
    }

    #[test]
    #[ignore = "requires native FlowANN SEARCH and BUILD support"]
    fn tiered_compressed_build_search_and_round_trip() {
        use super::super::{CompressionParams, GraphStorage, TieredGraphParams};
        let res = Resources::new().unwrap();
        let dataset = ndarray::Array2::from_shape_fn((2048, 32), |(r, c)| {
            (((r * 17 + c * 31) % 2017) as f32) / 2017.0
        });
        let params = IndexParams::builder()
            .nn_descent_with_iterations(10)
            .intermediate_graph_degree(32)
            .graph_degree(16)
            .graph_storage(GraphStorage::Tiered)
            .tiered(TieredGraphParams { validate: true, ..Default::default() })
            .build()
            .unwrap();
        let compression = CompressionParams::builder()
            .pq_dim(4)
            .vq_n_centers(16)
            .kmeans_n_iters(5)
            .build()
            .unwrap();
        let view = DatasetView::standard(&res, &*dataset).unwrap();
        let index = Index::build_compressed(&res, &params, &compression, &view).unwrap();
        drop(view);
        let queries =
            DeviceTensor::from_host(&res, &dataset.slice(s![0..8, ..]).to_owned()).unwrap();
        drop(dataset);
        let search = SearchParams::builder()
            .algo(super::super::SearchAlgo::MultiCta)
            .itopk_size(64)
            .max_iterations(32)
            .build()
            .unwrap();
        let mut neighbors = DeviceTensor::<u32>::zeros(&res, &[8, 1]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[8, 1]).unwrap();
        index.search(&res, &search, &queries, &mut neighbors, &mut distances).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let filename = dir.path().join("tiered-vpq.index");
        index.serialize(&res, &filename, true).unwrap();
        drop(index);
        let loaded = Index::deserialize_graph_and_dataset(&res, &filename).unwrap();
        assert_eq!(loaded.graph_storage().unwrap(), GraphStorage::Tiered);
        loaded.search(&res, &search, &queries, &mut neighbors, &mut distances).unwrap();
        let mut found = ndarray::Array2::<u32>::zeros((8, 1));
        neighbors.copy_to_host(&res, &mut found).unwrap();
        assert!(found.iter().all(|&id| id < 2048));
    }

    #[test]
    fn test_cagra_index() {
        let build_params = IndexParams::builder().build().unwrap();
        test_cagra(build_params);
    }

    #[test]
    fn explicit_views_classify_and_build_all_supported_kinds() {
        let res = Resources::new().unwrap();
        let params = IndexParams::builder().build().unwrap();
        let host_padded = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let host_standard = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES - 1),
            Uniform::new(0., 1.0).unwrap(),
        );
        let device_padded = DeviceTensor::from_host(&res, &host_padded).unwrap();
        let device_standard = DeviceTensor::from_host(&res, &host_standard).unwrap();

        let views = [
            (DatasetView::new(&res, &*host_padded).unwrap(), DatasetKind::HostPadded),
            (DatasetView::new(&res, &*host_standard).unwrap(), DatasetKind::HostStandard),
            (DatasetView::new(&res, &device_padded).unwrap(), DatasetKind::DevicePadded),
            (DatasetView::new(&res, &device_standard).unwrap(), DatasetKind::DeviceStandard),
        ];

        for (view, expected_kind) in &views {
            assert_eq!(view.dataset_kind().unwrap(), *expected_kind);
            let index = Index::build_from_dataset(&res, &params, view)
                .expect("every supported dataset kind should build");
            if *expected_kind == DatasetKind::DevicePadded {
                search_and_verify_self_neighbors(&res, &index, &host_padded, 4, 10);
            }
        }

        let owner = PaddedDataset::new(&res, &device_standard).unwrap();
        let index = Index::build_from_dataset(&res, &params, &owner).unwrap();
        search_and_verify_self_neighbors(&res, &index, &host_standard, 4, 10);
    }

    #[test]
    fn update_rejects_a_standard_view() {
        let res = Resources::new().unwrap();
        let dataset = ndarray::Array::<f32, _>::zeros((N_DATAPOINTS, N_FEATURES - 1));
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let params = IndexParams::builder().build().unwrap();
        let index = Index::build(&res, &params, &dataset_device).unwrap();
        let standard_view = DatasetView::new(&res, &dataset_device).unwrap();

        let err = index
            .update_dataset(&res, &standard_view)
            .expect_err("standard views cannot be attached for search");

        assert!(matches!(err, CagraError::Validation(_)), "unexpected error: {err:?}");
    }

    #[test]
    fn update_rebinds_the_index_to_new_backing_storage() {
        let res = Resources::new().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES - 1),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let params = IndexParams::builder().build().unwrap();
        let index = Index::build(&res, &params, &dataset_device).unwrap();
        let owner = PaddedDataset::new(&res, &dataset_device).unwrap();

        let index = index.update_dataset(&res, &owner).unwrap();
        drop(dataset_device);

        search_and_verify_self_neighbors(&res, &index, &dataset, 4, 10);
    }

    /// Test bitset-filtered search: exclude odd-indexed rows, verify they don't appear.
    #[test]
    fn test_cagra_search_filtered() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::builder().build().unwrap();

        let n_datapoints = 256;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );

        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to create cagra index");

        // Build a bitset that includes only even-indexed rows
        let n_words = n_datapoints.div_ceil(32);
        let mut bitset_host = ndarray::Array::<u32, _>::zeros(ndarray::Ix1(n_words));
        for i in 0..n_datapoints {
            if i % 2 == 0 {
                bitset_host[i / 32] |= 1u32 << (i % 32);
            }
        }
        let bitset = DeviceTensor::from_host(&res, &bitset_host).unwrap();

        // Query with the first 4 even-indexed rows
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries * 2;2, ..]).to_owned(); // rows 0, 2, 4, 6
        let queries = DeviceTensor::from_host(&res, &queries).unwrap();

        let k = 10;
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<u32>::zeros(&res, &[n_queries, k]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::builder().build().unwrap();
        let filter = Filter::<Bitset>::new(&bitset).unwrap();

        index
            .search_filtered(
                &res,
                &search_params,
                &queries,
                &mut neighbors,
                &mut distances,
                &filter,
            )
            .unwrap();

        neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();

        // All returned neighbors must be even-indexed (odd rows are filtered out).
        for q in 0..n_queries {
            for n in 0..k {
                let neighbor_id = neighbors_host[[q, n]];
                assert_eq!(
                    neighbor_id % 2,
                    0,
                    "query {q}, neighbor {n}: got odd index {neighbor_id}, expected only even"
                );
            }
        }

        // First query (row 0) should find itself as the nearest neighbor.
        assert_eq!(neighbors_host[[0, 0]], 0);
    }

    /// Test that an index can be searched multiple times without rebuilding.
    /// This validates that `search()` takes `&self` instead of `self`.
    #[test]
    fn test_cagra_multiple_searches() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::builder().build().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to build cagra index");

        for _ in 0..3 {
            search_and_verify_self_neighbors(&res, &index, &dataset, 4, 5);
        }
    }

    #[test]
    fn padded_deserialization_keeps_dataset_alive_and_searches() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::builder().build().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to build cagra index");

        let filepath = std::env::temp_dir().join("test_cagra_index.bin");
        index.serialize(&res, &filepath, true).expect("failed to serialize cagra index");
        drop(index);
        drop(dataset_device);

        let loaded = Index::deserialize_graph_and_dataset(&res, &filepath)
            .expect("failed to deserialize cagra index");
        assert_eq!(loaded.dataset().unwrap().dataset_kind().unwrap(), DatasetKind::DevicePadded);

        let queries =
            DeviceTensor::from_host(&res, &dataset.slice(s![0..1, ..]).to_owned()).unwrap();
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((1, 1));
        let mut neighbors = DeviceTensor::<u32>::zeros(&res, &[1, 1]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[1, 1]).unwrap();
        let search_params = SearchParams::builder().build().unwrap();
        loaded
            .search(&res, &search_params, &queries, &mut neighbors, &mut distances)
            .expect("padded deserialized index should be searchable");
        neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();
        assert_eq!(neighbors_host[[0, 0]], 0);

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_standard_serialize_deserialize_and_attach() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::builder().build().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES - 1),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to build standard cagra index");

        let filepath = std::env::temp_dir().join("test_cagra_standard_index.bin");
        index.serialize(&res, &filepath, true).expect("failed to serialize cagra index");
        drop(index);
        drop(dataset_device);

        let loaded = Index::deserialize_graph_and_dataset(&res, &filepath)
            .expect("failed to deserialize standard cagra index");
        assert_eq!(loaded.dataset().unwrap().dataset_kind().unwrap(), DatasetKind::DeviceStandard);

        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let owner = PaddedDataset::new(&res, &dataset_device).unwrap();
        let index = loaded.update_dataset(&res, &owner).unwrap();
        drop(dataset_device);
        search_and_verify_self_neighbors(&res, &index, &dataset, 4, 10);

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn graph_only_deserialization_rejects_search_until_attachment() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::builder().build().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device).unwrap();
        let filepath = std::env::temp_dir().join("test_cagra_graph_only.bin");
        index.serialize(&res, &filepath, false).unwrap();
        drop(index);

        let err = Index::deserialize_graph_and_dataset(&res, &filepath)
            .expect_err("graph-only file must not deserialize a dataset");
        assert!(err.to_string().contains("no dataset"), "unexpected error: {err:?}");

        let loaded = Index::deserialize_graph(&res, &filepath).unwrap();
        assert!(!loaded.has_dataset());

        let queries =
            DeviceTensor::from_host(&res, &dataset.slice(s![0..1, ..]).to_owned()).unwrap();
        let mut neighbors = DeviceTensor::<u32>::zeros(&res, &[1, 1]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[1, 1]).unwrap();
        let search_params = SearchParams::builder().build().unwrap();
        let err = loaded
            .search(&res, &search_params, &queries, &mut neighbors, &mut distances)
            .expect_err("graph-only index must reject search");
        assert!(matches!(err, CagraError::Validation(_)), "unexpected error: {err:?}");

        let view = DatasetView::new(&res, &dataset_device).unwrap();
        assert_eq!(view.dataset_kind().unwrap(), DatasetKind::DevicePadded);
        let index = loaded.update_dataset(&res, &view).unwrap();
        search_and_verify_self_neighbors(&res, &index, &dataset, 4, 10);

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_to_hnswlib() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::builder().build().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to build cagra index");

        let filepath = std::env::temp_dir().join("test_cagra_index_hnsw.bin");
        index
            .serialize_to_hnswlib(&res, &filepath)
            .expect("failed to serialize cagra index to hnswlib format");

        assert!(filepath.exists(), "serialized hnswlib index file should exist");
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized hnswlib index file should not be empty"
        );

        let _ = std::fs::remove_file(&filepath);
    }

    /// Passing a filename containing an interior NUL byte must surface as an
    /// `InvalidPath` error rather than panicking inside the serializer.
    #[test]
    fn test_cagra_serialize_rejects_interior_nul() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::builder().build().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to build cagra index");

        // `PathBuf::from` on Unix preserves arbitrary bytes, so we can embed a
        // NUL byte in the path and confirm the helper rejects it.
        let bad_path = std::path::PathBuf::from("/tmp/has\0nul.bin");
        let err = index
            .serialize(&res, &bad_path, true)
            .expect_err("serialize should reject paths with interior NUL");
        assert!(matches!(err, CagraError::InvalidPath(_)), "expected InvalidPath, got {err:?}");
    }
}
