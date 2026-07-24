/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::CString;
use std::io::{Write, stderr};
use std::marker::PhantomData;
use std::path::Path;

use super::{CagraError, IndexParams, SearchParams};
use crate::dlpack::{AsDlTensor, AsDlTensorMut, DLTensorView, DLTensorViewMut};
use crate::error::check_cuvs;
use crate::neighbors::filters::{Bitset, Filter, with_filter};
use crate::resources::Resources;

type Result<T> = std::result::Result<T, CagraError>;

/// A CAGRA approximate nearest neighbor index.
///
/// The lifetime `'d` ties this index to the underlying dataset,
/// passed at construction time. The C library may store a non-owning view
/// of properly aligned device-resident data, so the dataset must outlive
/// the index. When an index is deserialized from disk, the data is
/// self-contained and its lifetime is `'static`.
#[derive(Debug)]
pub struct Index<'d> {
    handle: ffi::cuvsCagraIndex_t,
    _dataset: PhantomData<&'d ()>,
}

fn is_device_compatible(device_type: ffi::DLDeviceType) -> bool {
    matches!(device_type, ffi::DLDeviceType::kDLCUDA | ffi::DLDeviceType::kDLCUDAManaged)
}

fn is_host_compatible(device_type: ffi::DLDeviceType) -> bool {
    matches!(device_type, ffi::DLDeviceType::kDLCPU | ffi::DLDeviceType::kDLCUDAHost)
}

/// User-owned padded dataset handle for CAGRA search attachment.
///
/// This mirrors the C API contract: callers allocate padded storage explicitly
/// and keep it alive for as long as attached indices need it.
#[derive(Debug)]
pub struct PaddedDataset {
    handle: ffi::cuvsDatasetPadded_t,
}

/// User-owned standard dataset handle returned by CAGRA deserialization.
#[derive(Debug)]
pub struct StandardDataset {
    handle: ffi::cuvsDatasetStandard_t,
}

/// Non-owning padded dataset view handle.
#[derive(Debug)]
pub struct PaddedDatasetView {
    handle: ffi::cuvsDatasetPaddedView_t,
}

impl PaddedDataset {
    /// Create a padded dataset handle from a source dataset tensor.
    pub fn new<T>(res: &Resources, dataset: &T) -> Result<Self>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        unsafe {
            let mut dataset_c = dataset.to_c();
            let device_type = dataset_c.inner.dl_tensor.device.device_type;
            let mut padded = std::mem::MaybeUninit::<ffi::cuvsDatasetPadded_t>::uninit();
            if is_device_compatible(device_type) {
                check_cuvs(ffi::cuvsDatasetMakeDevicePadded(
                    res.handle(),
                    dataset_c.as_mut_ptr(),
                    padded.as_mut_ptr(),
                ))?;
            } else if is_host_compatible(device_type) {
                check_cuvs(ffi::cuvsDatasetMakeHostPadded(
                    res.handle(),
                    dataset_c.as_mut_ptr(),
                    padded.as_mut_ptr(),
                ))?;
            } else {
                return Err(CagraError::Validation(format!(
                    "unsupported dataset device type for padded dataset: {device_type:?}"
                )));
            }
            Ok(Self { handle: padded.assume_init() })
        }
    }
}

impl PaddedDatasetView {
    /// Create a padded dataset view handle from a source dataset tensor.
    pub fn new<T>(res: &Resources, dataset: &T) -> Result<Self>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        unsafe {
            let mut dataset_c = dataset.to_c();
            let device_type = dataset_c.inner.dl_tensor.device.device_type;
            let mut padded = std::mem::MaybeUninit::<ffi::cuvsDatasetPaddedView_t>::uninit();
            if is_device_compatible(device_type) {
                check_cuvs(ffi::cuvsDatasetMakeDevicePaddedView(
                    res.handle(),
                    dataset_c.as_mut_ptr(),
                    padded.as_mut_ptr(),
                ))?;
            } else if is_host_compatible(device_type) {
                check_cuvs(ffi::cuvsDatasetMakeHostPaddedView(
                    res.handle(),
                    dataset_c.as_mut_ptr(),
                    padded.as_mut_ptr(),
                ))?;
            } else {
                return Err(CagraError::Validation(format!(
                    "unsupported dataset device type for padded dataset view: {device_type:?}"
                )));
            }
            Ok(Self { handle: padded.assume_init() })
        }
    }
}

/// Non-owning standard dataset view handle for symmetry with C API dataset handles.
#[derive(Debug)]
pub struct StandardDatasetView {
    handle: ffi::cuvsDatasetStandardView_t,
}

/// Typed output selector for CAGRA deserialization.
#[derive(Debug)]
pub enum DeserializeOutput<'a> {
    Padded(&'a mut Option<PaddedDataset>),
    Standard(&'a mut Option<StandardDataset>),
}

impl StandardDatasetView {
    /// Create a standard dataset view handle from a source dataset tensor.
    pub fn new<T>(res: &Resources, dataset: &T) -> Result<Self>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        unsafe {
            let mut dataset_c = dataset.to_c();
            let device_type = dataset_c.inner.dl_tensor.device.device_type;
            let mut standard = std::mem::MaybeUninit::<ffi::cuvsDatasetStandardView_t>::uninit();
            if is_device_compatible(device_type) {
                check_cuvs(ffi::cuvsDatasetMakeDeviceStandardView(
                    res.handle(),
                    dataset_c.as_mut_ptr(),
                    standard.as_mut_ptr(),
                ))?;
            } else if is_host_compatible(device_type) {
                check_cuvs(ffi::cuvsDatasetMakeHostStandardView(
                    res.handle(),
                    dataset_c.as_mut_ptr(),
                    standard.as_mut_ptr(),
                ))?;
            } else {
                return Err(CagraError::Validation(format!(
                    "unsupported dataset device type for standard dataset view: {device_type:?}"
                )));
            }
            Ok(Self { handle: standard.assume_init() })
        }
    }
}

/// Convert a filesystem path into a `CString` suitable for the cuVS C API,
/// returning [`CagraError::InvalidPath`] for a path with an interior NUL byte.
fn path_to_cstring(path: &Path) -> Result<CString> {
    Ok(CString::new(path.as_os_str().as_encoded_bytes())?)
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
        let dataset = dataset.as_dl_tensor()?;
        let index = Index::create_handle()?;
        unsafe {
            let mut dataset_c = dataset.to_c();
            let mut view_kind = std::mem::MaybeUninit::<ffi::cuvsDatasetViewKind_t>::uninit();
            check_cuvs(ffi::cuvsCagraGetDatasetViewKind(
                dataset_c.as_mut_ptr(),
                view_kind.as_mut_ptr(),
            ))?;

            match view_kind.assume_init() {
                ffi::cuvsDatasetViewKind_t::CUVS_DATASET_VIEW_KIND_DEVICE_PADDED => {
                    let dataset_view = PaddedDatasetView::new(res, &dataset)?;
                    check_cuvs(ffi::cuvsCagraBuildDevicePadded(
                        res.handle(),
                        params.handle(),
                        dataset_view.handle,
                        index.handle,
                    ))?;
                }
                ffi::cuvsDatasetViewKind_t::CUVS_DATASET_VIEW_KIND_DEVICE_STANDARD => {
                    let dataset_view = StandardDatasetView::new(res, &dataset)?;
                    check_cuvs(ffi::cuvsCagraBuildDeviceStandard(
                        res.handle(),
                        params.handle(),
                        dataset_view.handle,
                        index.handle,
                    ))?;
                }
                ffi::cuvsDatasetViewKind_t::CUVS_DATASET_VIEW_KIND_HOST_PADDED => {
                    let dataset_view = PaddedDatasetView::new(res, &dataset)?;
                    check_cuvs(ffi::cuvsCagraBuildHostPadded(
                        res.handle(),
                        params.handle(),
                        dataset_view.handle,
                        index.handle,
                    ))?;
                }
                ffi::cuvsDatasetViewKind_t::CUVS_DATASET_VIEW_KIND_HOST_STANDARD => {
                    let dataset_view = StandardDatasetView::new(res, &dataset)?;
                    check_cuvs(ffi::cuvsCagraBuildHostStandard(
                        res.handle(),
                        params.handle(),
                        dataset_view.handle,
                        index.handle,
                    ))?;
                }
            }
        }
        Ok(index)
    }

    fn create_handle() -> Result<Index<'d>> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsCagraIndex_t>::uninit();
            check_cuvs(ffi::cuvsCagraIndexCreate(index.as_mut_ptr()))?;
            Ok(Index { handle: index.assume_init(), _dataset: PhantomData })
        }
    }

    /// Attaches a user-owned padded device dataset and converts the index to
    /// a search-ready padded-device layout.
    pub fn attach_dataset(
        &mut self,
        res: &Resources,
        padded_dataset: &PaddedDatasetView,
    ) -> Result<()> {
        if padded_dataset.handle.is_null() {
            return Err(CagraError::Validation("padded dataset view is uninitialized".to_string()));
        }
        unsafe {
            check_cuvs(ffi::cuvsCagraAttachDataset(
                res.handle(),
                padded_dataset.handle,
                self.handle,
            ))?;
        }
        Ok(())
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
        self.search_impl(res, params, &queries, &mut neighbors, &mut distances, None)
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
        self.search_impl(res, params, &queries, &mut neighbors, &mut distances, Some(filter))
    }

    fn search_impl(
        &self,
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
                    self.handle,
                    queries.to_c().as_mut_ptr(),
                    neighbors.to_c().as_mut_ptr(),
                    distances.to_c().as_mut_ptr(),
                    prefilter,
                )
            })?;
            Ok(())
        })
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
    /// # Example:
    /// ```ignore
    /// use cuvs::Resources;
    /// use cuvs::neighbors::cagra::{DeserializeOutput, Index, IndexParams};
    ///
    /// fn serialize_example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let res = Resources::new()?;
    ///
    ///     // Build an index (using some dataset)
    ///     let build_params = IndexParams::builder().build()?;
    ///     // let index = Index::build(&res, &build_params, &dataset)?;
    ///
    ///     // Save the index to disk (including the dataset)
    ///     // index.serialize(&res, "/path/to/index.bin", true)?;
    ///
    ///     // Later, load the index from disk
    ///     let mut loaded_index = Index::new()?;
    ///     let mut out_dataset = None;
    ///     Index::deserialize(
    ///         &res,
    ///         "/path/to/index.bin",
    ///         &mut loaded_index,
    ///         DeserializeOutput::Standard(&mut out_dataset),
    ///     )?;
    ///
    ///     // The loaded index can be used for search just like the original
    ///     Ok(())
    /// }
    /// ```
    pub fn serialize<P: AsRef<Path>>(
        &self,
        res: &Resources,
        filename: P,
        include_dataset: bool,
    ) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        check_cuvs(unsafe {
            ffi::cuvsCagraSerialize(res.handle(), c_filename.as_ptr(), self.handle, include_dataset)
        })?;
        Ok(())
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
        let c_filename = path_to_cstring(filename.as_ref())?;
        check_cuvs(unsafe {
            ffi::cuvsCagraSerializeToHnswlib(res.handle(), c_filename.as_ptr(), self.handle)
        })?;
        Ok(())
    }

    /// Load a CAGRA index from file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The path of the file that stores the index
    pub fn deserialize<P: AsRef<Path>>(
        res: &Resources,
        filename: P,
        index: &mut Index<'static>,
        out_dataset: DeserializeOutput<'_>,
    ) -> Result<()> {
        match out_dataset {
            DeserializeOutput::Padded(out) => Self::deserialize_padded(res, filename, index, out),
            DeserializeOutput::Standard(out) => {
                Self::deserialize_standard(res, filename, index, out)
            }
        }
    }

    fn deserialize_padded<P: AsRef<Path>>(
        res: &Resources,
        filename: P,
        index: &mut Index<'static>,
        out_dataset: &mut Option<PaddedDataset>,
    ) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        let mut out: ffi::cuvsDatasetPadded_t = std::ptr::null_mut();
        unsafe {
            check_cuvs(ffi::cuvsCagraDeserializePadded(
                res.handle(),
                c_filename.as_ptr(),
                index.handle,
                &mut out,
            ))?;
        }
        *out_dataset = if out.is_null() { None } else { Some(PaddedDataset { handle: out }) };
        Ok(())
    }

    fn deserialize_standard<P: AsRef<Path>>(
        res: &Resources,
        filename: P,
        index: &mut Index<'static>,
        out_dataset: &mut Option<StandardDataset>,
    ) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        let mut out: ffi::cuvsDatasetStandard_t = std::ptr::null_mut();
        unsafe {
            check_cuvs(ffi::cuvsCagraDeserializeStandard(
                res.handle(),
                c_filename.as_ptr(),
                index.handle,
                &mut out,
            ))?;
        }
        *out_dataset = if out.is_null() { None } else { Some(StandardDataset { handle: out }) };
        Ok(())
    }
}

impl Drop for Index<'_> {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraIndexDestroy(self.handle) }) {
            write!(stderr(), "failed to call cagraIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

impl Drop for PaddedDataset {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsDatasetPaddedDestroy(self.handle) }) {
            write!(stderr(), "failed to call cuvsDatasetPaddedDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

impl Drop for PaddedDatasetView {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsDatasetPaddedViewDestroy(self.handle) }) {
            write!(stderr(), "failed to call cuvsDatasetPaddedViewDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

impl Drop for StandardDataset {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsDatasetStandardDestroy(self.handle) }) {
            write!(stderr(), "failed to call cuvsDatasetStandardDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

impl Drop for StandardDatasetView {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsDatasetStandardViewDestroy(self.handle) }) {
            write!(stderr(), "failed to call cuvsDatasetStandardViewDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbors::filters::{Bitset, Filter};
    use crate::test_utils::DeviceTensor;
    use ndarray::s;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    const N_DATAPOINTS: usize = 256;
    const N_FEATURES: usize = 16;

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

    #[test]
    fn test_cagra_index() {
        let build_params = IndexParams::builder().build().unwrap();
        test_cagra(build_params);
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
    fn test_cagra_serialize_deserialize() {
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
        index.serialize(&res, &filepath, false).expect("failed to serialize cagra index");

        assert!(filepath.exists(), "serialized index file should exist");
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized index file should not be empty"
        );

        let mut loaded_index = Index::create_handle().expect("failed to create index");
        let mut out_dataset: Option<StandardDataset> = None;
        Index::deserialize(
            &res,
            &filepath,
            &mut loaded_index,
            DeserializeOutput::Standard(&mut out_dataset),
        )
        .expect("failed to deserialize cagra index");

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_without_dataset() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::builder().build().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to build cagra index");

        let filepath = std::env::temp_dir().join("test_cagra_index_no_dataset.bin");
        index
            .serialize(&res, &filepath, false)
            .expect("failed to serialize cagra index without dataset");

        assert!(filepath.exists(), "serialized index file should exist");

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
