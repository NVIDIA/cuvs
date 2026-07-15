/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CAGRA example with a user-provided GPU tensor.
//!
//! This demonstrates how to feed your own device memory into cuVS by
//! implementing the public [`AsDlTensor`]/[`AsDlTensorMut`] traits. The
//! [`CudaTensor`] type manages device memory directly through the CUDA runtime
//! (`cudaMalloc`/`cudaFree`) and copies to/from host arrays with `cudaMemcpyAsync`
//! on the cuVS stream, reusing the resources handle's `get_cuda_stream`/
//! `sync_stream` for stream access and synchronization.
//!
//! A real application would likely rely on a helper crate such as `cudarc`
//! and its `CudaSlice`.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::os::raw::c_int;

use cuvs::Resources;
use cuvs::cagra::{Index, IndexParams, SearchParams};
use cuvs::dlpack::{
    AsDlTensor, AsDlTensorMut, DLDevice, DLDeviceType, DLPackError, DLTensorView, DLTensorViewMut,
    DType,
};

use ndarray::s;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

type ExampleResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Minimal CUDA runtime FFI
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type cudaError_t = c_int;
const CUDA_SUCCESS: cudaError_t = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    fn cudaFree(ptr: *mut c_void) -> cudaError_t;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: cuvs_sys::cudaStream_t,
    ) -> cudaError_t;
}

fn check_cuda(status: cudaError_t) -> ExampleResult<()> {
    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("CUDA runtime error: {status}").into())
    }
}

// ---------------------------------------------------------------------------
// A custom device tensor backed by the CUDA runtime
// ---------------------------------------------------------------------------

struct CudaTensor<T: DType> {
    data: *mut c_void,
    shape: Vec<i64>,
    bytes: usize,
    _marker: PhantomData<T>,
}

impl<T: DType> CudaTensor<T> {
    /// Allocate an uninitialized device buffer (used for search outputs).
    fn alloc(shape: &[usize]) -> ExampleResult<Self> {
        let bytes = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let mut data: *mut c_void = std::ptr::null_mut();
        check_cuda(unsafe { cudaMalloc(&mut data, bytes) })?;
        Ok(Self {
            data,
            shape: shape.iter().map(|&d| d as i64).collect(),
            bytes,
            _marker: PhantomData,
        })
    }

    /// Copy a contiguous host array onto the device.
    fn from_host<D>(res: &Resources, host: &ndarray::ArrayRef<T, D>) -> ExampleResult<Self>
    where
        D: ndarray::Dimension,
    {
        if !host.is_standard_layout() {
            return Err("host array must be contiguous (row-major)".into());
        }
        let tensor = Self::alloc(host.shape())?;

        let stream = res.get_cuda_stream()?;
        check_cuda(unsafe {
            cudaMemcpyAsync(
                tensor.data,
                host.as_ptr() as *const c_void,
                tensor.bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        })?;
        res.sync_stream()?;

        Ok(tensor)
    }

    /// Copy the device buffer back into a contiguous host array.
    fn to_host<D>(&self, res: &Resources, host: &mut ndarray::ArrayRef<T, D>) -> ExampleResult<()>
    where
        D: ndarray::Dimension,
    {
        if !host.is_standard_layout() {
            return Err("host array must be contiguous (row-major)".into());
        }

        let stream = res.get_cuda_stream()?;
        check_cuda(unsafe {
            cudaMemcpyAsync(
                host.as_mut_ptr() as *mut c_void,
                self.data,
                self.bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        })?;
        res.sync_stream()?;

        Ok(())
    }
}

impl<T: DType> Drop for CudaTensor<T> {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe { cudaFree(self.data) };
        }
    }
}

impl<T: DType> AsDlTensor for CudaTensor<T> {
    fn as_dl_tensor(&self) -> std::result::Result<DLTensorView<'_>, DLPackError> {
        unsafe {
            DLTensorView::from_raw_parts(
                self.data,
                DLDevice { device_type: DLDeviceType::kDLCUDA, device_id: 0 },
                &self.shape,
                None,
                T::dl_dtype(),
            )
        }
    }
}

impl<T: DType> AsDlTensorMut for CudaTensor<T> {
    fn as_dl_tensor_mut(&mut self) -> std::result::Result<DLTensorViewMut<'_>, DLPackError> {
        unsafe {
            DLTensorViewMut::from_raw_parts(
                self.data,
                DLDevice { device_type: DLDeviceType::kDLCUDA, device_id: 0 },
                &self.shape,
                None,
                T::dl_dtype(),
            )
        }
    }
}

/// Example showing how to index and search data with CAGRA.
fn cagra_example() -> ExampleResult<()> {
    let res = Resources::new()?;

    // Create a new random dataset to index and copy it to the device.
    let n_datapoints = 65536;
    let n_features = 512;
    let dataset_host = ndarray::Array::<f32, _>::random(
        (n_datapoints, n_features),
        Uniform::new(0., 1.0).unwrap(),
    );
    let dataset = CudaTensor::from_host(&res, &dataset_host)?;

    // Build the CAGRA index.
    let build_params = IndexParams::new()?;
    let index = Index::build(&res, &build_params, &dataset)?;
    println!("Indexed {n_datapoints}x{n_features} datapoints into cagra index");

    // Use the first 4 points as queries; each should be its own nearest neighbor.
    let n_queries = 4;
    let k = 10;
    let queries_host = dataset_host.slice(s![0..n_queries, ..]).to_owned();
    let queries = CudaTensor::from_host(&res, &queries_host)?;

    let mut neighbors = CudaTensor::<u32>::alloc(&[n_queries, k])?;
    let mut distances = CudaTensor::<f32>::alloc(&[n_queries, k])?;

    let search_params = SearchParams::new()?;
    index.search(&res, &search_params, &queries, &mut neighbors, &mut distances)?;

    // Copy the results back to the host.
    let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    neighbors.to_host(&res, &mut neighbors_host)?;
    distances.to_host(&res, &mut distances_host)?;

    println!("Neighbors {neighbors_host:?}");
    println!("Distances {distances_host:?}");
    Ok(())
}

fn main() {
    if let Err(e) = cagra_example() {
        println!("Failed to run CAGRA: {e:?}");
    }
}
