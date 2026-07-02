# cuvs 26.06.00 (3 Jun 2026)

### 🚨 Breaking Changes
* Default to static linking of libcudart by @bdice in https://github.com/rapidsai/cuvs/pull/1627
* Remove JIT+LTO fragment database by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1927
* Use static cudart by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1931
* Always build with JIT+LTO by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1923
* Migrate RMM usage to CCCL MR design by @bdice in https://github.com/rapidsai/cuvs/pull/1990
* Exposition of KMeans param object for PQ in C++ by @lowener in https://github.com/rapidsai/cuvs/pull/2005
* [Cleanup] Combine Batched and Regular KMeans Impl by @tarang-jain in https://github.com/rapidsai/cuvs/pull/2015
* Preserve input memory location for NN Descent by @jinsolp in https://github.com/rapidsai/cuvs/pull/1928
### 🐛 Bug Fixes
* Fix CCCL compilation error by @viclafargue in https://github.com/rapidsai/cuvs/pull/1963
* Forward-merge release/26.04 into main by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1971
* Forward-merge release/26.04 into main by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1980
* Remove dangling pointers in JIT Fragments by @divyegala in https://github.com/rapidsai/cuvs/pull/1988
* Add `head_rev` to cuvs recipe by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1993
* Fix potential OOB access in CAGRA search when graph size < dataset size by @irina-resh-nvda in https://github.com/rapidsai/cuvs/pull/1780
* Fix MG kmeans intertia_check n_iters by @aamijar in https://github.com/rapidsai/cuvs/pull/2020
* Fix cuvs_bench pytest pareto assert by @aamijar in https://github.com/rapidsai/cuvs/pull/2027
* Fix nightly build matrix by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/2054
* Fix vulnerable index deserialization by @lowener in https://github.com/rapidsai/cuvs/pull/2068
* Fix symbol export kmeans by @aamijar in https://github.com/rapidsai/cuvs/pull/2070
* Fix argmin/argmax based on the distance type by @achirkin in https://github.com/rapidsai/cuvs/pull/2016
* Remove unneeded request for CUDA device link phase by @robertmaynard in https://github.com/rapidsai/cuvs/pull/2077
* Update Faiss and DiskANN Patch to Use C++20 by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1796
* Fix brute force Rust index dataset lifetime by @yan-zaretskiy in https://github.com/rapidsai/cuvs/pull/2083
* Fix segfault cuvs bench by @aamijar in https://github.com/rapidsai/cuvs/pull/2088
* Fix cagra::optimize modifying the state of raft::resources by @achirkin in https://github.com/rapidsai/cuvs/pull/2103
* Add direct target dependency when embedding fatbins by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/2106
* Fix check for PQ vectorized load by @lowener in https://github.com/rapidsai/cuvs/pull/2107
* Fix workspace usage by @mfoerste4 in https://github.com/rapidsai/cuvs/pull/2135
* Add missing visibility controls in IVF SQ by @divyegala in https://github.com/rapidsai/cuvs/pull/2141
### 📖 Documentation
* Elaborate on fragment architecture in JIT+LTO documentation by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1991
* Add Cluster and Distance sections to C documentation by @lowener in https://github.com/rapidsai/cuvs/pull/1955
* Adding CAGRA merge to the documentation by @viclafargue in https://github.com/rapidsai/cuvs/pull/1942
* [Doc Update] CAGRA Memory Footprint by @singhmanas1 in https://github.com/rapidsai/cuvs/pull/1300
* Align docs with pluggable benchmark API by @jnke2016 in https://github.com/rapidsai/cuvs/pull/1891
* Add docs for cagra mem usage with NN Descent build algo by @jinsolp in https://github.com/rapidsai/cuvs/pull/2000
* Fix minor typos in ``cuvs-bench`` source build docs by @jrbourbeau in https://github.com/rapidsai/cuvs/pull/2006
* Fix `cuvs-bench` docker images in docs by @jrbourbeau in https://github.com/rapidsai/cuvs/pull/2003
* Update JIT+LTO guide to reflect new automatic embedding system by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/2045
* [Docs] Convert Sphinx docs to Fern by @cjnolet in https://github.com/rapidsai/cuvs/pull/2067
* Add UDF Usage and Developer docs by @divyegala in https://github.com/rapidsai/cuvs/pull/2030
* [DOC] Adding API guides for core cuVS types by @cjnolet in https://github.com/rapidsai/cuvs/pull/2117
### 🚀 New Features
* [REVIEW] Add L1 support to NN-Descent by @yan-zaretskiy in https://github.com/rapidsai/cuvs/pull/1898
* PCA C and Python API by @aamijar in https://github.com/rapidsai/cuvs/pull/1987
* Introduce UDF Architecture by @divyegala in https://github.com/rapidsai/cuvs/pull/1804
* JIT LTO Cagra Search by @divyegala in https://github.com/rapidsai/cuvs/pull/1807
* Expose supported brute force metrics in `all_neighbors` by @jinsolp in https://github.com/rapidsai/cuvs/pull/1827
* [REVIEW] Generalize and improve cagra::optimize by @mfoerste4 in https://github.com/rapidsai/cuvs/pull/1830
* IVF-SQ C++ API by @viclafargue in https://github.com/rapidsai/cuvs/pull/1865
### 🛠️ Improvements
* Use PQ API in CAGRA-Q + SCANN by @lowener in https://github.com/rapidsai/cuvs/pull/1746
* Speed up recall calculation in cuVS Bench for large top-K by @jamxia155 in https://github.com/rapidsai/cuvs/pull/1816
* Update codespell Version in pre-commit-config by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1920
* Forward-merge release/26.04 into main by @gforsyth in https://github.com/rapidsai/cuvs/pull/1936
* Refactor `StaticFatbinFragmentEntry` to use tags by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1970
* Replace cudaMemcpy2DAsync Calls with raft::copy_matrix by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1976
* update pip devcontainers' base image tags by @trxcllnt in https://github.com/rapidsai/cuvs/pull/1985
* Refactor instantiation matrices to generate at build time by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1984
* Add option to enable "sve" optimization level on armv9 by @LizYou in https://github.com/rapidsai/cuvs/pull/1121
* Improve cuvs-bench doc and add executable dir option by @tfeher in https://github.com/rapidsai/cuvs/pull/681
* Enforce type safety in JIT+LTO launcher by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1997
* Add KDE kernel by @Intron7 in https://github.com/rapidsai/cuvs/pull/1915
* Coderabbit integration by @benfred in https://github.com/rapidsai/cuvs/pull/1908
* Refactor fatbin registration to use common input file by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/2008
* Update to clang 20.1.8 by @bdice in https://github.com/rapidsai/cuvs/pull/2009
* JIT+LTO IVF-PQ compute similarity by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1957
* Refactor JIT+LTO kernels by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/2021
* feat(rust): add serialize/deserialize support for CAGRA index by @zbennett10 in https://github.com/rapidsai/cuvs/pull/1840
* Use new compute-matrix workflow by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/2034
* Reuse minClusterAndDistance Helper for Balanced KMeans by @tarang-jain in https://github.com/rapidsai/cuvs/pull/2001
* feat(rust): add search_with_filter to CAGRA Index by @jamie8johnson in https://github.com/rapidsai/cuvs/pull/2019
* [REVIEW] Drop extra copy in `get_last_error_text` by @jakirkham in https://github.com/rapidsai/cuvs/pull/2044
* FIX: disable warpspeed scan by @mfoerste4 in https://github.com/rapidsai/cuvs/pull/2062
* Use `token.rapids.nvidia.com` when issuing S3 bucket creds in devcontainers by @trxcllnt in https://github.com/rapidsai/cuvs/pull/2047
* Remove `NO_CUDART_DEP` property by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/2065
* Switch the remaining C++17 components to C++20 by @achirkin in https://github.com/rapidsai/cuvs/pull/2063
* fix(ci): resolve all zizmor findings and add zizmor pre-commit checks by @gforsyth in https://github.com/rapidsai/cuvs/pull/2053
* fix(ci): declare explicit secrets in `publish-rust.yaml` by @gforsyth in https://github.com/rapidsai/cuvs/pull/2069
* [REVIEW] Rewrite cuvs-sys build to discover pre-installed cuVS via cmake-package by @yan-zaretskiy in https://github.com/rapidsai/cuvs/pull/2022
* Fix symbol export by @vyasr in https://github.com/rapidsai/cuvs/pull/2052
* fix(ci): add explicit `actions: write` permission for `telemetry-summarize`
 by @gforsyth in https://github.com/rapidsai/cuvs/pull/2075
* [REVIEW] Improve 1-NN performance with split GEMM/reduction kernels on Blackwell by @vinaydes in https://github.com/rapidsai/cuvs/pull/1768
* Build and test with CUDA 13.2.0 by @bdice in https://github.com/rapidsai/cuvs/pull/2072
* Centralize shared utilities across benchmark backends by @jnke2016 in https://github.com/rapidsai/cuvs/pull/2040
* Persistent CAGRA: benchmark group and bad config warnings by @achirkin in https://github.com/rapidsai/cuvs/pull/2091
* Multi-GPU Batched KMeans by @viclafargue in https://github.com/rapidsai/cuvs/pull/2017
* IVF-SQ C API by @viclafargue in https://github.com/rapidsai/cuvs/pull/1910
* skip CuPy 14.1.0 by @jameslamb in https://github.com/rapidsai/cuvs/pull/2142

## New Contributors
* @singhmanas1 made their first contribution in https://github.com/rapidsai/cuvs/pull/1300
* @LizYou made their first contribution in https://github.com/rapidsai/cuvs/pull/1121
* @jamie8johnson made their first contribution in https://github.com/rapidsai/cuvs/pull/2019

**Full Changelog**: https://github.com/rapidsai/cuvs/compare/v26.06.00a...release/26.06

# cuvs 26.04.00 (8 Apr 2026)

### 🚨 Breaking Changes
* Use HNSW GPU Hierarchy by Default by @julianmi in https://github.com/rapidsai/cuvs/pull/1617
* Backport "Default to static linking of libcudart" by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1918
* enforce a floor on libnvjitlink, build wheels with CUDA 13.0.x, test wheels against mix of CTK versions by @jameslamb in https://github.com/rapidsai/cuvs/pull/1862
### 🐛 Bug Fixes
* Add float16 support for CAGRA extend by @lowener in https://github.com/rapidsai/cuvs/pull/1620
* Fix C examples CUDAToolkit dependency by @tfeher in https://github.com/rapidsai/cuvs/pull/1626
* clang-tidy: remove AnalyzeTemporaryDtors key by @achirkin in https://github.com/rapidsai/cuvs/pull/1778
* Fix build.sh: build cuvs_c for examples and fix --gpu-arch parsing by @achirkin in https://github.com/rapidsai/cuvs/pull/1779
* Better handling of batching of search in MG replicated mode by @viclafargue in https://github.com/rapidsai/cuvs/pull/1718
* Add nvjitlink to cuda-toolkit pip extras for cusparse compatibility by @bdice in https://github.com/rapidsai/cuvs/pull/1794
* [REVIEW] Fix: for balanced kmeans use grid.x for adjust_centers to avoid grid.y overflow for >262K centroids by @Nischal1729 in https://github.com/rapidsai/cuvs/pull/1805
* Fix thrust header by @aamijar in https://github.com/rapidsai/cuvs/pull/1817
* Fix persistent CAGRA regressions by @achirkin in https://github.com/rapidsai/cuvs/pull/1800
* Fix thrust header by @aamijar in https://github.com/rapidsai/cuvs/pull/1825
* Set `cudaFuncAttributeMaxDynamicSharedMemorySize` with thread-safety by @mythrocks in https://github.com/rapidsai/cuvs/pull/1771
* [REVIEW] Move from `thrust::make_counting_iterator` to `cuda::make_counting_iterator` by @mythrocks in https://github.com/rapidsai/cuvs/pull/1826
* Fixed cuvs benchmark debug build issue (linker step fail) by @irina-resh-nvda in https://github.com/rapidsai/cuvs/pull/1599
* Graph degree equals intermediate graph degree bug fix by @irina-resh-nvda in https://github.com/rapidsai/cuvs/pull/1834
* FAISS patch for `thrust_counting_iterator.h` by @aamijar in https://github.com/rapidsai/cuvs/pull/1844
* Use 1D grid calculations in `epsilon_neighborhood` by @divyegala in https://github.com/rapidsai/cuvs/pull/1847
* Fix setting CAGRA graph build algo to iterative search by default by @achirkin in https://github.com/rapidsai/cuvs/pull/1864
* Faiss suppress warning 611 by @aamijar in https://github.com/rapidsai/cuvs/pull/1879
* Pin faiss to 1.14.0 by @aamijar in https://github.com/rapidsai/cuvs/pull/1885
* Make some minor fixes to JIT+LTO functionality by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1896
* Revert "Default to static linking of libcudart" by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1925
* Fix hanging pytests by @aamijar in https://github.com/rapidsai/cuvs/pull/1924
* Disallow programmatic stream serialization in JIT kernel launches by @divyegala in https://github.com/rapidsai/cuvs/pull/1932
* IVF-Flat: fix irrelevant assert in the fused kernel mode by @achirkin in https://github.com/rapidsai/cuvs/pull/1941
* [REVIEW] cuVS bench: Fix cudaFuncSetAttribute not being called when CAGRA search switches kernel variants by @irina-resh-nvda in https://github.com/rapidsai/cuvs/pull/1851
* Pin openblas for aarch64 by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1965
* Null JIT kernel launch config by @divyegala in https://github.com/rapidsai/cuvs/pull/1974
* [BUG] Fix Vamana Serialization by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1966
### 📖 Documentation
* remove docs references to cuvs-bench-datasets, install a C compiler in pre-commit env by @jameslamb in https://github.com/rapidsai/cuvs/pull/1736
* Adding simple tarball install to build and install docs by @cjnolet in https://github.com/rapidsai/cuvs/pull/1868
* Fixing nccl link in tarball install instructions by @cjnolet in https://github.com/rapidsai/cuvs/pull/1887
* Add developer and user guides for JIT by @divyegala in https://github.com/rapidsai/cuvs/pull/1876
* PCA docs by @aamijar in https://github.com/rapidsai/cuvs/pull/1949
* Fix a few typos and ``.rst`` link syntax by @jrbourbeau in https://github.com/rapidsai/cuvs/pull/1973
* Update docs footer year by @aamijar in https://github.com/rapidsai/cuvs/pull/1958
* Doc improvements by @aamijar in https://github.com/rapidsai/cuvs/pull/1978
### 🚀 New Features
* JIT compile `interleaved_scan_kernel` for CUDA 13 by @divyegala in https://github.com/rapidsai/cuvs/pull/1405
* [REVIEW] L1 distance support for iterative search CAGRA build by @yan-zaretskiy in https://github.com/rapidsai/cuvs/pull/1831
* [FEA] Inertia Computation for Balanced KMeans and Add Option for Weighted Inertia by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1880
* PCA preprocessor by @aamijar in https://github.com/rapidsai/cuvs/pull/1808
* [FEA] Add Batching to KMeans by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1886
### 🛠️ Improvements
* Automatic Partition Count Derivation for ACE  by @julianmi in https://github.com/rapidsai/cuvs/pull/1603
* Add filter for cagra::merge by @benfred in https://github.com/rapidsai/cuvs/pull/1496
* Ivf_flat extends golang APIs by @cpegeric in https://github.com/rapidsai/cuvs/pull/1600
* Drop Python 3.10 support by @gforsyth in https://github.com/rapidsai/cuvs/pull/1748
* tighten wheel size limits, expand CI-skipping logic, other small build changes by @jameslamb in https://github.com/rapidsai/cuvs/pull/1751
* Migrate hash strategy to use the new cuco::static_map by @PointKernel in https://github.com/rapidsai/cuvs/pull/1462
* Update raft headers by @aamijar in https://github.com/rapidsai/cuvs/pull/1763
* remove pip.conf migration code in CI scripts, update CI-skipping rules by @jameslamb in https://github.com/rapidsai/cuvs/pull/1760
* Convert non-type template parameters to runtime parameters in CAGRA search to cut binary size by @seunghwak in https://github.com/rapidsai/cuvs/pull/1498
* Rename lanczos by @aamijar in https://github.com/rapidsai/cuvs/pull/1759
* CI: build with CUDA 13.1.1 by @jameslamb in https://github.com/rapidsai/cuvs/pull/1766
* Fixes for stricter compilers by @maxwbuckley in https://github.com/rapidsai/cuvs/pull/1703
* fix cpu_search call by including `k` argument by @benfred in https://github.com/rapidsai/cuvs/pull/1785
* Use GHA id-token for `sccache-dist` auth token by @trxcllnt in https://github.com/rapidsai/cuvs/pull/1790
* Remove `cagra_optimize.hpp` by @aamijar in https://github.com/rapidsai/cuvs/pull/1791
* Remove unused use_norms constant from kernel_sm60.cuh by @maxwbuckley in https://github.com/rapidsai/cuvs/pull/1705
* Remove unused variable read_idx from query loop by @maxwbuckley in https://github.com/rapidsai/cuvs/pull/1706
* remove gitutils by @jameslamb in https://github.com/rapidsai/cuvs/pull/1797
* refactor: build wheels and conda packages using Python limited API by @gforsyth in https://github.com/rapidsai/cuvs/pull/1788
* Add pluggable backend architecture to cuvs-bench by @jnke2016 in https://github.com/rapidsai/cuvs/pull/1536
* Use Specific CCCL Includes by @divyegala in https://github.com/rapidsai/cuvs/pull/1806
* Replace `thrust::tuple` with `cuda::std::tuple` by @miscco in https://github.com/rapidsai/cuvs/pull/1811
* check-nightly-ci: update to new version by @jameslamb in https://github.com/rapidsai/cuvs/pull/1813
* check-nightly-ci: remove testing config by @jameslamb in https://github.com/rapidsai/cuvs/pull/1824
* Refactor JIT LTO kernel generation by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1812
* Move `test_compute_matrix_product.py` to `cpp/tests` by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1836
* Drop uses of `thrust/functional.h` by @miscco in https://github.com/rapidsai/cuvs/pull/1835
* Better `ncv` param spectral embedding edge case by @aamijar in https://github.com/rapidsai/cuvs/pull/1828
* Modernize the uses of raft in cuVS by @achirkin in https://github.com/rapidsai/cuvs/pull/1837
* refactor(limited api): add explicit `wheel.py-api` to `pyproject.toml`
 by @gforsyth in https://github.com/rapidsai/cuvs/pull/1852
* Update Cython lower bound pin to 3.2.2 by @vyasr in https://github.com/rapidsai/cuvs/pull/1858
* Add support for Python 3.14 by @gforsyth in https://github.com/rapidsai/cuvs/pull/1845
* fix(rust): change Index::search to take &self instead of self by @zbennett10 in https://github.com/rapidsai/cuvs/pull/1839
* Remove pytest upper bound pin by @vyasr in https://github.com/rapidsai/cuvs/pull/1867
* Readme: vecflow paper in references by @aamijar in https://github.com/rapidsai/cuvs/pull/1874
* Refactor knn graph build params to remove `graph_build_types.hpp` by @jinsolp in https://github.com/rapidsai/cuvs/pull/1235
* Remove IndexWrapper by @aamijar in https://github.com/rapidsai/cuvs/pull/1792
* Use C linkage for JIT LTO kernels by @divyegala in https://github.com/rapidsai/cuvs/pull/1909
* Prevent nested parallelism in HNSW bench by @julianmi in https://github.com/rapidsai/cuvs/pull/1895
* Add a script to check for breaking C ABI changes by @benfred in https://github.com/rapidsai/cuvs/pull/1749
* Improve the IVF-PQ Coarse Batch Size Workspace Estimation by @julianmi in https://github.com/rapidsai/cuvs/pull/1937
* ScaNN: Fix AVQ prefetch by @rmaschal in https://github.com/rapidsai/cuvs/pull/1899

## New Contributors
* @cpegeric made their first contribution in https://github.com/rapidsai/cuvs/pull/1600
* @PointKernel made their first contribution in https://github.com/rapidsai/cuvs/pull/1462
* @seunghwak made their first contribution in https://github.com/rapidsai/cuvs/pull/1498
* @Nischal1729 made their first contribution in https://github.com/rapidsai/cuvs/pull/1805
* @jnke2016 made their first contribution in https://github.com/rapidsai/cuvs/pull/1536
* @zbennett10 made their first contribution in https://github.com/rapidsai/cuvs/pull/1839
* @jrbourbeau made their first contribution in https://github.com/rapidsai/cuvs/pull/1973

**Full Changelog**: https://github.com/rapidsai/cuvs/compare/v26.04.00a...release/26.04

# cuvs 26.02.00 (4 Feb 2026)

### 🚨 Breaking Changes
* Use CCCL's mdspan implementation by @divyegala in https://github.com/rapidsai/cuvs/pull/1605
* Add filter for cagra::merge  by @benfred in https://github.com/rapidsai/cuvs/pull/1755
### 🐛 Bug Fixes
* fix(ci): remove unknown parameter `name` from rocky8 build job by @gforsyth in https://github.com/rapidsai/cuvs/pull/1554
* CMake check for FAISS use in benchmarks by @irina-resh-nvda in https://github.com/rapidsai/cuvs/pull/1591
* Fix overflow in `preprocess_data_kernel` of NN Descent by @jinsolp in https://github.com/rapidsai/cuvs/pull/1596
* cmake is missing `sparse/gram.cu` gtest by @aamijar in https://github.com/rapidsai/cuvs/pull/1611
* Correctly specify the ninja-build version to download by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1634
* Remove Flaky Python ACE Example by @julianmi in https://github.com/rapidsai/cuvs/pull/1641
* Fix integer overflow in get_free_host_memory() on systems with >2TB RAM by @bdice in https://github.com/rapidsai/cuvs/pull/1636
* Fix libclang download for Rust, CUDA initialization for C tests by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1652
* Skip Python Multi-GPU Doctests by @julianmi in https://github.com/rapidsai/cuvs/pull/1671
* [Bug] Fix memory allocation test in CAGRA graph optimization by @enp1s0 in https://github.com/rapidsai/cuvs/pull/1675
* Fix multi-GPU All Neighbors memory coherence issue on older platforms by @viclafargue in https://github.com/rapidsai/cuvs/pull/1713
### 📖 Documentation
* Clean `raft/neighbors` comments by @aamijar in https://github.com/rapidsai/cuvs/pull/1651
* Clean `raft::neighbors` comments by @aamijar in https://github.com/rapidsai/cuvs/pull/1665
### 🚀 New Features
* Assign the c/ folder to the the c code ownder group by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1573
* Add arm64 builds to the libcuvs_c rocky8 matrix by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1570
* [FEA]  Enforce tighter link restrictions on libcuvs_c by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1614
* [FEA] IVF-PQ Build Factories for Precomputed Centroids and Codebooks by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1483
* Add HNSW ACE build method by @julianmi in https://github.com/rapidsai/cuvs/pull/1597
* [FEA] C + Python API for IVF-PQ Build Factories with Precomputed Centroids by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1664
* Add arm64 builds to the libcuvs_c build.yaml matrix by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1701
* pre-built libcuvs_c.so now use the new ABI major/minor values by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1708
* Correct base release for cuvs abi 1 major by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1724
* Add support for PQ preprocessing API by @lowener in https://github.com/rapidsai/cuvs/pull/1278
* [FEA] IVF-PQ to Write Flat PQ Codes by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1607
* Enable collation of all license files into pre-built binaries by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1679
### 🛠️ Improvements
* Update FAISS patch for RMM memory resource header migration by @bdice in https://github.com/rapidsai/cuvs/pull/1566
* Use strict priority in CI conda tests by @bdice in https://github.com/rapidsai/cuvs/pull/1583
* Update FAISS from 1.12.0 to 1.13.0 by @bdice in https://github.com/rapidsai/cuvs/pull/1585
* Use strict priority in CI conda tests by @bdice in https://github.com/rapidsai/cuvs/pull/1606
* Deduplicate `{unpack/pack}_list_data_kernel` by @jinsolp in https://github.com/rapidsai/cuvs/pull/1609
* Expose NN Descent fp16 data type support to python by @jinsolp in https://github.com/rapidsai/cuvs/pull/1616
* Remove alpha specs from non-RAPIDS dependencies by @bdice in https://github.com/rapidsai/cuvs/pull/1618
* Enable merge barriers by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1625
* Add devcontainer fallback for C++ test location by @bdice in https://github.com/rapidsai/cuvs/pull/1631
* Dispatch to use fp32 distance computation in NN Descent depending on data dimensions by @jinsolp in https://github.com/rapidsai/cuvs/pull/1415
* Prepare cuvs for removal of deprecated raft apis by @aamijar in https://github.com/rapidsai/cuvs/pull/1610
* Shorten the Test Duration of test_cagra_ace.py by @julianmi in https://github.com/rapidsai/cuvs/pull/1640
* Optional seed spectral embedding by @aamijar in https://github.com/rapidsai/cuvs/pull/1639
* FAISS patch for removed raft headers by @aamijar in https://github.com/rapidsai/cuvs/pull/1654
* Empty commit to trigger a build by @bdice in https://github.com/rapidsai/cuvs/pull/1662
* Spectral Clustering dataset api by @aamijar in https://github.com/rapidsai/cuvs/pull/1653
* Hide kernel symbols  by @divyegala in https://github.com/rapidsai/cuvs/pull/1663
* Update cuVS to c++20 by @divyegala in https://github.com/rapidsai/cuvs/pull/1649
* Move faiss_select from raft to cuvs by @aamijar in https://github.com/rapidsai/cuvs/pull/1658
* Fix DiskANN Override by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1666
* Use SPDX license identifiers in pyproject.toml, bump build dependency floors by @jameslamb in https://github.com/rapidsai/cuvs/pull/1667
* Add CUDA 13.1 support by @bdice in https://github.com/rapidsai/cuvs/pull/1642
* Remove faiss patch by @aamijar in https://github.com/rapidsai/cuvs/pull/1668
* Expose pq list data to python by @benfred in https://github.com/rapidsai/cuvs/pull/1428
* Add rust bindings for vamana index by @benfred in https://github.com/rapidsai/cuvs/pull/1608
* build and test against CUDA 13.1.0 by @jameslamb in https://github.com/rapidsai/cuvs/pull/1677
* [Improvement] Allow Configurable DRAM in DiskANN SSD Wrapper by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1598
* prefer CUDA 13.1 devcontainers, react to some cutlass removals in RAFT by @jameslamb in https://github.com/rapidsai/cuvs/pull/1686
* Empty commit to trigger a build by @jameslamb in https://github.com/rapidsai/cuvs/pull/1694
* Spectral Embedding `nnz_t` by @aamijar in https://github.com/rapidsai/cuvs/pull/1628
* Avoid installing CCCL headers in wheels by @divyegala in https://github.com/rapidsai/cuvs/pull/1692
* Use main shared-workflows branch by @jameslamb in https://github.com/rapidsai/cuvs/pull/1707
* Add Helper to Create NumPy Files by @julianmi in https://github.com/rapidsai/cuvs/pull/1645
* Spectral Embedding with `all_neighbors` by @aamijar in https://github.com/rapidsai/cuvs/pull/1693
* Deduplicate `calc_chunk_indices_kernel` by @jinsolp in https://github.com/rapidsai/cuvs/pull/1657
* wheel builds: react to changes in pip's handling of build constraints by @mmccarty in https://github.com/rapidsai/cuvs/pull/1710
* fix(build): build package on merge to `release/*` branch by @gforsyth in https://github.com/rapidsai/cuvs/pull/1733
* Add a `transform` function to ivf_pq by @benfred in https://github.com/rapidsai/cuvs/pull/1732

## New Contributors
* @irina-resh-nvda made their first contribution in https://github.com/rapidsai/cuvs/pull/1591

**Full Changelog**: https://github.com/rapidsai/cuvs/compare/v26.02.00a...release/26.02

# cuvs 25.12.00 (10 Dec 2025)

### 🚨 Breaking Changes
* Using `all_neighbors` for mutual reachability by @jinsolp in https://github.com/rapidsai/cuvs/pull/1234
* Refactor libcuvs_c header and source locations by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1357
* Update cagra C API enums to have more long term stable values by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1436
* Require CUDA 12.2+ by @jakirkham in https://github.com/rapidsai/cuvs/pull/1476
* Remove mutual_reachability_graph Public API by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1481
### 🐛 Bug Fixes
* Allow compilation when OpenMP is disabled by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1346
* Deallocation should be noexcept by @bdice in https://github.com/rapidsai/cuvs/pull/1416
* ANN_BENCH: Don't throw in noexcept do_deallocate by @achirkin in https://github.com/rapidsai/cuvs/pull/1417
* Remove unneeded cutlass public build/install dependency by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1411
* cuvs_static properly adds C include dir to target_include_directories by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1426
* Fix C/Python serialization for FP16. Add python tests by @lowener in https://github.com/rapidsai/cuvs/pull/1429
* Properly guard usage of openmp function calls  by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1435
* Update nlohmann-json to 3.12.0 by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1449
* Fix test params for filtered ivf_flat by @aamijar in https://github.com/rapidsai/cuvs/pull/1463
* [REVIEW][Java][Bug] Fix `CuVSMatrix#getRow` to take strides into consideration by @ldematte in https://github.com/rapidsai/cuvs/pull/1442
* Adding more tests and I find the compression settings are broken in the Golang API by @maxwbuckley in https://github.com/rapidsai/cuvs/pull/1472
* Fix binary quantizer host transform bounds and stream order by @achirkin in https://github.com/rapidsai/cuvs/pull/1473
* [BUG] Check if dynamic batching conf is null by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1459
* Include cagra search algorithms in libcuvs_static.a by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1457
* [Java] Disable flaky test `testFloatSerialization` by @mythrocks in https://github.com/rapidsai/cuvs/pull/1503
* Add libaio to cuvs-bench-cpu, improve bench dependencies by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1480
* [Java] Fix `CagraBuildAndSearchIT` concurrent deallocation errors by @ldematte in https://github.com/rapidsai/cuvs/pull/1510
* Pin Cython pre-3.2.0 and PyTest pre-9 by @jakirkham in https://github.com/rapidsai/cuvs/pull/1528
* Fix a bug in compute_distance_00_generate.py by @enp1s0 in https://github.com/rapidsai/cuvs/pull/1532
* Check stride information in from_dlpack by @benfred in https://github.com/rapidsai/cuvs/pull/1458
* refactored update-version.sh to handle new branching strategy by @rockhowse in https://github.com/rapidsai/cuvs/pull/1535
* fixed bug with update-version.sh by @rockhowse in https://github.com/rapidsai/cuvs/pull/1556
* fix(ci): remove unsupported `name` parameter from custom job def by @gforsyth in https://github.com/rapidsai/cuvs/pull/1560
* Remove need to have rapids_logger headers installed to use clib by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1527
* Fix BruteForce serialize test by @lowener in https://github.com/rapidsai/cuvs/pull/1568
* [C] Fix: `cuvsRMMMemoryResourceReset` sets a valid resource by @ldematte in https://github.com/rapidsai/cuvs/pull/1540
* Fix typo in date field by @vyasr in https://github.com/rapidsai/cuvs/pull/1563
* SNMG ANN build with OpenMP nested parallelism by @viclafargue in https://github.com/rapidsai/cuvs/pull/1526
### 📖 Documentation
* Fix table of content documentation by @lowener in https://github.com/rapidsai/cuvs/pull/1427
* Docs Spectral Clustering by @aamijar in https://github.com/rapidsai/cuvs/pull/1490
* Use current system architecture in conda environment creation command by @bdice in https://github.com/rapidsai/cuvs/pull/1499
* Updating README for release by @cjnolet in https://github.com/rapidsai/cuvs/pull/1584
### 🚀 New Features
* Unify binding headers by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1413
* CAGRA: decouple source idx type from graph idx type and add a mapping between them by @achirkin in https://github.com/rapidsai/cuvs/pull/1251
* Add SOVERSION information to libcuvs_c by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1474
* Spectral Clustering by @aamijar in https://github.com/rapidsai/cuvs/pull/1425
* ANN_BENCH: integrate NVTX statistics by @achirkin in https://github.com/rapidsai/cuvs/pull/1529
* Add Augmented Core Extraction Algorithm by @julianmi in https://github.com/rapidsai/cuvs/pull/1404
* Extend CI to build and test x86 libcuvs_c tarballs by @robertmaynard in https://github.com/rapidsai/cuvs/pull/1524
* Introduce `libcuvs-headers` and `libcuvs-static` CMake components and conda packages by @divyegala in https://github.com/rapidsai/cuvs/pull/1494
* [Java] Bindings, tests and benchmarks for RMM pooled memory by @ldematte in https://github.com/rapidsai/cuvs/pull/1453
### 🛠️ Improvements
* Update `RAPIDS_BRANCH`, codify changes in `update-version.sh` by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1368
* Move eigen_solvers from raft by @aamijar in https://github.com/rapidsai/cuvs/pull/1402
* CosineExpanded Distance Metric for CAGRA by @tarang-jain in https://github.com/rapidsai/cuvs/pull/197
* Enable `sccache-dist` connection pool by @trxcllnt in https://github.com/rapidsai/cuvs/pull/1431
* Use pinned_host_memory_resource instead of pinned_memory_resource. by @bdice in https://github.com/rapidsai/cuvs/pull/1434
* Use main in RAPIDS_BRANCH by @bdice in https://github.com/rapidsai/cuvs/pull/1439
* Use main shared-workflows branch by @bdice in https://github.com/rapidsai/cuvs/pull/1444
* [Review] ScaNN: Add option for AVQ/Noise Shaping to bfloat16 quantization by @rmaschal in https://github.com/rapidsai/cuvs/pull/1354
* Add NVTX annotations to CAGRA knn graph build stage by @achirkin in https://github.com/rapidsai/cuvs/pull/1443
* [Review][C] Export the ability to get/set the log level to the C API by @ldematte in https://github.com/rapidsai/cuvs/pull/1375
* [Java] Test indexing and serialization with integral (byte) dataset by @ldematte in https://github.com/rapidsai/cuvs/pull/1366
* Use SPDX for all copyright headers by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1446
* [REVIEW][Java] Log level api by @ldematte in https://github.com/rapidsai/cuvs/pull/1376
* [Review] [Java] Disable flaky Java tests to reduce CI churn by @mythrocks in https://github.com/rapidsai/cuvs/pull/1469
* Refactor rapids_cpm_package_details to rapids_cpm_package_info by @bdice in https://github.com/rapidsai/cuvs/pull/1433
* Fix golang test name and add more logging and error checking. by @maxwbuckley in https://github.com/rapidsai/cuvs/pull/1471
* Use an explicit std::min for byte alignment calculation by @maxwbuckley in https://github.com/rapidsai/cuvs/pull/1465
* Replace custom cuda_pinned_resource with RMM's pinned_host_memory_resource by @bdice in https://github.com/rapidsai/cuvs/pull/1466
* Add patch for FAISS memory resources by @bdice in https://github.com/rapidsai/cuvs/pull/1477
* Improved CAGRA build parameter heuristics by @achirkin in https://github.com/rapidsai/cuvs/pull/1448
* Decouple C++ library from C library by @divyegala in https://github.com/rapidsai/cuvs/pull/1488
* Use `RAPIDS_BRANCH` in cmake-format invocations that need rapids-cmake configs by @bdice in https://github.com/rapidsai/cuvs/pull/1475
* eigen `tolerance` option Spectral Embedding by @aamijar in https://github.com/rapidsai/cuvs/pull/1493
* Update to CUDA 13.0.2 by @bdice in https://github.com/rapidsai/cuvs/pull/1506
* Single Linkage to Use all_neighbors API to build the KNN graph by @tarang-jain in https://github.com/rapidsai/cuvs/pull/1507
* Migrate to new CCCL memory resource interface by @bdice in https://github.com/rapidsai/cuvs/pull/1502
* Remove unused headers scann by @lowener in https://github.com/rapidsai/cuvs/pull/1508
* Update RMM includes from `<rmm/mr/device/*>` to `<rmm/mr/*>` by @bdice in https://github.com/rapidsai/cuvs/pull/1538
* Use ruff-check, ruff-format instead of black, flake8 by @KyleFromNVIDIA in https://github.com/rapidsai/cuvs/pull/1500
* Set memory pool from Python multi-GPU resource by @viclafargue in https://github.com/rapidsai/cuvs/pull/1530
* [Java] Fix format with spotless by @ldematte in https://github.com/rapidsai/cuvs/pull/1539
* fix bad version update by @trxcllnt in https://github.com/rapidsai/cuvs/pull/1555
* Deduplicate KMeans instantiations by @divyegala in https://github.com/rapidsai/cuvs/pull/1565
* Improve memory usage in `build_mr_linkage` by @jinsolp in https://github.com/rapidsai/cuvs/pull/1550
* Use `sccache-dist` build cluster for conda and wheel builds by @trxcllnt in https://github.com/rapidsai/cuvs/pull/1495
* [Java] One PinnedMemoryBuffer per CuVSResourcesImpl by @ldematte in https://github.com/rapidsai/cuvs/pull/1441
* [Java] Relax cuVS version matching by @ldematte in https://github.com/rapidsai/cuvs/pull/1544

## New Contributors
* @maxwbuckley made their first contribution in https://github.com/rapidsai/cuvs/pull/1471
* @rockhowse made their first contribution in https://github.com/rapidsai/cuvs/pull/1535

**Full Changelog**: https://github.com/rapidsai/cuvs/compare/v25.12.00a...release/25.12

# cuvs 25.10.00 (8 Oct 2025)

## 🚨 Breaking Changes

- [Java] Uniform toHost/toDevice to work across all CuVSMatrix classes ([#1328](https://github.com/rapidsai/cuvs/pull/1328)) [@ldematte](https://github.com/ldematte)
- Use int64_t for getters in the cagra/ivf_flat c-api ([#1272](https://github.com/rapidsai/cuvs/pull/1272)) [@benfred](https://github.com/benfred)
- [REVIEW][Java] Rename destroyIndex() to close(), extend AutoCloseable ([#1252](https://github.com/rapidsai/cuvs/pull/1252)) [@ldematte](https://github.com/ldematte)
- Removing deprecated batching code specific to NN Descent ([#1249](https://github.com/rapidsai/cuvs/pull/1249)) [@jinsolp](https://github.com/jinsolp)
- MG C API ([#1160](https://github.com/rapidsai/cuvs/pull/1160)) [@viclafargue](https://github.com/viclafargue)

## 🐛 Bug Fixes

- Update `cpp/cmake/config.json` for `ConfigureTest()` ([#1385](https://github.com/rapidsai/cuvs/pull/1385)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix k-means++ root rank race condition ([#1359](https://github.com/rapidsai/cuvs/pull/1359)) [@csadorf](https://github.com/csadorf)
- Fix inadvertent uses of copy constructor in mdarrays across cuVS ([#1330](https://github.com/rapidsai/cuvs/pull/1330)) [@achirkin](https://github.com/achirkin)
- Refactor CMakeLists to support static only builds of libcuvs ([#1317](https://github.com/rapidsai/cuvs/pull/1317)) [@robertmaynard](https://github.com/robertmaynard)
- Fix pq_compressed file created by SSD Vamana build ([#1313](https://github.com/rapidsai/cuvs/pull/1313)) [@bkarsin](https://github.com/bkarsin)
- Revert &quot;[Java]Binary and scalar quantization ([#1104)&quot; (#1274](https://github.com/rapidsai/cuvs/pull/1104)&quot; (#1274)) [@mythrocks](https://github.com/mythrocks)
- Fix balanced kmeans API ([#1268](https://github.com/rapidsai/cuvs/pull/1268)) [@enp1s0](https://github.com/enp1s0)
- [BUG] NEIGHBORS_ALL_NEIGHBORS_TEST build ignores --no-mg ([#1230](https://github.com/rapidsai/cuvs/pull/1230)) [@enp1s0](https://github.com/enp1s0)
- Add error handling for CPU-only bench ([#1203](https://github.com/rapidsai/cuvs/pull/1203)) [@lowener](https://github.com/lowener)
- Make duplicate removal in all neighbors robust to distance drift across batches ([#1185](https://github.com/rapidsai/cuvs/pull/1185)) [@jinsolp](https://github.com/jinsolp)
- Fix single GPU sharded search merge ([#1094](https://github.com/rapidsai/cuvs/pull/1094)) [@viclafargue](https://github.com/viclafargue)
- Processing mutual reachability dist in connect_knn_graph for host data ([#1093](https://github.com/rapidsai/cuvs/pull/1093)) [@jinsolp](https://github.com/jinsolp)
- [Java][Fix] Multithreaded querying fails without synchronization ([#1082](https://github.com/rapidsai/cuvs/pull/1082)) [@chatman](https://github.com/chatman)

## 📖 Documentation

- Add Vamana C rst Docs to toctree ([#1399](https://github.com/rapidsai/cuvs/pull/1399)) [@tarang-jain](https://github.com/tarang-jain)
- Regroup overloads in docs ([#1377](https://github.com/rapidsai/cuvs/pull/1377)) [@lowener](https://github.com/lowener)
- [DOCS] Update IVF Docs with Supported Distance Metrics ([#1365](https://github.com/rapidsai/cuvs/pull/1365)) [@tarang-jain](https://github.com/tarang-jain)
- Add note about Linux-only support for cuVS pre-compiled packages ([#1306](https://github.com/rapidsai/cuvs/pull/1306)) [@stic](https://github.com/stic)
- Documentation Fixes ([#1301](https://github.com/rapidsai/cuvs/pull/1301)) [@benfred](https://github.com/benfred)
- Docs for spectral embedding ([#1299](https://github.com/rapidsai/cuvs/pull/1299)) [@aamijar](https://github.com/aamijar)
- Improve NN Descent documentation ([#1246](https://github.com/rapidsai/cuvs/pull/1246)) [@jinsolp](https://github.com/jinsolp)
- Add NN.extend and kmeans to Python doc ([#1217](https://github.com/rapidsai/cuvs/pull/1217)) [@lowener](https://github.com/lowener)

## 🚀 New Features

- [REVIEW] [Java] Option to build fat-jars with native dependencies included ([#1296](https://github.com/rapidsai/cuvs/pull/1296)) [@mythrocks](https://github.com/mythrocks)
- Updates needed when building with CUDA 13 ([#1219](https://github.com/rapidsai/cuvs/pull/1219)) [@robertmaynard](https://github.com/robertmaynard)
- [Java] Add CAGRA index graph accessor/build from graph (host memory) ([#1216](https://github.com/rapidsai/cuvs/pull/1216)) [@ldematte](https://github.com/ldematte)
- [Feat] Add Dockerfile for reproducible installation ([#1195](https://github.com/rapidsai/cuvs/pull/1195)) [@rhdong](https://github.com/rhdong)
- Vamana C / Python API ([#1112](https://github.com/rapidsai/cuvs/pull/1112)) [@tarang-jain](https://github.com/tarang-jain)
- Mutual Reachability in all-neighbors API ([#1016](https://github.com/rapidsai/cuvs/pull/1016)) [@jinsolp](https://github.com/jinsolp)
- CAGRA Build + DiskANN Search cuvs-bench Wrapper ([#899](https://github.com/rapidsai/cuvs/pull/899)) [@tarang-jain](https://github.com/tarang-jain)

## 🛠️ Improvements

- Patch FAISS for missing Thrust includes ([#1398](https://github.com/rapidsai/cuvs/pull/1398)) [@bdice](https://github.com/bdice)
- Empty commit to trigger a build ([#1363](https://github.com/rapidsai/cuvs/pull/1363)) [@msarahan](https://github.com/msarahan)
- [Review][Java] Refactor: extract interface from CuVSMatrixBaseImpl ([#1361](https://github.com/rapidsai/cuvs/pull/1361)) [@ldematte](https://github.com/ldematte)
- use CUDA 13.0.1 CI images ([#1353](https://github.com/rapidsai/cuvs/pull/1353)) [@jameslamb](https://github.com/jameslamb)
- Treat warnings as errors in doc builds + MG ANNs Python API doc update ([#1350](https://github.com/rapidsai/cuvs/pull/1350)) [@viclafargue](https://github.com/viclafargue)
- [Java] Support row strides in CuVSMatrix ([#1345](https://github.com/rapidsai/cuvs/pull/1345)) [@ldematte](https://github.com/ldematte)
- [Java] Adding tests to use CuVSDeviceMatrix (device memory) directly as a CagraIndex input dataset ([#1340](https://github.com/rapidsai/cuvs/pull/1340)) [@ldematte](https://github.com/ldematte)
- Improve performance of assigning clusters in batched all_neighbors ([#1336](https://github.com/rapidsai/cuvs/pull/1336)) [@jinsolp](https://github.com/jinsolp)
- Configure repo for automatic release notes generation ([#1334](https://github.com/rapidsai/cuvs/pull/1334)) [@AyodeAwe](https://github.com/AyodeAwe)
- [Java] Buffered device matrix builder ([#1332](https://github.com/rapidsai/cuvs/pull/1332)) [@ldematte](https://github.com/ldematte)
- Remove UCX-Py/UCXX from release script ([#1331](https://github.com/rapidsai/cuvs/pull/1331)) [@pentschev](https://github.com/pentschev)
- [Java] Uniform toHost/toDevice to work across all CuVSMatrix classes ([#1328](https://github.com/rapidsai/cuvs/pull/1328)) [@ldematte](https://github.com/ldematte)
- [Java] Add `libcuvs` &lt;-&gt; cuvs-java version check ([#1327](https://github.com/rapidsai/cuvs/pull/1327)) [@ldematte](https://github.com/ldematte)
- update dependencies: use cuda-toolkit wheels ([#1326](https://github.com/rapidsai/cuvs/pull/1326)) [@jameslamb](https://github.com/jameslamb)
- Add cagra.extend bindings for python ([#1324](https://github.com/rapidsai/cuvs/pull/1324)) [@benfred](https://github.com/benfred)
- [Java] Make `cudaGetDeviceProperties` compatible with CUDA 12 and 13 based on symbol presence ([#1323](https://github.com/rapidsai/cuvs/pull/1323)) [@ldematte](https://github.com/ldematte)
- Use branch-25.10 again ([#1319](https://github.com/rapidsai/cuvs/pull/1319)) [@jameslamb](https://github.com/jameslamb)
- [Review][Java] Add detailed error message for `libcuvs` load failure to UnsupportedProvider/UnsupportedOperationExceptions ([#1316](https://github.com/rapidsai/cuvs/pull/1316)) [@ldematte](https://github.com/ldematte)
- [Java] Add reason(s) to UnsupportedProvider/UnsupportedOperationExceptions ([#1314](https://github.com/rapidsai/cuvs/pull/1314)) [@ldematte](https://github.com/ldematte)
- MG Python API ([#1307](https://github.com/rapidsai/cuvs/pull/1307)) [@viclafargue](https://github.com/viclafargue)
- Fix redundant memset ([#1305](https://github.com/rapidsai/cuvs/pull/1305)) [@vinaydes](https://github.com/vinaydes)
- Expose pq-centers to C and Python ([#1303](https://github.com/rapidsai/cuvs/pull/1303)) [@benfred](https://github.com/benfred)
- Port `raft::neighbors::epsilon_neighborhood` to cuvs ([#1294](https://github.com/rapidsai/cuvs/pull/1294)) [@aamijar](https://github.com/aamijar)
- ScaNN: Overlapped gather for AVQ ([#1286](https://github.com/rapidsai/cuvs/pull/1286)) [@rmaschal](https://github.com/rmaschal)
- Update rapids-dependency-file-generator ([#1285](https://github.com/rapidsai/cuvs/pull/1285)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- [Review][Java] Expand `CagraIndex#search` to accept more types (int8 and int32) ([#1283](https://github.com/rapidsai/cuvs/pull/1283)) [@ldematte](https://github.com/ldematte)
- All neighbors C and Python bindings ([#1282](https://github.com/rapidsai/cuvs/pull/1282)) [@viclafargue](https://github.com/viclafargue)
- Build and test with CUDA 13.0.0 ([#1273](https://github.com/rapidsai/cuvs/pull/1273)) [@jameslamb](https://github.com/jameslamb)
- Use int64_t for getters in the cagra/ivf_flat c-api ([#1272](https://github.com/rapidsai/cuvs/pull/1272)) [@benfred](https://github.com/benfred)
- [Java][C] Expose GPUInfo ([#1267](https://github.com/rapidsai/cuvs/pull/1267)) [@ldematte](https://github.com/ldematte)
- Add rust bindings for kmeans ([#1266](https://github.com/rapidsai/cuvs/pull/1266)) [@benfred](https://github.com/benfred)
- [REVIEW] Add a public API for CAGRA graph optimize ([#1260](https://github.com/rapidsai/cuvs/pull/1260)) [@abc99lr](https://github.com/abc99lr)
- Fix debug build ([#1258](https://github.com/rapidsai/cuvs/pull/1258)) [@lowener](https://github.com/lowener)
- Run `cargo fmt` in the pre-commit hooks ([#1255](https://github.com/rapidsai/cuvs/pull/1255)) [@benfred](https://github.com/benfred)
- [REVIEW][Java] Rename destroyIndex() to close(), extend AutoCloseable ([#1252](https://github.com/rapidsai/cuvs/pull/1252)) [@ldematte](https://github.com/ldematte)
- Removing deprecated batching code specific to NN Descent ([#1249](https://github.com/rapidsai/cuvs/pull/1249)) [@jinsolp](https://github.com/jinsolp)
- Pin Latest Faiss Version ([#1247](https://github.com/rapidsai/cuvs/pull/1247)) [@tarang-jain](https://github.com/tarang-jain)
- Use build cluster in devcontainers ([#1240](https://github.com/rapidsai/cuvs/pull/1240)) [@trxcllnt](https://github.com/trxcllnt)
- Use rapids_cuda_enable_fatbin_compression ([#1239](https://github.com/rapidsai/cuvs/pull/1239)) [@robertmaynard](https://github.com/robertmaynard)
- [Improvement] Replace Calls to thrust::transform with raft::linalg ([#1238](https://github.com/rapidsai/cuvs/pull/1238)) [@tarang-jain](https://github.com/tarang-jain)
- [Java] CuVSMatrix for device memory ([#1232](https://github.com/rapidsai/cuvs/pull/1232)) [@ldematte](https://github.com/ldematte)
- Drop log level for CAGRA trace messages ([#1229](https://github.com/rapidsai/cuvs/pull/1229)) [@mythrocks](https://github.com/mythrocks)
- Update rapids_config to handle user defined branch name ([#1227](https://github.com/rapidsai/cuvs/pull/1227)) [@robertmaynard](https://github.com/robertmaynard)
- Reduce noise in Java tests ([#1226](https://github.com/rapidsai/cuvs/pull/1226)) [@mythrocks](https://github.com/mythrocks)
- Update another instance of rapids-build-backend ([#1225](https://github.com/rapidsai/cuvs/pull/1225)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix missing sync_stream in ScaNN build ([#1224](https://github.com/rapidsai/cuvs/pull/1224)) [@rmaschal](https://github.com/rmaschal)
- [Java] Exception-safe RMM Allocations ([#1215](https://github.com/rapidsai/cuvs/pull/1215)) [@mythrocks](https://github.com/mythrocks)
- Update rapids-build-backend to 0.4.1 ([#1214](https://github.com/rapidsai/cuvs/pull/1214)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Removing filter templates in ivfpq for binary size reduction ([#1211](https://github.com/rapidsai/cuvs/pull/1211)) [@jinsolp](https://github.com/jinsolp)
- ci(labeler): update labeler action to [@v5 ([#1210](https://github.com/rapidsai/cuvs/pull/1210)) @gforsyth](https://github.com/v5 ([#1210](https://github.com/rapidsai/cuvs/pull/1210)) @gforsyth)
- [Java] Add decorator for synchronized CuVSResource access ([#1209](https://github.com/rapidsai/cuvs/pull/1209)) [@ldematte](https://github.com/ldematte)
- Remove unused templated instances ([#1204](https://github.com/rapidsai/cuvs/pull/1204)) [@vinaydes](https://github.com/vinaydes)
- Spectral Embedding precomputed graph api ([#1197](https://github.com/rapidsai/cuvs/pull/1197)) [@aamijar](https://github.com/aamijar)
- cuvs RAPIDS dependency now use the new rapids-cmake branch info ([#1189](https://github.com/rapidsai/cuvs/pull/1189)) [@robertmaynard](https://github.com/robertmaynard)
- Allow latest OS in devcontainers ([#1169](https://github.com/rapidsai/cuvs/pull/1169)) [@bdice](https://github.com/bdice)
- MG C API ([#1160](https://github.com/rapidsai/cuvs/pull/1160)) [@viclafargue](https://github.com/viclafargue)
- Merge support for tiered index ([#1155](https://github.com/rapidsai/cuvs/pull/1155)) [@benfred](https://github.com/benfred)
- Update build infra to support new branching strategy ([#1143](https://github.com/rapidsai/cuvs/pull/1143)) [@robertmaynard](https://github.com/robertmaynard)
- [Java]Binary and scalar quantization ([#1104](https://github.com/rapidsai/cuvs/pull/1104)) [@punAhuja](https://github.com/punAhuja)
- Improve Vamana index build performance and recall ([#1032](https://github.com/rapidsai/cuvs/pull/1032)) [@bkarsin](https://github.com/bkarsin)
- Use GCC 14 in conda builds. ([#1030](https://github.com/rapidsai/cuvs/pull/1030)) [@vyasr](https://github.com/vyasr)
- Update SNMG ANN API testing ([#1023](https://github.com/rapidsai/cuvs/pull/1023)) [@viclafargue](https://github.com/viclafargue)

# cuvs 25.08.00 (6 Aug 2025)

## 🚨 Breaking Changes

- [Java] Using functions for mapping ([#1007](https://github.com/rapidsai/cuvs/pull/1007)) [@ldematte](https://github.com/ldematte)
- Remove CUDA 11 from dependencies.yaml ([#962](https://github.com/rapidsai/cuvs/pull/962)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Common knn graph build params ([#949](https://github.com/rapidsai/cuvs/pull/949)) [@jinsolp](https://github.com/jinsolp)
- stop uploading packages to downloads.rapids.ai ([#940](https://github.com/rapidsai/cuvs/pull/940)) [@jameslamb](https://github.com/jameslamb)

## 🐛 Bug Fixes

- [BUG] Fix Large Batch Preprocessing for NN Descent ([#1166](https://github.com/rapidsai/cuvs/pull/1166)) [@tarang-jain](https://github.com/tarang-jain)
- Fix TieredIndexIT,  Scoped resource access change was missed ([#1156](https://github.com/rapidsai/cuvs/pull/1156)) [@punAhuja](https://github.com/punAhuja)
- Fix for `hellinger` metric ([#1128](https://github.com/rapidsai/cuvs/pull/1128)) [@viclafargue](https://github.com/viclafargue)
- Fix go CI failures ([#1114](https://github.com/rapidsai/cuvs/pull/1114)) [@benfred](https://github.com/benfred)
- Conditionally post process distances in NN Descent for use with distance epilogue ([#1108](https://github.com/rapidsai/cuvs/pull/1108)) [@jinsolp](https://github.com/jinsolp)
- [BUG] Disable flaky cpp test `AnnIVFFlatTestF_half.AnnIVFFlat/21` ([#1092](https://github.com/rapidsai/cuvs/pull/1092)) [@julianmi](https://github.com/julianmi)
- [BUG] Fix device access check in build_sorted_mst ([#1083](https://github.com/rapidsai/cuvs/pull/1083)) [@tarang-jain](https://github.com/tarang-jain)
- Pin Faiss Version to Last Stable Release ([#1029](https://github.com/rapidsai/cuvs/pull/1029)) [@tarang-jain](https://github.com/tarang-jain)
- Fix verify-copyright precommit hook ([#1017](https://github.com/rapidsai/cuvs/pull/1017)) [@benfred](https://github.com/benfred)
- Adding GH_TOKEN pass-through to summarize job ([#1011](https://github.com/rapidsai/cuvs/pull/1011)) [@msarahan](https://github.com/msarahan)
- Reuse `&lt;version_config.h&gt;` instead of separate version constants in `c_api.h` ([#1003](https://github.com/rapidsai/cuvs/pull/1003)) [@mythrocks](https://github.com/mythrocks)
- IVF-PQ coarse search: fix integer overflow and avoid excessive batch sizes ([#999](https://github.com/rapidsai/cuvs/pull/999)) [@achirkin](https://github.com/achirkin)
- Fix C compile error in `tiered_index.h` ([#996](https://github.com/rapidsai/cuvs/pull/996)) [@mythrocks](https://github.com/mythrocks)
- Fix for NN Descent negative or NaN distances ([#994](https://github.com/rapidsai/cuvs/pull/994)) [@jinsolp](https://github.com/jinsolp)
- Adding int64 search for MG CAGRA ([#975](https://github.com/rapidsai/cuvs/pull/975)) [@viclafargue](https://github.com/viclafargue)
- [ANN_BENCH] Fix OpenMP Sections in Wrappers ([#911](https://github.com/rapidsai/cuvs/pull/911)) [@tarang-jain](https://github.com/tarang-jain)
- [BUG] cuvs-bench data_export fixes ([#863](https://github.com/rapidsai/cuvs/pull/863)) [@tarang-jain](https://github.com/tarang-jain)

## 📖 Documentation

- [ANN_BENCH] [DOCS] Add Vamana / DiskANN to cuvs-bench Docs ([#1164](https://github.com/rapidsai/cuvs/pull/1164)) [@tarang-jain](https://github.com/tarang-jain)
- Add Golang build instructions to build.rst ([#1116](https://github.com/rapidsai/cuvs/pull/1116)) [@mythrocks](https://github.com/mythrocks)
- Minor correction for Java test instructions ([#1081](https://github.com/rapidsai/cuvs/pull/1081)) [@mythrocks](https://github.com/mythrocks)
- add docs on CI workflow inputs ([#1075](https://github.com/rapidsai/cuvs/pull/1075)) [@jameslamb](https://github.com/jameslamb)
- [REVIEW][Java] Add instructions on how to run a single test ([#1004](https://github.com/rapidsai/cuvs/pull/1004)) [@ldematte](https://github.com/ldematte)

## 🚀 New Features

- Binary CAGRA with NN Descent ([#1133](https://github.com/rapidsai/cuvs/pull/1133)) [@tarang-jain](https://github.com/tarang-jain)
- CAGRA build heuristics for HNSW ([#1125](https://github.com/rapidsai/cuvs/pull/1125)) [@achirkin](https://github.com/achirkin)
- BitwiseHamming distance for NN Descent ([#1101](https://github.com/rapidsai/cuvs/pull/1101)) [@jinsolp](https://github.com/jinsolp)
- `ReachabilityPostProcess` distance epilogue for NN Descent ([#1073](https://github.com/rapidsai/cuvs/pull/1073)) [@jinsolp](https://github.com/jinsolp)
- Add brute force to `all_neighbors` API ([#1062](https://github.com/rapidsai/cuvs/pull/1062)) [@jinsolp](https://github.com/jinsolp)
- [Java] Support for tiered index ([#1028](https://github.com/rapidsai/cuvs/pull/1028)) [@punAhuja](https://github.com/punAhuja)
- Enable `build_sorted_mst` with data on host memory ([#997](https://github.com/rapidsai/cuvs/pull/997)) [@jinsolp](https://github.com/jinsolp)
- Add C-API method to fetch cuVS version from Java. ([#935](https://github.com/rapidsai/cuvs/pull/935)) [@mythrocks](https://github.com/mythrocks)
- Spectral Embedding ([#871](https://github.com/rapidsai/cuvs/pull/871)) [@aamijar](https://github.com/aamijar)
- [FEA] Build Single Linkage API ([#820](https://github.com/rapidsai/cuvs/pull/820)) [@tarang-jain](https://github.com/tarang-jain)

## 🛠️ Improvements

- Check shape is initialized in cuvsMatrixSliceRows ([#1193](https://github.com/rapidsai/cuvs/pull/1193)) [@benfred](https://github.com/benfred)
- Add error checking for our C example code ([#1171](https://github.com/rapidsai/cuvs/pull/1171)) [@benfred](https://github.com/benfred)
- Removing all references to CUDA 11 from codebase ([#1150](https://github.com/rapidsai/cuvs/pull/1150)) [@cjnolet](https://github.com/cjnolet)
- [Java]Fixed TieredIndexParams creation/destruction, calling cuvsTieredIndexParamsCreate ([#1147](https://github.com/rapidsai/cuvs/pull/1147)) [@punAhuja](https://github.com/punAhuja)
- Fix update-version ([#1135](https://github.com/rapidsai/cuvs/pull/1135)) [@AyodeAwe](https://github.com/AyodeAwe)
- Reducing binary size in `ivf_pq` by deduplicating kernels ([#1130](https://github.com/rapidsai/cuvs/pull/1130)) [@jinsolp](https://github.com/jinsolp)
- fix(docker): use versioned `-latest` tag for all `rapidsai` images ([#1129](https://github.com/rapidsai/cuvs/pull/1129)) [@gforsyth](https://github.com/gforsyth)
- SCaNN Index build ([#1120](https://github.com/rapidsai/cuvs/pull/1120)) [@rmaschal](https://github.com/rmaschal)
- [Java] Extend `Dataset` to work as an output data container ([#1111](https://github.com/rapidsai/cuvs/pull/1111)) [@ldematte](https://github.com/ldematte)
- [Java] Fix HNSW params allocation ([#1110](https://github.com/rapidsai/cuvs/pull/1110)) [@ldematte](https://github.com/ldematte)
- [Java] Fix CAGRA params allocation ([#1109](https://github.com/rapidsai/cuvs/pull/1109)) [@ldematte](https://github.com/ldematte)
- [Java] Fix POM ([#1106](https://github.com/rapidsai/cuvs/pull/1106)) [@ldematte](https://github.com/ldematte)
- Enforce java codeowners for managing contents of `java/` ([#1103](https://github.com/rapidsai/cuvs/pull/1103)) [@cjnolet](https://github.com/cjnolet)
- Reduce binary size of refine functions ([#1095](https://github.com/rapidsai/cuvs/pull/1095)) [@tfeher](https://github.com/tfeher)
- [Java] Introduce scoped resource access ([#1089](https://github.com/rapidsai/cuvs/pull/1089)) [@ldematte](https://github.com/ldematte)
- Expose graph and dataset accessors for CAGRA to C/Python ([#1086](https://github.com/rapidsai/cuvs/pull/1086)) [@benfred](https://github.com/benfred)
- Add warning for unused GPU when `n_clusters &lt; n_ranks` in batch `all_neighbors` ([#1072](https://github.com/rapidsai/cuvs/pull/1072)) [@jinsolp](https://github.com/jinsolp)
- [Java] Tidy up `MemorySegment`s lifecycle ([#1069](https://github.com/rapidsai/cuvs/pull/1069)) [@ldematte](https://github.com/ldematte)
- [Java] Refactor SearchResults implementation classes ([#1067](https://github.com/rapidsai/cuvs/pull/1067)) [@ldematte](https://github.com/ldematte)
- Use CUDA 12.9 in Conda, Devcontainers, Spark, GHA, etc. ([#1063](https://github.com/rapidsai/cuvs/pull/1063)) [@jakirkham](https://github.com/jakirkham)
- Exporting changed `graph_build_params` namespace into `all_neighbors` ([#1060](https://github.com/rapidsai/cuvs/pull/1060)) [@jinsolp](https://github.com/jinsolp)
- hnsw::from_cagra: avoid allocating the graph twice ([#1057](https://github.com/rapidsai/cuvs/pull/1057)) [@achirkin](https://github.com/achirkin)
- Improve memory efficiency for returning NN Descent distances ([#1053](https://github.com/rapidsai/cuvs/pull/1053)) [@jinsolp](https://github.com/jinsolp)
- Deprecation notice for batching code specific to NN Descent ([#1052](https://github.com/rapidsai/cuvs/pull/1052)) [@jinsolp](https://github.com/jinsolp)
- Remove nvidia and dask channels ([#1050](https://github.com/rapidsai/cuvs/pull/1050)) [@vyasr](https://github.com/vyasr)
- [Java] Enforce Java code format standard ([#1049](https://github.com/rapidsai/cuvs/pull/1049)) [@narangvivek10](https://github.com/narangvivek10)
- Make rust publish run after conda upload ([#1047](https://github.com/rapidsai/cuvs/pull/1047)) [@AyodeAwe](https://github.com/AyodeAwe)
- refactor(cuda11): remove cuda11-only conda channels, cleanup docs ([#1046](https://github.com/rapidsai/cuvs/pull/1046)) [@gforsyth](https://github.com/gforsyth)
- [Review][Java] Fix random segabort/segfault/double free problems ([#1045](https://github.com/rapidsai/cuvs/pull/1045)) [@ldematte](https://github.com/ldematte)
- [Java] Add Dataset based on `MemorySegment` ([#1034](https://github.com/rapidsai/cuvs/pull/1034)) [@ldematte](https://github.com/ldematte)
- [Java] Add Java API benchmarks ([#1033](https://github.com/rapidsai/cuvs/pull/1033)) [@ldematte](https://github.com/ldematte)
- fix(cli): Add validation for search-mode parameter ([#1026](https://github.com/rapidsai/cuvs/pull/1026)) [@mayani-nv](https://github.com/mayani-nv)
- [Java] Encapsulate on-heap float arrays into `Dataset` ([#1024](https://github.com/rapidsai/cuvs/pull/1024)) [@ldematte](https://github.com/ldematte)
- refactor(shellcheck): fix all remaining warnings/errors ([#1019](https://github.com/rapidsai/cuvs/pull/1019)) [@gforsyth](https://github.com/gforsyth)
- Instantiate only specific RAFT linewise kernels ([#1018](https://github.com/rapidsai/cuvs/pull/1018)) [@aamijar](https://github.com/aamijar)
- Update nightly CI check to allow 30 days of failing nightly build. ([#1008](https://github.com/rapidsai/cuvs/pull/1008)) [@cjnolet](https://github.com/cjnolet)
- [Java] Using functions for mapping ([#1007](https://github.com/rapidsai/cuvs/pull/1007)) [@ldematte](https://github.com/ldematte)
- [REVIEW][Java] Refactor CagraBuildAndSearchIT to explicitly express different execution modes ([#1006](https://github.com/rapidsai/cuvs/pull/1006)) [@ldematte](https://github.com/ldematte)
- [REVIEW][Java] Add jextract artifacts to gitignore ([#1005](https://github.com/rapidsai/cuvs/pull/1005)) [@ldematte](https://github.com/ldematte)
- Remove pytest pin ([#998](https://github.com/rapidsai/cuvs/pull/998)) [@vyasr](https://github.com/vyasr)
- [java] Utility function for `cudaMemcpy` ([#983](https://github.com/rapidsai/cuvs/pull/983)) [@mythrocks](https://github.com/mythrocks)
- [java] Copy `pom.xml` to `target/` on build ([#981](https://github.com/rapidsai/cuvs/pull/981)) [@mythrocks](https://github.com/mythrocks)
- Remove CUDA 11 from dependencies.yaml ([#962](https://github.com/rapidsai/cuvs/pull/962)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- refactor(rattler): remove cuda11 options and general cleanup ([#961](https://github.com/rapidsai/cuvs/pull/961)) [@gforsyth](https://github.com/gforsyth)
- Remove CUDA 11 devcontainers and update CI scripts ([#960](https://github.com/rapidsai/cuvs/pull/960)) [@bdice](https://github.com/bdice)
- Common knn graph build params ([#949](https://github.com/rapidsai/cuvs/pull/949)) [@jinsolp](https://github.com/jinsolp)
- stop uploading packages to downloads.rapids.ai ([#940](https://github.com/rapidsai/cuvs/pull/940)) [@jameslamb](https://github.com/jameslamb)
- Instantiate only specific RAFT reduction kernels ([#925](https://github.com/rapidsai/cuvs/pull/925)) [@divyegala](https://github.com/divyegala)
- [Java] Replace the C wrapper layer with the Panama API-based Java implementation ([#916](https://github.com/rapidsai/cuvs/pull/916)) [@narangvivek10](https://github.com/narangvivek10)
- Forward-merge branch-25.06 into branch-25.08 ([#909](https://github.com/rapidsai/cuvs/pull/909)) [@gforsyth](https://github.com/gforsyth)
- Forward-merge branch-25.06 into branch-25.08 ([#897](https://github.com/rapidsai/cuvs/pull/897)) [@gforsyth](https://github.com/gforsyth)
- Serialize Vamana index with SSD sector alignment per MSFT DiskANN format, generate quantized dataset for integration with DiskANN ([#846](https://github.com/rapidsai/cuvs/pull/846)) [@jamxia155](https://github.com/jamxia155)
- Add mean and median threshold modes to binary quantization ([#716](https://github.com/rapidsai/cuvs/pull/716)) [@enp1s0](https://github.com/enp1s0)

# cuvs 25.06.00 (5 Jun 2025)

## 🚨 Breaking Changes

- [Java] Adding support for prefiltering in CAGRA ([#870](https://github.com/rapidsai/cuvs/pull/870)) [@punAhuja](https://github.com/punAhuja)

## 🐛 Bug Fixes

- Fix failing CAGRA merge google tests for 25.06 ([#974](https://github.com/rapidsai/cuvs/pull/974)) [@rhdong](https://github.com/rhdong)
- Fix for recent NCCL resource update ([#968](https://github.com/rapidsai/cuvs/pull/968)) [@viclafargue](https://github.com/viclafargue)
- Revert &quot;Fix kmeans::predict argument order ([#915)&quot; (#951](https://github.com/rapidsai/cuvs/pull/915)&quot; (#951)) [@divyegala](https://github.com/divyegala)
- IVF-PQ tests: fix segfault when accessing empty lists ([#933](https://github.com/rapidsai/cuvs/pull/933)) [@achirkin](https://github.com/achirkin)
- batch_load_iterator shall copy data located in host memory ([#926](https://github.com/rapidsai/cuvs/pull/926)) [@tfeher](https://github.com/tfeher)
- Fix kmeans::predict argument order ([#915](https://github.com/rapidsai/cuvs/pull/915)) [@enp1s0](https://github.com/enp1s0)
- ANN_BENCH: fix the reported core count ([#896](https://github.com/rapidsai/cuvs/pull/896)) [@achirkin](https://github.com/achirkin)
- CUVS_ANN_BENCH_SINGLE_EXE: find the mg library ([#890](https://github.com/rapidsai/cuvs/pull/890)) [@achirkin](https://github.com/achirkin)
- [FIX] Revert negated InnerProduct distance for NN Descent ([#859](https://github.com/rapidsai/cuvs/pull/859)) [@jinsolp](https://github.com/jinsolp)
- IVF-PQ tests: fix segfault when accessing empty lists ([#838](https://github.com/rapidsai/cuvs/pull/838)) [@achirkin](https://github.com/achirkin)
- [CuVS-Java] Automate panama bindings generation, Include IVF_PQ parameters in CAGRA index parameters and other changes ([#831](https://github.com/rapidsai/cuvs/pull/831)) [@narangvivek10](https://github.com/narangvivek10)
- Ann-bench: fix unsafe lazy blobs ([#828](https://github.com/rapidsai/cuvs/pull/828)) [@achirkin](https://github.com/achirkin)
- Fix test_brute_force conversion ([#821](https://github.com/rapidsai/cuvs/pull/821)) [@lowener](https://github.com/lowener)
- [FEA] New algos and updates/corrections to Faiss cuvs-bench ([#677](https://github.com/rapidsai/cuvs/pull/677)) [@tarang-jain](https://github.com/tarang-jain)

## 📖 Documentation

- All-neighbors API docs ([#944](https://github.com/rapidsai/cuvs/pull/944)) [@jinsolp](https://github.com/jinsolp)
- cagra.rst: fitered -&gt; filtered ([#866](https://github.com/rapidsai/cuvs/pull/866)) [@eli-b](https://github.com/eli-b)
- [DOCS] Fix Integration Docs (Faiss) ([#782](https://github.com/rapidsai/cuvs/pull/782)) [@tarang-jain](https://github.com/tarang-jain)

## 🚀 New Features

- Add IVF-PQ support in CAGRA index params in Python ([#918](https://github.com/rapidsai/cuvs/pull/918)) [@lowener](https://github.com/lowener)
- [Feat] Expose C API for CAGRA `merge` ([#860](https://github.com/rapidsai/cuvs/pull/860)) [@rhdong](https://github.com/rhdong)
- Use NCCL wheels from PyPI for CUDA 12 builds ([#827](https://github.com/rapidsai/cuvs/pull/827)) [@divyegala](https://github.com/divyegala)
- Add support for half in CAGRA+HNSW ([#813](https://github.com/rapidsai/cuvs/pull/813)) [@lowener](https://github.com/lowener)
- [CI] Enable Java test in CI workflow ([#805](https://github.com/rapidsai/cuvs/pull/805)) [@rhdong](https://github.com/rhdong)
- Wrapper for all-neighbors knn graph building ([#785](https://github.com/rapidsai/cuvs/pull/785)) [@jinsolp](https://github.com/jinsolp)
- Add support of half dtype in IVF-FLAT ([#730](https://github.com/rapidsai/cuvs/pull/730)) [@lowener](https://github.com/lowener)
- IVF-PQ: low-precision coarse search ([#715](https://github.com/rapidsai/cuvs/pull/715)) [@achirkin](https://github.com/achirkin)
- [Feat] Add support of logical merge in Cagra ([#713](https://github.com/rapidsai/cuvs/pull/713)) [@rhdong](https://github.com/rhdong)

## 🛠️ Improvements

- Update score calculation for CAGRA-Q instance selection ([#938](https://github.com/rapidsai/cuvs/pull/938)) [@enp1s0](https://github.com/enp1s0)
- [FEA] Use Native Brute Force for Sparse Pairwise KNN ([#927](https://github.com/rapidsai/cuvs/pull/927)) [@tarang-jain](https://github.com/tarang-jain)
- use &#39;rapids-init-pip&#39; in wheel CI, other CI changes ([#917](https://github.com/rapidsai/cuvs/pull/917)) [@jameslamb](https://github.com/jameslamb)
- ANN_BENCH: Avoid repeated calls to raft::get_device_for_address in CAGRA search ([#908](https://github.com/rapidsai/cuvs/pull/908)) [@achirkin](https://github.com/achirkin)
- [Java] New off-heap Dataset support for CAGRA and Bruteforce ([#902](https://github.com/rapidsai/cuvs/pull/902)) [@chatman](https://github.com/chatman)
- Finish CUDA 12.9 migration and use branch-25.06 workflows ([#901](https://github.com/rapidsai/cuvs/pull/901)) [@bdice](https://github.com/bdice)
- Update to clang 20 ([#898](https://github.com/rapidsai/cuvs/pull/898)) [@bdice](https://github.com/bdice)
- get Java artifacts from GitHub Actions artifact store ([#893](https://github.com/rapidsai/cuvs/pull/893)) [@jameslamb](https://github.com/jameslamb)
- Quote head_rev in conda recipes ([#892](https://github.com/rapidsai/cuvs/pull/892)) [@bdice](https://github.com/bdice)
- [Java] Exposing  merge API for multiple CAGRA indices ([#891](https://github.com/rapidsai/cuvs/pull/891)) [@punAhuja](https://github.com/punAhuja)
- Expose ivf-flat centers to python/c ([#888](https://github.com/rapidsai/cuvs/pull/888)) [@benfred](https://github.com/benfred)
- CUDA 12.9 use updated compression flags ([#887](https://github.com/rapidsai/cuvs/pull/887)) [@robertmaynard](https://github.com/robertmaynard)
- Expose ivf-pq centers to python/c ([#881](https://github.com/rapidsai/cuvs/pull/881)) [@benfred](https://github.com/benfred)
- Accept host inputs in python for ivf-pq build and extend ([#880](https://github.com/rapidsai/cuvs/pull/880)) [@benfred](https://github.com/benfred)
- Add tiered_index support ([#879](https://github.com/rapidsai/cuvs/pull/879)) [@benfred](https://github.com/benfred)
- Exclude librmm.so from auditwheel ([#878](https://github.com/rapidsai/cuvs/pull/878)) [@bdice](https://github.com/bdice)
- update.version.sh: remove broken reference, skip most CI on PRs that only modify update-version.sh ([#875](https://github.com/rapidsai/cuvs/pull/875)) [@jameslamb](https://github.com/jameslamb)
- Add support for Python 3.13 ([#874](https://github.com/rapidsai/cuvs/pull/874)) [@gforsyth](https://github.com/gforsyth)
- chore: lower wheel size threshold ([#872](https://github.com/rapidsai/cuvs/pull/872)) [@gforsyth](https://github.com/gforsyth)
- [Java] Adding support for prefiltering in CAGRA ([#870](https://github.com/rapidsai/cuvs/pull/870)) [@punAhuja](https://github.com/punAhuja)
- Change snmg index to use updated multi gpu resource API ([#869](https://github.com/rapidsai/cuvs/pull/869)) [@jinsolp](https://github.com/jinsolp)
- run shellcheck on all files, other small pre-commit updates ([#865](https://github.com/rapidsai/cuvs/pull/865)) [@jameslamb](https://github.com/jameslamb)
- Fix IVF PQ build metric for CAGRA ([#862](https://github.com/rapidsai/cuvs/pull/862)) [@lowener](https://github.com/lowener)
- ANN_BENCH: Expose parallel_mode parameter of FAISS CPU IVF implementation ([#861](https://github.com/rapidsai/cuvs/pull/861)) [@achirkin](https://github.com/achirkin)
- Specify matplotlib version ([#839](https://github.com/rapidsai/cuvs/pull/839)) [@benfred](https://github.com/benfred)
- Use random tmp names for index files in tests ([#837](https://github.com/rapidsai/cuvs/pull/837)) [@achirkin](https://github.com/achirkin)
- Download build artifacts from Github for CI ([#834](https://github.com/rapidsai/cuvs/pull/834)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- Add NN-Descent return_distances functionality to python/C ([#833](https://github.com/rapidsai/cuvs/pull/833)) [@benfred](https://github.com/benfred)
- Optimize hnsw::from_cagra&lt;GPU&gt; ([#826](https://github.com/rapidsai/cuvs/pull/826)) [@achirkin](https://github.com/achirkin)
- Use vendored RAPIDS.cmake in example code. ([#824](https://github.com/rapidsai/cuvs/pull/824)) [@bdice](https://github.com/bdice)
- refactor(rattler): enable strict channel priority for builds ([#823](https://github.com/rapidsai/cuvs/pull/823)) [@gforsyth](https://github.com/gforsyth)
- Reduce device memory usage for CAGRA&#39;s graph optimization process (2-hop detour counting) ([#822](https://github.com/rapidsai/cuvs/pull/822)) [@anaruse](https://github.com/anaruse)
- [cuvs_bench] distinguish search label from build label in data_export.py ([#818](https://github.com/rapidsai/cuvs/pull/818)) [@jiangyinzuo](https://github.com/jiangyinzuo)
- Vendor RAPIDS.cmake ([#816](https://github.com/rapidsai/cuvs/pull/816)) [@bdice](https://github.com/bdice)
- Update libcuvs libraft ver to 25.06 in conda env ([#808](https://github.com/rapidsai/cuvs/pull/808)) [@jinsolp](https://github.com/jinsolp)
- Moving NN Descent class and struct declarations to `nn_descent_gnnd.hpp` ([#803](https://github.com/rapidsai/cuvs/pull/803)) [@jinsolp](https://github.com/jinsolp)
- Remove `[@rapidsai/cuvs-build-codeowners` ([#783](https://github.com/rapidsai/cuvs/pull/783)) @KyleFromNVIDIA](https://github.com/rapidsai/cuvs-build-codeowners` ([#783](https://github.com/rapidsai/cuvs/pull/783)) @KyleFromNVIDIA)
- Moving wheel builds to specified location and uploading build artifacts to Github ([#777](https://github.com/rapidsai/cuvs/pull/777)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- Remove unused raft cagra header in add_nodes.cuh ([#741](https://github.com/rapidsai/cuvs/pull/741)) [@jiangyinzuo](https://github.com/jiangyinzuo)
- Expose kmeans to python ([#729](https://github.com/rapidsai/cuvs/pull/729)) [@benfred](https://github.com/benfred)
- Update cuvs to properly create a NCCL::NCCL target ([#720](https://github.com/rapidsai/cuvs/pull/720)) [@robertmaynard](https://github.com/robertmaynard)
- Optimize euclidean distance in host refine phase ([#689](https://github.com/rapidsai/cuvs/pull/689)) [@anstellaire](https://github.com/anstellaire)
- Moving MG functions into unified API + `raft::device_resources_snmg` as device resource type for MG functions ([#454](https://github.com/rapidsai/cuvs/pull/454)) [@viclafargue](https://github.com/viclafargue)
- Moving random ball cover ([#218](https://github.com/rapidsai/cuvs/pull/218)) [@cjnolet](https://github.com/cjnolet)

# cuvs 25.04.00 (9 Apr 2025)

## 🚨 Breaking Changes

- Use new rapids-logger library ([#644](https://github.com/rapidsai/cuvs/pull/644)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Use new build_patch_only argument ([#780](https://github.com/rapidsai/cuvs/pull/780)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix resource import so generate_groundtruth works ([#774](https://github.com/rapidsai/cuvs/pull/774)) [@nvrohanv](https://github.com/nvrohanv)
- Relax max duplicates in batched NN Descent ([#770](https://github.com/rapidsai/cuvs/pull/770)) [@jinsolp](https://github.com/jinsolp)
- [BUG] Fix graph index sorting in CAGRA graph build by NN Descent ([#763](https://github.com/rapidsai/cuvs/pull/763)) [@enp1s0](https://github.com/enp1s0)
- cuvs-bench-cpu: avoid &#39;mkl&#39; dependency on aarch64 ([#750](https://github.com/rapidsai/cuvs/pull/750)) [@jameslamb](https://github.com/jameslamb)
- ANN_BENCH: Fix segfault in CAGRA wrapper when moving its graph ([#733](https://github.com/rapidsai/cuvs/pull/733)) [@achirkin](https://github.com/achirkin)
- Fix duplicate indices in batch NN Descent ([#702](https://github.com/rapidsai/cuvs/pull/702)) [@jinsolp](https://github.com/jinsolp)

## 📖 Documentation

- Go module - Usage docs ([#779](https://github.com/rapidsai/cuvs/pull/779)) [@AyodeAwe](https://github.com/AyodeAwe)

## 🚀 New Features

- `L2SqrtExpanded` metric support for NN Descent ([#790](https://github.com/rapidsai/cuvs/pull/790)) [@jinsolp](https://github.com/jinsolp)
- CAGRA search: int64_t indices in the API ([#769](https://github.com/rapidsai/cuvs/pull/769)) [@achirkin](https://github.com/achirkin)
- Expose IVF PQ params in CAGRA C API ([#768](https://github.com/rapidsai/cuvs/pull/768)) [@divyegala](https://github.com/divyegala)
- [Feat] Expose search with bitset C API for Brute Force ([#717](https://github.com/rapidsai/cuvs/pull/717)) [@rhdong](https://github.com/rhdong)
- Diskann Benchmarking Wrapper ([#260](https://github.com/rapidsai/cuvs/pull/260)) [@tarang-jain](https://github.com/tarang-jain)

## 🛠️ Improvements

- Use L4 instead of V100 ([#797](https://github.com/rapidsai/cuvs/pull/797)) [@bdice](https://github.com/bdice)
- Rust publishing updates ([#789](https://github.com/rapidsai/cuvs/pull/789)) [@AyodeAwe](https://github.com/AyodeAwe)
- Removing print statement ([#786](https://github.com/rapidsai/cuvs/pull/786)) [@jinsolp](https://github.com/jinsolp)
- Support passing a dataset to `FromCagra` ([#767](https://github.com/rapidsai/cuvs/pull/767)) [@ajit283](https://github.com/ajit283)
- add function to build specific gpu-arches [REVIEW] ([#766](https://github.com/rapidsai/cuvs/pull/766)) [@nvrohanv](https://github.com/nvrohanv)
- feat(libcuvs): port libcuvs to rattler-build ([#751](https://github.com/rapidsai/cuvs/pull/751)) [@gforsyth](https://github.com/gforsyth)
- Consolidate Conda environment for Rust ([#745](https://github.com/rapidsai/cuvs/pull/745)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- rename &#39;sklearn&#39; conda dependency to &#39;scikit-learn&#39; ([#743](https://github.com/rapidsai/cuvs/pull/743)) [@jameslamb](https://github.com/jameslamb)
- Reduce memory consumption of scalar quantizer ([#736](https://github.com/rapidsai/cuvs/pull/736)) [@mfoerste4](https://github.com/mfoerste4)
- Use conda-build instead of conda-mambabuild ([#723](https://github.com/rapidsai/cuvs/pull/723)) [@bdice](https://github.com/bdice)
- Expand NVTX annotations in CAGRA build and HNSW ([#711](https://github.com/rapidsai/cuvs/pull/711)) [@achirkin](https://github.com/achirkin)
- Replace `cub::TransformInputIterator` with `thrust::transform_iterator` ([#707](https://github.com/rapidsai/cuvs/pull/707)) [@miscco](https://github.com/miscco)
- Consolidate more Conda solves in CI ([#701](https://github.com/rapidsai/cuvs/pull/701)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Require CMake 3.30.4 ([#691](https://github.com/rapidsai/cuvs/pull/691)) [@robertmaynard](https://github.com/robertmaynard)
- Create Conda CI test env in one step ([#684](https://github.com/rapidsai/cuvs/pull/684)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Use shared-workflows branch-25.04 ([#669](https://github.com/rapidsai/cuvs/pull/669)) [@bdice](https://github.com/bdice)
- Add `shellcheck` to pre-commit and fix warnings ([#662](https://github.com/rapidsai/cuvs/pull/662)) [@gforsyth](https://github.com/gforsyth)
- Add build_type input field for test.yaml ([#661](https://github.com/rapidsai/cuvs/pull/661)) [@gforsyth](https://github.com/gforsyth)
- Use `rapids-pip-retry` in CI jobs that might need retries ([#648](https://github.com/rapidsai/cuvs/pull/648)) [@gforsyth](https://github.com/gforsyth)
- Use new rapids-logger library ([#644](https://github.com/rapidsai/cuvs/pull/644)) [@vyasr](https://github.com/vyasr)
- disallow fallback to Make in Python builds ([#636](https://github.com/rapidsai/cuvs/pull/636)) [@jameslamb](https://github.com/jameslamb)
- Add `verify-codeowners` hook ([#633](https://github.com/rapidsai/cuvs/pull/633)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Forward-merge branch-25.02 into branch-25.04 ([#632](https://github.com/rapidsai/cuvs/pull/632)) [@bdice](https://github.com/bdice)
- Account for RAFT sparse types updates ([#629](https://github.com/rapidsai/cuvs/pull/629)) [@divyegala](https://github.com/divyegala)
- Migrate to NVKS for amd64 CI runners ([#627](https://github.com/rapidsai/cuvs/pull/627)) [@bdice](https://github.com/bdice)

# cuvs 25.02.00 (13 Feb 2025)

## 🚨 Breaking Changes

- update pip devcontainers to UCX 1.18, small update-version.sh fixes ([#604](https://github.com/rapidsai/cuvs/pull/604)) [@jameslamb](https://github.com/jameslamb)
- Improve the performance of CAGRA new vector addition with the default params ([#569](https://github.com/rapidsai/cuvs/pull/569)) [@enp1s0](https://github.com/enp1s0)
- Update for raft logger changes ([#540](https://github.com/rapidsai/cuvs/pull/540)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Fix ann-bench dataset blob integer overflow leading to incorrect data copy beyond 4B elems ([#671](https://github.com/rapidsai/cuvs/pull/671)) [@achirkin](https://github.com/achirkin)
- Fix ann-bench deadlocking on HNSW destruction due to task locks ([#667](https://github.com/rapidsai/cuvs/pull/667)) [@achirkin](https://github.com/achirkin)
- cuvs-bench Fixes ([#654](https://github.com/rapidsai/cuvs/pull/654)) [@tarang-jain](https://github.com/tarang-jain)
- Fix std::lock_guard use for gcc 14 support ([#639](https://github.com/rapidsai/cuvs/pull/639)) [@enp1s0](https://github.com/enp1s0)
- Fix indexing bug when using parallelism to build CPU hierarchy in HNSW ([#620](https://github.com/rapidsai/cuvs/pull/620)) [@divyegala](https://github.com/divyegala)
- add runtime dependency on libcuvs in cuvs wheels ([#615](https://github.com/rapidsai/cuvs/pull/615)) [@jameslamb](https://github.com/jameslamb)
- Temporarily skip CUDA 11 wheel CI ([#599](https://github.com/rapidsai/cuvs/pull/599)) [@bdice](https://github.com/bdice)
- [Fix] l2_exp random fail in half-float32 mixed precision on self-neighboring ([#596](https://github.com/rapidsai/cuvs/pull/596)) [@rhdong](https://github.com/rhdong)
- Add CAGRA InnerProduct test and fix a bug ([#595](https://github.com/rapidsai/cuvs/pull/595)) [@enp1s0](https://github.com/enp1s0)
- fix cuvs_bench.run --groups options ([#592](https://github.com/rapidsai/cuvs/pull/592)) [@jiangyinzuo](https://github.com/jiangyinzuo)
- Fix cagra_hnsw serialization when dataset is not part of index ([#591](https://github.com/rapidsai/cuvs/pull/591)) [@tfeher](https://github.com/tfeher)
- fix create_pointset for throughput mode ([#589](https://github.com/rapidsai/cuvs/pull/589)) [@jiangyinzuo](https://github.com/jiangyinzuo)
- Fix the use of constexpr in the dynamic batching header ([#582](https://github.com/rapidsai/cuvs/pull/582)) [@achirkin](https://github.com/achirkin)
- Reduce the recall threshold for IVF-PQ low-precision LUT inner product  tests ([#573](https://github.com/rapidsai/cuvs/pull/573)) [@achirkin](https://github.com/achirkin)
- Small fixes to docs and pairwise distances ([#570](https://github.com/rapidsai/cuvs/pull/570)) [@cjnolet](https://github.com/cjnolet)
- [BUG] Fix CAGRA graph optimization bug ([#565](https://github.com/rapidsai/cuvs/pull/565)) [@enp1s0](https://github.com/enp1s0)
- Fix broken link to python doc ([#564](https://github.com/rapidsai/cuvs/pull/564)) [@lowener](https://github.com/lowener)
- Fix cagra::extend error message ([#532](https://github.com/rapidsai/cuvs/pull/532)) [@enp1s0](https://github.com/enp1s0)
- Fix Grace-specific issues in CAGRA ([#527](https://github.com/rapidsai/cuvs/pull/527)) [@achirkin](https://github.com/achirkin)

## 📖 Documentation

- add docs for nn_descent ([#668](https://github.com/rapidsai/cuvs/pull/668)) [@Intron7](https://github.com/Intron7)
- Fixing small typo in cuvs bench docs ([#586](https://github.com/rapidsai/cuvs/pull/586)) [@cjnolet](https://github.com/cjnolet)
- Fix typos in README ([#543](https://github.com/rapidsai/cuvs/pull/543)) [@nvanbenschoten](https://github.com/nvanbenschoten)
- Use nvidia-sphinx-theme for docs ([#528](https://github.com/rapidsai/cuvs/pull/528)) [@benfred](https://github.com/benfred)

## 🚀 New Features

- Add deep-100M to datasets.yaml for cuvs-bench ([#670](https://github.com/rapidsai/cuvs/pull/670)) [@tarang-jain](https://github.com/tarang-jain)
- Expose configuration singleton as a global context for ann-bench algos ([#647](https://github.com/rapidsai/cuvs/pull/647)) [@achirkin](https://github.com/achirkin)
- ANN_BENCH enhanced dataset support ([#624](https://github.com/rapidsai/cuvs/pull/624)) [@achirkin](https://github.com/achirkin)
- [Feat] Add Support for Index `merge` in CAGRA ([#618](https://github.com/rapidsai/cuvs/pull/618)) [@rhdong](https://github.com/rhdong)
- HNSW GPU hierarchy ([#616](https://github.com/rapidsai/cuvs/pull/616)) [@divyegala](https://github.com/divyegala)
- CAGRA binary Hamming distance support ([#610](https://github.com/rapidsai/cuvs/pull/610)) [@enp1s0](https://github.com/enp1s0)
- Add cuda 12.8 support ([#605](https://github.com/rapidsai/cuvs/pull/605)) [@robertmaynard](https://github.com/robertmaynard)
- Add support for refinement with `uint32_t` index type ([#563](https://github.com/rapidsai/cuvs/pull/563)) [@lowener](https://github.com/lowener)
- [Feat] Support `bitset` filter for Brute Force ([#560](https://github.com/rapidsai/cuvs/pull/560)) [@rhdong](https://github.com/rhdong)
- Remove upper bounds on cuda-python to allow 12.6.2 and 11.8.5 ([#508](https://github.com/rapidsai/cuvs/pull/508)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Add filtering to python for ivf_flat ([#664](https://github.com/rapidsai/cuvs/pull/664)) [@benfred](https://github.com/benfred)
- Expose binary quantizer to C and Python ([#660](https://github.com/rapidsai/cuvs/pull/660)) [@benfred](https://github.com/benfred)
- Add telemetry ([#652](https://github.com/rapidsai/cuvs/pull/652)) [@gforsyth](https://github.com/gforsyth)
- Revert docs builds to CI latest tag. ([#643](https://github.com/rapidsai/cuvs/pull/643)) [@bdice](https://github.com/bdice)
- Add float16 support in python for cagra/brute_force/ivf_pq and scalar quantizer ([#637](https://github.com/rapidsai/cuvs/pull/637)) [@benfred](https://github.com/benfred)
- Expose NN-Descent to C and Python ([#635](https://github.com/rapidsai/cuvs/pull/635)) [@benfred](https://github.com/benfred)
- Revert CUDA 12.8 shared workflow branch changes ([#630](https://github.com/rapidsai/cuvs/pull/630)) [@vyasr](https://github.com/vyasr)
- cuvs-java: Rework the api to be Java 21 friendly ([#628](https://github.com/rapidsai/cuvs/pull/628)) [@ChrisHegarty](https://github.com/ChrisHegarty)
- Build and test with CUDA 12.8.0 ([#621](https://github.com/rapidsai/cuvs/pull/621)) [@bdice](https://github.com/bdice)
- Add Scalar Quantization to the c and python apis ([#617](https://github.com/rapidsai/cuvs/pull/617)) [@benfred](https://github.com/benfred)
- Iteratively build graph index ([#612](https://github.com/rapidsai/cuvs/pull/612)) [@anaruse](https://github.com/anaruse)
- update pip devcontainers to UCX 1.18, small update-version.sh fixes ([#604](https://github.com/rapidsai/cuvs/pull/604)) [@jameslamb](https://github.com/jameslamb)
- Reduce CAGRA test runtime ([#602](https://github.com/rapidsai/cuvs/pull/602)) [@bdice](https://github.com/bdice)
- Revert &quot;Temporarily skip CUDA 11 wheel CI&quot; ([#601](https://github.com/rapidsai/cuvs/pull/601)) [@bdice](https://github.com/bdice)
- introduce libcuvs wheels ([#594](https://github.com/rapidsai/cuvs/pull/594)) [@jameslamb](https://github.com/jameslamb)
- Normalize whitespace ([#593](https://github.com/rapidsai/cuvs/pull/593)) [@bdice](https://github.com/bdice)
- Rename test to tests. ([#590](https://github.com/rapidsai/cuvs/pull/590)) [@bdice](https://github.com/bdice)
- Use cuda.bindings layout. ([#588](https://github.com/rapidsai/cuvs/pull/588)) [@bdice](https://github.com/bdice)
- run_cuvs_pytests.sh uses proper test dir ([#584](https://github.com/rapidsai/cuvs/pull/584)) [@robertmaynard](https://github.com/robertmaynard)
- expose col-major bfknn to python ([#575](https://github.com/rapidsai/cuvs/pull/575)) [@benfred](https://github.com/benfred)
- Run cuvs-bench pytests and end-to-end tests in CI ([#574](https://github.com/rapidsai/cuvs/pull/574)) [@dantegd](https://github.com/dantegd)
- Expose col-major pairwise distances to python ([#572](https://github.com/rapidsai/cuvs/pull/572)) [@benfred](https://github.com/benfred)
- Improve the performance of CAGRA new vector addition with the default params ([#569](https://github.com/rapidsai/cuvs/pull/569)) [@enp1s0](https://github.com/enp1s0)
- Improve filtering documentation ([#568](https://github.com/rapidsai/cuvs/pull/568)) [@lowener](https://github.com/lowener)
- Use GCC 13 in CUDA 12 conda builds. ([#567](https://github.com/rapidsai/cuvs/pull/567)) [@bdice](https://github.com/bdice)
- Allow brute_force::build to work on host matrix dataset ([#562](https://github.com/rapidsai/cuvs/pull/562)) [@benfred](https://github.com/benfred)
- FAISS with cuVS enabled in cuvs-bench ([#561](https://github.com/rapidsai/cuvs/pull/561)) [@tarang-jain](https://github.com/tarang-jain)
- Vamana build improvement and added docs ([#558](https://github.com/rapidsai/cuvs/pull/558)) [@bkarsin](https://github.com/bkarsin)
- Support raft&#39;s logger targets ([#557](https://github.com/rapidsai/cuvs/pull/557)) [@vyasr](https://github.com/vyasr)
- Get Breathe from conda again ([#554](https://github.com/rapidsai/cuvs/pull/554)) [@vyasr](https://github.com/vyasr)
- Check if nightlies have succeeded recently enough ([#548](https://github.com/rapidsai/cuvs/pull/548)) [@vyasr](https://github.com/vyasr)
- Add support for float16 to the python pairwise distance api ([#547](https://github.com/rapidsai/cuvs/pull/547)) [@benfred](https://github.com/benfred)
- Additional Distances for CAGRA C and Python API ([#546](https://github.com/rapidsai/cuvs/pull/546)) [@tarang-jain](https://github.com/tarang-jain)
- remove setup.cfg files, other packaging cleanup ([#544](https://github.com/rapidsai/cuvs/pull/544)) [@jameslamb](https://github.com/jameslamb)
- Fix CI for python cuvs_bench ([#541](https://github.com/rapidsai/cuvs/pull/541)) [@benfred](https://github.com/benfred)
- Update for raft logger changes ([#540](https://github.com/rapidsai/cuvs/pull/540)) [@vyasr](https://github.com/vyasr)
- Change brute_force api to match ivf*/cagra ([#536](https://github.com/rapidsai/cuvs/pull/536)) [@benfred](https://github.com/benfred)
- Branch 25.02 merge 24.12 ([#526](https://github.com/rapidsai/cuvs/pull/526)) [@benfred](https://github.com/benfred)
- Update cuda-python lower bounds to 12.6.2 / 11.8.5 ([#524](https://github.com/rapidsai/cuvs/pull/524)) [@bdice](https://github.com/bdice)
- Automatic adjustment of itopk size according to filtering rate ([#509](https://github.com/rapidsai/cuvs/pull/509)) [@anaruse](https://github.com/anaruse)
- prefer system install of UCX in devcontainers ([#501](https://github.com/rapidsai/cuvs/pull/501)) [@jameslamb](https://github.com/jameslamb)
- Adapt to rmm logger changes ([#499](https://github.com/rapidsai/cuvs/pull/499)) [@vyasr](https://github.com/vyasr)
- Require approval to run CI on draft PRs ([#498](https://github.com/rapidsai/cuvs/pull/498)) [@bdice](https://github.com/bdice)
- Remove RAFT BUILD_ANN_BENCH option ([#497](https://github.com/rapidsai/cuvs/pull/497)) [@bdice](https://github.com/bdice)
- Update example code fetching rapids-cmake to use CUVS instead of RAFT ([#493](https://github.com/rapidsai/cuvs/pull/493)) [@bdice](https://github.com/bdice)
- Improve multi-CTA algorithm ([#492](https://github.com/rapidsai/cuvs/pull/492)) [@anaruse](https://github.com/anaruse)
- Add filtering for CAGRA to C API ([#452](https://github.com/rapidsai/cuvs/pull/452)) [@ajit283](https://github.com/ajit283)
- Initial cut for a cuVS Java API ([#450](https://github.com/rapidsai/cuvs/pull/450)) [@chatman](https://github.com/chatman)
- Add breaking change workflow trigger ([#442](https://github.com/rapidsai/cuvs/pull/442)) [@AyodeAwe](https://github.com/AyodeAwe)
- Expose `extend()` in C API ([#276](https://github.com/rapidsai/cuvs/pull/276)) [@ajit283](https://github.com/ajit283)
- Go API - [WIP] ([#212](https://github.com/rapidsai/cuvs/pull/212)) [@ajit283](https://github.com/ajit283)

# cuvs 24.12.00 (11 Dec 2024)

## 🚨 Breaking Changes

- HNSW CPU Hierarchy ([#465](https://github.com/rapidsai/cuvs/pull/465)) [@divyegala](https://github.com/divyegala)
- Use dashes in cuvs-bench package name. ([#417](https://github.com/rapidsai/cuvs/pull/417)) [@bdice](https://github.com/bdice)

## 🐛 Bug Fixes

- Skip IVF-PQ packing test for lists with not enough data ([#512](https://github.com/rapidsai/cuvs/pull/512)) [@achirkin](https://github.com/achirkin)
- [BUG] Fix CAGRA filter ([#489](https://github.com/rapidsai/cuvs/pull/489)) [@enp1s0](https://github.com/enp1s0)
- Add `kIsSingleSource` to `PairwiseDistanceEpilogueElementwise` ([#485](https://github.com/rapidsai/cuvs/pull/485)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix include errors, header, and unsafe locks in iface.hpp ([#467](https://github.com/rapidsai/cuvs/pull/467)) [@achirkin](https://github.com/achirkin)
- Fix an OOB error in device-side cuvs::neighbors::refine and CAGRA kern_prune ([#460](https://github.com/rapidsai/cuvs/pull/460)) [@achirkin](https://github.com/achirkin)
- Put a ceiling on cuda-python ([#445](https://github.com/rapidsai/cuvs/pull/445)) [@bdice](https://github.com/bdice)
- Enable NVTX in cuvs-cagra-search component ([#439](https://github.com/rapidsai/cuvs/pull/439)) [@achirkin](https://github.com/achirkin)
- BUG: CAGRA multi-cta illegal access with bad queries ([#438](https://github.com/rapidsai/cuvs/pull/438)) [@achirkin](https://github.com/achirkin)
- Fix index overflow in edge cases of CAGRA graph optimize ([#435](https://github.com/rapidsai/cuvs/pull/435)) [@achirkin](https://github.com/achirkin)
- Fix correct call to brute force in generate groundtruth of cuvs-bench ([#427](https://github.com/rapidsai/cuvs/pull/427)) [@dantegd](https://github.com/dantegd)
- Use Python for sccache hit rate computation. ([#420](https://github.com/rapidsai/cuvs/pull/420)) [@bdice](https://github.com/bdice)
- Add `click` package to `cuvs-bench` conda recipe ([#408](https://github.com/rapidsai/cuvs/pull/408)) [@divyegala](https://github.com/divyegala)
- Fix NVTX annotations ([#400](https://github.com/rapidsai/cuvs/pull/400)) [@achirkin](https://github.com/achirkin)

## 📖 Documentation

- [Doc] Fix CAGRA search sample code ([#484](https://github.com/rapidsai/cuvs/pull/484)) [@enp1s0](https://github.com/enp1s0)
- Fix broken link in README.md references ([#473](https://github.com/rapidsai/cuvs/pull/473)) [@Azurethi](https://github.com/Azurethi)
- Adding tech stack to docs ([#448](https://github.com/rapidsai/cuvs/pull/448)) [@cjnolet](https://github.com/cjnolet)
- Fix Question Retrieval notebook ([#352](https://github.com/rapidsai/cuvs/pull/352)) [@lowener](https://github.com/lowener)

## 🚀 New Features

- Add C++ API scalar quantization ([#494](https://github.com/rapidsai/cuvs/pull/494)) [@mfoerste4](https://github.com/mfoerste4)
- HNSW CPU Hierarchy ([#465](https://github.com/rapidsai/cuvs/pull/465)) [@divyegala](https://github.com/divyegala)
- Add serialization API to brute-force ([#461](https://github.com/rapidsai/cuvs/pull/461)) [@lowener](https://github.com/lowener)
- Add Question Retrieval notebook using Milvus ([#451](https://github.com/rapidsai/cuvs/pull/451)) [@lowener](https://github.com/lowener)
- Migrate feature diff for NN Descent from RAFT to cuVS ([#421](https://github.com/rapidsai/cuvs/pull/421)) [@divyegala](https://github.com/divyegala)
- Add --no-lap-sync cmd option to ann-bench ([#405](https://github.com/rapidsai/cuvs/pull/405)) [@achirkin](https://github.com/achirkin)
- Add `InnerProduct` and `CosineExpanded` metric support in NN Descent ([#177](https://github.com/rapidsai/cuvs/pull/177)) [@divyegala](https://github.com/divyegala)

## 🛠️ Improvements

- Update cuvs to match raft&#39;s cutlass changes ([#516](https://github.com/rapidsai/cuvs/pull/516)) [@vyasr](https://github.com/vyasr)
- add a README for wheels ([#504](https://github.com/rapidsai/cuvs/pull/504)) [@jameslamb](https://github.com/jameslamb)
- Move check_input_array from pylibraft ([#474](https://github.com/rapidsai/cuvs/pull/474)) [@benfred](https://github.com/benfred)
- use different wheel-size thresholds based on CUDA version ([#469](https://github.com/rapidsai/cuvs/pull/469)) [@jameslamb](https://github.com/jameslamb)
- Modify cuvs-bench to be able to generate ground truth in CPU systems ([#466](https://github.com/rapidsai/cuvs/pull/466)) [@dantegd](https://github.com/dantegd)
- enforce wheel size limits, README formatting in CI ([#464](https://github.com/rapidsai/cuvs/pull/464)) [@jameslamb](https://github.com/jameslamb)
- Moving spectral embedding and kernel gramm APIs to cuVS ([#463](https://github.com/rapidsai/cuvs/pull/463)) [@cjnolet](https://github.com/cjnolet)
- Migrate sparse knn and distances code from raft ([#457](https://github.com/rapidsai/cuvs/pull/457)) [@benfred](https://github.com/benfred)
- Don&#39;t presume pointers location infers usability. ([#441](https://github.com/rapidsai/cuvs/pull/441)) [@robertmaynard](https://github.com/robertmaynard)
- call `enable_testing` in root CMakeLists.txt ([#437](https://github.com/rapidsai/cuvs/pull/437)) [@robertmaynard](https://github.com/robertmaynard)
- CAGRA tech debt: distance descriptor and workspace memory ([#436](https://github.com/rapidsai/cuvs/pull/436)) [@achirkin](https://github.com/achirkin)
- Add ci run_ scripts needed for build infra ([#434](https://github.com/rapidsai/cuvs/pull/434)) [@robertmaynard](https://github.com/robertmaynard)
- Use environment variables in cache hit rate computation. ([#422](https://github.com/rapidsai/cuvs/pull/422)) [@bdice](https://github.com/bdice)
- Use dashes in cuvs-bench package name. ([#417](https://github.com/rapidsai/cuvs/pull/417)) [@bdice](https://github.com/bdice)
- We need to enable the c_api by default ([#416](https://github.com/rapidsai/cuvs/pull/416)) [@robertmaynard](https://github.com/robertmaynard)
- print sccache stats in builds ([#413](https://github.com/rapidsai/cuvs/pull/413)) [@jameslamb](https://github.com/jameslamb)
- make conda installs in CI stricter ([#406](https://github.com/rapidsai/cuvs/pull/406)) [@jameslamb](https://github.com/jameslamb)
- Ivf c example ([#404](https://github.com/rapidsai/cuvs/pull/404)) [@abner-ma](https://github.com/abner-ma)
- Prune workflows based on changed files ([#392](https://github.com/rapidsai/cuvs/pull/392)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- [WIP] Add pinned memory resource to C API ([#311](https://github.com/rapidsai/cuvs/pull/311)) [@ajit283](https://github.com/ajit283)
- Dynamic Batching ([#261](https://github.com/rapidsai/cuvs/pull/261)) [@achirkin](https://github.com/achirkin)

# cuvs 24.10.00 (9 Oct 2024)

## 🐛 Bug Fixes

- Use 64 bit types for dataset size calculation in CAGRA graph optimizer ([#380](https://github.com/rapidsai/cuvs/pull/380)) [@tfeher](https://github.com/tfeher)
- Remove EXPLICIT_INSTANTIATE_ONLY macros ([#358](https://github.com/rapidsai/cuvs/pull/358)) [@achirkin](https://github.com/achirkin)
- Fix order of operations for cosine IVF Flat ([#329](https://github.com/rapidsai/cuvs/pull/329)) [@lowener](https://github.com/lowener)
- Exclude any kernel symbol that uses cutlass ([#314](https://github.com/rapidsai/cuvs/pull/314)) [@benfred](https://github.com/benfred)
- [Fix] pin raft dependent to rapidsai ([#299](https://github.com/rapidsai/cuvs/pull/299)) [@rhdong](https://github.com/rhdong)
- Fix dataset dimension in IVF-PQ C wrappers ([#292](https://github.com/rapidsai/cuvs/pull/292)) [@tfeher](https://github.com/tfeher)
- Fix python ivf-pq for int8/uint8 dtypes ([#271](https://github.com/rapidsai/cuvs/pull/271)) [@benfred](https://github.com/benfred)
- FP16 API for CAGRA and IVF-PQ ([#264](https://github.com/rapidsai/cuvs/pull/264)) [@tfeher](https://github.com/tfeher)

## 📖 Documentation

- More doc updates for 24.10 ([#396](https://github.com/rapidsai/cuvs/pull/396)) [@cjnolet](https://github.com/cjnolet)
- fix 404 in documentation link in readme ([#395](https://github.com/rapidsai/cuvs/pull/395)) [@benfred](https://github.com/benfred)
- Improving getting started materials ([#342](https://github.com/rapidsai/cuvs/pull/342)) [@cjnolet](https://github.com/cjnolet)
- Fix broken examples link in README. ([#326](https://github.com/rapidsai/cuvs/pull/326)) [@bdice](https://github.com/bdice)
- Recommend `miniforge` for conda install ([#325](https://github.com/rapidsai/cuvs/pull/325)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Port remaining scripts to `cuvs_bench` ([#368](https://github.com/rapidsai/cuvs/pull/368)) [@divyegala](https://github.com/divyegala)
- [Feat] Relative change with `bitset` API feature #2439 in raft ([#350](https://github.com/rapidsai/cuvs/pull/350)) [@rhdong](https://github.com/rhdong)
- cuvs_bench plotting functions ([#347](https://github.com/rapidsai/cuvs/pull/347)) [@dantegd](https://github.com/dantegd)
- CosineExpanded Metric for IVF-PQ (normalize inputs) ([#346](https://github.com/rapidsai/cuvs/pull/346)) [@tarang-jain](https://github.com/tarang-jain)
- Python API for CAGRA+HNSW ([#246](https://github.com/rapidsai/cuvs/pull/246)) [@divyegala](https://github.com/divyegala)
- C API for CAGRA+HNSW ([#240](https://github.com/rapidsai/cuvs/pull/240)) [@divyegala](https://github.com/divyegala)
- SNMG ANN ([#231](https://github.com/rapidsai/cuvs/pull/231)) [@viclafargue](https://github.com/viclafargue)
- [FEA] Support for half-float mixed precise in brute-force ([#225](https://github.com/rapidsai/cuvs/pull/225)) [@rhdong](https://github.com/rhdong)

## 🛠️ Improvements

- Remove cuvs-cagra-search from cuvs_static link ([#388](https://github.com/rapidsai/cuvs/pull/388)) [@benfred](https://github.com/benfred)
- Add a static library for cuvs ([#382](https://github.com/rapidsai/cuvs/pull/382)) [@benfred](https://github.com/benfred)
- Put the ann-bench large_workspace_resource in managed memory ([#372](https://github.com/rapidsai/cuvs/pull/372)) [@achirkin](https://github.com/achirkin)
- Add multigpu kmeans fit function ([#348](https://github.com/rapidsai/cuvs/pull/348)) [@benfred](https://github.com/benfred)
- Update update-version.sh to use packaging lib ([#344](https://github.com/rapidsai/cuvs/pull/344)) [@AyodeAwe](https://github.com/AyodeAwe)
- remove NCCL pins in build and test environments ([#341](https://github.com/rapidsai/cuvs/pull/341)) [@jameslamb](https://github.com/jameslamb)
- Vamana/DiskANN index build ([#339](https://github.com/rapidsai/cuvs/pull/339)) [@bkarsin](https://github.com/bkarsin)
- Use CI workflow branch &#39;branch-24.10&#39; again ([#331](https://github.com/rapidsai/cuvs/pull/331)) [@jameslamb](https://github.com/jameslamb)
- fix style checks on Python 3.12 ([#328](https://github.com/rapidsai/cuvs/pull/328)) [@jameslamb](https://github.com/jameslamb)
- Update flake8 to 7.1.1. ([#327](https://github.com/rapidsai/cuvs/pull/327)) [@bdice](https://github.com/bdice)
- Add function for calculating the mutual_reachability_graph ([#323](https://github.com/rapidsai/cuvs/pull/323)) [@benfred](https://github.com/benfred)
- Simplify libcuvs conda recipe. ([#322](https://github.com/rapidsai/cuvs/pull/322)) [@bdice](https://github.com/bdice)
- Refactor dependencies.yaml to use depends-on pattern. ([#321](https://github.com/rapidsai/cuvs/pull/321)) [@bdice](https://github.com/bdice)
- Update Python versions in cuvs_bench pyproject.toml. ([#318](https://github.com/rapidsai/cuvs/pull/318)) [@bdice](https://github.com/bdice)
- Brute force knn tile size heuristic ([#316](https://github.com/rapidsai/cuvs/pull/316)) [@mfoerste4](https://github.com/mfoerste4)
- Euclidean distance example ([#315](https://github.com/rapidsai/cuvs/pull/315)) [@abner-ma](https://github.com/abner-ma)
- Migrate trustworthiness and silhouette_score stats from RAFT ([#313](https://github.com/rapidsai/cuvs/pull/313)) [@benfred](https://github.com/benfred)
- Add support for Python 3.12 ([#312](https://github.com/rapidsai/cuvs/pull/312)) [@jameslamb](https://github.com/jameslamb)
- Add `managed` option for RMM Pool memory resource to C API ([#305](https://github.com/rapidsai/cuvs/pull/305)) [@ajit283](https://github.com/ajit283)
- Update rapidsai/pre-commit-hooks ([#303](https://github.com/rapidsai/cuvs/pull/303)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Expose search function with pre-filter for ANN ([#302](https://github.com/rapidsai/cuvs/pull/302)) [@lowener](https://github.com/lowener)
- Drop Python 3.9 support ([#301](https://github.com/rapidsai/cuvs/pull/301)) [@jameslamb](https://github.com/jameslamb)
- Use CUDA math wheels ([#298](https://github.com/rapidsai/cuvs/pull/298)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove NumPy &lt;2 pin ([#297](https://github.com/rapidsai/cuvs/pull/297)) [@seberg](https://github.com/seberg)
- CAGRA - separable compilation for distance computation ([#296](https://github.com/rapidsai/cuvs/pull/296)) [@achirkin](https://github.com/achirkin)
- Updating example notebooks ([#294](https://github.com/rapidsai/cuvs/pull/294)) [@cjnolet](https://github.com/cjnolet)
- Add RMM Pool memory resource to C API ([#285](https://github.com/rapidsai/cuvs/pull/285)) [@ajit283](https://github.com/ajit283)
- Update pre-commit hooks ([#283](https://github.com/rapidsai/cuvs/pull/283)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Improve update-version.sh ([#282](https://github.com/rapidsai/cuvs/pull/282)) [@bdice](https://github.com/bdice)
- Use tool.scikit-build.cmake.version, set scikit-build-core minimum-version ([#280](https://github.com/rapidsai/cuvs/pull/280)) [@jameslamb](https://github.com/jameslamb)
- Add cuvs_bench.run python code and build ([#279](https://github.com/rapidsai/cuvs/pull/279)) [@dantegd](https://github.com/dantegd)
- Add cuvs-bench to dependencies and conda environments ([#275](https://github.com/rapidsai/cuvs/pull/275)) [@dantegd](https://github.com/dantegd)
- Update pip devcontainers to UCX v1.17.0 ([#262](https://github.com/rapidsai/cuvs/pull/262)) [@jameslamb](https://github.com/jameslamb)
- Adding example for tuning build and search params using Optuna ([#257](https://github.com/rapidsai/cuvs/pull/257)) [@dpadmanabhan03](https://github.com/dpadmanabhan03)
- Fixed link to build docs and corrected ivf_flat_example ([#255](https://github.com/rapidsai/cuvs/pull/255)) [@mmccarty](https://github.com/mmccarty)
- Merge branch-24.08 into branch-24.10 ([#254](https://github.com/rapidsai/cuvs/pull/254)) [@jameslamb](https://github.com/jameslamb)
- Persistent CAGRA kernel ([#215](https://github.com/rapidsai/cuvs/pull/215)) [@achirkin](https://github.com/achirkin)
- [FEA] Support for Cosine distance in IVF-Flat ([#179](https://github.com/rapidsai/cuvs/pull/179)) [@lowener](https://github.com/lowener)

# cuvs 24.08.00 (7 Aug 2024)

## 🚨 Breaking Changes

- Allow serialization on streams ([#173](https://github.com/rapidsai/cuvs/pull/173)) [@benfred](https://github.com/benfred)

## 🐛 Bug Fixes

- Remove fp16 kernels that have no public entry point ([#268](https://github.com/rapidsai/cuvs/pull/268)) [@tfeher](https://github.com/tfeher)
- Use `raft::util::popc(...)` public API ([#249](https://github.com/rapidsai/cuvs/pull/249)) [@divyegala](https://github.com/divyegala)
- Enable building FAISS main statically ([#241](https://github.com/rapidsai/cuvs/pull/241)) [@tarang-jain](https://github.com/tarang-jain)
- CAGRA bench: use device-side refinement when the data is on device ([#228](https://github.com/rapidsai/cuvs/pull/228)) [@achirkin](https://github.com/achirkin)
- Rename `.devcontainer`s for CUDA 12.5 ([#224](https://github.com/rapidsai/cuvs/pull/224)) [@jakirkham](https://github.com/jakirkham)
- Fix a CAGRA graph opt bug ([#192](https://github.com/rapidsai/cuvs/pull/192)) [@enp1s0](https://github.com/enp1s0)

## 📖 Documentation

- fix library name in docs (&#39;cuvs&#39; not &#39;pycuvs&#39;) ([#193](https://github.com/rapidsai/cuvs/pull/193)) [@jameslamb](https://github.com/jameslamb)

## 🚀 New Features

- Add cuvs_bench python folder, config files and constraints ([#244](https://github.com/rapidsai/cuvs/pull/244)) [@dantegd](https://github.com/dantegd)
- Add MST optimization to guarantee the connectivity of CAGRA graphs ([#237](https://github.com/rapidsai/cuvs/pull/237)) [@anaruse](https://github.com/anaruse)
- Moving over C++ API of CAGRA+hnswlib from RAFT ([#229](https://github.com/rapidsai/cuvs/pull/229)) [@divyegala](https://github.com/divyegala)
- [FEA] expose python &amp; C API for prefiltered brute force ([#174](https://github.com/rapidsai/cuvs/pull/174)) [@rhdong](https://github.com/rhdong)
- CAGRA new vector addition ([#151](https://github.com/rapidsai/cuvs/pull/151)) [@enp1s0](https://github.com/enp1s0)

## 🛠️ Improvements

- [Opt] introduce the `masked_matmul` to prefiltered brute force. ([#251](https://github.com/rapidsai/cuvs/pull/251)) [@rhdong](https://github.com/rhdong)
- Add more info to ANN_BENCH context ([#248](https://github.com/rapidsai/cuvs/pull/248)) [@achirkin](https://github.com/achirkin)
- split up CUDA-suffixed dependencies in dependencies.yaml ([#247](https://github.com/rapidsai/cuvs/pull/247)) [@jameslamb](https://github.com/jameslamb)
- Fix pinning to a different RAFT tag ([#235](https://github.com/rapidsai/cuvs/pull/235)) [@benfred](https://github.com/benfred)
- Use workflow branch 24.08 again ([#234](https://github.com/rapidsai/cuvs/pull/234)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- chore: update search_plan.cuh ([#232](https://github.com/rapidsai/cuvs/pull/232)) [@eltociear](https://github.com/eltociear)
- Enable kernel &amp; memcpy overlapping in IVF index building ([#230](https://github.com/rapidsai/cuvs/pull/230)) [@abc99lr](https://github.com/abc99lr)
- CAGRA: reduce argument count in select_and_run() kernel wrappers ([#227](https://github.com/rapidsai/cuvs/pull/227)) [@achirkin](https://github.com/achirkin)
- Mark the rust brute force unittest as flaky ([#226](https://github.com/rapidsai/cuvs/pull/226)) [@benfred](https://github.com/benfred)
- Add python bindings for ivf-* extend functions ([#220](https://github.com/rapidsai/cuvs/pull/220)) [@benfred](https://github.com/benfred)
- Build and test with CUDA 12.5.1 ([#219](https://github.com/rapidsai/cuvs/pull/219)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add col-major support for brute force knn ([#217](https://github.com/rapidsai/cuvs/pull/217)) [@benfred](https://github.com/benfred)
- Add CUDA_STATIC_MATH_LIBRARIES ([#216](https://github.com/rapidsai/cuvs/pull/216)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- skip CMake 3.30.0 ([#214](https://github.com/rapidsai/cuvs/pull/214)) [@jameslamb](https://github.com/jameslamb)
- Complete Migration of IVF Helpers / Features from RAFT ([#213](https://github.com/rapidsai/cuvs/pull/213)) [@tarang-jain](https://github.com/tarang-jain)
- Use verify-alpha-spec hook ([#209](https://github.com/rapidsai/cuvs/pull/209)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fixes for publishing rust package to crates.io ([#207](https://github.com/rapidsai/cuvs/pull/207)) [@benfred](https://github.com/benfred)
- Add rust example ([#206](https://github.com/rapidsai/cuvs/pull/206)) [@benfred](https://github.com/benfred)
- Adding IVF examples ([#203](https://github.com/rapidsai/cuvs/pull/203)) [@cjnolet](https://github.com/cjnolet)
- Fix compilation error when _CLK_BREAKDOWN is defined in cagra. ([#202](https://github.com/rapidsai/cuvs/pull/202)) [@jiangyinzuo](https://github.com/jiangyinzuo)
- DOC: update notebook link ([#191](https://github.com/rapidsai/cuvs/pull/191)) [@raybellwaves](https://github.com/raybellwaves)
- Change cagra.build_index to cagra.build ([#187](https://github.com/rapidsai/cuvs/pull/187)) [@benfred](https://github.com/benfred)
- Add python serialization API&#39;s for ivf-pq and ivf_flat ([#186](https://github.com/rapidsai/cuvs/pull/186)) [@benfred](https://github.com/benfred)
- resolve dependency-file-generator warning, rapids-build-backend followup ([#185](https://github.com/rapidsai/cuvs/pull/185)) [@jameslamb](https://github.com/jameslamb)
- Adopt CI/packaging codeowners ([#183](https://github.com/rapidsai/cuvs/pull/183)) [@raydouglass](https://github.com/raydouglass)
- Scaling workspace resources ([#181](https://github.com/rapidsai/cuvs/pull/181)) [@achirkin](https://github.com/achirkin)
- Remove text builds of documentation ([#180](https://github.com/rapidsai/cuvs/pull/180)) [@vyasr](https://github.com/vyasr)
- Add refine to the Python and C api&#39;s ([#175](https://github.com/rapidsai/cuvs/pull/175)) [@benfred](https://github.com/benfred)
- Allow serialization on streams ([#173](https://github.com/rapidsai/cuvs/pull/173)) [@benfred](https://github.com/benfred)
- Forward-merge branch-24.06 into branch-24.08 ([#169](https://github.com/rapidsai/cuvs/pull/169)) [@benfred](https://github.com/benfred)
- Use rapids-build-backend ([#145](https://github.com/rapidsai/cuvs/pull/145)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- ANN_BENCH ([#130](https://github.com/rapidsai/cuvs/pull/130)) [@achirkin](https://github.com/achirkin)
- Enable random subsampling ([#122](https://github.com/rapidsai/cuvs/pull/122)) [@tfeher](https://github.com/tfeher)

# cuvs 24.06.00 (5 Jun 2024)

## 🐛 Bug Fixes

- Fix CAGRA OOM handling ([#167](https://github.com/rapidsai/cuvs/pull/167)) [@tfeher](https://github.com/tfeher)
- Pass through raft static CMake var ([#111](https://github.com/rapidsai/cuvs/pull/111)) [@vyasr](https://github.com/vyasr)
- Fix rust docs build ([#84](https://github.com/rapidsai/cuvs/pull/84)) [@benfred](https://github.com/benfred)

## 📖 Documentation

- chore: update Doxyfile ([#162](https://github.com/rapidsai/cuvs/pull/162)) [@eltociear](https://github.com/eltociear)
- cuVS docs updates for release ([#161](https://github.com/rapidsai/cuvs/pull/161)) [@cjnolet](https://github.com/cjnolet)
- update: fix RAFT URL in README ([#91](https://github.com/rapidsai/cuvs/pull/91)) [@hurutoriya](https://github.com/hurutoriya)
- Update the developer&#39;s guide with new copyright hook ([#81](https://github.com/rapidsai/cuvs/pull/81)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add `lucene-cuvs` to integrations section of docs ([#73](https://github.com/rapidsai/cuvs/pull/73)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- Add `refine` to public API ([#154](https://github.com/rapidsai/cuvs/pull/154)) [@lowener](https://github.com/lowener)
- [FEA] support of prefiltered brute force ([#146](https://github.com/rapidsai/cuvs/pull/146)) [@rhdong](https://github.com/rhdong)
- Migrate IVF-Flat from RAFT ([#94](https://github.com/rapidsai/cuvs/pull/94)) [@divyegala](https://github.com/divyegala)
- Migrate IVF-PQ from RAFT to cuVS ([#86](https://github.com/rapidsai/cuvs/pull/86)) [@lowener](https://github.com/lowener)

## 🛠️ Improvements

- Expose serialization to the python / c-api ([#164](https://github.com/rapidsai/cuvs/pull/164)) [@benfred](https://github.com/benfred)
- Select k instantiations ([#159](https://github.com/rapidsai/cuvs/pull/159)) [@benfred](https://github.com/benfred)
- fix devcontainer name for codespaces ([#153](https://github.com/rapidsai/cuvs/pull/153)) [@trxcllnt](https://github.com/trxcllnt)
- Accept host_mdspan for IVF-PQ build and extend ([#148](https://github.com/rapidsai/cuvs/pull/148)) [@tfeher](https://github.com/tfeher)
- Add pairwise_distance api&#39;s for C, Python and Rust ([#142](https://github.com/rapidsai/cuvs/pull/142)) [@benfred](https://github.com/benfred)
- Changing RAFT_EXPLICT_* to CUVS_EXPLITI_* ([#141](https://github.com/rapidsai/cuvs/pull/141)) [@cjnolet](https://github.com/cjnolet)
- Speed-up rust build ([#138](https://github.com/rapidsai/cuvs/pull/138)) [@benfred](https://github.com/benfred)
- Removing `libraft.so` from libcuvs dependencies ([#132](https://github.com/rapidsai/cuvs/pull/132)) [@cjnolet](https://github.com/cjnolet)
- CAGRA API update and allow async host refinement ([#131](https://github.com/rapidsai/cuvs/pull/131)) [@mfoerste4](https://github.com/mfoerste4)
- Fix rust api docs ([#119](https://github.com/rapidsai/cuvs/pull/119)) [@benfred](https://github.com/benfred)
- Migrate BFKNN from raft ([#118](https://github.com/rapidsai/cuvs/pull/118)) [@benfred](https://github.com/benfred)
- Fix IVF-PQ helper functions ([#116](https://github.com/rapidsai/cuvs/pull/116)) [@lowener](https://github.com/lowener)
- Migrate `raft::cluster` to `cuvs::cluster` ([#115](https://github.com/rapidsai/cuvs/pull/115)) [@cjnolet](https://github.com/cjnolet)
- hide RAFT #pragma deprecation warnings ([#114](https://github.com/rapidsai/cuvs/pull/114)) [@trxcllnt](https://github.com/trxcllnt)
- Enable Warnings as errors in Python tests ([#102](https://github.com/rapidsai/cuvs/pull/102)) [@mroeschke](https://github.com/mroeschke)
- Remove libnvjitlink dependency. ([#97](https://github.com/rapidsai/cuvs/pull/97)) [@bdice](https://github.com/bdice)
- Migrate to `{{ stdlib(&quot;c&quot;) }}` ([#93](https://github.com/rapidsai/cuvs/pull/93)) [@hcho3](https://github.com/hcho3)
- update: replace to cuvs from RAFT in PULL_REQUEST_TEMPLATE ([#92](https://github.com/rapidsai/cuvs/pull/92)) [@hurutoriya](https://github.com/hurutoriya)
- Add python and rust bindings for Ivf-Pq ([#90](https://github.com/rapidsai/cuvs/pull/90)) [@benfred](https://github.com/benfred)
- add --rm and --name to devcontainer run args ([#89](https://github.com/rapidsai/cuvs/pull/89)) [@trxcllnt](https://github.com/trxcllnt)
- Update pip devcontainers to UCX v1.15.0 ([#88](https://github.com/rapidsai/cuvs/pull/88)) [@trxcllnt](https://github.com/trxcllnt)
- Remove gtest from dependencies.yaml ([#87](https://github.com/rapidsai/cuvs/pull/87)) [@robertmaynard](https://github.com/robertmaynard)
- Moving and renaming distance namespaces from raft -&gt; cuvs ([#85](https://github.com/rapidsai/cuvs/pull/85)) [@cjnolet](https://github.com/cjnolet)
- Use static gtest ([#83](https://github.com/rapidsai/cuvs/pull/83)) [@robertmaynard](https://github.com/robertmaynard)
- Add python and rust bindings for Ivf-Flat ([#82](https://github.com/rapidsai/cuvs/pull/82)) [@benfred](https://github.com/benfred)
- Forward merge branch-24.04 to branch-24.06 ([#80](https://github.com/rapidsai/cuvs/pull/80)) [@benfred](https://github.com/benfred)
- Update devcontainers to use cuda12.2 ([#72](https://github.com/rapidsai/cuvs/pull/72)) [@benfred](https://github.com/benfred)
- Forward merge branch-24.04 to branch-24.06 ([#71](https://github.com/rapidsai/cuvs/pull/71)) [@benfred](https://github.com/benfred)
- Enable forward-merger ops-bot plugin ([#70](https://github.com/rapidsai/cuvs/pull/70)) [@benfred](https://github.com/benfred)
- Adds missing files to `update-version.sh` ([#69](https://github.com/rapidsai/cuvs/pull/69)) [@AyodeAwe](https://github.com/AyodeAwe)
- Add Cagra-Q compression to the python and rust api&#39;s ([#68](https://github.com/rapidsai/cuvs/pull/68)) [@benfred](https://github.com/benfred)
- ConfigureCUDA.cmake now sets CUVS_ prefixed variables ([#66](https://github.com/rapidsai/cuvs/pull/66)) [@robertmaynard](https://github.com/robertmaynard)
- Enable all tests for `arm` jobs ([#63](https://github.com/rapidsai/cuvs/pull/63)) [@galipremsagar](https://github.com/galipremsagar)
