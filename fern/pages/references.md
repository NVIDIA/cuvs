# References

Many state-of-the-art implementations of vector search, vector preprocessing, vector compression, and vector clustering algorithms influenced the creation of NVIDIA cuVS. These papers describe core algorithms and GPU primitives used throughout NVIDIA cuVS, from graph-based approximate nearest-neighbor search to clustering, sparse neighborhood methods, top-k selection, and filtered vector search.

Use this page when citing the research behind NVIDIA cuVS algorithms or when looking for deeper technical background on the methods implemented in the library.

## CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs

[Paper](https://arxiv.org/abs/2308.15136)

CAGRA introduces a GPU-accelerated graph construction and approximate nearest-neighbor search algorithm. It is the main research foundation for NVIDIA cuVS CAGRA, a graph-based vector search index optimized for fast GPU index build and high-throughput GPU search.

```bibtex
@inproceedings{ootomo2024cagra,
  author    = {Ootomo, Hiroyuki and Naruse, Akira and Nolet, Corey and Wang, Ray and Feher, Tamas and Wang, Yong},
  title     = {CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs},
  booktitle = {2024 IEEE 40th International Conference on Data Engineering (ICDE)},
  pages     = {4236--4247},
  year      = {2024},
  doi       = {10.1109/ICDE60146.2024.00323}
}
```

## Parallel Top-K Algorithms on GPU: A Comprehensive Study and New Methods

[Paper](https://doi.org/10.1145/3581784.3607062)

This paper studies GPU top-k selection and introduces AIR top-K and GridSelect. Efficient top-k selection is a core primitive for nearest-neighbor search because search algorithms often need to keep only the best candidate neighbors out of a much larger set.

```bibtex
@inproceedings{zhang2023parallelTopK,
  author    = {Zhang, Jingrong and Naruse, Akira and Li, Xipeng and Wang, Yong},
  title     = {Parallel Top-K Algorithms on GPU: A Comprehensive Study and New Methods},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  series    = {SC '23},
  pages     = {76:1--76:13},
  year      = {2023},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3581784.3607062}
}
```

## Fast k-NN Graph Construction by GPU Based NN-Descent

[Paper](https://doi.org/10.1145/3459637.3482344)

This paper adapts NN-Descent to GPU architecture for fast approximate k-nearest-neighbor graph construction. It provides background for NVIDIA cuVS NN-Descent and for workflows that use k-NN graphs as intermediate structures.

```bibtex
@inproceedings{wang2021fastKnnGraph,
  author    = {Wang, Hui and Zhao, Wan-Lei and Zeng, Xiangxiang and Yang, Jianye},
  title     = {Fast K-NN Graph Construction by GPU Based NN-Descent},
  booktitle = {Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
  series    = {CIKM '21},
  pages     = {1929--1938},
  year      = {2021},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3459637.3482344}
}
```

## cuSLINK: Single-linkage Agglomerative Clustering on the GPU

[Paper](https://arxiv.org/abs/2306.16354)

cuSLINK reformulates single-linkage agglomerative clustering for the GPU. It connects clustering with nearest-neighbor graph construction, spanning trees, and dendrogram extraction, which makes it relevant to NVIDIA cuVS clustering and graph-building routines.

```bibtex
@misc{nolet2023cuslink,
  title         = {cuSLINK: Single-linkage Agglomerative Clustering on the GPU},
  author        = {Corey J. Nolet and Divye Gala and Alex Fender and Mahesh Doijade and Joe Eaton and Edward Raff and John Zedlewski and Brad Rees and Tim Oates},
  year          = {2023},
  eprint        = {2306.16354},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## GPU Semiring Primitives for Sparse Neighborhood Methods

[Paper](https://arxiv.org/abs/2104.06357)

This paper presents GPU semiring primitives for sparse vector operations and neighborhood methods. These primitives provide background for sparse-distance and sparse-neighborhood workflows that can appear in vector search, preprocessing, and machine-learning pipelines.

```bibtex
@article{nolet2021semiring,
  author  = {Nolet, Corey J. and Gala, Divye and Raff, Edward and Eaton, Joe and Rees, Brad and Zedlewski, John and Oates, Tim},
  title   = {GPU Semiring Primitives for Sparse Neighborhood Methods},
  journal = {arXiv preprint arXiv:2104.06357},
  year    = {2021}
}
```

## VecFlow: A High-Performance Vector Data Management System for Filtered-Search on GPUs

[Paper](https://arxiv.org/abs/2506.00812)

VecFlow studies filtered approximate nearest-neighbor search on GPUs. It is useful background for NVIDIA cuVS filtered-search work and for systems that combine vector indexes with structured metadata filters.

```bibtex
@article{vecflow2025,
  author  = {Xi, Jingyi and Mo, Chenghao and Karsin, Ben and Chirkin, Artem and Li, Mingqin and Zhang, Minjia},
  title   = {VecFlow: A High-Performance Vector Data Management System for Filtered-Search on GPUs},
  journal = {arXiv preprint arXiv:2506.00812},
  year    = {2025}
}
```

## GPU-Native Approximate Nearest Neighbor Search with IVF-RaBitQ: Fast Index Build and Search

[Paper](https://arxiv.org/abs/2602.23999)

IVF-RaBitQ is a GPU-native ANNS solution that integrates the cluster-based IVF method with RaBitQ quantization into an efficient GPU index build and search pipeline.

```bibtex
@article{shi2026ivfrabitq,
  author  = {Shi, Jifan and Gao, Jianyang and Xia, James and Feh{\'e}r, Tam{\'a}s B{\'e}la and Long, Cheng},
  title   = {GPU-Native Approximate Nearest Neighbor Search with IVF-RaBitQ: Fast Index Build and Search},
  journal = {arXiv preprint arXiv:2602.23999},
  year    = {2026}
}
```
