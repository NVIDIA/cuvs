#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Convert serialized cuVS-ScaNN index artifacts into the protobuf structs that
OSS ScaNN needs for search.

This is a packaged, importable version of the conversion in the
``scann_protobuf_example.ipynb`` notebook (authored by Robert Maschal). Given a
directory holding the files that ``cuvs::neighbors::scann::serialize`` writes
(``cuvs_metadata.bin``, ``centers.npy``, ``pq_codebook.npy``,
``datapoint_to_token.npy``, ``hashed_dataset*.npy`` and, for two-level indexes,
``coarse_*.npy``), :func:`build_scann_assets` produces the remaining artifacts:

* ``serialized_partitioner.pb`` - the k-means tree
* ``ah_codebook.pb`` - the product-quantization codebooks
* ``scann_config.pb`` - the search config
* ``scann_assets.pbtxt`` - the asset manifest

The ScaNN protobuf modules are imported lazily inside this module so importing
it does not require ``scann`` until a conversion actually runs.
"""

import os

import numpy as np

from scann.data_format.features_pb2 import GenericFeatureVector
from scann.partitioning.partitioner_pb2 import SerializedPartitioner
from scann.proto.centers_pb2 import CentersForAllSubspaces, CentersForSubspace
from scann.proto.hash_pb2 import AsymmetricHasherConfig
from scann.proto.partitioning_pb2 import (
    BottomUpTopLevelPartitioner,
    DatabaseSpillingConfig,
    PartitioningConfig,
    QuerySpillingConfig,
)
from scann.proto.projection_pb2 import ProjectionConfig
from scann.proto.scann_pb2 import ScannConfig
from scann.trees.kmeans_tree.kmeans_tree_pb2 import SerializedKMeansTree


class ScannParams(object):
    """
    Dataclass storing param values during index build
    """

    def __init__(
        self,
        num_leaves,
        num_leaves_to_search,
        partitioning_eta,
        soar_lambda,
        partitioning_training_size,
        pq_dim,
        pq_training_size,
        pq_threshold,
        bf16,
        bf16_threshold,
        n_coarse_clusters,
        n_coarse_clusters_to_search,
    ):
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        self.partitioning_eta = partitioning_eta
        self.soar_lambda = soar_lambda
        self.partitioning_training_size = partitioning_training_size
        self.pq_dim = pq_dim
        self.pq_training_size = pq_training_size
        self.pq_threshold = pq_threshold
        self.bf16 = bf16
        self.bf16_threshold = bf16_threshold
        self.n_coarse_clusters = n_coarse_clusters
        self.n_coarse_clusters_to_search = n_coarse_clusters_to_search


class CuvsScannIndex(object):
    """
    Class storing cuvs-Scann index build artifacts and metadata required
    for producing protobuf artifacts for OSS ScaNN search
    """

    def __init__(self, scann_dir):
        metadata_path = os.path.join(scann_dir, "cuvs_metadata.bin")
        with open(metadata_path, "rb") as f:
            self.version = np.load(f).item()
            self.dim = np.load(f).item()
            self.pq_dim = np.load(f).item()
            self.num_subspaces = int(self.dim / self.pq_dim)
            self.num_levels = np.load(f).item()

        centers_path = os.path.join(scann_dir, "centers.npy")
        with open(centers_path, "rb") as f:
            self.centers = np.load(f)

        self.coarse_centers = None
        coarse_centers_path = os.path.join(scann_dir, "coarse_centers.npy")
        if os.path.exists(coarse_centers_path):
            with open(coarse_centers_path, "rb") as f:
                self.coarse_centers = np.load(f)

        self.coarse_labels = None
        coarse_labels_path = os.path.join(scann_dir, "coarse_labels.npy")
        if os.path.exists(coarse_labels_path):
            with open(coarse_labels_path, "rb") as f:
                self.coarse_labels = np.load(f)

            self.coarse_labels = self.coarse_labels.reshape(
                self.centers.shape[0], -1
            )

        pq_codebook_path = os.path.join(scann_dir, "pq_codebook.npy")
        with open(pq_codebook_path, "rb") as f:
            self.pq_codebook = np.load(f)


def build_tree_level(level, centers, labels=None):
    inverted_labels = {}

    if labels is not None:
        for i in range(0, labels.shape[0]):
            for j in range(0, labels.shape[1]):
                label = labels[i][j].item()

                if label == -1:
                    continue

                if label not in inverted_labels:
                    inverted_labels[label] = []

                inverted_labels[label].append(i)

    centers_list = []
    children_list = []

    for i in range(0, centers.shape[0]):
        center = SerializedKMeansTree.Center()
        center.dimension.extend(centers[i, :].tolist())
        centers_list.append(center)

        child = SerializedKMeansTree.Node()
        child.leaf_id = i
        child.learned_spilling_threshold = float("nan")

        if i in inverted_labels:
            child.indices.extend(inverted_labels[i])

        children_list.append(child)

    del level.root.centers[:]
    del level.root.children[:]

    level.root.centers.extend(centers_list)
    level.root.children.extend(children_list)
    level.root.learned_spilling_threshold = float("nan")


def build_partitioner_from_idx(idx, partitioner=None):
    """
    Build the kmeans tree structure from cuvs-Scann Index

    Parameters
    ----------
    idx : CuvsScannIndex
    partitioner : SerializedPartitioner, optional

    Returns
    -------
    SerializedPartitioner
        Stores the kmeans tree structure from the cuvs index.
    """
    if partitioner is None:
        partitioner = SerializedPartitioner()

    kmeans_tree_partitioner = partitioner.kmeans
    bottom_level = kmeans_tree_partitioner.kmeans_tree

    partitioner.n_tokens = idx.centers.shape[0]

    build_tree_level(bottom_level, idx.centers, None)

    if idx.num_levels == 2:
        next_bottom_up_level = kmeans_tree_partitioner.next_bottom_up_level
        build_tree_level(
            next_bottom_up_level.kmeans_tree,
            idx.coarse_centers,
            idx.coarse_labels,
        )

    return partitioner


def serialized_partitioner_from_index(scann_dir, idx):
    """
    Produce serialized_partitioner.pb from cuvs-Scann index

    Parameters
    ----------
    scann_dir : str
        Directory where scann artifacts are stored.
    idx : CuvsScannIndex
    """

    sp_path = os.path.join(scann_dir, "serialized_partitioner.pb")

    sp = SerializedPartitioner()
    sp = build_partitioner_from_idx(idx, sp)
    sp_str = sp.SerializeToString()

    with open(sp_path, "wb") as f:
        f.write(sp_str)


def pq_codebook_from_index(scann_dir, idx):
    """
    Produce ah_codebook.pb from cuvs-Scann index

    Parameters
    ----------
    scann_dir : str
        Directory where scann artifacts are stored.
    idx : CuvsScannIndex
    """

    codebook_path = os.path.join(scann_dir, "ah_codebook.pb")

    codebook = CentersForAllSubspaces()
    codebook.quantization_scheme = (
        AsymmetricHasherConfig.QuantizationScheme.PRODUCT
    )

    sub_dim = idx.pq_dim
    subspace_centers = []

    for j in range(0, idx.num_subspaces):
        gfvs = []
        sub_start = sub_dim * j
        sub_end = sub_start + sub_dim
        for i in range(0, idx.pq_codebook.shape[0]):
            gfv = GenericFeatureVector()
            gfv.feature_type = GenericFeatureVector.FeatureType.FLOAT
            gfv.feature_value_float.extend(
                idx.pq_codebook[i][sub_start:sub_end]
            )

            gfvs.append(gfv)

        subspace_center = CentersForSubspace()
        subspace_center.center.extend(gfvs)
        subspace_centers.append(subspace_center)

    codebook.subspace_centers.extend(subspace_centers)

    codebook_str = codebook.SerializeToString()

    with open(codebook_path, "wb") as f:
        f.write(codebook_str)


def populate_partitioner_config(partitioning, idx, params):
    partitioning.num_children = params.num_leaves
    partitioning.min_cluster_size = 50
    partitioning.max_clustering_iterations = 12
    partitioning.single_machine_center_initialization = PartitioningConfig.SingleMachineCenterInitializationType.RANDOM_INITIALIZATION
    partitioning.partitioning_distance.distance_measure = "SquaredL2Distance"

    partitioning.query_spilling.max_spill_centers = 50
    partitioning.query_spilling.spilling_type = (
        QuerySpillingConfig.FIXED_NUMBER_OF_CENTERS
    )

    partitioning.database_spilling.spilling_type = (
        DatabaseSpillingConfig.SpillingType.TWO_CENTER_ORTHOGONALITY_AMPLIFIED
    )
    partitioning.database_spilling.orthogonality_amplification_lambda = (
        params.soar_lambda
    )

    partitioning.expected_sample_size = params.partitioning_training_size
    partitioning.query_tokenization_distance_override.distance_measure = (
        "DotProductDistance"
    )
    partitioning.partitioning_type = (
        PartitioningConfig.PartitioningType.GENERIC
    )

    partitioning.query_tokenization_type = (
        PartitioningConfig.TokenizationType.FLOAT
    )
    partitioning.avq = params.partitioning_eta

    if idx.num_levels == 2:
        bottom_up_partitioner = partitioning.bottom_up_top_level_partitioner

        bottom_up_partitioner.enabled = True
        bottom_up_partitioner.num_centroids = idx.coarse_centers.shape[0]
        bottom_up_partitioner.num_centroids_to_search = (
            params.n_coarse_clusters_to_search
        )
        bottom_up_partitioner.quantization = (
            BottomUpTopLevelPartitioner.Quantization.FLOAT32
        )
        bottom_up_partitioner.avq = params.partitioning_eta
        bottom_up_partitioner.soar.enabled = True

        # lambda is a reserved word in python
        setattr(bottom_up_partitioner.soar, "lambda", params.soar_lambda)
        bottom_up_partitioner.soar.overretrieve_factor = 2


def populate_hashing_config(hashing, idx, params):
    hashing.num_clusters_per_block = 16
    hashing.max_clustering_iterations = 10
    hashing.quantization_distance.distance_measure = "SquaredL2Distance"
    hashing.lookup_type = AsymmetricHasherConfig.LookupType.INT8_LUT16
    hashing.use_residual_quantization = True
    hashing.noise_shaping_threshold = params.pq_threshold
    hashing.expected_sample_size = params.pq_training_size
    hashing.use_global_topn = idx.num_subspaces <= 256

    hashing.fixed_point_lut_conversion_options.float_to_int_conversion_method = AsymmetricHasherConfig.FixedPointLUTConversionOptions.ROUND
    hashing.projection.projection_type = ProjectionConfig.ProjectionType.CHUNK
    hashing.projection.num_blocks = idx.num_subspaces
    hashing.projection.num_dims_per_block = idx.pq_dim
    hashing.projection.input_dim = idx.dim


def scann_config_from_params(idx, scann_dir, params):
    """
    Produce scann_config.pb

    Parameters
    ----------
    idx : CuvsScannIndex
    scann_dir : str
        Directory where scann artifacts are stored.
    params : ScannParams
        Index build parameters.
    """
    scann_config_path = os.path.join(scann_dir, "scann_config.pb")

    scann_config = ScannConfig()
    scann_config.num_neighbors = 10
    scann_config.distance_measure.distance_measure = "DotProductDistance"

    populate_partitioner_config(scann_config.partitioning, idx, params)

    populate_hashing_config(scann_config.hash.asymmetric_hash, idx, params)

    scann_config.exact_reordering.approx_num_neighbors = 100
    scann_config.exact_reordering.bfloat16.enabled = params.bf16

    if params.bf16:
        scann_config.exact_reordering.bfloat16.noise_shaping_threshold = 0.2

    with open(scann_config_path, "wb") as f:
        f.write(scann_config.SerializeToString())


def scann_assets_file(root_dir, params):
    """
    Produce scann_assets.pbtxt

    Parameters
    ----------
    root_dir : str
        Path where ScaNN assets are stored.
    params : ScannParams
    """

    bf_16 = """assets {
        asset_type: BF16_DATASET_NPY
        asset_path: "bf16_dataset.npy"
    }
    """
    assets_str = """
    assets {{
      asset_type: AH_CENTERS
      asset_path: "ah_codebook.pb"
    }}
    assets {{
      asset_type: PARTITIONER
      asset_path: "serialized_partitioner.pb"
    }}
    assets {{
      asset_type: TOKENIZATION_NPY
      asset_path: "datapoint_to_token.npy"
    }}
    assets {{
      asset_type: AH_DATASET_NPY
      asset_path: "hashed_dataset.npy"
    }}
    assets {{
      asset_type: AH_DATASET_SOAR_NPY
      asset_path: "hashed_dataset_soar.npy"
    }}
    assets {{
      asset_type: DATASET_NPY
      asset_path: "dataset.npy"
    }}
    {bf16_clause}
    """.format(bf16_clause=bf_16 if params.bf16 else "")

    assets_path = os.path.join(root_dir, "scann_assets.pbtxt")

    with open(assets_path, "w") as f:
        f.write(assets_str)


def build_scann_assets(scann_dir, params):
    """
    Given a cuVS-ScaNN index, produce the remaining artifacts required
    for consumption into OSS ScaNN for search. this includes:
    * ah_codebook.pb (the product quantization codebook)
    * serialized_partitioner.pb (the kmeans tree)
    * scann_config.pb (the config)
    * scann_assets.pbtxt (a list of scann assets and their filenames)

    Parameters
    ----------
    scann_dir : str
        The directory where cuvs artifacts are stored.
    params : ScannParams
        Parameters used during index build.
    """

    idx = CuvsScannIndex(scann_dir)

    serialized_partitioner_from_index(scann_dir, idx)
    pq_codebook_from_index(scann_dir, idx)
    scann_config_from_params(idx, scann_dir, params)
    scann_assets_file(scann_dir, params)
