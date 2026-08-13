#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Base classes and data structures for the cuvs-bench plugin system.

This module defines the abstract interface that all benchmark backends must implement,
along with standardized data structures for datasets and results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..orchestrator.config_loaders import IndexConfig


class Dataset:
    """
    Dataset representation for benchmarking.

    Supports two usage patterns:

    1. File-path based (C++ backend): The orchestrator passes file paths
       and the backend reads them directly via ``base_file``, ``query_file``,
       etc. Vectors are never loaded into Python.

    2. Array based (Python-native backends like OpenSearch, Elasticsearch):
       Backends access ``training_vectors``, ``query_vectors``, etc. If vectors
       were not provided directly but a file path exists, they are loaded
       lazily on first access. This keeps file I/O invisible to backends.

    Parameters
    ----------
    name : str
        Dataset name (e.g., "glove-100-inner")
    training_vectors : Optional[np.ndarray]
        Training vectors for index building, shape (n_vectors, dims)
    query_vectors : Optional[np.ndarray]
        Query vectors for search, shape (n_queries, dims)
    groundtruth_neighbors : Optional[np.ndarray]
        Ground truth neighbor IDs, shape (n_queries, k_gt)
    groundtruth_distances : Optional[np.ndarray]
        Ground truth distances, shape (n_queries, k_gt)
    distance_metric : str
        Distance metric ("euclidean", "inner_product", "cosine")
    base_file : Optional[str]
        Path to base vectors file
    query_file : Optional[str]
        Path to query vectors file
    groundtruth_neighbors_file : Optional[str]
        Path to ground truth neighbors file
    groundtruth_distances_file : Optional[str]
        Path to ground truth distances file
    metadata : Optional[Dict[str, Any]]
        Additional dataset metadata like {"subset_size": 10000}
    """

    def __init__(
        self,
        name: str,
        training_vectors: Optional[np.ndarray] = None,
        query_vectors: Optional[np.ndarray] = None,
        groundtruth_neighbors: Optional[np.ndarray] = None,
        groundtruth_distances: Optional[np.ndarray] = None,
        distance_metric: str = "euclidean",
        base_file: Optional[str] = None,
        query_file: Optional[str] = None,
        groundtruth_neighbors_file: Optional[str] = None,
        groundtruth_distances_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        # Vectors are stored privately to support lazy loading.
        # If the caller provides vectors directly, they are used as-is.
        # If empty and a file path is provided, the corresponding property
        # loads vectors from the file on first access. This keeps the
        # loading logic invisible to backends: they just access
        # dataset.training_vectors and get a numpy array regardless of
        # whether it was passed in or loaded from disk.
        self._training_vectors = (
            training_vectors
            if training_vectors is not None
            else np.empty((0, 0))
        )
        self._query_vectors = (
            query_vectors if query_vectors is not None else np.empty((0, 0))
        )
        self._groundtruth_neighbors = groundtruth_neighbors
        self._groundtruth_distances = groundtruth_distances
        self.distance_metric = distance_metric
        self.base_file = base_file
        self.query_file = query_file
        self.groundtruth_neighbors_file = groundtruth_neighbors_file
        self.groundtruth_distances_file = groundtruth_distances_file
        self.metadata = metadata or {}

    @property
    def training_vectors(self) -> np.ndarray:
        """Training vectors for index building.

        Loaded from base_file on first access if not provided directly.
        """
        if self._training_vectors.size == 0 and self.base_file:
            from ._utils import load_vectors

            self._training_vectors = load_vectors(
                self.base_file, self.metadata.get("subset_size")
            )
        return self._training_vectors

    @training_vectors.setter
    def training_vectors(self, value: Optional[np.ndarray]) -> None:
        """Set training vectors directly."""
        self._training_vectors = (
            value if value is not None else np.empty((0, 0))
        )

    @property
    def query_vectors(self) -> np.ndarray:
        """Query vectors for search.

        Loaded from query_file on first access if not provided directly.
        """
        if self._query_vectors.size == 0 and self.query_file:
            from ._utils import load_vectors

            self._query_vectors = load_vectors(self.query_file)
        return self._query_vectors

    @query_vectors.setter
    def query_vectors(self, value: Optional[np.ndarray]) -> None:
        """Set query vectors directly."""
        self._query_vectors = value if value is not None else np.empty((0, 0))

    @property
    def groundtruth_neighbors(self) -> Optional[np.ndarray]:
        """Ground truth neighbor IDs.

        Loaded from groundtruth_neighbors_file on first access if not
        provided directly.
        """
        if (
            self._groundtruth_neighbors is None
            and self.groundtruth_neighbors_file
        ):
            from ._utils import load_vectors

            self._groundtruth_neighbors = load_vectors(
                self.groundtruth_neighbors_file
            )
        return self._groundtruth_neighbors

    @groundtruth_neighbors.setter
    def groundtruth_neighbors(self, value: Optional[np.ndarray]) -> None:
        """Set ground truth neighbors directly."""
        self._groundtruth_neighbors = value

    @property
    def groundtruth_distances(self) -> Optional[np.ndarray]:
        """Ground truth distances.

        Loaded from groundtruth_distances_file on first access if not
        provided directly.
        """
        if (
            self._groundtruth_distances is None
            and self.groundtruth_distances_file
        ):
            from ._utils import load_vectors

            self._groundtruth_distances = load_vectors(
                self.groundtruth_distances_file
            )
        return self._groundtruth_distances

    @groundtruth_distances.setter
    def groundtruth_distances(self, value: Optional[np.ndarray]) -> None:
        """Set ground truth distances directly."""
        self._groundtruth_distances = value

    @property
    def dims(self) -> int:
        """Vector dimensionality."""
        if self.training_vectors.size == 0:
            return 0
        return self.training_vectors.shape[1]

    @property
    def n_base(self) -> int:
        """Number of base vectors."""
        return self.training_vectors.shape[0]

    @property
    def n_queries(self) -> int:
        """Number of query vectors."""
        return self.query_vectors.shape[0]


@dataclass
class BuildResult:
    """
    Results from index building phase.

    Attributes
    ----------
    index_path : str
        Path to the built index
    build_time_seconds : float
        Time taken to build the index (seconds)
    index_size_bytes : int
        Size of the built index (bytes)
    algorithm : str
        Algorithm name (e.g., "cuvs_ivf_flat")
    build_params : Dict[str, Any]
        Build parameters used (e.g., {"nlist": 1024, "niter": 20})
    metadata : Dict[str, Any]
        Additional metrics (e.g., GPU time, CPU time, memory usage)
    success : bool
        Whether the build succeeded
    error_message : Optional[str]
        Error message if build failed
    """

    index_path: str
    build_time_seconds: float
    index_size_bytes: int
    algorithm: str
    build_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable format (Google Benchmark compatible).

        Returns
        -------
        Dict[str, Any]
            Dictionary with benchmark results
        """
        return {
            "name": f"{self.algorithm}/build",
            "real_time": self.build_time_seconds,
            "time_unit": "s",
            "index_size": self.index_size_bytes,
            "success": self.success,
            **self.build_params,
            **self.metadata,
        }


@dataclass
class SearchResult:
    """
    Results from search phase.

    Attributes
    ----------
    neighbors : np.ndarray
        Neighbor IDs returned by search, shape (n_queries, k)
    distances : np.ndarray
        Distances to neighbors, shape (n_queries, k)
    search_time_ms : float
        Total search time (milliseconds)
    queries_per_second : float
        Query throughput (QPS)
    recall : float
        Recall@k metric (0.0 to 1.0)
    algorithm : str
        Algorithm name
    search_params : List[Dict[str, Any]]
        List of search parameter combinations used (e.g., [{"nprobe": 1}, {"nprobe": 5}])
        All are batched into one C++ command (matches runners.py behavior)
    latency_percentiles : Optional[Dict[str, float]]
        Latency percentiles in milliseconds (p50, p95, p99)
    gpu_time_seconds : Optional[float]
        GPU execution time (if applicable)
    cpu_time_seconds : Optional[float]
        CPU execution time
    metadata : Dict[str, Any]
        Additional metrics
    success : bool
        Whether the search succeeded
    error_message : Optional[str]
        Error message if search failed
    """

    neighbors: np.ndarray
    distances: np.ndarray
    search_time_ms: float
    queries_per_second: float
    recall: float
    algorithm: str
    search_params: List[Dict[str, Any]]
    latency_percentiles: Optional[Dict[str, float]] = None
    gpu_time_seconds: Optional[float] = None
    cpu_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable format (Google Benchmark compatible).

        Returns
        -------
        Dict[str, Any]
            Dictionary with benchmark results
        """
        result = {
            "name": f"{self.algorithm}/search",
            "real_time": self.search_time_ms,
            "time_unit": "ms",
            "items_per_second": self.queries_per_second,
            "Recall": self.recall,
            "success": self.success,
            "search_params": self.search_params,
            **self.metadata,
        }

        if self.latency_percentiles:
            result.update(self.latency_percentiles)

        if self.gpu_time_seconds is not None:
            result["GPU"] = self.gpu_time_seconds

        if self.cpu_time_seconds is not None:
            result["cpu_time"] = self.cpu_time_seconds

        return result


class BenchmarkBackend(ABC):
    """
    Abstract base class for all benchmark backends.

    All backends must implement this interface to be compatible with
    the cuvs-bench execution layer.

    Parameters
    ----------
    config : Dict[str, Any]
        Backend-specific configuration. Common keys include:

        Required (enforced by base class):
        - name : str - User-defined label for this index configuration
            e.g., "cuvs_ivf_flat_test.nlist1024.ratio1" or "cuvs_cagra.graph_degree64"

        C++ backend keys (CppGoogleBenchmarkBackend):
        - executable_path : str - Path to C++ benchmark executable (required)
        - data_prefix : str - Prefix for dataset paths (default: "")
        - warmup_time : float - Warmup time in seconds (default: 1.0)
        - dataset : str - Dataset name e.g., "sift-128-euclidean" (default: "")
        - output_filename : Tuple[str, str] - (build_name, search_name) (default: ("", ""))
        - algo : str - Algorithm name e.g., "cuvs_cagra" (default: "") FIXME: Should this be required?
        - requires_gpu : bool - Whether backend requires GPU (default: False)

        Network backend keys (e.g., Milvus, Qdrant):
        - host : str - Server hostname
        - port : int - Server port
        - api_key : str - Authentication key
        - requires_network : bool - Whether backend requires network (default: False)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize backend with configuration."""
        self.config = config

    @abstractmethod
    def build(
        self,
        dataset: Dataset,
        indexes: List["IndexConfig"],
        force: bool = False,
        dry_run: bool = False,
    ) -> BuildResult:
        """
        Build indexes from the given dataset.

        ALL indexes are batched into ONE command execution (matches runners.py).

        Parameters
        ----------
        dataset : Dataset
            Dataset with base vectors and metadata
        indexes : List[IndexConfig]
            List of index configurations to build together. Internally creates a temp
            config file and builds all indexes in one command.
            FIXME: Might be specific to C++ backend.
        force : bool, optional
            If True, rebuild even if index exists; if False, skip if exists
        dry_run : bool, optional
            If True, print command without executing

        Returns
        -------
        BuildResult
            Build timing, index size, and metadata

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        pass

    @abstractmethod
    def search(
        self,
        dataset: Dataset,
        indexes: List["IndexConfig"],
        k: int,
        batch_size: int = 10000,
        mode: str = "latency",
        force: bool = False,
        search_threads: Optional[int] = None,
        dry_run: bool = False,
    ) -> SearchResult:
        """
        Search for nearest neighbors using the built indexes.

        ALL indexes are batched into ONE command execution (matches runners.py).
        Each index has its own search_params list, so total benchmarks =
        sum(len(idx.search_params) for idx in indexes).

        Parameters
        ----------
        dataset : Dataset
            Dataset with query vectors and ground truth
        indexes : List[IndexConfig]
            List of index configurations, each with its own search_params
            FIXME: Might be specific to C++ backend.
        k : int
            Number of neighbors to return per query
        batch_size : int, optional
            Number of queries to process at once (default: 10000)
        mode : str, optional
            "latency" (measure individual query latency with percentiles) or
            "throughput" (measure overall QPS) (default: "latency")
        force : bool, optional
            Whether to force the execution regardless of existing results (default: False)
        search_threads : Optional[int], optional
            Number of threads to use for searching (default: None)
        dry_run : bool, optional
            If True, print command without executing

        Returns
        -------
        SearchResult
            Search timing, results, and recall metrics

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        pass

    def initialize(self) -> None:
        """
        Initialize backend (called once before any build/search operations).

        Note: C++ backends handle initialization internally via subprocess.

        Use this for:
        - Establishing persistent connections (for network backends like Milvus)
        - Loading shared resources (for Python-native backends)
        - Pre-allocating memory pools

        Default implementation does nothing. Override if needed.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up backend resources (called once after all benchmarks complete).

        Note: C++ backends handle cleanup internally via subprocess.

        Use this for:
        - Closing network connections (for backends like Milvus, Elasticsearch)
        - Deleting temporary files
        - Freeing shared resources

        Default implementation does nothing. Override if needed.
        """
        pass

    def _check_gpu_available(self) -> bool:
        """
        Check if GPU is available (same logic as run.py rmm_present()).

        Returns
        -------
        bool
            True if RMM/GPU is available, False otherwise
        """
        try:
            import rmm  # noqa: F401

            return True
        except ImportError:
            return False

    def _check_network_available(self) -> bool:
        """
        Check if network is available.

        Default returns False to ensure network backends implement their own check.
        Network backends MUST override with actual connectivity checks.

        Returns
        -------
        bool
            True if network is available, False otherwise
        """
        return False

    def _pre_flight_check(self) -> Optional[str]:
        """
        Run pre-flight checks before build/search operations.

        Returns
        -------
        Optional[str]
            Skip reason if checks fail (e.g., "no_gpu", "no_network"),
            None if all checks pass and operation should proceed.
        """
        if self.requires_gpu and not self._check_gpu_available():
            return "no_gpu"
        if self.requires_network and not self._check_network_available():
            return "no_network"
        return None

    @property
    def name(self) -> str:
        """
        User-defined name for this index configuration.

        Must be provided in the config dict. Matches 'name' field in runners.py config.

        Returns
        -------
        str
            User-defined index name

        Raises
        ------
        ValueError
            If 'name' is not provided in config
        """
        if "name" not in self.config:
            raise ValueError(
                "'name' is required in config (user-defined index label)"
            )
        return self.config["name"]

    @property
    @abstractmethod
    def algo(self) -> str:
        """
        Algorithm name (e.g., 'cuvs_ivf_flat', 'cuvs_cagra', 'hnswlib').

        Used for logging, result organization, and mapping to executables.

        Returns
        -------
        str
            Algorithm name

        Raises
        ------
        NotImplementedError
            This is an abstract property that must be implemented by subclasses
        """
        pass

    @property
    def requires_gpu(self) -> bool:
        """
        Whether this backend requires GPU acceleration.

        Reads from config["requires_gpu"], matching algorithms.yaml structure.

        Note
        ----
        If requires_gpu=True and no GPU is available (checked via RMM import),
        the backend will be skipped during pre-flight checks. This matches the
        legacy run.py behavior where GPU-requiring algorithms are not executed
        on CPU-only systems.

        Returns
        -------
        bool
            True if backend requires GPU, False otherwise (default)
        """
        return self.config.get("requires_gpu", False)

    @property
    def requires_network(self) -> bool:
        """
        Whether this backend requires network connectivity.

        Reads from config["requires_network"], similar to requires_gpu.

        Returns
        -------
        bool
            True if backend needs network (e.g., remote VDB), False otherwise (default)
        """
        return self.config.get("requires_network", False)
