#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Backend registry for the cuvs-bench plugin system.

This module provides a central registry for discovering, registering, and
instantiating benchmark backends.
"""

from typing import Dict, Type, Optional
from pathlib import Path
import importlib
import importlib.metadata
import yaml

from .base import BenchmarkBackend

# Entry point group names for plugin discovery
_BACKENDS_GROUP = "cuvs_bench.backends"
_CONFIG_LOADERS_GROUP = "cuvs_bench.config_loaders"

_OPTIONAL_BACKEND_EXTRAS = {
    "elastic": ("elasticsearch", "pip install cuvs-bench[elastic]"),
    "opensearch": ("opensearchpy", "pip install cuvs-bench[opensearch]"),
}


def _optional_backend_install_hint(name: str) -> str:
    """Return an install hint for optional backends."""
    extra = _OPTIONAL_BACKEND_EXTRAS.get(name)
    if extra is None:
        return ""
    return f" Install with: {extra[1]}"


def _rewrite_optional_backend_import_error(
    name: str, error: ImportError
) -> ImportError | None:
    """Rewrite ImportError with an install hint for known optional backends."""
    extra = _OPTIONAL_BACKEND_EXTRAS.get(name)
    if extra is None:
        return None

    module_name, install_cmd = extra
    message = str(error).lower()
    if module_name.lower() in message:
        backend_name = name.capitalize()
        return ImportError(
            f"{backend_name} backend requires the '{name}' extra. "
            f"Install with: {install_cmd}"
        )
    return None


class BackendRegistry:
    """
    Central registry for all benchmark backends.

    The registry maintains a mapping of backend names to backend classes,
    and provides methods for registering, discovering, and instantiating backends.

    Supports:
    - Built-in backends (auto-registered at initialization)
    - User-defined plugins via YAML configuration
    - Dynamic loading from Python modules

    Examples
    --------
    >>> registry = BackendRegistry()
    >>> backend = registry.get_backend("cpp_gbench", config={"executable_path": "..."})
    >>> backend.build(dataset, params, index_path)
    """

    def __init__(self):
        """Initialize the registry with an empty backend mapping."""
        self._backends: Dict[str, Type[BenchmarkBackend]] = {}

    def register(
        self, name: str, backend_class: Type[BenchmarkBackend]
    ) -> None:
        """
        Register a backend class with the given name.

        Parameters
        ----------
        name : str
            Unique backend name (e.g., "cpp_gbench", "milvus")
        backend_class : Type[BenchmarkBackend]
            Backend class that inherits from BenchmarkBackend

        Raises
        ------
        TypeError
            If backend_class does not inherit from BenchmarkBackend
        ValueError
            If a backend with this name is already registered

        Examples
        --------
        >>> from .base import BenchmarkBackend
        >>> class MyBackend(BenchmarkBackend):
        ...     def build(self, ...): pass
        ...     def search(self, ...): pass
        ...     @property
        ...     def name(self) -> str: return "my_backend"
        >>> registry = BackendRegistry()
        >>> registry.register("my_backend", MyBackend)
        """
        if not issubclass(backend_class, BenchmarkBackend):
            raise TypeError(
                f"{backend_class.__name__} must inherit from BenchmarkBackend"
            )

        if name in self._backends:
            raise ValueError(
                f"Backend '{name}' is already registered. "
                f"Use unregister() first if you want to replace it."
            )

        self._backends[name] = backend_class
        print(
            f"[Registry] Registered backend: {name} ({backend_class.__name__})"
        )

    def unregister(self, name: str) -> None:
        """
        Unregister a backend by name.

        Parameters
        ----------
        name : str
            Backend name to unregister

        Raises
        ------
        KeyError
            If backend is not registered
        """
        if name not in self._backends:
            raise KeyError(f"Backend '{name}' is not registered")

        del self._backends[name]
        print(f"[Registry] Unregistered backend: {name}")

    def get_backend(self, name: str, config: Dict) -> BenchmarkBackend:
        """
        Get a backend instance by name.

        Parameters
        ----------
        name : str
            Backend name
        config : Dict
            Backend-specific configuration

        Returns
        -------
        BenchmarkBackend
            Instantiated backend

        Raises
        ------
        ValueError
            If backend is not registered

        Examples
        --------
        >>> registry = BackendRegistry()
        >>> backend = registry.get_backend("cpp_gbench", {
        ...     "executable_path": "/path/to/CUVS_IVF_FLAT_ANN_BENCH"
        ... })
        >>> backend.name
        'cuvs_ivf_flat'
        """
        if name not in self._backends:
            available = ", ".join(self._backends.keys())
            raise ValueError(
                f"Backend '{name}' not found. Available backends: {available or '(none)'}"
            )

        backend_class = self._backends[name]
        return backend_class(config)

    def is_registered(self, name: str) -> bool:
        """
        Check if a backend is registered.

        Parameters
        ----------
        name : str
            Backend name

        Returns
        -------
        bool
            True if registered, False otherwise
        """
        return name in self._backends

    def list_backends(self) -> Dict[str, Type[BenchmarkBackend]]:
        """
        List all registered backends.

        Returns
        -------
        Dict[str, Type[BenchmarkBackend]]
            Copy of the backend registry mapping

        Examples
        --------
        >>> registry = BackendRegistry()
        >>> backends = registry.list_backends()
        >>> for name, cls in backends.items():
        ...     print(f"{name}: {cls.__name__}")
        """
        return self._backends.copy()

    def load_from_module(
        self, module_path: str, class_name: str, backend_name: str
    ) -> None:
        """
        Dynamically load a backend from a Python module.

        This enables loading external plugins without modifying core code.

        Parameters
        ----------
        module_path : str
            Python module path (e.g., "custom_backends.qdrant")
        class_name : str
            Class name within the module (e.g., "QdrantBackend")
        backend_name : str
            Name to register the backend under (e.g., "qdrant")

        Raises
        ------
        ModuleNotFoundError
            If the module cannot be imported
        AttributeError
            If the class does not exist in the module
        TypeError
            If the class does not inherit from BenchmarkBackend

        Examples
        --------
        >>> registry = BackendRegistry()
        >>> registry.load_from_module(
        ...     module_path="custom_backends.qdrant",
        ...     class_name="QdrantBackend",
        ...     backend_name="qdrant"
        ... )
        [Registry] Registered backend: qdrant (QdrantBackend)
        """
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Cannot import module '{module_path}': {e}"
            )

        try:
            backend_class = getattr(module, class_name)
        except AttributeError:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_path}'"
            )

        self.register(backend_name, backend_class)

    def load_from_config(self, config_path: Path) -> None:
        """
        Load backends from a YAML configuration file.

        The YAML file should have the following structure:

        .. code-block:: yaml

            backends:
              - name: qdrant
                module: custom_backends.qdrant
                class: QdrantBackend
              - name: elasticsearch
                module: custom_backends.elasticsearch
                class: ElasticsearchBackend

        Parameters
        ----------
        config_path : Path
            Path to YAML configuration file

        Raises
        ------
        FileNotFoundError
            If config file does not exist
        yaml.YAMLError
            If config file is not valid YAML
        KeyError
            If required fields are missing from config

        Examples
        --------
        >>> registry = BackendRegistry()
        >>> registry.load_from_config(Path("config/custom_backends.yaml"))
        [Registry] Registered backend: qdrant (QdrantBackend)
        [Registry] Registered backend: elasticsearch (ElasticsearchBackend)
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        backends_config = config.get("backends", [])
        if not backends_config:
            print(f"[Registry] Warning: No backends found in {config_path}")
            return

        for backend_config in backends_config:
            try:
                name = backend_config["name"]
                module_path = backend_config["module"]
                class_name = backend_config["class"]
            except KeyError as e:
                print(
                    f"[Registry] Warning: Skipping backend due to missing field {e}: "
                    f"{backend_config}"
                )
                continue

            try:
                self.load_from_module(module_path, class_name, name)
            except (ModuleNotFoundError, AttributeError, TypeError) as e:
                print(
                    f"[Registry] Warning: Failed to load backend '{name}': {e}"
                )


# Global registry instance
_global_registry: Optional[BackendRegistry] = None


def get_registry() -> BackendRegistry:
    """
    Get the global backend registry instance (singleton pattern).

    Returns
    -------
    BackendRegistry
        The global registry

    Examples
    --------
    >>> registry = get_registry()
    >>> backend = registry.get_backend("cpp_gbench", config)
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = BackendRegistry()
    return _global_registry


def register_backend(name: str, backend_class: Type[BenchmarkBackend]) -> None:
    """
    Convenience function to register a backend with the global registry.

    Parameters
    ----------
    name : str
        Unique backend name
    backend_class : Type[BenchmarkBackend]
        Backend class

    Examples
    --------
    >>> from .base import BenchmarkBackend
    >>> class MyBackend(BenchmarkBackend):
    ...     def build(self, ...): pass
    ...     def search(self, ...): pass
    ...     @property
    ...     def name(self) -> str: return "my_backend"
    >>> register_backend("my_backend", MyBackend)
    [Registry] Registered backend: my_backend (MyBackend)
    """
    registry = get_registry()
    registry.register(name, backend_class)


def get_backend(name: str, config: Dict) -> BenchmarkBackend:
    """
    Convenience function to get a backend instance from the global registry.

    Parameters
    ----------
    name : str
        Backend name
    config : Dict
        Backend configuration

    Returns
    -------
    BenchmarkBackend
        Instantiated backend

    Examples
    --------
    >>> backend = get_backend("cpp_gbench", {"executable_path": "..."})
    >>> backend.build(dataset, params, index_path)
    """
    registry = get_registry()
    return registry.get_backend(name, config)


def _try_load_plugin(name: str) -> None:
    """
    Try to load backend and config loader from entry points for the given name.

    Plugins register themselves when their entry point is loaded.
    Raises ImportError with install instructions if the plugin requires
    an optional dependency that is not installed.
    """
    registry = get_registry()
    for group in (_BACKENDS_GROUP, _CONFIG_LOADERS_GROUP):
        try:
            eps = importlib.metadata.entry_points(group=group)
        except TypeError:
            eps = importlib.metadata.entry_points().get(group, [])
        if hasattr(eps, "select"):  # Python 3.10+
            eps = list(eps.select(name=name))
        else:
            eps = [e for e in eps if e.name == name]
        for ep in eps:
            if group == _BACKENDS_GROUP and name in registry._backends:
                break
            if (
                group == _CONFIG_LOADERS_GROUP
                and name in _CONFIG_LOADER_REGISTRY
            ):
                break
            try:
                ep.load()()
            except ImportError as e:
                rewritten = _rewrite_optional_backend_import_error(name, e)
                if rewritten is not None:
                    raise rewritten from e
                raise
            if (
                name in registry._backends
                and name in _CONFIG_LOADER_REGISTRY
            ):
                return


def get_backend_class(name: str) -> Type[BenchmarkBackend]:
    """
    Get the backend class (not instance) from the global registry.

    If the backend is not registered, attempts to load it from entry points
    (e.g., optional plugins like elastic).

    Parameters
    ----------
    name : str
        Backend name

    Returns
    -------
    Type[BenchmarkBackend]
        Backend class
    """
    registry = get_registry()
    if name not in registry._backends:
        _try_load_plugin(name)
    if name not in registry._backends:
        available = ", ".join(registry._backends.keys())
        hint = _optional_backend_install_hint(name)
        raise ValueError(
            f"Backend '{name}' not found. Available backends: {available or '(none)'}.{hint}"
        )
    return registry._backends[name]


def list_backends() -> Dict[str, Type[BenchmarkBackend]]:
    """Return all registered backends."""
    registry = get_registry()
    return registry.list_backends()


# ============================================================================
# Config Loader Registry
# ============================================================================

# Simple module-level registry for config loaders
# (Config loaders are simpler than backends, don't need full class-based registry)
# Note: Module level variable that persists for the lifetime of the Python
# process.
_CONFIG_LOADER_REGISTRY: Dict[str, Type] = {}


def register_config_loader(name: str, loader_class: Type) -> None:
    """
    Register a config loader class.

    Parameters
    ----------
    name : str
        Backend name (e.g., 'cpp_gbench', 'milvus')
    loader_class : Type
        Config loader class to register

    Examples
    --------
    >>> register_config_loader("cpp_gbench", CppGBenchConfigLoader)
    """
    _CONFIG_LOADER_REGISTRY[name] = loader_class
    print(
        f"[Registry] Registered config loader: {name} ({loader_class.__name__})"
    )


def get_config_loader(name: str) -> Type:
    """
    Get a registered config loader class by name.

    If the config loader is not registered, attempts to load it from entry points
    (e.g., optional plugins like elastic).

    Parameters
    ----------
    name : str
        Backend name

    Returns
    -------
    Type
        Config loader class

    Raises
    ------
    ValueError
        If config loader is not registered
    """
    if name not in _CONFIG_LOADER_REGISTRY:
        _try_load_plugin(name)
    if name not in _CONFIG_LOADER_REGISTRY:
        available = ", ".join(_CONFIG_LOADER_REGISTRY.keys()) or "none"
        hint = _optional_backend_install_hint(name)
        raise ValueError(
            f"Unknown config loader for backend: '{name}'. Available: {available}.{hint}"
        )
    return _CONFIG_LOADER_REGISTRY[name]


def list_config_loaders() -> Dict[str, Type]:
    """Return all registered config loaders."""
    return dict(_CONFIG_LOADER_REGISTRY)


def unregister_config_loader(name: str) -> None:
    """
    Unregister a config loader by name (primarily for testing).

    Parameters
    ----------
    name : str
        Backend name to unregister
    """
    if name in _CONFIG_LOADER_REGISTRY:
        del _CONFIG_LOADER_REGISTRY[name]
