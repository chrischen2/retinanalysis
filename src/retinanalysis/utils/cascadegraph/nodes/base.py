"""Base classes for the CascadeGraph computation graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.optimize import minimize


class ModelNode(ABC):
    """Base class for all nodes in a directed computation graph.

    A ModelNode graph should be acyclic. During graph traversal, data are
    passed from upstream to downstream nodes in a list, where each entry
    contains input from one parent.

    Subclasses must implement ``return_output``.
    """

    def __init__(self) -> None:
        self.upstream: list[ModelNode] = []

    def process_upstream(self) -> Any:
        """Initiate graph traversal from this node."""
        return self._process_parents()

    def _process_parents(self) -> Any:
        """Recursively traverse the computation graph."""
        if len(self.upstream) == 0:
            return self.return_output()
        inputs = []
        for parent in self.upstream:
            result = parent._process_parents()
            # Unpack single-element lists
            if isinstance(result, list) and len(result) == 1:
                result = result[0]
            inputs.append(result)
        return self.return_output(inputs)

    @abstractmethod
    def return_output(self, inputs: list | None = None) -> Any:
        """Return this node's output given upstream inputs."""
        ...


class ParameterizedNode(ModelNode, ABC):
    """A ModelNode with optimizable free parameters.

    Subclasses must:
    - Define ``free_param_names`` as a class-level list of parameter names.
    - Store each free parameter as an instance attribute.
    - Implement ``process_temp_params(params, input_data, dt=None)``.
    """

    free_param_names: list[str] = []

    def __init__(
        self,
        free_params: np.ndarray | dict | None = None,
        other_params: dict | None = None,
    ) -> None:
        super().__init__()
        if free_params is not None:
            self.write_free_params(free_params)
        if other_params is not None:
            for name, value in other_params.items():
                setattr(self, name, value)

    def process(self, input_data: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Process input using stored parameters."""
        params = self.get_free_params(as_dict=True)
        for name, val in params.items():
            if val is None:
                raise ValueError(f"Parameter '{name}' is not set")
        if dt is not None:
            return self.process_temp_params(params, input_data, dt)
        return self.process_temp_params(params, input_data)

    def optimize_params(
        self,
        params0: np.ndarray,
        input_data: np.ndarray,
        target: np.ndarray,
        dt: float | None = None,
        options: dict | None = None,
    ) -> np.ndarray:
        """Optimize free parameters to minimize MSE via Nelder-Mead."""

        def objective(params_vec: np.ndarray) -> float:
            pdict = self.param_vec_to_dict(params_vec)
            if dt is not None:
                prediction = self.process_temp_params(pdict, input_data, dt)
            else:
                prediction = self.process_temp_params(pdict, input_data)
            return float(np.sum((target - prediction) ** 2))

        minimize_opts = options or {}
        result = minimize(objective, params0, method="Nelder-Mead", options=minimize_opts)
        self.write_free_params(result.x)
        return result.x

    def write_free_params(self, params: np.ndarray | dict) -> None:
        """Set free parameters from a vector or dict."""
        if isinstance(params, dict):
            params = self.param_dict_to_vec(params)
        params = np.asarray(params).ravel()
        if len(params) != len(self.free_param_names):
            raise ValueError(
                f"Expected {len(self.free_param_names)} parameters, got {len(params)}"
            )
        for name, val in zip(self.free_param_names, params):
            setattr(self, name, float(val))

    def get_free_params(self, as_dict: bool = False) -> np.ndarray | dict:
        """Retrieve free parameters as a vector or dict."""
        values = [getattr(self, name, None) for name in self.free_param_names]
        if as_dict:
            return dict(zip(self.free_param_names, values))
        return np.array([v if v is not None else np.nan for v in values])

    def param_vec_to_dict(self, params: np.ndarray) -> dict:
        """Convert parameter vector to dict."""
        params = np.asarray(params).ravel()
        if len(params) != len(self.free_param_names):
            raise ValueError(
                f"Expected {len(self.free_param_names)} parameters, got {len(params)}"
            )
        return dict(zip(self.free_param_names, params))

    def param_dict_to_vec(self, params: dict) -> np.ndarray:
        """Convert parameter dict to vector."""
        return np.array([params[name] for name in self.free_param_names])

    @abstractmethod
    def process_temp_params(
        self, params: dict, input_data: np.ndarray, dt: float | None = None
    ) -> np.ndarray:
        """Process input using the given parameters (not stored ones)."""
        ...

    def return_output(self, inputs: list | None = None) -> Any:
        """Default return_output for parameterized nodes — subclasses may override."""
        if inputs is not None and len(inputs) == 1:
            return self.process(inputs[0])
        raise NotImplementedError("Subclass must override return_output")


class HyperNode(ParameterizedNode, ABC):
    """A ParameterizedNode that contains a sub-graph of other nodes.

    Subclasses must implement ``subnodes()`` which builds and returns a dict
    of the contained nodes.
    """

    def __init__(
        self,
        free_params: np.ndarray | dict | None = None,
        other_params: dict | None = None,
    ) -> None:
        super().__init__(free_params, other_params)
        self._subnodes_protected = self.subnodes()

    @abstractmethod
    def subnodes(self) -> dict:
        """Build and return the sub-node graph as a dict."""
        ...
