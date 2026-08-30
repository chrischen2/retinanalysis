"""Nonlinearity nodes — static nonlinear transformations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from ..nodes.base import ModelNode, ParameterizedNode


class PolyfitNlNode(ModelNode):
    """Nonlinearity described by polynomial evaluation.

    Uses centered/scaled polynomial (like MATLAB's polyval with mu).
    """

    def __init__(
        self,
        coeff: np.ndarray | None = None,
        mu: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.coeff = coeff
        self.mu = mu

    @property
    def degree(self) -> int | None:
        if self.coeff is None:
            return None
        return len(self.coeff) - 1

    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Evaluate polynomial with centering/scaling."""
        if self.mu is not None:
            x_scaled = (input_data - self.mu[0]) / self.mu[1]
        else:
            x_scaled = input_data
        return np.polyval(self.coeff, x_scaled)

    def fit_to_sample(
        self, xarray: np.ndarray, yarray: np.ndarray, degree: int
    ) -> None:
        """Fit polynomial to sampled data with centering/scaling."""
        # Center and scale x for numerical stability (like MATLAB polyfit with mu)
        mu_center = np.mean(xarray)
        mu_scale = np.std(xarray)
        if mu_scale == 0:
            mu_scale = 1.0
        self.mu = np.array([mu_center, mu_scale])
        x_scaled = (xarray - mu_center) / mu_scale
        self.coeff = np.polyfit(x_scaled, yarray, degree)

    def return_output(self, inputs: list | None = None) -> np.ndarray:
        if inputs is None or len(inputs) != 1:
            raise ValueError("PolyfitNlNode expects exactly one input")
        return self.process(inputs[0])


class SigmoidNlNode(ParameterizedNode):
    """Nonlinearity described by cumulative normal density function.

    Formula: alpha * normcdf(beta * x + gamma) + epsilon

    Free parameters: alpha, beta, gamma, epsilon
    """

    free_param_names = ["alpha", "beta", "gamma", "epsilon"]

    def __init__(
        self,
        free_params: np.ndarray | dict | None = None,
        other_params: dict | None = None,
    ) -> None:
        self.alpha: float | None = None
        self.beta: float | None = None
        self.gamma: float | None = None
        self.epsilon: float | None = None
        super().__init__(free_params, other_params)

    @staticmethod
    def _sigmoid(params_vec: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate sigmoid: alpha * normcdf(beta * x + gamma) + epsilon."""
        alpha, beta, gamma, epsilon = params_vec[0], params_vec[1], params_vec[2], params_vec[3]
        return alpha * norm.cdf(beta * x + gamma) + epsilon

    def process_temp_params(
        self, params: dict | np.ndarray, input_data: np.ndarray, dt: float | None = None
    ) -> np.ndarray:
        if isinstance(params, dict):
            pvec = np.array([params["alpha"], params["beta"], params["gamma"], params["epsilon"]])
        else:
            pvec = np.asarray(params).ravel()
        return self._sigmoid(pvec, input_data)

    def fit_to_sample(
        self,
        xarray: np.ndarray,
        yarray: np.ndarray,
        params0: np.ndarray | None = None,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
        optim_iters: int = 5,
    ) -> np.ndarray:
        """Fit sigmoid parameters to sampled input-output relationship."""
        if params0 is None:
            params0 = np.array([2 * np.max(yarray), 0.1, -1.0, -1.0])
        if lower_bounds is None:
            lower_bounds = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        if upper_bounds is None:
            upper_bounds = np.array([np.inf, np.inf, np.inf, np.max(yarray)])

        current_params = np.asarray(params0).copy()
        for _ in range(optim_iters):
            result = least_squares(
                lambda p: self._sigmoid(p, xarray) - yarray,
                current_params,
                bounds=(lower_bounds, upper_bounds),
                max_nfev=2400,
            )
            current_params = result.x

        self.write_free_params(current_params)
        return current_params

    def return_output(self, inputs: list | None = None) -> np.ndarray:
        if inputs is None or len(inputs) != 1:
            raise ValueError("SigmoidNlNode expects exactly one input")
        return self.process(inputs[0])
