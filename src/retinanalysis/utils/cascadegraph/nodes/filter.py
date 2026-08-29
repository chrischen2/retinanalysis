"""Filter nodes — temporal filtering via convolution."""

from __future__ import annotations

from typing import Any

import numpy as np

from cascadegraph.nodes.base import ModelNode, ParameterizedNode
from cascadegraph.utils.filters import convolve_filter_with_stim


class FilterNode(ModelNode):
    """Temporal filter described by a vector in the time domain."""

    def __init__(
        self,
        filter_vec: np.ndarray | None = None,
        has_anticausal_half: bool = False,
    ) -> None:
        super().__init__()
        self.filter = filter_vec
        self.has_anticausal_half = has_anticausal_half

    def process(self, stim: np.ndarray) -> np.ndarray:
        return convolve_filter_with_stim(self.filter, stim, self.has_anticausal_half)

    def return_output(self, inputs: list | None = None) -> np.ndarray:
        if inputs is None or len(inputs) != 1:
            raise ValueError("FilterNode expects exactly one input")
        return self.process(inputs[0])


class ParamFilterNode(ParameterizedNode):
    """Parameterized temporal filter.

    Free parameters: numFilt, tauR, tauD, tauP, phi
    """

    free_param_names = ["numFilt", "tauR", "tauD", "tauP", "phi"]

    def __init__(
        self,
        free_params: np.ndarray | dict | None = None,
        other_params: dict | None = None,
    ) -> None:
        self.numFilt: float | None = None
        self.tauR: float | None = None
        self.tauD: float | None = None
        self.tauP: float | None = None
        self.phi: float | None = None
        self.dt_stored: float | None = None
        super().__init__(free_params, other_params)

    def get_filter(self, num_points: int, dt: float) -> np.ndarray:
        """Compute the filter using stored parameters."""
        params = self.get_free_params(as_dict=True)
        return self.get_filter_with_params(params, num_points, dt)

    @staticmethod
    def get_filter_with_params(params: dict, num_points: int, dt: float) -> np.ndarray:
        """Compute filter from parameter dict."""
        t = np.arange(1, num_points + 1) * dt
        tauR = abs(params["tauR"])
        filt = (
            ((t / tauR) ** params["numFilt"] / (1 + (t / tauR) ** params["numFilt"]))
            * np.exp(-t / params["tauD"])
            * np.cos(2 * np.pi * t / params["tauP"] + 2 * np.pi * params["phi"] / 360)
        )
        max_abs = max(abs(filt.max()), abs(filt.min()))
        if max_abs > 0:
            filt = filt / max_abs
        return filt

    def process_temp_params(
        self, params: dict, stim: np.ndarray, dt: float | None = None
    ) -> np.ndarray:
        """Convolve parameterized filter with stimulus."""
        if dt is None:
            raise ValueError("ParamFilterNode requires dt")
        # stim is 2D (epochs x time) — process each row
        stim_2d = np.atleast_2d(stim)
        num_points = stim_2d.shape[1]
        filt = self.get_filter_with_params(params, num_points, dt)
        # FFT-based convolution per row
        result = np.real(np.fft.ifft(np.fft.fft(stim_2d, axis=1) * np.fft.fft(filt), axis=1))
        return result if stim.ndim == 2 else result.squeeze(axis=0)

    def return_output(self, inputs: list | None = None) -> np.ndarray:
        if inputs is None or len(inputs) != 1:
            raise ValueError("ParamFilterNode expects exactly one input")
        if self.dt_stored is None:
            raise ValueError("ParamFilterNode.return_output requires dt_stored to be set")
        return self.process(inputs[0], self.dt_stored)
