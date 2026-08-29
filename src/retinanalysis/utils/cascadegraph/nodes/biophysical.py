"""Biophysical model nodes — rod photoreceptor models."""

from __future__ import annotations

import numpy as np

from cascadegraph.nodes.base import ParameterizedNode


class RodLinearNode(ParameterizedNode):
    """Linear model of rod photoreceptor.

    Free parameters: scFact, tauR, tauD
    Fixed parameters: darkCurrent
    """

    free_param_names = ["scFact", "tauR", "tauD"]

    def __init__(
        self,
        free_params: np.ndarray | dict | None = None,
        other_params: dict | None = None,
    ) -> None:
        self.scFact: float | None = None
        self.tauR: float | None = None
        self.tauD: float | None = None
        self.darkCurrent: float | None = None
        self.dt_stored: float | None = None
        super().__init__(free_params, other_params)

    def process_temp_params(
        self, params: dict, stim: np.ndarray, dt: float | None = None
    ) -> np.ndarray:
        if dt is None:
            raise ValueError("RodLinearNode requires dt")
        stim_2d = np.atleast_2d(stim)
        num_points = stim_2d.shape[1]
        t = np.arange(1, num_points + 1) * dt
        filt = (
            params["scFact"]
            * ((t / params["tauR"]) ** 3 / (1 + (t / params["tauR"]) ** 3))
            * np.exp(-t / params["tauD"])
        )
        result = np.real(
            np.fft.ifft(np.fft.fft(stim_2d, axis=1) * np.fft.fft(filt), axis=1)
        ) - self.darkCurrent
        return result if stim.ndim == 2 else result.squeeze(axis=0)

    def return_output(self, inputs: list | None = None) -> np.ndarray:
        if inputs is None or len(inputs) != 1:
            raise ValueError("RodLinearNode expects exactly one input")
        if self.dt_stored is None:
            raise ValueError("RodLinearNode.return_output requires dt_stored to be set")
        return self.process(inputs[0], self.dt_stored)


class RodBiophysNode(ParameterizedNode):
    """Biophysical model of rod photoreceptor.

    Free parameters: beta, hillaffinity, sigma, gamma, eta
    Fixed parameters: betaSlow, hillcoef, darkCurrent
    Constants: cdark=0.5, cgmphill=3, cgmp2cur=10e-3
    """

    free_param_names = ["beta", "hillaffinity", "sigma", "gamma", "eta"]

    # Class-level constants
    CDARK = 0.5
    CGMPHILL = 3
    CGMP2CUR = 10e-3

    def __init__(
        self,
        free_params: np.ndarray | dict | None = None,
        other_params: dict | None = None,
    ) -> None:
        self.beta: float | None = None
        self.hillaffinity: float | None = None
        self.sigma: float | None = None
        self.gamma: float | None = None
        self.eta: float | None = None
        self.betaSlow: float | None = None
        self.hillcoef: float | None = None
        self.darkCurrent: float | None = None
        self.dt_stored: float | None = None
        super().__init__(free_params, other_params)

    def process_temp_params(
        self, params: dict, stim: np.ndarray, dt: float | None = None
    ) -> np.ndarray:
        if dt is None:
            raise ValueError("RodBiophysNode requires dt")
        if not isinstance(params, dict):
            params = self.param_vec_to_dict(params)

        phi = params["sigma"]
        gdark = (2 * self.darkCurrent / self.CGMP2CUR) ** (1 / self.CGMPHILL)
        cur2ca = params["beta"] * self.CDARK / self.darkCurrent
        smax = (
            params["eta"] / phi * gdark
            * (1 + (self.CDARK / params["hillaffinity"]) ** self.hillcoef)
        )

        stim = np.asarray(stim).ravel()
        num_pts = len(stim)

        r = np.zeros(num_pts)
        p = np.zeros(num_pts)
        g = np.zeros(num_pts)
        c = np.zeros(num_pts)
        s = np.zeros(num_pts)
        cslow = np.zeros(num_pts)

        r[0] = 0.0
        p[0] = params["eta"] / phi
        g[0] = gdark
        c[0] = self.CDARK
        s[0] = gdark * params["eta"] / phi
        cslow[0] = self.CDARK

        for i in range(1, num_pts):
            r[i] = r[i - 1] + dt * (-params["sigma"] * r[i - 1]) + params["gamma"] * stim[i - 1]
            p[i] = p[i - 1] + dt * (r[i - 1] + params["eta"] - phi * p[i - 1])
            g[i] = g[i - 1] + dt * (s[i - 1] - p[i - 1] * g[i - 1])
            I_val = self.CGMP2CUR * g[i - 1] ** self.CGMPHILL / (1 + cslow[i - 1] / self.CDARK)
            c[i] = c[i - 1] + dt * (cur2ca * I_val - params["beta"] * c[i - 1])
            s[i] = smax / (1 + (c[i] / params["hillaffinity"]) ** self.hillcoef)
            cslow[i] = cslow[i - 1] - dt * self.betaSlow * (cslow[i - 1] - c[i - 1])

        prediction = -self.CGMP2CUR * g ** self.CGMPHILL / (1 + cslow / self.CDARK)
        return prediction

    def return_output(self, inputs: list | None = None) -> np.ndarray:
        if inputs is None or len(inputs) != 1:
            raise ValueError("RodBiophysNode expects exactly one input")
        if self.dt_stored is None:
            raise ValueError("RodBiophysNode.return_output requires dt_stored to be set")
        return self.process(inputs[0], self.dt_stored)
