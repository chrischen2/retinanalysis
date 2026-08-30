"""Composite (HyperNode) models — multi-node cascade graphs."""

from __future__ import annotations

import numpy as np

from ..nodes.base import HyperNode
from ..nodes.data import DataNode
from ..nodes.filter import ParamFilterNode
from ..nodes.nonlinearity import SigmoidNlNode
from ..nodes.operators import SumNode


class LnHyperNode(HyperNode):
    """Full linear-nonlinear cascade model.

    Contains: DataNode -> ParamFilterNode -> SigmoidNlNode

    Free parameters (9 total):
        Filter: numFilt, tauR, tauD, tauP, phi
        Nonlinearity: alpha, beta, gamma, epsilon
    """

    free_param_names = [
        "numFilt", "tauR", "tauD", "tauP", "phi",
        "alpha", "beta", "gamma", "epsilon",
    ]

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
        self.alpha: float | None = None
        self.beta: float | None = None
        self.gamma: float | None = None
        self.epsilon: float | None = None
        self.dt_stored: float | None = None
        super().__init__(free_params, other_params)

    def subnodes(self) -> dict:
        """Build the LN cascade sub-graph."""
        nodes = {}
        nodes["input"] = DataNode()
        nodes["filter"] = ParamFilterNode(
            np.array([
                self.numFilt or 0, self.tauR or 0, self.tauD or 0,
                self.tauP or 0, self.phi or 0,
            ])
        )
        nodes["nonlinearity"] = SigmoidNlNode(
            np.array([
                self.alpha or 0, self.beta or 0,
                self.gamma or 0, self.epsilon or 0,
            ])
        )
        if self.dt_stored is not None:
            nodes["filter"].dt_stored = self.dt_stored
        # Connect graph
        nodes["filter"].upstream.append(nodes["input"])
        nodes["nonlinearity"].upstream.append(nodes["filter"])
        return nodes

    def process_temp_params(
        self, params: dict, stim: np.ndarray, dt: float | None = None
    ) -> np.ndarray:
        if dt is None:
            raise ValueError("LnHyperNode requires dt")
        sn = self._subnodes_protected
        sn["input"].data = stim
        sn["filter"].dt_stored = dt
        sn["filter"].write_free_params(
            np.array([params["numFilt"], params["tauR"], params["tauD"],
                       params["tauP"], params["phi"]])
        )
        sn["nonlinearity"].write_free_params(
            np.array([params["alpha"], params["beta"], params["gamma"], params["epsilon"]])
        )
        return sn["nonlinearity"].process_upstream()

    def return_output(self, inputs: list | None = None) -> np.ndarray:
        if inputs is None or len(inputs) != 1:
            raise ValueError("LnHyperNode expects exactly one input")
        if self.dt_stored is None:
            raise ValueError("LnHyperNode.return_output requires dt_stored to be set")
        return self.process(inputs[0], self.dt_stored)


class TwoArmLnHyperNode(HyperNode):
    """Two-arm linear-nonlinear cascade model.

    Structure:
        Input -> Filter1 -> Sum -> Nonlinearity1 -> Output
        Input -> Filter2 -> Nonlinearity2 -> Sum

    Free parameters (18 total):
        Filter1: numFilt1, tauR1, tauD1, tauP1, phi1
        Filter2: numFilt2, tauR2, tauD2, tauP2, phi2
        NL1: alpha1, beta1, gamma1, epsilon1
        NL2: alpha2, beta2, gamma2, epsilon2
    """

    free_param_names = [
        "numFilt1", "tauR1", "tauD1", "tauP1", "phi1",
        "numFilt2", "tauR2", "tauD2", "tauP2", "phi2",
        "alpha1", "beta1", "gamma1", "epsilon1",
        "alpha2", "beta2", "gamma2", "epsilon2",
    ]

    def __init__(
        self,
        free_params: np.ndarray | dict | None = None,
        other_params: dict | None = None,
    ) -> None:
        self.numFilt1: float | None = None
        self.tauR1: float | None = None
        self.tauD1: float | None = None
        self.tauP1: float | None = None
        self.phi1: float | None = None
        self.numFilt2: float | None = None
        self.tauR2: float | None = None
        self.tauD2: float | None = None
        self.tauP2: float | None = None
        self.phi2: float | None = None
        self.alpha1: float | None = None
        self.beta1: float | None = None
        self.gamma1: float | None = None
        self.epsilon1: float | None = None
        self.alpha2: float | None = None
        self.beta2: float | None = None
        self.gamma2: float | None = None
        self.epsilon2: float | None = None
        self.dt_stored: float | None = None
        super().__init__(free_params, other_params)

    def subnodes(self) -> dict:
        """Build the two-arm LN cascade sub-graph."""
        nodes = {}
        nodes["input"] = DataNode()
        nodes["filter1"] = ParamFilterNode(
            np.array([
                self.numFilt1 or 0, self.tauR1 or 0, self.tauD1 or 0,
                self.tauP1 or 0, self.phi1 or 0,
            ])
        )
        nodes["filter2"] = ParamFilterNode(
            np.array([
                self.numFilt2 or 0, self.tauR2 or 0, self.tauD2 or 0,
                self.tauP2 or 0, self.phi2 or 0,
            ])
        )
        nodes["nonlinearity1"] = SigmoidNlNode(
            np.array([
                self.alpha1 or 0, self.beta1 or 0,
                self.gamma1 or 0, self.epsilon1 or 0,
            ])
        )
        nodes["nonlinearity2"] = SigmoidNlNode(
            np.array([
                self.alpha2 or 0, self.beta2 or 0,
                self.gamma2 or 0, self.epsilon2 or 0,
            ])
        )
        nodes["sum"] = SumNode()

        if self.dt_stored is not None:
            nodes["filter1"].dt_stored = self.dt_stored
            nodes["filter2"].dt_stored = self.dt_stored

        # Connect graph
        nodes["filter1"].upstream.append(nodes["input"])
        nodes["filter2"].upstream.append(nodes["input"])
        nodes["nonlinearity2"].upstream.append(nodes["filter2"])
        nodes["sum"].upstream.append(nodes["filter1"])
        nodes["sum"].upstream.append(nodes["nonlinearity2"])
        nodes["nonlinearity1"].upstream.append(nodes["sum"])
        return nodes

    def process_temp_params(
        self, params: dict, stim: np.ndarray, dt: float | None = None
    ) -> np.ndarray:
        if dt is None:
            raise ValueError("TwoArmLnHyperNode requires dt")
        sn = self._subnodes_protected
        sn["input"].data = stim
        sn["filter1"].dt_stored = dt
        sn["filter2"].dt_stored = dt
        sn["filter1"].write_free_params(
            np.array([params["numFilt1"], params["tauR1"], params["tauD1"],
                       params["tauP1"], params["phi1"]])
        )
        sn["filter2"].write_free_params(
            np.array([params["numFilt2"], params["tauR2"], params["tauD2"],
                       params["tauP2"], params["phi2"]])
        )
        sn["nonlinearity1"].write_free_params(
            np.array([params["alpha1"], params["beta1"], params["gamma1"], params["epsilon1"]])
        )
        sn["nonlinearity2"].write_free_params(
            np.array([params["alpha2"], params["beta2"], params["gamma2"], params["epsilon2"]])
        )
        return sn["nonlinearity1"].process_upstream()

    def return_output(self, inputs: list | None = None) -> np.ndarray:
        if inputs is None or len(inputs) != 1:
            raise ValueError("TwoArmLnHyperNode expects exactly one input")
        if self.dt_stored is None:
            raise ValueError("TwoArmLnHyperNode.return_output requires dt_stored to be set")
        return self.process(inputs[0], self.dt_stored)
