"""Operator nodes — element-wise operations on inputs."""

from __future__ import annotations

import numpy as np

from cascadegraph.nodes.base import ModelNode


class SumNode(ModelNode):
    """Sums all upstream inputs element-wise."""

    def return_output(self, inputs: list | None = None) -> np.ndarray:
        if inputs is None or len(inputs) == 0:
            raise ValueError("SumNode requires at least one input")
        result = np.asarray(inputs[0], dtype=float)
        for inp in inputs[1:]:
            arr = np.asarray(inp, dtype=float)
            if arr.shape != result.shape:
                raise ValueError("All inputs to SumNode must have the same shape")
            result = result + arr
        return result


class NegativeNode(ModelNode):
    """Multiplies all inputs by -1, returning a list if multiple inputs."""

    def return_output(self, inputs: list | None = None) -> np.ndarray | list:
        if inputs is None or len(inputs) == 0:
            raise ValueError("NegativeNode requires at least one input")
        negated = [-np.asarray(x, dtype=float) for x in inputs]
        return negated if len(negated) > 1 else negated[0]
