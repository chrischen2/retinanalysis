"""DataNode — stores and returns data."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..nodes.base import ModelNode


class DataNode(ModelNode):
    """A terminal node that stores and returns data when queried."""

    def __init__(self, data: np.ndarray | None = None) -> None:
        super().__init__()
        self.data = data

    def return_output(self, inputs: list | None = None) -> np.ndarray:
        if len(self.upstream) > 0:
            raise ValueError("DataNode is a terminal node and should not have parents")
        return self.data
