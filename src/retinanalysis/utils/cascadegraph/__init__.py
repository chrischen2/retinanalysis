"""CascadeGraph - Computation graph framework for cascade models.

Vendored from https://github.com/chrischen2/cascadegraph (``python/cascadegraph``,
v0.1.0, upstream commit fb68536) so the analysis notebooks in this repository do
not depend on a sibling checkout being present and on ``sys.path``.

This is an unmodified copy. Fixes belong upstream and should be re-vendored
here rather than edited in place, so the two cannot drift.
"""

from .nodes.base import ModelNode, ParameterizedNode, HyperNode
from .nodes.data import DataNode
from .nodes.filter import FilterNode, ParamFilterNode
from .nodes.nonlinearity import PolyfitNlNode, SigmoidNlNode
from .nodes.operators import SumNode, NegativeNode
from .nodes.composite import LnHyperNode, TwoArmLnHyperNode
from .nodes.biophysical import RodLinearNode, RodBiophysNode

from .utils.filters import compute_filter, convolve_filter_with_stim
from .utils.nonlinearity import sample_nl
from .utils.metrics import compute_variance_explained
from .utils.signal import (
    apply_frequency_cutoff,
    apply_frequency_cutoff_to_fft,
    baseline_subtract,
)

__all__ = [
    # Core nodes
    "ModelNode",
    "ParameterizedNode",
    "HyperNode",
    "DataNode",
    "FilterNode",
    "ParamFilterNode",
    "PolyfitNlNode",
    "SigmoidNlNode",
    "SumNode",
    "NegativeNode",
    "LnHyperNode",
    "TwoArmLnHyperNode",
    "RodLinearNode",
    "RodBiophysNode",
    # Utilities
    "compute_filter",
    "convolve_filter_with_stim",
    "sample_nl",
    "compute_variance_explained",
    "apply_frequency_cutoff",
    "apply_frequency_cutoff_to_fft",
    "baseline_subtract",
]
