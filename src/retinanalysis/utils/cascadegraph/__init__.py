"""CascadeGraph - Computation graph framework for cascade models.

Vendored from https://github.com/chrischen2/cascadegraph (``python/cascadegraph``,
v0.1.0, upstream commit fb68536) so the analysis notebooks in this repository do
not depend on a sibling checkout being present and on ``sys.path``.

This is an unmodified copy. Fixes belong upstream and should be re-vendored
here rather than edited in place, so the two cannot drift.
"""

from cascadegraph.nodes.base import ModelNode, ParameterizedNode, HyperNode
from cascadegraph.nodes.data import DataNode
from cascadegraph.nodes.filter import FilterNode, ParamFilterNode
from cascadegraph.nodes.nonlinearity import PolyfitNlNode, SigmoidNlNode
from cascadegraph.nodes.operators import SumNode, NegativeNode
from cascadegraph.nodes.composite import LnHyperNode, TwoArmLnHyperNode
from cascadegraph.nodes.biophysical import RodLinearNode, RodBiophysNode

from cascadegraph.utils.filters import compute_filter, convolve_filter_with_stim
from cascadegraph.utils.nonlinearity import sample_nl
from cascadegraph.utils.metrics import compute_variance_explained
from cascadegraph.utils.signal import (
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
