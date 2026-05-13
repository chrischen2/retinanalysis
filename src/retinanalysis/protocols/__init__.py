"""Per-protocol analysis modules.

Each Symphony protocol's higher-level analysis lives in its own module
here, so we can keep the demo notebooks lean (one notebook per protocol)
and reuse the same analysis across them.

A protocol module typically exposes:

* ``analyze(pipeline, **kwargs)`` — returns a results dict (PSTHs by
  condition, summary stats, ...).
* ``plot_psth_by_condition(pipeline, ...)`` — plots that align spikes
  with stim conditions.

Lower-level utilities (raster, PSTH kernel, cell-type verification,
mosaic/stim overlay) live in :mod:`retinanalysis.utils` so they can be
shared across protocols.
"""
