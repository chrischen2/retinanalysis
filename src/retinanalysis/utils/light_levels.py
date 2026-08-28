"""Measured maximum light levels for normalized Stage display stimuli.

The values here are calibrated R*/s at a spatial-stimulus intensity of ``1``
for the OLED/LightCrafter display.  Fixed filters (the ``EL`` names) and
the motorized filter wheel are separate parts of a setting.  Measurements are
returned verbatim when available; otherwise only the wheel component is
inferred, using its nominal ``10**OD`` attenuation for the same fixed-filter
stack.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Final, Mapping


LightSetting = tuple[tuple[str, ...], float]


# Measured maxima supplied for rigs E and G.  Filter order is intentionally
# irrelevant.  The duplicate G measurement for EL06 + EL2 + FW1 (7700 and
# 7600) is represented by the later supplied value, 7600.
VISUAL_STIMULUS_MAX: Final[Mapping[str, Mapping[LightSetting, float]]] = {
    "E": {
        ((), 0.0): 30_000_000.0,
        (("EL3",), 0.0): 30_000.0,
        (("EL3",), 0.5): 10_000.0,
        (("EL3",), 1.0): 3_000.0,
        (("EL3",), 3.0): 60.0,
        (("EL3",), 4.0): 6.0,
    },
    "G": {
        ((), 0.0): 30_250_000.0,
        (("EL03", "EL2"), 0.0): 62_000.0,
        (("EL06", "EL2"), 0.0): 77_000.0,
        (("EL2",), 0.5): 100_000.0,
        (("EL03", "EL2"), 0.5): 19_000.0,
        (("EL03", "EL2"), 1.0): 6_200.0,
        (("EL06", "EL2"), 0.5): 24_000.0,
        (("EL06", "EL2"), 1.0): 7_600.0,
    },
}

_RIG_ALIASES = {
    "E": "E",
    "RIG E": "E",
    "FRED_DATA": "E",
    "G": "G",
    "RIG G": "G",
    "CHRIS_DATA": "G",
}
_EXPERIMENT_RIG = re.compile(
    r"^\d{4}-?\d{2}-?\d{2}[_-]?([A-Za-z])(?:[_-].*)?$"
)
_FILTER_SPLIT = re.compile(r"\s*(?:\+|,|_)\s*|\s+")
_FILTER_NAME = re.compile(r"^EL(\d+(?:\.\d+)?)$", re.IGNORECASE)
_FW_NAME = re.compile(r"^(?:FW)?(\d+(?:\.\d+)?)$", re.IGNORECASE)


def _normalize_rig(rig: Any) -> str:
    text = str(rig).strip()
    alias = _RIG_ALIASES.get(text.upper())
    if alias is not None:
        return alias
    match = _EXPERIMENT_RIG.match(text)
    if match and match.group(1).upper() in VISUAL_STIMULUS_MAX:
        return match.group(1).upper()
    choices = "E, G, fred_data, chris_data, or an E/G experiment name"
    raise ValueError(f"Unknown visual-stimulus rig {rig!r}; expected {choices}")


def _normalize_filter(name: Any) -> str:
    match = _FILTER_NAME.fullmatch(str(name).strip())
    if match is None:
        raise ValueError(f"Unknown fixed-filter name {name!r}; expected an EL name")
    token = match.group(1)
    # EL3 and EL03 are different installed filters in these measurements.
    return f"EL{token.upper()}"


def _normalize_setting(ndfs: Any) -> tuple[tuple[str, ...], float | None]:
    if ndfs is None:
        return (), None
    if isinstance(ndfs, str):
        text = ndfs.strip()
        if not text or text.lower() in {"none", "no ndfs", "no ndf"}:
            return (), None
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid fixed-filter JSON {ndfs!r}") from exc
            if not isinstance(decoded, list):
                raise ValueError("Fixed-filter JSON must contain a list")
            values = decoded
        else:
            values = [value for value in _FILTER_SPLIT.split(text) if value]
    else:
        values = list(ndfs)
    fixed = []
    embedded_wheels = []
    for value in values:
        text = str(value).strip()
        if text.upper().startswith("FW"):
            embedded_wheels.append(_normalize_wheel(text))
        else:
            fixed.append(_normalize_filter(text))
    if len(embedded_wheels) > 1:
        raise ValueError("Only one filter-wheel value may be supplied")
    return tuple(sorted(fixed)), (embedded_wheels[0] if embedded_wheels else None)


def _normalize_wheel(value: Any) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0
    match = _FW_NAME.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(
            f"Invalid filter-wheel value {value!r}; expected an OD such as 0.5 or FW1"
        )
    wheel = float(match.group(1))
    if not math.isfinite(wheel) or wheel < 0:
        raise ValueError("Filter-wheel OD must be finite and non-negative")
    return wheel


def visual_stimulus_max(
    rig: Any,
    ndfs: Any = None,
    filter_wheel_ndf: Any = None,
    *,
    infer_filter_wheel: bool = True,
) -> float:
    """Return maximum OLED/LightCrafter R*/s at normalized intensity 1.

    Parameters
    ----------
    rig
        ``"E"``/``"fred_data"``, ``"G"``/``"chris_data"``, or an
        experiment name such as ``"2026-08-10_G"``.
    ndfs
        Fixed EL filter(s), supplied as a sequence or text such as
        ``"EL06 + EL2"``.  This may include the wheel, for example
        ``"EL3+FW0.5"``.  Filter order does not affect the lookup.
    filter_wheel_ndf
        Numeric wheel optical density, or text such as ``"FW0.5"``.
    infer_filter_wheel
        If true (default), infer an unmeasured wheel setting from the
        lowest-wheel measurement for the same EL stack.  Exact measurements
        always take precedence.

    Raises
    ------
    KeyError
        If the rig/fixed-filter combination has no measured basis, or if the
        exact wheel setting is absent and inference is disabled.
    """
    rig_key = _normalize_rig(rig)
    filters, embedded_wheel = _normalize_setting(ndfs)
    # A separately supplied numeric value is the protected FilterWheel hardware
    # reading. Any FW token embedded in the manually threaded fixed-NDF string
    # is descriptive stale text and must never overrule that direct reading.
    wheel = (_normalize_wheel(filter_wheel_ndf)
             if filter_wheel_ndf is not None else embedded_wheel)
    measurements = VISUAL_STIMULUS_MAX[rig_key]
    exact = measurements.get((filters, wheel))
    if exact is not None:
        return exact

    candidates = sorted(
        (measured_wheel, maximum)
        for (measured_filters, measured_wheel), maximum in measurements.items()
        if measured_filters == filters
    )
    setting = " + ".join(filters) if filters else "no fixed filters"
    if not candidates:
        raise KeyError(f"No measured visual-stimulus maximum for rig {rig_key}, {setting}")
    if not infer_filter_wheel:
        raise KeyError(
            f"No exact visual-stimulus maximum for rig {rig_key}, {setting}, FW{wheel:g}"
        )

    basis_wheel, basis_maximum = candidates[0]
    return basis_maximum / 10.0 ** (wheel - basis_wheel)


def _read_epoch_filter_wheel_parameters(block_ids):
    """Fetch epoch IDs and parameters for the requested parent blocks."""
    from retinanalysis.config import schema

    epochs = schema.Epoch() & [
        {"parent_id": int(block_id)} for block_id in block_ids
    ]
    if not len(epochs):
        return []
    return epochs.fetch("id", "parent_id", "parameters", as_dict=True)


def filter_wheel_ndf_from_epoch_parameters(parameters, *, context="epochs"):
    """Return one protected numeric FilterWheel value across all epochs.

    Missing values return NaN. Invalid or conflicting recorded values raise so
    callers never silently select the first epoch. Embedded ``FW`` labels in
    other parameters are deliberately ignored.
    """
    import numpy as np
    import pandas as pd

    values = []
    invalid = 0
    for item in parameters:
        item = item if isinstance(item, Mapping) else {}
        if "NDF" not in item or item["NDF"] is None:
            continue
        value = pd.to_numeric(item["NDF"], errors="coerce")
        if pd.isna(value):
            invalid += 1
        else:
            values.append(float(value))
    if invalid:
        raise ValueError(f"{context} contain {invalid} invalid FilterWheel reading(s)")
    unique = sorted(set(values))
    if len(unique) > 1:
        joined = ", ".join(f"{value:g}" for value in unique)
        raise ValueError(f"{context} have conflicting FilterWheel metadata: {joined}")
    return unique[0] if unique else np.nan


def _block_filter_wheel_table(block_ids, epoch_rows=None, *, on_conflict="raise"):
    """Consolidate protected per-epoch FilterWheel metadata by parent block."""
    import pandas as pd

    if on_conflict not in {"raise", "report"}:
        raise ValueError("on_conflict must be 'raise' or 'report'")

    ids = [int(block_id) for block_id in block_ids]
    rows = (_read_epoch_filter_wheel_parameters(ids)
            if epoch_rows is None else list(epoch_rows))
    grouped = {block_id: [] for block_id in ids}
    for row in rows:
        block_id = int(row["parent_id"])
        if block_id in grouped:
            grouped[block_id].append(row)

    columns = [
        "block_id", "filter_wheel_ndf", "filter_wheel_status", "n_epochs",
        "n_filter_wheel_readings",
    ]
    result = []
    for block_id in ids:
        block_epochs = grouped[block_id]
        values = []
        n_recorded = 0
        n_invalid = 0
        for epoch in block_epochs:
            parameters = epoch.get("parameters")
            parameters = parameters if isinstance(parameters, Mapping) else {}
            if "NDF" not in parameters or parameters["NDF"] is None:
                continue
            n_recorded += 1
            value = pd.to_numeric(parameters["NDF"], errors="coerce")
            if pd.isna(value):
                n_invalid += 1
            else:
                values.append(float(value))
        unique = sorted(set(values))
        if len(unique) > 1:
            joined = ", ".join(f"{value:g}" for value in unique)
            if on_conflict == "raise":
                raise ValueError(
                    f"Epoch block {block_id} has conflicting FilterWheel metadata: {joined}"
                )
            status = f"conflict: {joined}"
        elif n_invalid:
            status = "invalid"
        elif not block_epochs:
            status = "no epochs"
        elif not unique:
            status = "not recorded"
        elif n_recorded < len(block_epochs):
            status = "partially recorded"
        else:
            status = "recorded"
        result.append({
            "block_id": block_id,
            "filter_wheel_ndf": unique[0] if len(unique) == 1 else float("nan"),
            "filter_wheel_status": status,
            "n_epochs": len(block_epochs),
            "n_filter_wheel_readings": len(values),
        })
    return pd.DataFrame(result, columns=columns)


def _selected_block_filter_wheel_table(wanted, epoch_rows=None):
    """Validate condition-specific wheel values against protected epoch metadata.

    ``wanted`` contains ``block_id`` and ``filter_wheel_ndf``. A block may occur
    more than once when its hardware setting changed between epochs. Each
    requested value must occur in at least one protected ``parameters['NDF']``
    reading; embedded Stage ``FW`` text never enters this check.
    """
    import numpy as np
    import pandas as pd

    ids = wanted["block_id"].astype(int).drop_duplicates().tolist()
    rows = (_read_epoch_filter_wheel_parameters(ids)
            if epoch_rows is None else list(epoch_rows))
    grouped = {block_id: [] for block_id in ids}
    for row in rows:
        block_id = int(row["parent_id"])
        if block_id not in grouped:
            continue
        parameters = row.get("parameters")
        parameters = parameters if isinstance(parameters, Mapping) else {}
        value = pd.to_numeric(parameters.get("NDF"), errors="coerce")
        if pd.notna(value):
            grouped[block_id].append(float(value))

    result = []
    for requested in wanted.itertuples(index=False):
        block_id = int(requested.block_id)
        selected = float(requested.filter_wheel_ndf)
        values = grouped[block_id]
        if not np.isfinite(selected):
            if values:
                available = ", ".join(f"{value:g}" for value in sorted(set(values)))
                raise ValueError(
                    f"Epoch block {block_id} requested a missing FilterWheel value "
                    f"but protected readings exist: {available}"
                )
            result.append({
                "block_id": block_id,
                "filter_wheel_ndf": float("nan"),
                "filter_wheel_status": "not recorded",
                "n_epochs": 0,
                "n_filter_wheel_readings": 0,
            })
            continue
        matches = sum(np.isclose(value, selected) for value in values)
        if not matches:
            available = ", ".join(f"{value:g}" for value in sorted(set(values)))
            raise ValueError(
                f"Epoch block {block_id} has no protected FilterWheel reading "
                f"at {selected:g}; recorded values: {available or 'none'}"
            )
        result.append({
            "block_id": block_id,
            "filter_wheel_ndf": selected,
            "filter_wheel_status": (
                "recorded" if len(set(values)) == 1 else "selected from mixed block"),
            "n_epochs": len(values),
            "n_filter_wheel_readings": len(values),
        })
    return pd.DataFrame(result)


def _epoch_fixed_ndf_table(block_ids, epoch_rows=None):
    """Fallback fixed filters from protected epoch parameters.

    Raw Stage device configuration remains authoritative. This table is used
    only when a block has no Stage setting (for example an LED-only rig). Empty
    flattened ``ndfs`` values are ignored because another device can overwrite
    the active stimulus value with an empty list. Embedded ``FW`` tokens are
    retained only as ignored provenance; they never become wheel measurements.
    """
    import pandas as pd
    from retinanalysis.utils.isomerization import split_stage_ndfs

    ids = [int(block_id) for block_id in block_ids]
    rows = (_read_epoch_filter_wheel_parameters(ids)
            if epoch_rows is None else list(epoch_rows))
    grouped = {block_id: [] for block_id in ids}
    for row in rows:
        block_id = int(row["parent_id"])
        if block_id in grouped:
            grouped[block_id].append(row)

    result = []
    for block_id in ids:
        fixed_values = []
        embedded_values = []
        for epoch in grouped[block_id]:
            parameters = epoch.get("parameters")
            parameters = parameters if isinstance(parameters, Mapping) else {}
            raw = parameters.get("ndfs")
            if raw is None or not str(raw).strip() or str(raw).strip() in {"[]", "()"}:
                continue
            fixed, embedded = split_stage_ndfs(raw)
            fixed = tuple(sorted(fixed))
            if fixed and fixed not in fixed_values:
                fixed_values.append(fixed)
            embedded_values.extend(embedded)
        if len(fixed_values) > 1:
            joined = " | ".join(" + ".join(value) for value in fixed_values)
            raise ValueError(
                f"Epoch block {block_id} has conflicting fixed-NDF metadata: {joined}"
            )
        result.append({
            "block_id": block_id,
            "epoch_fixed_ndfs": fixed_values[0] if fixed_values else (),
            "ignored_epoch_fw_tokens": tuple(sorted(set(embedded_values))),
        })
    return pd.DataFrame(result)


def read_block_light_settings(blocks, block_id=None, *, amp="Amp1", verbose=True,
                              on_filter_wheel_conflict="raise"):
    """Read one trusted fixed-NDF + FilterWheel setting per epoch block.

    ``blocks`` may be a frame containing ``exp_name`` and ``block_id``, or one
    experiment name when ``block_id`` is supplied separately. If the frame also
    contains ``filter_wheel_ndf``, each row is treated as an epoch-level wheel
    condition and validated against the protected readings; this allows a block
    whose hardware setting changed mid-run to be split without guessing. Fixed EL filters
    come from the raw Stage device configurator.  Any ``FW`` token found in
    that list is ignored.  The numeric wheel OD comes independently from the
    protected ``FilterWheel:NDF`` metadata and is checked across every epoch in
    the block before being consolidated.

    The returned frame has one row per requested block. By default conflicting
    protected readings raise. ``on_filter_wheel_conflict='report'`` keeps the
    block with a missing numeric wheel and an explicit conflict status, which
    is useful for broad discovery tables; condition analysis should split the
    epochs or retain the default error.

    ``fixed_ndfs`` and
    ``filter_wheel_ndf`` can be passed directly to :func:`visual_stimulus_max`;
    ``ndf_combination`` is the readable combined setting.

    Examples
    --------
    >>> settings = read_block_light_settings("2026-08-10_G", 12345)
    >>> row = settings.iloc[0]
    >>> visual_stimulus_max(row.rig, row.fixed_ndfs, row.filter_wheel_ndf)
    """
    import pandas as pd
    from retinanalysis.SCutils.recording_mode import stage_ndf_table
    from retinanalysis.utils.isomerization import split_stage_ndfs

    if on_filter_wheel_conflict not in {"raise", "report"}:
        raise ValueError(
            "on_filter_wheel_conflict must be 'raise' or 'report'")

    if isinstance(blocks, str):
        if block_id is None:
            raise TypeError("block_id is required when blocks is an experiment name")
        wanted = pd.DataFrame({"exp_name": [blocks], "block_id": [block_id]})
    elif block_id is not None:
        raise TypeError("block_id must be omitted when blocks is a table")
    else:
        wanted = pd.DataFrame(blocks).copy()
    required = {"exp_name", "block_id"}
    missing = required.difference(wanted.columns)
    if missing:
        raise ValueError(f"blocks is missing required column(s): {', '.join(sorted(missing))}")
    has_selected_wheel = "filter_wheel_ndf" in wanted.columns
    wanted_columns = ["exp_name", "block_id"]
    if has_selected_wheel:
        wanted_columns.append("filter_wheel_ndf")
    wanted = wanted[wanted_columns].drop_duplicates().reset_index(drop=True)
    wanted["block_id"] = pd.to_numeric(wanted["block_id"], errors="raise").astype(int)
    if has_selected_wheel:
        wanted["filter_wheel_ndf"] = pd.to_numeric(
            wanted["filter_wheel_ndf"], errors="raise").astype(float)
    elif wanted["block_id"].duplicated().any():
        duplicates = wanted.loc[wanted["block_id"].duplicated(False), "block_id"].tolist()
        raise ValueError(f"block_id must identify one experiment; duplicated: {duplicates}")

    output_columns = [
        "exp_name", "block_id", "rig", "stage_ndfs", "fixed_ndfs", "filter_wheel_ndf",
        "ndf_combination", "filter_wheel_status", "n_epochs",
        "n_filter_wheel_readings", "fixed_ndf_source", "ignored_stage_fw_tokens",
        "ignored_epoch_fw_tokens",
    ]
    if wanted.empty:
        return pd.DataFrame(columns=output_columns)

    fixed_table = stage_ndf_table(wanted, amp=amp, verbose=verbose)
    wheel_table = (_selected_block_filter_wheel_table(wanted)
                   if has_selected_wheel else
                   _block_filter_wheel_table(
                       wanted["block_id"].tolist(),
                       on_conflict=on_filter_wheel_conflict))
    if has_selected_wheel:
        result = (wanted.merge(fixed_table, on="block_id", how="left",
                               validate="many_to_one")
                  .merge(wheel_table, on=["block_id", "filter_wheel_ndf"],
                         how="left", validate="one_to_one"))
    else:
        result = (wanted.merge(fixed_table, on="block_id", how="left",
                               validate="one_to_one")
                  .merge(wheel_table, on="block_id", how="left",
                         validate="one_to_one"))

    fixed_values = []
    embedded_values = []
    for raw in result["stage_ndfs"].fillna(""):
        fixed, embedded = split_stage_ndfs(raw)
        fixed_values.append(tuple(sorted(fixed)))
        embedded_values.append(tuple(embedded))
    result["fixed_ndfs"] = fixed_values
    result["ignored_stage_fw_tokens"] = embedded_values
    result["fixed_ndf_source"] = [
        "stage" if str(raw).strip() else "not recorded"
        for raw in result["stage_ndfs"].fillna("")
    ]
    result["ignored_epoch_fw_tokens"] = [()] * len(result)

    # LED-only recordings have no Stage device setting. Preserve their active
    # stimulus fixed filters from epoch parameters, but never use an embedded
    # FW label as the wheel value.
    missing_stage = result["fixed_ndf_source"].eq("not recorded")
    if missing_stage.any():
        fallback = _epoch_fixed_ndf_table(
            result.loc[missing_stage, "block_id"].drop_duplicates().tolist())
        fallback = fallback.set_index("block_id")
        for index in result.index[missing_stage]:
            block = int(result.at[index, "block_id"])
            fixed = fallback.at[block, "epoch_fixed_ndfs"]
            ignored = fallback.at[block, "ignored_epoch_fw_tokens"]
            result.at[index, "fixed_ndfs"] = fixed
            result.at[index, "ignored_epoch_fw_tokens"] = ignored
            if fixed:
                result.at[index, "fixed_ndf_source"] = "epoch fallback"
    result["rig"] = result["exp_name"].astype(str).str.extract(
        r"_([A-Za-z])(?:_\d+)?$", expand=False).str.upper()
    result["ndf_combination"] = [
        " + ".join([*fixed, f"FW{wheel:g}"])
        if pd.notna(wheel) else
        " + ".join([*fixed, "FW conflict"])
        if str(status).startswith("conflict:") else
        (" + ".join(fixed) or "not recorded")
        for fixed, wheel, status in zip(
            result["fixed_ndfs"], result["filter_wheel_ndf"],
            result["filter_wheel_status"])
    ]
    return result[output_columns]


__all__ = [
    "VISUAL_STIMULUS_MAX", "filter_wheel_ndf_from_epoch_parameters",
    "read_block_light_settings", "visual_stimulus_max",
]
