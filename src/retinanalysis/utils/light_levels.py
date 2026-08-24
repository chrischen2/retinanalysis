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
    if embedded_wheel is not None and filter_wheel_ndf is not None:
        explicit_wheel = _normalize_wheel(filter_wheel_ndf)
        if explicit_wheel != embedded_wheel:
            raise ValueError("Conflicting embedded and explicit filter-wheel values")
    wheel = (embedded_wheel if embedded_wheel is not None
             else _normalize_wheel(filter_wheel_ndf))
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


__all__ = ["VISUAL_STIMULUS_MAX", "visual_stimulus_max"]
