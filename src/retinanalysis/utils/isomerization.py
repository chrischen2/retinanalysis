"""Symphony light calibration and photoreceptor-isomerization utilities.

The numerical conversion is a direct Python translation of Rieke Lab's
``convisom.m`` (used by ``IsomerizationsConverter.m``).  Metadata helpers keep
fixed neutral-density filters separate from the motorized filter wheel: an
``FW3`` token embedded in a Stage ``ndfs`` setting is *not* used as a wheel
reading; the numeric ``FilterWheel:NDF`` configuration is authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PLANCK_CONSTANT = 6.62607004e-34
SPEED_OF_LIGHT = 299_792_458.0

RIG_BY_SUFFIX = {
    # Provisional mapping confirmed by the user on 2026-08-12. Keep this in
    # one visible table because the historical rig assignment may be revised.
    "G": "shared_two_photon",
    "E": "confocal",
    "B": "two_photon",
    "F": "confocal",
    "C": "mea",
}

# Values are optical density (OD), copied from the corresponding base rig
# descriptions in riekelab-package. Display colour-dependent maps are left to
# the caller because the exact add-on rig class can change with date.
_LED_NDF_OD = {
    "shared_two_photon": {
        "red": dict(zip(("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"),
                        (.9884, .9910, 1.9023, 2.0200, 3.9784, .3, .6, 1.14, 1.99))),
        "uv": dict(zip(("G1", "G2", "G3", "G4", "G6", "G7", "G8", "G9"),
                       (1.0060, 1.0524, 2.1342, 2.6278, .28, .59, 1.25, 2.23))),
        "blue": dict(zip(("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"),
                         (1.0171, 1.0428, 2.0749, 2.1623, 4.2439, .26, .61, 1.22, 2.17))),
    },
    "two_photon": {
        "red": dict(zip(("B1", "B2", "B3", "B4", "B5", "B11"),
                        (.29, .61, 1.01, 2.08, 4.41, 3.94))),
        "uv": dict(zip(("B1", "B2", "B3", "B4", "B5", "B6", "B7"),
                       (.29, .71, 1.21, 2.54, 4.58, 2.71, 5.13))),
        "blue": dict(zip(("B1", "B2", "B3", "B4", "B5", "B8", "B9"),
                         (.29, .60, 1.02, 2.41, 4.58, 2.20, 4.32))),
    },
    "old_slice": {
        "red": dict(zip(("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F12"),
                        (.3081, .2842, .6371, 1.0571, 1.8768, 1.944, 3.752, 2.84))),
        "green": dict(zip(("F1", "F2", "F3", "F4", "F5", "F8", "F9", "F12"),
                          (.3059, .2862, .5869, 1.0955, 1.9804, 1.8555, 3.6936, 3.6))),
        "uv": dict(zip(("F1", "F2", "F3", "F4", "F5", "F10", "F11", "F12"),
                       (.3011, .2828, .5367, 1.127, 2.0587, 1.7208, 3.7415, 4.06))),
    },
    "confocal": {
        "red": dict(zip(("E1", "E2", "E3", "E4", "E5", "E10", "E11", "E12"),
                        (.24, .63, .94, 2.02, 3.43, 1.86, 3.73, .3))),
        "uv": dict(zip(("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E12"),
                       (.26, .52, .89, 2.30, 4.20, 1.88, 3.92, .28))),
        "green": dict(zip(("E1", "E2", "E3", "E4", "E5", "E8", "E9", "E12"),
                          (.26, .58, .93, 2.19, 4.11, 1.85, 3.8, .3))),
    },
    "mea": {
        "uv": dict(zip(("C1", "C2", "C3", "C4", "C5"),
                       (.2768, .5076, .9281, 2.1275, 2.5022))),
        "blue": dict(zip(("C1", "C2", "C3", "C4", "C5"),
                         (.2663, .5389, .9569, 2.0810, 2.3747))),
        "green": dict(zip(("C1", "C2", "C3", "C4", "C5"),
                          (.2866, .5933, .9675, 1.9279, 2.1372))),
    },
}

_LIGHTCRAFTER_NDF_OD = {
    "auto": {"EL1": .97, "EL2": 2.11, "EL3": 4.23},
    "red": {"EL1": .98, "EL2": 2.06, "EL3": 4.09},
    "green": {"EL1": .99, "EL2": 2.14, "EL3": 4.29},
    "blue": {"EL1": .99, "EL2": 2.16, "EL3": 4.40},
}

_MICRODISPLAY_NDF_OD = {
    ("two_photon", "above"): {
        "white": dict(zip(("B1", "B2", "B3", "B4", "B12", "B13"),
                          (.26, .60, .98, 2.21, .27, 1.03))),
        "red": dict(zip(("B1", "B2", "B3", "B4", "B12", "B13"),
                        (.26, .60, .97, 2.09, .27, 1.01))),
        "green": dict(zip(("B1", "B2", "B3", "B4", "B12", "B13"),
                          (.26, .61, .98, 2.22, .27, 1.03))),
        "blue": dict(zip(("B1", "B2", "B3", "B4", "B12", "B13"),
                         (.26, .60, .97, 2.24, .27, 1.04))),
    },
    ("mea", "below"): {
        "white": dict(zip(("E1", "E2", "E3", "E4", "E12"),
                          (.26, .59, .94, 2.07, .30))),
        "red": dict(zip(("E1", "E2", "E3", "E4", "E12"),
                        (.26, .61, .94, 2.05, .29))),
        "green": dict(zip(("E1", "E2", "E3", "E4", "E12"),
                          (.26, .58, .94, 2.12, .29))),
        "blue": dict(zip(("E1", "E2", "E3", "E4", "E12"),
                         (.26, .57, .93, 2.13, .29))),
    },
}

COLLECTING_AREAS = {
    "primate": {
        "lCone": (.37, .60), "mCone": (.37, .60),
        "sCone": (.37, .60), "rod": (1.0, 1.0),
    },
    "mouse": {
        "mCone": (.20, 1.0), "sCone": (.20, 1.0), "rod": (.50, .87),
    },
}

_FW_TOKEN = re.compile(r"^FW(?:\d+(?:\.\d+)?)?$", re.IGNORECASE)
_EXP_RIG = re.compile(r"^\d{4}-?\d{2}-?\d{2}[_-]?([A-Za-z])(?:[_-].*)?$")


@dataclass(frozen=True)
class FluxFactor:
    """One dated flux calibration, in (nW/intensity)/um^2."""

    factor: float
    calibration_time: datetime
    path: Path


def infer_rig_name(exp_name: str) -> str:
    """Infer the calibration rig from an experiment name/date suffix."""
    match = _EXP_RIG.match(Path(str(exp_name)).stem)
    if not match or match.group(1).upper() not in RIG_BY_SUFFIX:
        raise ValueError(f"Cannot infer a known rig suffix from {exp_name!r}")
    return RIG_BY_SUFFIX[match.group(1).upper()]


def parse_ndfs(value: Any) -> tuple[str, ...]:
    """Normalize a Symphony ``ndfs`` value (JSON string, list, or text)."""
    if value is None:
        return ()
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            value = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = re.split(r"\s*,\s*", text)
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return tuple(str(item).strip() for item in value if str(item).strip())


def split_stage_ndfs(value: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(fixed_filters, embedded_FW_tokens)`` from a Stage setting."""
    fixed, wheel_tokens = [], []
    for name in parse_ndfs(value):
        (wheel_tokens if _FW_TOKEN.match(name) else fixed).append(name)
    return tuple(fixed), tuple(wheel_tokens)


def ndf_attenuation(ndfs: Iterable[str], attenuations: Mapping[str, float],
                    filter_wheel_ndf: float = 0.0) -> float:
    """Return transmission, using the actual numeric wheel OD separately."""
    total_od = float(filter_wheel_ndf)
    missing = []
    for name in ndfs:
        if _FW_TOKEN.match(str(name)):
            continue
        if name not in attenuations:
            missing.append(str(name))
        else:
            total_od += float(attenuations[name])
    if missing:
        raise KeyError(f"No attenuation for NDF(s): {', '.join(missing)}")
    return 10.0 ** (-total_od)


def led_ndf_attenuations(rig: str, device_name: str) -> Mapping[str, float]:
    """Get the base-rig LED NDF map for a device such as ``Red LED``."""
    colour = str(device_name).split()[0].lower()
    try:
        return _LED_NDF_OD[rig][colour]
    except KeyError as exc:
        raise KeyError(f"No base-rig LED NDF map for {rig!r}/{device_name!r}") from exc


def isomerizations_per_watt(device_spectrum: Sequence[Sequence[float]],
                            photoreceptor_spectrum: Sequence[Sequence[float]]) -> float:
    """Calculate isomerizations/s per watt, matching MATLAB ``convisom``."""
    device = np.asarray(device_spectrum, dtype=float)
    receptor = np.asarray(photoreceptor_spectrum, dtype=float)
    if device.ndim != 2 or receptor.ndim != 2 or device.shape[1] < 2 or receptor.shape[1] < 2:
        raise ValueError("Spectra must be two-column wavelength/value arrays")
    if receptor.shape[0] < 2:
        raise ValueError("Photoreceptor spectrum requires at least two wavelengths")

    dwl = device[:, 0].copy()
    rwl = receptor[:, 0].copy()
    if np.nanmax(dwl) > 1:
        dwl *= 1e-9
    if np.nanmax(rwl) > 1:
        rwl *= 1e-9
    dval = np.interp(rwl, dwl, device[:, 1], left=np.nan, right=np.nan)
    if np.isnan(dval).any():
        raise ValueError("Device spectrum does not cover the photoreceptor wavelengths")
    dval = np.maximum(dval, 0)
    rval = np.maximum(receptor[:, 1], 0)
    widths = np.diff(rwl)
    widths = np.r_[widths, widths[-1]]
    numerator = np.sum(dval * rval * widths)
    denominator = np.sum(dval * (PLANCK_CONSTANT * SPEED_OF_LIGHT / rwl) * widths)
    if denominator <= 0:
        raise ValueError("Device spectrum has no positive power over the receptor spectrum")
    return float(numerator / denominator)


def convert_isomerizations(value: Any, input_units: str, flux_factor: float,
                            device_spectrum: Sequence[Sequence[float]],
                            photoreceptor_spectrum: Sequence[Sequence[float]],
                            collecting_area: float, ndfs: Iterable[str] = (),
                            attenuations: Mapping[str, float] | None = None,
                            filter_wheel_ndf: float = 0.0):
    """Convert intensity/volts to isomerizations/s, or the inverse.

    ``flux_factor`` is in ``(nW / intensity) / um^2`` and ``collecting_area``
    in ``um^2``, matching the MATLAB calibration resources.
    """
    transmission = ndf_attenuation(ndfs, attenuations or {}, filter_wheel_ndf)
    isom_per_watt = isomerizations_per_watt(device_spectrum, photoreceptor_spectrum)
    watts_per_intensity = float(flux_factor) * float(collecting_area) * 1e-9
    scale = watts_per_intensity * isom_per_watt * transmission
    values = np.asarray(value)
    units = input_units.lower()
    if units == "isom":
        output = values / scale
    elif units in {"volts", "intensity"}:
        output = values * scale
    else:
        raise ValueError("input_units must be 'isom', 'volts', or 'intensity'")
    return output.item() if output.ndim == 0 else output


# A discoverable name for users looking for the MATLAB module equivalent.
isomerizations_converter = convert_isomerizations


def collecting_area(species: str, receptor: str, light_path: str,
                    preparation_type: str) -> float:
    """Select collecting area using the orientation logic in the MATLAB UI."""
    species_key = "primate" if str(species).lower().startswith(("m.", "primate", "macaca")) else str(species).lower()
    try:
        photoreceptor_side, ganglion_side = COLLECTING_AREAS[species_key][receptor]
    except KeyError as exc:
        raise KeyError(f"Unknown species/receptor combination: {species!r}/{receptor!r}") from exc
    prep = str(preparation_type).lower()
    if "shredded" in prep or "slice" in prep:
        orientation = "lateral"
    elif "cone" in prep and "up" in prep:
        orientation = "up"
    elif "rpe" in prep or "ganglion" in prep or "rgc" in prep:
        orientation = "down"
    else:
        raise ValueError(f"Cannot infer retinal orientation from {preparation_type!r}")
    path = str(light_path).lower()
    if (path == "below" and orientation in {"down", "lateral"}) or (path == "above" and orientation in {"up", "lateral"}):
        return photoreceptor_side
    if (path == "below" and orientation == "up") or (path == "above" and orientation == "down"):
        return ganglion_side
    raise ValueError(f"Unknown light path {light_path!r}")


def resolve_calibration_resources(path: str | os.PathLike[str] | None = None) -> Path:
    """Locate a checkout of Rieke-Lab/calibration-resources."""
    candidates = [path, os.environ.get("RIEKELAB_CALIBRATION_RESOURCES")]
    package = Path("/Users/chrischen/Documents/GitHub/riekelab-package-master")
    candidates.extend((package.parent / "calibration-resources", package / "calibration-resources"))
    for candidate in candidates:
        if candidate and (Path(candidate) / "rigs").is_dir() and (Path(candidate) / "sources").is_dir():
            return Path(candidate)
    tried = "\n  ".join(str(Path(value)) for value in candidates if value)
    raise FileNotFoundError(
        "calibration-resources checkout not found; pass its path or set "
        f"RIEKELAB_CALIBRATION_RESOURCES. Tried:\n  {tried}"
    )


def load_spectrum(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a two-column Rieke Lab spectrum resource."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected a two-column spectrum in {path}")
    return data[:, :2]


def select_flux_factor(path: str | os.PathLike[str], experiment_date: str | datetime) -> FluxFactor:
    """Select the most recent calibration at or before an experiment date."""
    import pandas as pd

    if isinstance(experiment_date, str):
        match = re.match(r"^(\d{4})-?(\d{2})-?(\d{2})", experiment_date)
        when = pd.Timestamp("-".join(match.groups())) if match else pd.Timestamp(experiment_date)
        # An experiment date means the complete day; a same-day calibration
        # should therefore be eligible even when no experiment time is given.
        if match and len(experiment_date) <= 12:
            when = when + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    else:
        when = pd.Timestamp(experiment_date)
    table = pd.read_csv(path, sep="\t")
    if not {"date", "factor"}.issubset(table.columns):
        raise ValueError(f"Flux table lacks date/factor columns: {path}")
    table["_date"] = pd.to_datetime(table["date"], format="%d-%b-%Y %H:%M:%S")
    table["factor"] = pd.to_numeric(table["factor"], errors="coerce")
    valid = table[(table["_date"] <= when) & (table["factor"] > 0)].sort_values("_date")
    if valid.empty:
        raise ValueError(
            f"No positive flux calibration on or before {when.date()} in {path}"
        )
    row = valid.iloc[-1]
    return FluxFactor(float(row["factor"]), row["_date"].to_pydatetime(), Path(path))


def _species_key(species: str) -> str:
    text = str(species).strip().lower()
    if text.startswith(("m.", "macaca", "primate")):
        return "primate"
    if text.startswith(("mouse", "mus ")):
        return "mouse"
    if text.startswith("zebrafish"):
        return "zebrafish"
    return text


def _require_file(candidates: Sequence[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot load {description}; tried:\n  {tried}")


def _device_calibration_files(row: Mapping[str, Any], color: str,
                              resources: Path) -> tuple[Path, Path, Mapping[str, float], str]:
    rig = str(row.get("rig"))
    device = str(row.get("device"))
    path = str(row.get("light_path") or "").lower()
    gain = str(row.get("gain") or "").lower()
    brightness = str(row.get("brightness") or "").lower()
    rig_dir = resources / "rigs" / rig
    lower = device.lower()

    if "led" in lower and "lightcrafter" not in lower:
        device_color = device.split()[0].lower()
        spectrum = _require_file([rig_dir / f"{device_color}_led_spectrum.txt"],
                                 f"{rig}/{device} spectrum")
        names = []
        if gain:
            if path == "above":
                names.append(f"{device_color}_led_above_{gain}_flux_factors.txt")
            names.append(f"{device_color}_led_{gain}_flux_factors.txt")
        names.append(f"{device_color}_led_flux_factors.txt")
        flux = _require_file([rig_dir / name for name in names],
                             f"{rig}/{device} flux factors (gain={gain or 'not recorded'})")
        return spectrum, flux, led_ndf_attenuations(rig, device), device_color

    if "lightcrafter" in lower:
        selected_color = "auto" if color in {"", "white", "achromatic"} else color
        stem = f"lightcrafter_{path or 'above'}_{selected_color}"
        spectrum = _require_file([rig_dir / f"{stem}_spectrum.txt"],
                                 f"{rig}/{device} {selected_color} spectrum")
        flux = _require_file([rig_dir / f"{stem}_flux_factors.txt"],
                             f"{rig}/{device} {selected_color} flux factors")
        return spectrum, flux, _LIGHTCRAFTER_NDF_OD[selected_color], selected_color

    if "microdisplay" in lower:
        selected_color = "white" if color in {"", "auto", "achromatic"} else color
        stem = f"microdisplay_{path or 'below'}"
        spectrum = _require_file([rig_dir / f"{stem}_{selected_color}_spectrum.txt"],
                                 f"{rig}/{device} {selected_color} spectrum")
        if brightness not in {"low", "medium", "high"}:
            raise ValueError(
                f"Cannot choose {rig}/{device} flux factors: microdisplayBrightness "
                f"is {brightness or 'not recorded'!r}; expected low, medium, or high"
            )
        flux = _require_file([rig_dir / f"{stem}_{brightness}_flux_factors.txt"],
                             f"{rig}/{device} {brightness} flux factors")
        try:
            attenuations = _MICRODISPLAY_NDF_OD[(rig, path)][selected_color]
        except KeyError:
            if row.get("named_ndfs"):
                raise KeyError(
                    f"No microdisplay NDF map for rig={rig!r}, path={path!r}, "
                    f"color={selected_color!r}"
                )
            attenuations = {}
        return spectrum, flux, attenuations, selected_color

    raise ValueError(f"Unsupported optical stimulus device {device!r}")


def convert_experiment_receptors(metadata: Mapping[str, Any] | str | os.PathLike[str],
                                 configuration: Mapping[str, Any], value: Any,
                                 input_units: str = "intensity", color: str = "auto",
                                 calibration_root: str | os.PathLike[str] | None = None):
    """Convert one experiment/device configuration for every species receptor.

    Returns one row per receptor with the selected calibration provenance.
    Missing resources raise a detailed exception naming every attempted path;
    :func:`isomerization_converter_widget` prints that exception in its output.
    """
    import pandas as pd

    row = configuration
    resources = resolve_calibration_resources(calibration_root)
    species = _species_key(str(row.get("species") or ""))
    if species not in COLLECTING_AREAS:
        raise ValueError(f"No receptor definitions for loaded species {row.get('species')!r}")
    spectrum_path, flux_path, attenuations, selected_color = _device_calibration_files(
        row, str(color).lower(), resources
    )
    device_spectrum = load_spectrum(spectrum_path)
    flux = select_flux_factor(flux_path, str(row.get("exp_name")))
    fixed_ndfs = parse_ndfs(row.get("named_ndfs"))
    wheel = float(row.get("filter_wheel_ndf") or 0.0)
    results = []
    receptor_files = {
        "lCone": "l_cone_spectrum.txt", "mCone": "m_cone_spectrum.txt",
        "sCone": "s_cone_spectrum.txt", "rod": "rod_spectrum.txt",
    }
    for receptor in COLLECTING_AREAS[species]:
        receptor_path = _require_file(
            [resources / "sources" / species / receptor_files[receptor]],
            f"{species}/{receptor} spectrum",
        )
        area = collecting_area(species, receptor, str(row.get("light_path")),
                               str(row.get("preparation_type")))
        output = convert_isomerizations(
            value, input_units, flux.factor, device_spectrum,
            load_spectrum(receptor_path), area, fixed_ndfs, attenuations, wheel,
        )
        results.append({
            "receptor": receptor,
            "output": output,
            "output_units": "intensity" if input_units.lower() == "isom" else "isom/s",
            "collecting_area_um2": area,
            "calibration_date": flux.calibration_time,
            "flux_factor": flux.factor,
            "device_color": selected_color,
            "named_ndfs": ", ".join(fixed_ndfs),
            "filter_wheel_ndf": wheel,
            "flux_file": str(flux.path),
            "spectrum_file": str(spectrum_path),
        })
    return pd.DataFrame(results)


def _load_metadata(metadata: Mapping[str, Any] | str | os.PathLike[str]) -> Mapping[str, Any]:
    if isinstance(metadata, Mapping):
        return metadata
    path = Path(metadata)
    if not path.is_file():
        from retinanalysis.config.settings import find_path
        exp_name = path.stem
        resolved = Path(find_path("meta", f"{exp_name}.json"))
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Cannot load metadata for {exp_name}; tried {path} and {resolved}"
            )
        path = resolved
    with open(path) as stream:
        return json.load(stream)


def epoch_group_ndf_table(metadata: Mapping[str, Any] | str | os.PathLike[str]):
    """Return distinct per-epoch-group light configurations from parsed JSON.

    Fixed filters and embedded ``FWx`` labels come from each optical device's
    ``ndfs`` setting. The calculation-ready ``filter_wheel_ndf`` comes only
    from the actual FilterWheel background configuration.
    """
    import pandas as pd

    metadata_path = None if isinstance(metadata, Mapping) else Path(metadata)
    root = _load_metadata(metadata)
    # The root JSON label is commonly the project/source label (for example
    # "Primate"), not the experiment name. The canonical filename is reliable.
    exp_name = metadata_path.stem if metadata_path is not None else str(
        root.get("exp_name") or root.get("label") or ""
    )
    try:
        rig = infer_rig_name(exp_name)
    except ValueError:
        rig = None
    rows = []
    project_species = str(root.get("label") or "").strip()
    if project_species.lower() not in {"mouse", "primate", "zebrafish"}:
        project_species = None
    for animal in root.get("animals", []):
        species = animal.get("species") or project_species
        for prep in animal.get("preparations", []):
            prep_type = prep.get("preparationType")
            for cell in prep.get("cells", []):
                for group in cell.get("epoch_groups", []):
                    for block in group.get("epoch_blocks", []):
                        for epoch in block.get("epochs", []):
                            backgrounds = epoch.get("backgrounds") or {}
                            stimuli = epoch.get("stimuli") or {}
                            parameters = epoch.get("parameters") or {}
                            epoch_block_config = backgrounds.get("epochBlock") or {}
                            wheel = next((cfg for name, cfg in backgrounds.items()
                                          if str(name).split("@")[0] == "FilterWheel"), None)
                            wheel_present = wheel is not None and wheel.get("NDF") is not None
                            wheel_ndf = float(wheel["NDF"]) if wheel_present else 0.0
                            optical_configs = [(name, cfg, "background", False)
                                               for name, cfg in backgrounds.items()]
                            # JSON stimuli contain the waveform metadata while
                            # their active device settings (ndfs, lightPath,
                            # gain) are promoted into epoch.parameters.
                            for name, stimulus_cfg in stimuli.items():
                                cfg = dict(parameters)
                                cfg.update(stimulus_cfg or {})
                                optical_configs.append((name, cfg, "stimulus", True))
                            for device, cfg, config_source, is_stimulus in optical_configs:
                                base_name = str(device).split("@")[0]
                                if base_name in {"FilterWheel", "Amp1", "backgrounds", "epochBlock",
                                                 "properties", "protocolParameters", "responses", "stimuli"}:
                                    continue
                                is_light = ("ndfs" in cfg or "LED" in base_name or "Stage" in base_name
                                            or "display" in base_name.lower() or "LightCrafter" in base_name)
                                if not is_light:
                                    continue
                                fixed, embedded_fw = split_stage_ndfs(cfg.get("ndfs"))
                                rows.append({
                                    "exp_name": exp_name,
                                    "rig": rig,
                                    "species": species,
                                    "preparation_type": prep_type,
                                    "cell": cell.get("label"),
                                    "epoch_group": group.get("label"),
                                    "epoch_group_uuid": group.get("uuid"),
                                    "protocol": epoch.get("protocolID") or epoch_block_config.get("protocolID"),
                                    "epoch_block_uuid": block.get("uuid") or epoch_block_config.get("uuid"),
                                    "device": base_name,
                                    "configuration_source": config_source,
                                    "is_stimulus": is_stimulus,
                                    "light_path": cfg.get("lightPath"),
                                    "gain": cfg.get("gain"),
                                    "brightness": cfg.get("microdisplayBrightness"),
                                    "color": cfg.get("color") or cfg.get("chromaticClass"),
                                    "named_ndfs": ", ".join(fixed),
                                    "embedded_fw_tokens": ", ".join(embedded_fw),
                                    "filter_wheel_ndf": wheel_ndf,
                                    "filter_wheel_present": wheel_present,
                                })
    columns = ["exp_name", "rig", "species", "preparation_type", "cell",
               "epoch_group", "epoch_group_uuid", "protocol",
               "device", "configuration_source", "is_stimulus", "light_path",
               "gain", "brightness", "color", "named_ndfs",
               "embedded_fw_tokens", "filter_wheel_ndf", "filter_wheel_present"]
    if not rows:
        return pd.DataFrame(columns=columns + ["n_epoch_blocks", "n_epochs"])
    frame = pd.DataFrame(rows)
    group_cols = columns
    return (frame.groupby(group_cols, dropna=False, sort=False)
            .agg(n_epoch_blocks=("epoch_block_uuid", "nunique"),
                 n_epochs=("epoch_block_uuid", "size"))
            .reset_index())


def isomerization_converter_widget(metadata: Mapping[str, Any] | str | os.PathLike[str],
                                   calibration_root: str | os.PathLike[str] | None = None,
                                   show: bool = True):
    """Interactive experiment/device converter modeled after the MATLAB UI.

    The species and available optical devices are read from the experiment.
    The configuration menu selects the epoch group/block NDF state, and every
    receptor defined for the loaded species is reported together. Calibration
    failures are printed with the missing file paths rather than raised from
    an ipywidgets callback.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:
        raise ImportError("isomerization_converter_widget requires ipywidgets") from exc

    table = epoch_group_ndf_table(metadata)
    if table.empty:
        raise ValueError("No optical stimulus configurations found in the experiment metadata")
    devices = list(dict.fromkeys(table["device"].dropna().astype(str)))
    device = widgets.Dropdown(options=devices, description="Device:",
                              layout=widgets.Layout(width="420px"))
    configuration = widgets.Dropdown(description="Epoch group:",
                                     layout=widgets.Layout(width="760px"))
    color = widgets.Dropdown(description="Color:", layout=widgets.Layout(width="250px"))
    input_units = widgets.Dropdown(options=[("intensity", "intensity"),
                                            ("volts", "volts"),
                                            ("isom/s", "isom")],
                                       description="Input units:",
                                       layout=widgets.Layout(width="250px"))
    value = widgets.FloatText(value=1.0, description="Input value:",
                              layout=widgets.Layout(width="250px"))
    root_value = str(calibration_root or "")
    resource_path = widgets.Text(value=root_value, description="Calibration:",
                                 placeholder="path or RIEKELAB_CALIBRATION_RESOURCES",
                                 layout=widgets.Layout(width="760px"))
    calculate = widgets.Button(description="Calculate all receptors", icon="calculator",
                               button_style="primary")
    output = widgets.Output()
    species_values = ", ".join(str(v) for v in table["species"].dropna().unique()) or "unknown"
    rig_values = ", ".join(str(v) for v in table["rig"].dropna().unique()) or "unknown"
    header = widgets.HTML(
        f"<b>Isomerizations converter</b> &nbsp; Species: {species_values} &nbsp; "
        f"Rig: {rig_values} <i>(provisional suffix mapping)</i>"
    )

    def _colors_for_device(name: str):
        lower = name.lower()
        if "lightcrafter" in lower:
            return ["auto", "red", "green", "blue"]
        if "microdisplay" in lower:
            return ["white", "red", "green", "blue"]
        if "led" in lower:
            return [name.split()[0].lower()]
        return ["auto"]

    def _set_configurations(*_):
        subset = table[table["device"].eq(device.value)]
        options = []
        for index, row in subset.iterrows():
            source = "stimulus" if bool(row["is_stimulus"]) else "background"
            filters = row["named_ndfs"] or "no fixed NDF"
            wheel = f"FW={float(row['filter_wheel_ndf']):g}"
            protocol = str(row["protocol"]).split(".protocols.")[-1]
            label = (f"{row['cell']} | {row['epoch_group']} | {protocol} | "
                     f"{filters}, {wheel} | {source} "
                     f"({int(row['n_epoch_blocks'])} blocks, {int(row['n_epochs'])} epochs)")
            options.append((label, int(index)))
        configuration.options = options
        configuration.value = options[0][1] if options else None
        color.options = _colors_for_device(str(device.value))

    def _calculate(_=None):
        with output:
            clear_output(wait=True)
            if configuration.value is None:
                print("Select an epoch-group configuration first.")
                return
            row = table.loc[int(configuration.value)]
            try:
                result = convert_experiment_receptors(
                    metadata, row, value.value, input_units.value, color.value,
                    resource_path.value.strip() or None,
                )
                visible = ["receptor", "output", "output_units", "collecting_area_um2",
                           "calibration_date", "flux_factor", "device_color",
                           "named_ndfs", "filter_wheel_ndf"]
                display(result[visible])
            except Exception as exc:
                print(f"Could not calculate {row['exp_name']} / {row['device']}:")
                print(f"{type(exc).__name__}: {exc}")

    device.observe(_set_configurations, names="value")
    calculate.on_click(_calculate)
    _set_configurations()
    box = widgets.VBox([
        header, widgets.HBox([device, color]), configuration,
        widgets.HBox([input_units, value]), resource_path, calculate, output,
    ])
    box.ndf_table = table
    box.selectors = {
        "device": device, "configuration": configuration, "color": color,
        "input_units": input_units, "value": value,
        "calibration_root": resource_path,
    }
    box.calculate_button = calculate
    box.output = output
    if show:
        display(box)
    return box


def isomerization_converter_browser(experiments=None,
                                    calibration_root: str | os.PathLike[str] | None = None,
                                    show: bool = True):
    """Experiment pull-down that loads an :func:`isomerization_converter_widget`."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError("isomerization_converter_browser requires ipywidgets") from exc

    if experiments is None:
        from retinanalysis.SCutils.explore import _experiment_catalog
        experiments = _experiment_catalog()[["exp_name"]]
    if hasattr(experiments, "columns"):
        names = experiments["exp_name"].dropna().astype(str).drop_duplicates().tolist()
    else:
        names = list(dict.fromkeys(str(value) for value in experiments))
    if not names:
        raise ValueError("No experiments are available for the isomerization converter")
    experiment = widgets.Dropdown(options=names, description="Experiment:",
                                  layout=widgets.Layout(width="360px"))
    content = widgets.VBox()
    error = widgets.Output()

    def _load(*_):
        error.clear_output(wait=True)
        try:
            converter = isomerization_converter_widget(
                str(experiment.value), calibration_root=calibration_root, show=False
            )
            content.children = (converter,)
        except Exception as exc:
            content.children = ()
            with error:
                print(f"Could not load {experiment.value}: {type(exc).__name__}: {exc}")

    experiment.observe(_load, names="value")
    _load()
    box = widgets.VBox([experiment, content, error])
    box.experiment_selector = experiment
    box.converter_content = content
    if show:
        display(box)
    return box


__all__ = [
    "COLLECTING_AREAS", "FluxFactor", "RIG_BY_SUFFIX", "collecting_area",
    "convert_isomerizations", "epoch_group_ndf_table", "infer_rig_name",
    "isomerizations_converter", "isomerization_converter_widget",
    "isomerization_converter_browser",
    "isomerizations_per_watt", "led_ndf_attenuations", "load_spectrum",
    "ndf_attenuation", "parse_ndfs", "resolve_calibration_resources",
    "select_flux_factor", "split_stage_ndfs", "convert_experiment_receptors",
]
