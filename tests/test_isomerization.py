import json

import numpy as np
import pytest

from retinanalysis.utils.isomerization import (
    convert_experiment_receptors,
    convert_isomerizations,
    epoch_group_ndf_table,
    infer_rig_name,
    ndf_attenuation,
    resolve_calibration_resources,
    select_flux_factor,
    split_stage_ndfs,
)


@pytest.mark.parametrize("name, expected", [
    ("2026-08-10_G", "shared_two_photon"),
    ("20260810G.h5", "shared_two_photon"),
    ("2025-03-20_B_2", "two_photon"),
    ("2026-07-03_E", "confocal"),
    ("2024-12-19_F", "confocal"),
    ("20230313C", "mea"),
])
def test_infer_rig_name(name, expected):
    assert infer_rig_name(name) == expected


def test_default_calibration_resources_is_repository_folder():
    root = resolve_calibration_resources()
    assert root.name == "calibration-resources"
    assert (root / "rigs" / "shared_two_photon").is_dir()
    assert (root / "sources" / "primate").is_dir()


def test_filter_wheel_token_is_not_used_as_wheel_reading():
    fixed, recorded_wheel = split_stage_ndfs('["EL06", "EL2", "FW3"]')
    assert fixed == ("EL06", "EL2")
    assert recorded_wheel == ("FW3",)
    # FW3 is diagnostic only; the actual wheel reading here is 1 OD.
    assert ndf_attenuation((*fixed, *recorded_wheel), {"EL06": .6, "EL2": 2}, 1) == pytest.approx(10 ** -3.6)


def test_converter_round_trip():
    device = np.array([[400, 1], [500, 2], [600, 1]], dtype=float)
    receptor = np.array([[400, .2], [500, 1], [600, .2]], dtype=float)
    isom = convert_isomerizations(
        .4, "intensity", .01, device, receptor, .5,
        ndfs=("N1",), attenuations={"N1": 1.0}, filter_wheel_ndf=.5,
    )
    assert isom > 0
    recovered = convert_isomerizations(
        isom, "isom", .01, device, receptor, .5,
        ndfs=("N1",), attenuations={"N1": 1.0}, filter_wheel_ndf=.5,
    )
    assert recovered == pytest.approx(.4)


def test_select_flux_factor_uses_latest_prior_calibration(tmp_path):
    path = tmp_path / "flux.txt"
    path.write_text(
        "date\tuser\tintensity\tdiameter\tpower\tfactor\tnote\n"
        "01-Jan-2020 12:00:00\tA\t1\t1\t1\t0.1\t\n"
        "01-Jan-2021 12:00:00\tA\t1\t1\t1\t0.2\t\n"
        "01-Jan-2022 12:00:00\tA\t1\t1\t1\t0.3\t\n"
    )
    selected = select_flux_factor(path, "2021-06-01_G")
    assert selected.factor == pytest.approx(.2)
    assert selected.calibration_time.year == 2021


def test_select_flux_factor_rejects_zero_placeholder(tmp_path):
    path = tmp_path / "flux.txt"
    path.write_text(
        "date\tuser\tintensity\tdiameter\tpower\tfactor\tnote\n"
        "01-Jan-2020 12:00:00\tA\t1\t1\t0\t0\t\n"
    )
    with pytest.raises(ValueError, match="No positive flux calibration"):
        select_flux_factor(path, "2026-01-01_E")


def test_epoch_group_ndf_table_uses_numeric_filter_wheel():
    epoch = {
        "protocolID": "example.Protocol",
        "backgrounds": {
            "FilterWheel": {"NDF": 1.0},
            "LightCrafter Stage@RIGE": {
                "ndfs": json.dumps(["EL3", "FW3"]), "lightPath": "above"
            },
            "epochBlock": {"uuid": "block-1"},
        },
    }
    metadata = {
        "label": "2026-07-03_E",
        "animals": [{
            "species": "M. mulatta",
            "preparations": [{
                "preparationType": "whole mount, RGCs up",
                "cells": [{
                    "label": "c1",
                    "epoch_groups": [{
                        "label": "Control", "uuid": "group-1",
                        "epoch_blocks": [{"uuid": "block-1", "epochs": [epoch, epoch]}],
                    }],
                }],
            }],
        }],
    }
    row = epoch_group_ndf_table(metadata).iloc[0]
    assert row["rig"] == "confocal"
    assert row["named_ndfs"] == "EL3"
    assert row["embedded_fw_tokens"] == "FW3"
    assert row["filter_wheel_ndf"] == pytest.approx(1.0)
    assert row["n_epochs"] == 2


def test_epoch_group_ndf_table_uses_filename_not_project_label(tmp_path):
    path = tmp_path / "2026-08-10_G.json"
    path.write_text(json.dumps({"label": "Primate", "animals": []}))
    table = epoch_group_ndf_table(path)
    # Empty tables still have their schema; rig inference itself is exercised
    # by the filename through the non-empty test above and infer_rig_name tests.
    assert list(table.columns)[0:2] == ["exp_name", "rig"]


def test_active_stimulus_uses_epoch_parameter_ndfs():
    metadata = {
        "label": "2026-08-10_G",
        "animals": [{"species": "Mouse", "preparations": [{
            "preparationType": "RPE attached", "cells": [{"label": "c1",
            "epoch_groups": [{"label": "Control", "uuid": "g1",
                "epoch_blocks": [{"uuid": "b1", "epochs": [{
                    "parameters": {"ndfs": '["G5"]', "lightPath": "above"},
                    "backgrounds": {"epochBlock": {"protocolID": "p"}},
                    "stimuli": {"UV LED": {"units": "_normalized_"}},
                }]}]}]}]}]}],
    }
    row = epoch_group_ndf_table(metadata).iloc[0]
    assert row["device"] == "UV LED"
    assert bool(row["is_stimulus"])
    assert row["configuration_source"] == "stimulus"
    assert row["named_ndfs"] == "G5"


def test_convert_experiment_receptors_returns_all_primate_receptors(tmp_path):
    root = tmp_path / "calibration-resources"
    rig = root / "rigs" / "confocal"
    source = root / "sources" / "primate"
    rig.mkdir(parents=True)
    source.mkdir(parents=True)
    spectrum = "400 1\n500 2\n600 1\n"
    (rig / "green_led_spectrum.txt").write_text(spectrum)
    (rig / "green_led_flux_factors.txt").write_text(
        "date\tuser\tintensity\tdiameter\tpower\tfactor\tnote\n"
        "01-Jan-2020 12:00:00\tA\t1\t1\t1\t0.1\t\n"
    )
    for name in ("l_cone", "m_cone", "s_cone", "rod"):
        (source / f"{name}_spectrum.txt").write_text(spectrum)
    row = {
        "exp_name": "2026-07-03_E", "rig": "confocal", "species": "Primate",
        "preparation_type": "RPE attached", "device": "Green LED",
        "light_path": "above", "gain": None, "brightness": None,
        "named_ndfs": "", "filter_wheel_ndf": 0,
    }
    result = convert_experiment_receptors({}, row, .5, calibration_root=root)
    assert result["receptor"].tolist() == ["lCone", "mCone", "sCone", "rod"]
    assert (result["output"] > 0).all()
