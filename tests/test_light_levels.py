import pytest
import pandas as pd

import retinanalysis as ra
from retinanalysis.utils.light_levels import VISUAL_STIMULUS_MAX


@pytest.mark.parametrize(
    "rig, ndfs, wheel, expected",
    [
        ("E", None, 0, 30_000_000),
        ("rig E", "EL3", "FW0.5", 10_000),
        ("E", "EL3+FW1", None, 3_000),
        ("2026-08-10_G", ["EL2", "EL03"], 1, 6_200),
        ("G", '["EL06", "EL2", "FW0.5"]', None, 24_000),
        ("chris_data", "EL06 + EL2", 0.5, 24_000),
        ("G", "EL2+EL06", "FW1.0", 7_600),
    ],
)
def test_visual_stimulus_max_exact_measurements(rig, ndfs, wheel, expected):
    assert ra.visual_stimulus_max(rig, ndfs, wheel) == expected


def test_visual_stimulus_max_infers_only_filter_wheel_attenuation():
    assert ra.visual_stimulus_max("E", "EL3", 2) == pytest.approx(300)
    assert ra.visual_stimulus_max("chris_data", "EL2", 1.5) == pytest.approx(10_000)


def test_exact_measurement_precedes_nominal_wheel_inference():
    assert ra.visual_stimulus_max("E", "EL3", 3) == 60
    assert ra.visual_stimulus_max("G", "EL03+EL2", 0.5) == 19_000


def test_unmeasured_fixed_filter_combination_is_not_inferred():
    with pytest.raises(KeyError, match="No measured visual-stimulus maximum",):
        ra.visual_stimulus_max("E", "EL2", 1)


def test_exact_only_mode_rejects_unmeasured_wheel_setting():
    with pytest.raises(KeyError, match="No exact visual-stimulus maximum"):
        ra.visual_stimulus_max("G", "EL03+EL2", 2, infer_filter_wheel=False)


def test_lookup_table_is_available_from_utility_module():
    assert VISUAL_STIMULUS_MAX["E"][(('EL3',), 4.0)] == 6


def test_every_lookup_entry_round_trips_through_function():
    for rig, measurements in VISUAL_STIMULUS_MAX.items():
        for (filters, wheel), expected in measurements.items():
            assert ra.visual_stimulus_max(rig, filters, wheel) == expected


def test_conflicting_embedded_and_explicit_wheel_is_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        ra.visual_stimulus_max("E", "EL3+FW1", 0.5)


def test_block_filter_wheel_table_consolidates_all_epochs():
    from retinanalysis.utils.light_levels import _block_filter_wheel_table

    epochs = [
        {"id": 1, "parent_id": 10, "parameters": {"NDF": 0.5}},
        {"id": 2, "parent_id": 10, "parameters": {"NDF": "0.5"}},
        {"id": 3, "parent_id": 11, "parameters": {}},
    ]
    result = _block_filter_wheel_table([10, 11, 12], epoch_rows=epochs)

    assert result.loc[0].to_dict() == {
        "block_id": 10,
        "filter_wheel_ndf": 0.5,
        "filter_wheel_status": "recorded",
        "n_epochs": 2,
        "n_filter_wheel_readings": 2,
    }
    assert result.loc[1, "filter_wheel_status"] == "not recorded"
    assert result.loc[2, "filter_wheel_status"] == "no epochs"


def test_block_filter_wheel_table_rejects_conflict_within_block():
    from retinanalysis.utils.light_levels import _block_filter_wheel_table

    epochs = [
        {"id": 1, "parent_id": 10, "parameters": {"NDF": 0.5}},
        {"id": 2, "parent_id": 10, "parameters": {"NDF": 1.0}},
    ]
    with pytest.raises(ValueError, match="block 10.*0.5, 1"):
        _block_filter_wheel_table([10], epoch_rows=epochs)


def test_read_block_light_settings_combines_trusted_sources(monkeypatch):
    import retinanalysis.SCutils.recording_mode as recording_mode
    import retinanalysis.utils.light_levels as light_levels

    blocks = pd.DataFrame({
        "exp_name": ["2026-08-01_E", "2026-08-10_G"],
        "block_id": [10, 20],
    })
    monkeypatch.setattr(
        recording_mode,
        "stage_ndf_table",
        lambda *_args, **_kwargs: pd.DataFrame({
            "block_id": [10, 20],
            "stage_ndfs": ["EL3, FW4", "EL2, EL06, FW1"],
        }),
    )
    monkeypatch.setattr(
        light_levels,
        "_block_filter_wheel_table",
        lambda *_args, **_kwargs: pd.DataFrame({
            "block_id": [10, 20],
            "filter_wheel_ndf": [0.5, 1.0],
            "filter_wheel_status": ["recorded", "recorded"],
            "n_epochs": [3, 4],
            "n_filter_wheel_readings": [3, 4],
        }),
    )

    result = ra.read_block_light_settings(blocks, verbose=False)

    assert result[["block_id", "rig", "fixed_ndfs", "filter_wheel_ndf",
                   "ndf_combination"]].to_dict("records") == [
        {"block_id": 10, "rig": "E", "fixed_ndfs": ("EL3",),
         "filter_wheel_ndf": 0.5, "ndf_combination": "EL3 + FW0.5"},
        {"block_id": 20, "rig": "G", "fixed_ndfs": ("EL06", "EL2"),
         "filter_wheel_ndf": 1.0, "ndf_combination": "EL06 + EL2 + FW1"},
    ]
    assert result["ignored_stage_fw_tokens"].tolist() == [("FW4",), ("FW1",)]
    assert ra.visual_stimulus_max(
        result.loc[1, "rig"], result.loc[1, "fixed_ndfs"],
        result.loc[1, "filter_wheel_ndf"],
    ) == 7_600


def test_read_block_light_settings_empty_input():
    result = ra.read_block_light_settings(
        pd.DataFrame(columns=["exp_name", "block_id"]), verbose=False)
    assert result.empty
    assert "ndf_combination" in result
