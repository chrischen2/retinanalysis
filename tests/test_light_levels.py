import pytest

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
