import numpy as np
import pandas as pd

from retinanalysis.utils.cycle_average import (
    browse_cell_psths,
    summarize_template_outliers,
)


def _row(cell_id, condition, passes, *, evaluable=True, cell_type='OnM'):
    return {
        'cell_id': cell_id,
        'cell_type': cell_type,
        'condition': condition,
        'passes_template': passes,
        'template_evaluable': evaluable,
    }


def test_cross_condition_template_qc_is_conservative_and_propagates_labels():
    match = pd.DataFrame([
        _row(1, 'a', False), _row(1, 'b', False), _row(1, 'c', True),
        _row(2, 'a', False), _row(2, 'b', True),
        _row(3, 'a', False, evaluable=False),
    ])

    annotated, summary = summarize_template_outliers(
        match, candidate_cell_ids=[1, 2, 3, 4], min_conditions=2,
        min_pass_fraction=0.5, verbose=False,
    )
    by_id = summary.set_index('cell_id')

    assert bool(by_id.at[1, 'excluded_downstream'])
    assert not bool(by_id.at[2, 'excluded_downstream'])  # exactly half passes
    assert by_id.at[3, 'qc_status'] == 'kept_unscored'
    assert by_id.at[4, 'qc_status'] == 'kept_unscored'
    assert np.isnan(by_id.at[4, 'template_pass_fraction'])
    assert annotated.groupby('cell_id')['excluded_downstream'].first().to_dict() == {
        1: True, 2: False, 3: False,
    }


def test_cross_condition_template_qc_validates_inputs():
    match = pd.DataFrame([_row(1, 'a', True)])
    try:
        summarize_template_outliers(match, min_conditions=0, verbose=False)
    except ValueError as exc:
        assert 'min_conditions' in str(exc)
    else:
        raise AssertionError('expected min_conditions validation')


def test_psth_browser_accepts_an_empty_outlier_list(monkeypatch):
    import matplotlib.pyplot as plt
    import retinanalysis.utils.browse as browse

    def render_first(options, render, **kwargs):
        return render(options[0][1])

    def close_figure(fig):
        plt.close(fig)
        return b'png'

    monkeypatch.setattr(browse, 'png_browser', render_first)
    monkeypatch.setattr(browse, 'figure_to_png', close_figure)
    cells = pd.DataFrame({'cell_id': [1], 'cell_type': ['OnM']})

    saved = {}
    body, png = browse_cell_psths(
        cells, np.array([0.0, 1.0]), np.array([[1.0, 2.0]]),
        outlier_cell_ids=[], n_cols=1, figure_sink=saved,
    )

    assert body is None
    assert png == b'png'
    assert list(saved) == ['OnM']
    assert hasattr(saved['OnM'], 'savefig')
