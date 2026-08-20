import matplotlib

matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pytest

from retinanalysis.utils import style


@pytest.fixture(autouse=True)
def restore_rcparams():
    original = mpl.rcParams.copy()
    yield
    mpl.rcParams.update(original)
    plt.close('all')


def test_publication_style_matches_igor_defaults_and_accepts_overrides():
    style.apply_publication_style()

    assert mpl.rcParams['font.size'] == 10
    assert mpl.rcParams['font.weight'] == 'light'
    assert mpl.rcParams['font.family'][:2] == ['Helvetica Neue', 'Helvetica']
    assert mpl.rcParams['axes.linewidth'] == pytest.approx(0.5)
    assert mpl.rcParams['xtick.major.size'] == pytest.approx(2)
    assert mpl.rcParams['xtick.major.pad'] == pytest.approx(-1.5)
    assert mpl.rcParams['lines.linewidth'] == pytest.approx(1)
    assert mpl.rcParams['legend.frameon']

    style.apply_publication_style(font_size=9, rc={'axes.facecolor': '#eeeeee'})
    assert mpl.rcParams['font.size'] == 9
    assert mpl.rcParams['axes.facecolor'] == '#eeeeee'


def test_format_figure_applies_ticks_axes_margins_lines_font_and_legend():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 10], [0, 20], lw=4, label='response')
    ax.set_xlabel('time')
    ax.set_ylabel('rate')
    returned = style.format_figure(fig, legend=True)

    assert returned is fig
    assert isinstance(ax.xaxis.get_major_locator(), MaxNLocator)
    assert ax.xaxis.get_major_locator()._nbins == 3
    assert ax.spines['left'].get_linewidth() == pytest.approx(.5)
    assert ax.lines[0].get_linewidth() == pytest.approx(1)
    assert ax.xaxis.majorTicks[0].get_pad() == pytest.approx(-1.5)
    assert ax.xaxis.label.get_fontsize() == pytest.approx(10)
    assert ax.xaxis.label.get_fontweight() == 'light'
    assert ax.get_legend().get_frame_on()
    assert fig.subplotpars.left == pytest.approx(35 / (4 * 72))
    assert fig.subplotpars.bottom == pytest.approx(27 / (3 * 72))
    base = ax.get_subplotspec().get_position(fig)
    assert ax.get_position().x0 == pytest.approx(base.x0 + .05 * base.width)
    assert ax.get_position().y0 == pytest.approx(base.y0 + .05 * base.height)
    assert ax.get_position().x1 == pytest.approx(base.x1)
    assert ax.get_position().y1 == pytest.approx(base.y1)


def test_format_figure_preserves_fixed_ticks_and_allows_specific_overrides():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xticks([0, 1], ['disc', 'cone'])
    fixed_locator = ax.xaxis.get_major_locator()

    style.format_figure(
        fig, axis_enable=None, margins_pt=None, line_width=None,
        font_size=8, nticks=5, legend=False)

    assert ax.xaxis.get_major_locator() is fixed_locator
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ['disc', 'cone']
    assert ax.xaxis.label.get_fontsize() == pytest.approx(8)


def test_format_figure_formats_figure_level_legends():
    fig, ax = plt.subplots(figsize=(4, 3))
    line, = ax.plot([0, 1], label='response')
    figure_legend = fig.legend([line], ['response'])

    style.format_figure(fig, margins_pt=None, axis_enable=None,
                        font_size=9, legend_frame=True)

    assert figure_legend.get_frame_on()
    assert figure_legend.get_frame().get_linewidth() == pytest.approx(.5)
    assert figure_legend.get_texts()[0].get_fontsize() == pytest.approx(9)
