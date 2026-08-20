"""Tests for utils.igor_export — the Python side of makeAxisStructChris.m.

The on-disk conventions asserted here were read off a real MATLAB-exported file
(projectHDF5s/linCone/linDiscSpike10kMidget.h5): one group named after the file
stem, scalars as shape-(1,) float64, vectors as (N,), errorbar deltas as (N,2),
strings as fixed-length bytes.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from retinanalysis.utils import igor_export  # noqa: E402


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_sanitize_follows_matlab_rules():
    assert igor_export._sanitize('mean SEM', 1) == 'mean_SEM'
    assert igor_export._sanitize('DoG fit', 1) == 'DoG_fit'
    assert igor_export._sanitize('40', 1) == 'n40'          # numeric -> 'n' prefix
    assert igor_export._sanitize('a__b', 1) == 'a_b'        # collapse underscores
    assert igor_export._sanitize('', 7) == 'L007'           # unlabeled
    assert igor_export._sanitize('_child0', 3) == 'L003'    # matplotlib auto label
    assert igor_export._sanitize('!!!', 2) == 'wave2'       # nothing survives


def test_line_export_fields(ax):
    ax.plot([1, 2, 3], [4, 5, 6], label='trace', color='#ff0000', lw=2.0,
            ls='--', marker='o', ms=7)
    s = igor_export.axis_to_dict(ax)
    assert s['trace_X'].tolist() == [1, 2, 3]
    assert s['trace_Y'].tolist() == [4, 5, 6]
    assert np.allclose(s['trace_color'], [1, 0, 0])
    assert s['trace_linestyle'] == 1        # dashed
    assert s['trace_linewidth'] == 2.0
    assert s['trace_markerNum'] == 8        # 'o' -> 8, per makeAxisStructChris.m
    assert s['trace_markerSize'] == 7.0
    assert s['lineNames'] == ['trace']


def test_long_even_x_becomes_start_delta(ax):
    """MATLAB stores >500-point evenly sampled X as start/delta, not an X wave."""
    t = np.arange(0, 1, 0.001)             # 1000 points, constant step
    ax.plot(t, np.sin(t), label='psth')
    s = igor_export.axis_to_dict(ax)
    assert 'psth_X' not in s
    assert s['psth_start'] == pytest.approx(0.0)
    assert s['psth_delta'] == pytest.approx(0.001)
    assert len(s['psth_Y']) == 1000


def test_long_uneven_x_keeps_x_wave(ax):
    x = np.sort(np.random.RandomState(0).uniform(0, 1, 600))
    ax.plot(x, x, label='uneven')
    s = igor_export.axis_to_dict(ax)
    assert 'uneven_X' in s and 'uneven_start' not in s


def test_scatter_exports_marker_mode(ax):
    ax.scatter([1, 2], [3, 4], s=20, color='#00ff00', label='pts')
    s = igor_export.axis_to_dict(ax)
    assert s['pts_X'].tolist() == [1, 2]
    assert s['pts_mode'] == 3 and s['pts_marker'] == 19
    assert s['pts_markerSize'] == 20.0
    assert np.allclose(s['pts_markercolor'], [0, 1, 0])


def test_scatter_preserves_per_point_sizes_and_colors(ax):
    ax.scatter([1, 2], [3, 4], s=[10, 30],
               c=[[1, 0, 0], [0, 0, 1]], label='cells')

    s = igor_export.axis_to_dict(ax)

    assert s['cells_markerSize'].tolist() == [10, 30]
    assert np.allclose(s['cells_markercolor'], [[1, 0, 0], [0, 0, 1]])


def test_errorbar_yerr_recovered_from_segments(ax):
    y = np.array([1.0, 2.0, 3.0])
    err = np.array([0.1, 0.2, 0.3])
    ax.errorbar([1, 2, 3], y, yerr=err, fmt='o', label='mean',
                ecolor='red', elinewidth=1.5, capsize=4)
    s = igor_export.axis_to_dict(ax)
    assert 'meanerrY_Yerr' in s
    yerr = s['meanerrY_Yerr']
    assert yerr.shape == (3, 2)                     # [negative, positive]
    assert np.allclose(yerr[:, 0], err)
    assert np.allclose(yerr[:, 1], err)
    assert 'meanerrY_Xerr' not in s
    assert np.allclose(s['meanerrY_errorBarColor'], [1, 0, 0])
    assert s['meanerrY_errorBarWidth'] == pytest.approx(1.5)
    assert s['meanerrY_capSize'] == pytest.approx(4)


def test_errorbar_xerr_gets_errx_suffix(ax):
    ax.errorbar([1, 2], [1, 2], xerr=[0.5, 0.5], fmt='o', label='h')
    s = igor_export.axis_to_dict(ax)
    assert 'herrX_Xerr' in s
    assert np.allclose(s['herrX_Xerr'], 0.5)


def test_eventplot_rows_merge_into_one_raster(ax):
    ax.eventplot([np.array([0.1, 0.2]), np.array([0.3])], lineoffsets=[0, 1])
    s = igor_export.axis_to_dict(ax)
    assert s['raster_X'].tolist() == [0.1, 0.2, 0.3]
    assert s['raster_Y'].tolist() == [0.0, 0.0, 1.0]
    assert s['raster_mode'] == 3
    assert s['lineNames'] == ['raster']


def test_duplicate_labels_do_not_overwrite(ax):
    ax.plot([1, 2], [1, 2], label='dup')
    ax.plot([1, 2], [3, 4], label='dup')
    s = igor_export.axis_to_dict(ax)
    assert s['dup_Y'].tolist() == [1, 2]
    assert s['dup2_Y'].tolist() == [3, 4]
    assert s['lineNames'] == ['dup', 'dup2']


def test_axvspan_is_not_exported(ax):
    ax.plot([1, 2], [1, 2], label='keep')
    ax.axvspan(1.2, 1.8, color='y', alpha=0.2)
    s = igor_export.axis_to_dict(ax)
    assert s['lineNames'] == ['keep']


def test_axis_level_fields(ax):
    ax.plot([1, 2], [1, 2])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (Hz)')
    ax.set_yscale('log')
    ax.set_title('my title')
    s = igor_export.axis_to_dict(ax)
    assert s['Xlabel'] == 'Time (s)' and s['Ylabel'] == 'Rate (Hz)'
    assert s['Yscale'] == 1 and s['Xscale'] == 0
    assert s['FigureTitle'] == 'my title'
    assert len(s['Xlim']) == 2 and len(s['Ylim']) == 2
    assert s['XTickLabel'] and isinstance(s['XTickLabel'][0], str)
    assert s['HasColorbar'] == 0


def test_histogram_and_bar_containers_export_as_single_waves():
    fig, (hist_ax, bar_ax) = plt.subplots(1, 2)
    hist_ax.hist([0, 0, 1, 2], bins=[0, 1, 2, 3], label='counts')
    bar_ax.bar([1, 2], [3, 4], label='cells')

    hist = igor_export.axis_to_dict(hist_ax)
    bars = igor_export.axis_to_dict(bar_ax)

    assert hist['countshist_type'] == 'histogram'
    assert hist['countshist_mode'] == 6
    assert hist['countshist_numBins'] == 3
    assert hist['countshist_binLimits'].tolist() == [0, 3]
    assert hist['lineNames'] == ['countshist']
    assert bars['cellsbar_type'] == 'bar'
    assert bars['cellsbar_X'].tolist() == [1, 2]
    assert bars['cellsbar_Y'].tolist() == [3, 4]
    plt.close(fig)


def test_contour_exports_levels_paths_and_colormap(ax):
    z = np.arange(16, dtype=float).reshape(4, 4)
    ax.contour(z, levels=[3, 6, 9], cmap='viridis')

    s = igor_export.axis_to_dict(ax)
    prefix = s['lineNames'][0]

    assert prefix.startswith('contour')
    assert s[f'{prefix}_type'] == 'contour'
    assert s[f'{prefix}_LevelList'].tolist() == [3, 6, 9]
    assert s[f'{prefix}_X'].shape == s[f'{prefix}_Y'].shape
    assert s[f'{prefix}_cmap'].shape == (256, 3)


def test_3d_line_exports_z_limits_view_and_projection():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot([0, 1], [2, 3], [4, 5], label='trajectory')
    ax.set_zlabel('depth')
    ax.view_init(elev=20, azim=35)

    s = igor_export.axis_to_dict(ax)

    assert s['trajectory_Z'].tolist() == [4, 5]
    assert s['Zlabel'] == 'depth'
    assert len(s['Zlim']) == 2
    assert s['view_azimuth'] == pytest.approx(35)
    assert s['view_elevation'] == pytest.approx(20)
    assert s['Projection'] in ('orthographic', 'perspective')
    plt.close(fig)


def test_image_colorbar_metadata_and_colorbar_axis_is_not_exported(tmp_path):
    fig, ax = plt.subplots()
    image = ax.imshow(np.arange(9).reshape(3, 3), vmin=0, vmax=8)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label('contrast')

    s = igor_export.axis_to_dict(ax)
    paths = igor_export.export_figure_to_h5(
        fig, 'imageFigure', basedir=tmp_path, verbose=False)

    assert s['HasColorbar'] == 1
    assert s['ColorbarLabel'] == 'contrast'
    assert s['ColorbarLimits'].tolist() == [0, 8]
    assert len(paths) == 1 and paths[0].name == 'imageFigure.h5'
    plt.close(fig)


def test_hdf5_layout_matches_matlab(ax, tmp_path):
    """One group named after the file stem; MATLAB's dtypes and shapes."""
    import h5py

    ax.plot(np.arange(3.0), np.arange(3.0), label='trace', lw=1.5)
    ax.errorbar([1, 2], [1, 2], yerr=[0.1, 0.1], fmt='o', label='mean')
    ax.set_xlabel('x')
    path = igor_export.export_axis_to_h5(ax, 'figA', basedir=tmp_path, verbose=False)
    assert path == tmp_path / 'figA.h5'

    with h5py.File(path) as f:
        assert list(f.keys()) == ['figA']       # group name == file stem
        g = f['figA']
        assert g['trace_linewidth'].shape == (1,)          # scalars are (1,)
        assert g['trace_linewidth'].dtype == np.float64
        assert g['trace_X'].shape == (3,)                  # vectors are (N,)
        assert g['meanerrY_Yerr'].shape == (2, 2)          # deltas are (N,2)
        assert g['Xlabel'].dtype.kind == 'S'               # fixed-length bytes
        assert g['Xlabel'].shape == (1,)
        assert g['Xlabel'][()][0].decode() == 'x'
        assert g['lineNames'].dtype.kind == 'S'


def test_export_figure_one_file_per_axes(tmp_path):
    fig, axes = plt.subplots(2, 1)
    axes[0].plot([1, 2], [1, 2], label='top')
    axes[1].plot([1, 2], [2, 1], label='bottom')
    paths = igor_export.export_figure_to_h5(fig, 'multi', basedir=tmp_path, verbose=False)
    assert [p.name for p in paths] == ['multi1.h5', 'multi2.h5']
    plt.close(fig)


def test_export_figure_single_axes_keeps_plain_name(tmp_path):
    fig, ax1 = plt.subplots()
    ax1.plot([1, 2], [1, 2], label='only')
    paths = igor_export.export_figure_to_h5(fig, 'solo', basedir=tmp_path, verbose=False)
    assert [p.name for p in paths] == ['solo.h5']
    plt.close(fig)


def test_overwrite_guard(ax, tmp_path):
    ax.plot([1, 2], [1, 2], label='a')
    igor_export.export_axis_to_h5(ax, 'once', basedir=tmp_path, verbose=False)
    with pytest.raises(FileExistsError):
        igor_export.export_axis_to_h5(ax, 'once', basedir=tmp_path, overwrite=False,
                                      verbose=False)


def test_igor_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv('RA_IGOR_DIR', str(tmp_path / 'igor'))
    assert igor_export.igor_dir() == tmp_path / 'igor'
    monkeypatch.delenv('RA_IGOR_DIR')
    assert igor_export.igor_dir().name == 'igor_h5'


def test_top_level_igor_output_accepts_figure_or_axis(tmp_path):
    import retinanalysis as ra

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 2], label='trace')

    figure_paths = ra.igor_output(
        fig, 'fromFigure', basedir=tmp_path, verbose=False)
    axis_paths = ra.igor_output(
        ax, 'fromAxis', basedir=tmp_path, verbose=False)

    assert [path.name for path in figure_paths] == ['fromFigure.h5']
    assert [path.name for path in axis_paths] == ['fromAxis.h5']
    assert ra.igor_axis_struct(ax)['lineNames'] == ['trace']
    with pytest.raises(TypeError):
        ra.igor_output(object(), 'bad', basedir=tmp_path, verbose=False)
    plt.close(fig)
