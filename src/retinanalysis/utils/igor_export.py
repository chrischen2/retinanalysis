"""Export a matplotlib axes to an Igor-loadable HDF5 file.

Python counterpart of ``makeAxisStructChris.m``. It walks the artists of a
matplotlib ``Axes`` the way the MATLAB walks ``get(ax,'children')``, and writes
the same flat set of datasets, so the existing Igor procedures
(``DisplayFigFromMatlab``) load a Python figure exactly like a MATLAB one.

    from retinanalysis.utils import igor_export

    fig = es.plot_expanding_spots(res)
    igor_export.export_figure_to_h5(fig, 'expSpots2026_06_04_Cell2')

File layout, matching what MATLAB's ``hdf5write`` produces:

    /<file stem>/                      one group, named after the file
        Xlabel, Ylabel                 (1,)  fixed-length bytes
        Xlim, Ylim                     (2,)  float64
        Xscale, Yscale                 (1,)  float64, 1 = log
        lineNames                      (n,)  fixed-length bytes
        <prefix>_waveName              (1,)  fixed-length bytes
        <prefix>_X, <prefix>_Y         (N,)  float64
        <prefix>_Yerr                  (N,2) float64  [negative, positive]
        <prefix>_color                 (3,)  float64  RGB in 0-1
        <prefix>_linewidth ...         (1,)  float64

Igor's loader finds plots by listing every wave named ``*_Y`` and then looking
for the matching ``<prefix>_X``, ``<prefix>_color``, and so on — so the dataset
names are the contract, not the order.

Three things to know when building a figure you intend to export:

* **Label your artists.** ``plot(..., label='mean')`` becomes the Igor wave name
  (sanitized). Unlabeled artists fall back to ``L001``, ``L002``, ...
* **The figure is drawn first.** Tick labels and auto limits do not exist until
  a draw, so :func:`export_axis_to_h5` calls ``fig.canvas.draw()``.
* **Only data artists are exported** — lines, scatters, errorbars, event
  (raster) collections and images. Shaded ``axvspan`` regions, text and legends
  are decoration; recreate those in Igor.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ['export_axis_to_h5', 'export_figure_to_h5', 'axis_to_dict', 'igor_dir']

# makeAxisStructChris.m marker mapping (matplotlib marker -> Igor marker number).
MARKER_NUM = {'o': 8, '+': 0, '.': 19, '^': 17, 'x': 1, '*': 2, 's': 16, 'D': 18}
DEFAULT_MARKER_NUM = 19

# Igor line styles: 0 solid, 1 dashed.
LINESTYLE_NUM = {'-': 0, 'solid': 0, '--': 1, 'dashed': 1}

# X waves longer than this with a constant step are stored as start/delta, as
# the MATLAB does, instead of a full X wave.
TIMESERIES_MIN_POINTS = 500


def igor_dir() -> Path:
    """Where exported .h5 files land by default.

    ``RA_IGOR_DIR`` if set, else ``<OUTPUT_DIR>/igor_h5``. Point ``RA_IGOR_DIR``
    at the Igor project folder (the MATLAB hardcodes ``projectHDF5s/<project>/``)
    to drop files straight where Igor reads them.
    """
    env = os.environ.get('RA_IGOR_DIR', '')
    if env:
        return Path(env).expanduser()
    from retinanalysis.config.settings import OUTPUT_DIR
    return Path(OUTPUT_DIR) / 'igor_h5'


def _sanitize(label: str, index: int) -> str:
    """Igor/MATLAB-safe wave prefix, following makeAxisStructChris.m."""
    if not label or label.startswith('_'):  # matplotlib's auto labels
        return f'L{index:03d}'
    name = re.sub(r'[^a-zA-Z0-9_]', '', label)
    name = re.sub(r'_{2,}', '_', name)
    if not name:
        return f'wave{index}'
    try:  # a purely numeric label gets an 'n' so it stays a valid name
        float(name)
        return 'n' + name
    except ValueError:
        pass
    if not name[0].isalpha():
        name = 'w' + name
    return name


def _rgb(color) -> np.ndarray:
    from matplotlib.colors import to_rgb
    return np.asarray(to_rgb(color), dtype=float)


def _xy_fields(prefix: str, x, y) -> Dict[str, object]:
    """X/Y datasets, collapsing a long evenly-sampled X to start + delta."""
    out: Dict[str, object] = {}
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size > TIMESERIES_MIN_POINTS:
        d = np.diff(x)
        if d.size and np.allclose(d, d[0]):
            out[f'{prefix}_start'] = float(x[0])
            out[f'{prefix}_delta'] = float(d[0])
        else:
            out[f'{prefix}_X'] = x
    else:
        out[f'{prefix}_X'] = x
    out[f'{prefix}_Y'] = y
    return out


def _line_style_fields(prefix: str, artist) -> Dict[str, object]:
    out: Dict[str, object] = {}
    ls = artist.get_linestyle()
    if ls and ls != 'None':
        out[f'{prefix}_linestyle'] = LINESTYLE_NUM.get(ls, 0)
    try:
        out[f'{prefix}_color'] = _rgb(artist.get_color())
    except (AttributeError, ValueError):
        pass
    lw = artist.get_linewidth()
    if lw:
        out[f'{prefix}_linewidth'] = float(lw)
    marker = artist.get_marker()
    if marker and marker not in ('None', 'none', ''):
        out[f'{prefix}_markerNum'] = MARKER_NUM.get(marker, DEFAULT_MARKER_NUM)
        out[f'{prefix}_markerSize'] = float(artist.get_markersize())
        face = artist.get_markerfacecolor()
        if face and face not in ('none', 'auto'):
            out[f'{prefix}_markercolor'] = _rgb(face)
    return out


def _errorbar_deltas(bars, x, y):
    """Pull (neg, pos) deltas out of an errorbar's LineCollections.

    matplotlib keeps no yerr attribute, so read the drawn segments back:
    a vertical segment at constant x carries the y error and vice versa.
    """
    xerr = yerr = None
    for coll in bars:
        segs = coll.get_segments()
        if not segs:
            continue
        seg = np.array([[[s[0][0], s[0][1]], [s[-1][0], s[-1][1]]] for s in segs])
        dx = np.abs(seg[:, 1, 0] - seg[:, 0, 0])
        dy = np.abs(seg[:, 1, 1] - seg[:, 0, 1])
        if np.all(dx <= dy):  # vertical bars -> y error
            lo, hi = seg[:, 0, 1], seg[:, 1, 1]
            yerr = np.column_stack([y[:len(lo)] - lo, hi - y[:len(hi)]])
        else:                  # horizontal bars -> x error
            lo, hi = seg[:, 0, 0], seg[:, 1, 0]
            xerr = np.column_stack([x[:len(lo)] - lo, hi - x[:len(hi)]])
    return xerr, yerr


def axis_to_dict(ax, figure_title: Optional[str] = None) -> Dict[str, object]:
    """Flatten one matplotlib Axes into the field dict the Igor loader expects.

    Exposed separately from the HDF5 write so the mapping can be inspected or
    tested without touching disk.
    """
    from matplotlib.collections import EventCollection, LineCollection, PathCollection
    from matplotlib.container import ErrorbarContainer
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D

    ax.figure.canvas.draw()  # tick labels / auto limits are only real after a draw

    s: Dict[str, object] = {}
    used: set = set()

    def unique(prefix: str) -> str:
        """Two artists sharing a label would otherwise overwrite each other."""
        if prefix not in used:
            used.add(prefix)
            return prefix
        n = 2
        while f'{prefix}{n}' in used:
            n += 1
        used.add(f'{prefix}{n}')
        return f'{prefix}{n}'

    if ax.get_xlabel():
        s['Xlabel'] = ax.get_xlabel()
    if ax.get_ylabel():
        s['Ylabel'] = ax.get_ylabel()
    s['Xlim'] = np.asarray(ax.get_xlim(), dtype=float)
    s['Ylim'] = np.asarray(ax.get_ylim(), dtype=float)
    s['Zlim'] = np.array([-1.0, 1.0])           # MATLAB writes these for 2-D axes too
    s['Xscale'] = 1 if ax.get_xscale() == 'log' else 0
    s['Yscale'] = 1 if ax.get_yscale() == 'log' else 0
    s['Zscale'] = 0
    s['view_azimuth'] = 0
    s['view_elevation'] = 90

    names: List[str] = []
    index = 0

    # Errorbar containers first: their child Line2D/LineCollections must not be
    # exported a second time as plain lines.
    claimed = set()
    for container in ax.containers:
        if not isinstance(container, ErrorbarContainer):
            continue
        data_line, caps, bars = container.lines
        claimed.add(id(data_line))
        for group in (caps or (), bars or ()):
            for a in group:
                claimed.add(id(a))
        index += 1
        x = np.asarray(data_line.get_xdata(), dtype=float)
        y = np.asarray(data_line.get_ydata(), dtype=float)
        xerr, yerr = _errorbar_deltas(bars or (), x, y)
        prefix = unique(_sanitize(container.get_label(), index)
                        + ('errX' if xerr is not None and yerr is None else 'errY'))
        s.update(_xy_fields(prefix, x, y))
        s.update(_line_style_fields(prefix, data_line))
        if yerr is not None:
            s[f'{prefix}_Yerr'] = yerr
        if xerr is not None:
            s[f'{prefix}_Xerr'] = xerr
        s[f'{prefix}_waveName'] = prefix
        names.append(prefix)

    # eventplot() makes one EventCollection per row; a raster is far more useful
    # in Igor as a single marker trace than as one wave per epoch, so merge the
    # collections that share a label (matplotlib's auto labels -> one 'raster').
    event_groups: Dict[str, List[np.ndarray]] = {}
    event_colors: Dict[str, np.ndarray] = {}
    for artist in ax.get_children():
        if not isinstance(artist, EventCollection) or id(artist) in claimed:
            continue
        claimed.add(id(artist))
        segs = artist.get_segments()
        if not segs:
            continue
        label = artist.get_label()
        key = 'raster' if (not label or label.startswith('_')) else _sanitize(label, 0)
        pts = np.array([[np.mean([p[0] for p in seg]), np.mean([p[1] for p in seg])]
                        for seg in segs])
        event_groups.setdefault(key, []).append(pts)
        colors = np.atleast_2d(np.asarray(artist.get_color(), dtype=float))
        if colors.size:
            event_colors.setdefault(key, colors[0][:3])

    for key, chunks in event_groups.items():
        pts = np.vstack(chunks)
        prefix = unique(key)
        s.update(_xy_fields(prefix, pts[:, 0], pts[:, 1]))
        s[f'{prefix}_mode'] = 3      # Igor: markers only, no connecting line
        s[f'{prefix}_marker'] = 19
        s[f'{prefix}_markerSize'] = 1.5
        if key in event_colors:
            s[f'{prefix}_markercolor'] = event_colors[key]
        s[f'{prefix}_waveName'] = prefix
        names.append(prefix)

    for artist in ax.get_children():
        if id(artist) in claimed:
            continue

        if isinstance(artist, Line2D):
            x, y = artist.get_xdata(), artist.get_ydata()
            if len(np.atleast_1d(x)) == 0:
                continue
            index += 1
            prefix = unique(_sanitize(artist.get_label(), index))
            s.update(_xy_fields(prefix, x, y))
            s.update(_line_style_fields(prefix, artist))

        elif isinstance(artist, PathCollection):  # scatter
            offsets = np.asarray(artist.get_offsets(), dtype=float)
            if offsets.size == 0:
                continue
            index += 1
            prefix = unique(_sanitize(artist.get_label(), index))
            s.update(_xy_fields(prefix, offsets[:, 0], offsets[:, 1]))
            s[f'{prefix}_mode'] = 3          # Igor: markers only
            s[f'{prefix}_marker'] = 19
            sizes = np.asarray(artist.get_sizes(), dtype=float)
            if sizes.size:
                s[f'{prefix}_markerSize'] = float(sizes[0])
            face = artist.get_facecolor()
            if len(face):
                s[f'{prefix}_markercolor'] = np.asarray(face[0][:3], dtype=float)

        elif isinstance(artist, LineCollection):  # e.g. vlines / hlines
            segs = artist.get_segments()
            if not segs:
                continue
            index += 1
            prefix = unique(_sanitize(artist.get_label(), index))
            # One point per segment center; Igor has no segment-collection type.
            pts = np.array([[np.mean([p[0] for p in seg]),
                             np.mean([p[1] for p in seg])] for seg in segs])
            s.update(_xy_fields(prefix, pts[:, 0], pts[:, 1]))
            s[f'{prefix}_mode'] = 3
            s[f'{prefix}_marker'] = 19
            s[f'{prefix}_markerSize'] = 1.5
            # get_color() is a single RGBA for a uniformly colored collection
            # and an (n, 4) array when the events differ.
            colors = np.atleast_2d(np.asarray(artist.get_color(), dtype=float))
            if colors.size:
                s[f'{prefix}_markercolor'] = colors[0][:3]

        elif isinstance(artist, AxesImage):
            index += 1
            prefix = f'imageplot{index}'
            x0, x1, y0, y1 = artist.get_extent()
            s[f'{prefix}_X'] = np.array([x0, x1], dtype=float)
            s[f'{prefix}_Y'] = np.array([y0, y1], dtype=float)
            s[f'{prefix}_CData'] = np.asarray(artist.get_array(), dtype=float)
            s[f'{prefix}_cmap'] = np.asarray(
                artist.get_cmap()(np.linspace(0, 1, 256))[:, :3], dtype=float)
            s[f'{prefix}_waveName'] = prefix
            names.append(prefix)
            continue

        else:
            continue

        s[f'{prefix}_waveName'] = prefix
        names.append(prefix)

    s['lineNames'] = names
    s['XAxisLocation'] = ax.xaxis.get_ticks_position() or 'bottom'
    s['YAxisLocation'] = ax.yaxis.get_ticks_position() or 'left'
    s['XTickLabel'] = [t.get_text() for t in ax.get_xticklabels()]
    s['YTickLabel'] = [t.get_text() for t in ax.get_yticklabels()]
    s['XTickLabelRotation'] = float(ax.get_xticklabels()[0].get_rotation()) if \
        len(ax.get_xticklabels()) else 0.0
    s['YTickLabelRotation'] = float(ax.get_yticklabels()[0].get_rotation()) if \
        len(ax.get_yticklabels()) else 0.0

    if figure_title is None:
        figure_title = ax.get_title() or (ax.figure._suptitle.get_text()
                                          if ax.figure._suptitle else '')
    s['FigureTitle'] = figure_title
    return s


def _write_group(group, fields: Dict[str, object]) -> None:
    """Write one flat dict as datasets, matching MATLAB hdf5write's shapes.

    Scalars become shape (1,) float64 and strings fixed-length bytes, which is
    what the MATLAB-written files contain (verified against an exported .h5).
    """
    for key, value in fields.items():
        if isinstance(value, str):
            if not value:
                continue  # h5py cannot make a zero-length string dtype
            group[key] = np.array([value.encode('utf-8')])
        elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            encoded = [v.encode('utf-8') for v in value if v]
            if not encoded:
                continue
            width = max(len(v) for v in encoded)
            group[key] = np.array(encoded, dtype=f'S{width}')
        else:
            arr = np.asarray(value, dtype=float)
            group[key] = np.atleast_1d(arr)


def export_axis_to_h5(ax, name: str, basedir: Optional[os.PathLike] = None,
                      figure_title: Optional[str] = None,
                      overwrite: bool = True, verbose: bool = True) -> Path:
    """Write one matplotlib Axes to ``<basedir>/<name>.h5`` for Igor.

    The HDF5 contains a single group named ``name`` — Igor's loader uses that as
    the data-folder name, so it must match the file stem, exactly as the MATLAB
    does with ``exportStructToHDF5(s, [fname '.h5'], fname, ...)``.
    """
    import h5py

    base = Path(basedir) if basedir is not None else igor_dir()
    base.mkdir(parents=True, exist_ok=True)
    path = base / f'{name}.h5'
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} exists; pass overwrite=True to replace it')

    fields = axis_to_dict(ax, figure_title=figure_title)
    with h5py.File(path, 'w') as f:
        _write_group(f.create_group(name), fields)
    if verbose:
        print(f'wrote {path}  ({len(fields["lineNames"])} waves: '
              f'{", ".join(fields["lineNames"]) or "none"})')
    return path


def export_figure_to_h5(fig, name: str, basedir: Optional[os.PathLike] = None,
                        overwrite: bool = True, verbose: bool = True) -> List[Path]:
    """Export every axes of a figure — one file per axes, as the MATLAB does.

    A single-axes figure writes ``<name>.h5``; multiple axes write
    ``<name>1.h5``, ``<name>2.h5``, ... in figure order.
    """
    axes = [a for a in fig.axes if a.has_data()]
    if not axes:
        raise ValueError('figure has no axes with data to export')
    out = []
    for i, ax in enumerate(axes, start=1):
        stem = name if len(axes) == 1 else f'{name}{i}'
        out.append(export_axis_to_h5(ax, stem, basedir=basedir,
                                     overwrite=overwrite, verbose=verbose))
    return out
