"""Cell-by-cell visual QC GUI for the per-cell PNG archive.

Pairs each cell's raster + PSTH PNGs (written by
``save_per_cell_plots`` under ``OUTPUT_DIR/<exp>/<protocol>/cells/...``)
side-by-side in a Jupyter notebook and lets the user tag each cell
``good`` or ``bad`` via buttons. Tags are persisted incrementally to
``<OUTPUT_DIR>/<exp>/<protocol>/visual_qc.csv`` so the GUI is resumable
across sessions.

Downstream population analyses load the tags via
:func:`load_visual_qc` to filter to visually-inspected cells.

Usage (inside a notebook)::

    import retinanalysis as ra
    ra.browse_cells_qc('20220823C')          # interactive
    qc = ra.load_visual_qc()                 # pandas DataFrame across all exps
    good = qc[qc.tag == 'good']
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config.settings import OUTPUT_DIR


__all__ = ['browse_cells_qc', 'load_visual_qc', 'visual_qc_csv_path',
           'select_good_cells']


VQC_COLUMNS = ['exp_name', 'cell_id', 'cell_type', 'tag',
               'timestamp', 'inspector']


def visual_qc_csv_path(exp_name: str, protocol: str,
                       output_root: Optional[str] = None) -> Path:
    """Path to the visual_qc.csv for one experiment + protocol."""
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    return root / exp_name / protocol / 'visual_qc.csv'


def _discover_cells(exp_root: Path, protocol: str) -> List[dict]:
    """Walk ``<exp_root>/<protocol>/cells/<celltype>/`` for raster+psth pairs."""
    cells: List[dict] = []
    proto_dir = exp_root / protocol / 'cells'
    if not proto_dir.is_dir():
        return cells
    for ctype_dir in sorted(proto_dir.iterdir()):
        if not ctype_dir.is_dir():
            continue
        rasters, psths = {}, {}
        for f in ctype_dir.iterdir():
            if not f.name.startswith('cell_') or f.suffix != '.png':
                continue
            parts = f.stem.split('_')
            if len(parts) < 3:
                continue
            try:
                cid = int(parts[1])
            except ValueError:
                continue
            kind = parts[2]
            if kind == 'raster':
                rasters[cid] = f
            elif kind == 'psth':
                psths[cid] = f
        for cid in sorted(set(rasters) & set(psths)):
            cells.append({
                'cell_id': cid,
                'cell_type': ctype_dir.name,
                'raster_path': str(rasters[cid]),
                'psth_path': str(psths[cid]),
            })
    return cells


def _load_tags(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for r in df.itertuples():
        out[int(r.cell_id)] = {
            'tag': r.tag,
            'timestamp': getattr(r, 'timestamp', ''),
            'inspector': getattr(r, 'inspector', ''),
        }
    return out


def _save_tag(csv_path: Path, exp_name: str, cell_id: int, cell_type: str,
              tag: str, inspector: str) -> None:
    """Upsert one row by ``cell_id`` and rewrite the CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df['cell_id'] != cell_id]
    else:
        df = pd.DataFrame(columns=VQC_COLUMNS)
    new_row = {
        'exp_name': exp_name, 'cell_id': cell_id, 'cell_type': cell_type,
        'tag': tag,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'inspector': inspector,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df[VQC_COLUMNS]
    df.to_csv(csv_path, index=False)


def browse_cells_qc(
    exp_name: str,
    protocol: str = 'eye_movement_alt_bg',
    output_root: Optional[str] = None,
    inspector: Optional[str] = None,
    image_width: int = 600,
):
    """Open the cell-by-cell QC GUI for one experiment.

    Loads previously-saved tags from
    ``<output_root>/<exp_name>/<protocol>/visual_qc.csv`` and writes
    each new tag back immediately on click.

    Parameters
    ----------
    exp_name : str
        Experiment date string (e.g. ``'20220823C'``).
    protocol : str
        Subdir name under the experiment root. Default ``'eye_movement_alt_bg'``.
    output_root : str, optional
        Override ``OUTPUT_DIR``.
    inspector : str, optional
        Name stored in each tag record. Defaults to ``$USER``.
    image_width : int
        Display width (px) for each of the two PNGs.
    """
    import ipywidgets as widgets
    from IPython.display import display

    if output_root is None:
        output_root = OUTPUT_DIR
    if inspector is None:
        inspector = os.environ.get('USER', 'unknown')

    exp_root = Path(output_root) / exp_name
    cells = _discover_cells(exp_root, protocol)
    if not cells:
        print(f'No cell PNGs found under {exp_root}/{protocol}/cells/. '
              f'Run section 16 of chrisMain to generate the archive first.')
        return None

    csv_path = visual_qc_csv_path(exp_name, protocol, output_root)
    tags = _load_tags(csv_path)

    types = sorted({c['cell_type'] for c in cells})
    type_dropdown = widgets.Dropdown(
        options=['all'] + types, value='all', description='Type:',
        layout=widgets.Layout(width='240px'))
    only_untagged = widgets.Checkbox(
        value=False, description='Only untagged', indent=False)

    state = {'visible': list(range(len(cells))), 'i': 0}

    def compute_visible():
        idxs = list(range(len(cells)))
        if type_dropdown.value != 'all':
            idxs = [i for i in idxs
                    if cells[i]['cell_type'] == type_dropdown.value]
        if only_untagged.value:
            idxs = [i for i in idxs if cells[i]['cell_id'] not in tags]
        return idxs

    raster_w = widgets.Image(format='png', width=image_width)
    psth_w   = widgets.Image(format='png', width=image_width)
    info_label = widgets.HTML()
    save_path_label = widgets.HTML(
        f"<small>Tags persist to <code>{csv_path}</code></small>"
    )

    def render():
        idxs = state['visible']
        if not idxs:
            info_label.value = '<b>No cells match the current filter.</b>'
            raster_w.value = b''
            psth_w.value = b''
            return
        state['i'] = max(0, min(state['i'], len(idxs) - 1))
        cell = cells[idxs[state['i']]]
        with open(cell['raster_path'], 'rb') as f:
            raster_w.value = f.read()
        with open(cell['psth_path'], 'rb') as f:
            psth_w.value = f.read()
        existing = tags.get(cell['cell_id'])
        n_tagged = sum(1 for c in cells if c['cell_id'] in tags)
        n_good = sum(1 for t in tags.values() if t['tag'] == 'good')
        n_bad  = sum(1 for t in tags.values() if t['tag'] == 'bad')
        tag_html = ''
        if existing:
            color = {'good': 'green', 'bad': 'red'}.get(existing['tag'], 'gray')
            tag_html = (f'  <span style="color:{color}"><b>'
                        f'[{existing["tag"]}]</b></span> '
                        f'({existing["timestamp"]})')
        info_label.value = (
            f"<b>{state['i']+1} / {len(idxs)}</b> "
            f"(filter: {type_dropdown.value}"
            f"{', untagged' if only_untagged.value else ''})"
            f" &mdash; cell <b>{cell['cell_id']}</b> "
            f"({cell['cell_type']}){tag_html}"
            f" &mdash; totals: good={n_good}, bad={n_bad}, "
            f"tagged={n_tagged}/{len(cells)}"
        )

    def tag_and_advance(tag):
        idxs = state['visible']
        if not idxs:
            return
        cell = cells[idxs[state['i']]]
        _save_tag(csv_path, exp_name, cell['cell_id'], cell['cell_type'],
                  tag, inspector)
        tags[cell['cell_id']] = {
            'tag': tag,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'inspector': inspector,
        }
        if only_untagged.value:
            state['visible'] = compute_visible()
        else:
            state['i'] = min(state['i'] + 1, len(idxs) - 1)
        render()

    good_btn = widgets.Button(description='Good', button_style='success',
                              layout=widgets.Layout(width='110px'))
    bad_btn  = widgets.Button(description='Bad', button_style='danger',
                              layout=widgets.Layout(width='110px'))
    prev_btn = widgets.Button(description='◀ Prev',
                              layout=widgets.Layout(width='110px'))
    next_btn = widgets.Button(description='Next ▶',
                              layout=widgets.Layout(width='110px'))

    good_btn.on_click(lambda _: tag_and_advance('good'))
    bad_btn .on_click(lambda _: tag_and_advance('bad'))

    def go(delta):
        state['i'] += delta
        render()
    prev_btn.on_click(lambda _: go(-1))
    next_btn.on_click(lambda _: go(+1))

    def on_filter_change(_):
        state['visible'] = compute_visible()
        state['i'] = 0
        render()
    type_dropdown.observe(on_filter_change, 'value')
    only_untagged.observe(on_filter_change, 'value')

    state['visible'] = compute_visible()
    render()

    display(widgets.VBox([
        widgets.HBox([type_dropdown, only_untagged]),
        info_label,
        widgets.HBox([raster_w, psth_w]),
        widgets.HBox([prev_btn, bad_btn, good_btn, next_btn]),
        save_path_label,
    ]))


def select_good_cells(
    exp_names: Optional[List[str]] = None,
    output_root: Optional[str] = None,
    protocol: str = 'eye_movement_alt_bg',
    use_visual_qc: str = 'auto',
) -> pd.DataFrame:
    """Cells to feed into population analysis.

    Default behavior (``use_visual_qc='auto'``) is the **superset**: every
    cell that passed the automated protocol QC in section 16 (i.e. has a
    PNG pair on disk / appears in that experiment's ``index.csv``).

    If a ``visual_qc.csv`` *is* present for an experiment, it overrides
    the default for that experiment only — restricting the set to cells
    explicitly tagged ``good``. Experiments without a ``visual_qc.csv``
    keep the default (all QC-passers in).

    Parameters
    ----------
    exp_names : list[str], optional
        Restrict to these dates. Default: every subdir of ``output_root``
        that has an ``<exp>/<protocol>/index.csv``.
    output_root : str, optional
        Override ``OUTPUT_DIR``.
    protocol : str
        Subdir name. Default ``'eye_movement_alt_bg'``.
    use_visual_qc : {'auto', 'require', 'ignore'}
        - ``'auto'`` (default): apply visual QC only where it exists.
        - ``'require'``: raise if any selected experiment lacks a
          ``visual_qc.csv``.
        - ``'ignore'``: never read ``visual_qc.csv``; always return the
          full QC-passing set.

    Returns
    -------
    pandas.DataFrame
        Columns: ``exp_name, cell_id, cell_type, selection_source``,
        where ``selection_source`` is ``'visual_qc'`` (kept because a
        user tagged it good) or ``'qc_passes'`` (kept by the protocol QC
        fallback).
    """
    if use_visual_qc not in ('auto', 'require', 'ignore'):
        raise ValueError(f"use_visual_qc must be one of "
                         f"'auto', 'require', 'ignore'; got {use_visual_qc!r}")

    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    if exp_names is None:
        if not root.is_dir():
            return pd.DataFrame(columns=['exp_name', 'cell_id', 'cell_type',
                                         'selection_source'])
        exp_names = [p.name for p in sorted(root.iterdir()) if p.is_dir()]

    out_rows = []
    for exp in exp_names:
        index_path = root / exp / protocol / 'index.csv'
        if not index_path.exists():
            continue
        idx = pd.read_csv(index_path)[['cell_id', 'cell_type']].copy()
        idx['cell_id'] = idx['cell_id'].astype(int)
        idx['exp_name'] = exp

        vqc_path = root / exp / protocol / 'visual_qc.csv'
        if use_visual_qc != 'ignore' and vqc_path.exists():
            vqc = pd.read_csv(vqc_path)[['cell_id', 'tag']].copy()
            vqc['cell_id'] = vqc['cell_id'].astype(int)
            good_ids = set(vqc.loc[vqc['tag'] == 'good', 'cell_id'].astype(int))
            sub = idx[idx['cell_id'].isin(good_ids)].copy()
            sub['selection_source'] = 'visual_qc'
        elif use_visual_qc == 'require':
            raise FileNotFoundError(
                f"use_visual_qc='require' but no visual_qc.csv for "
                f"{exp} at {vqc_path}"
            )
        else:
            sub = idx.copy()
            sub['selection_source'] = 'qc_passes'

        out_rows.append(sub[['exp_name', 'cell_id', 'cell_type',
                             'selection_source']])

    if not out_rows:
        return pd.DataFrame(columns=['exp_name', 'cell_id', 'cell_type',
                                     'selection_source'])
    return pd.concat(out_rows, ignore_index=True)


def load_visual_qc(
    exp_names: Optional[List[str]] = None,
    output_root: Optional[str] = None,
    protocol: str = 'eye_movement_alt_bg',
) -> pd.DataFrame:
    """Concatenate ``visual_qc.csv`` files across experiments.

    Parameters
    ----------
    exp_names : list[str], optional
        Restrict to these dates. Default: every subdir of ``output_root``.
    output_root : str, optional
        Override ``OUTPUT_DIR``.
    protocol : str
        Subdir name. Default ``'eye_movement_alt_bg'``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``exp_name, cell_id, cell_type, tag, timestamp, inspector``.
        Empty DataFrame with the same columns when no logs are found.
    """
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    if exp_names is None:
        if not root.is_dir():
            return pd.DataFrame(columns=VQC_COLUMNS)
        exp_names = [p.name for p in sorted(root.iterdir()) if p.is_dir()]

    dfs = []
    for exp in exp_names:
        csv_path = root / exp / protocol / 'visual_qc.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if 'exp_name' not in df.columns or df['exp_name'].isna().any():
            df['exp_name'] = exp
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=VQC_COLUMNS)
    out = pd.concat(dfs, ignore_index=True)
    for col in VQC_COLUMNS:
        if col not in out.columns:
            out[col] = ''
    return out[VQC_COLUMNS]
