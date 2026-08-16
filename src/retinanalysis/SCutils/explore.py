"""Read-only helpers for browsing the single-cell (patch) side of the database.

These are the notebook-facing counterparts to ``utils.datajoint_utils``: they
query DataJoint and render the result as a compact HTML table instead of
letting pandas truncate a wide frame. Everything here is read-only — no table
in the schema is written or deleted.

Typical use from a notebook::

    from retinanalysis.SCutils import explore as sc

    sc.list_experiments()                       # every is_mea=0 experiment
    df = sc.find_blocks('ExpandingSpots')       # which dates ran a protocol
    df_exp = sc.summarize_experiment('2019-03-05_G')   # one date, as a tree

Also reachable as ``ra.sc_explore``.

The display helpers (:func:`scroll_table`, :func:`tree_table`,
:func:`compact_ids`) are generic — they take a DataFrame and are reusable for
any notebook table, single-cell or not.
"""
from __future__ import annotations

import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# DataJoint (and the DB connection) is imported inside the query functions, not
# here, so the display helpers stay usable — and importing this module stays
# cheap — without a database.

__all__ = [
    'compact_ids',
    'scroll_table',
    'tree_table',
    'list_experiments',
    'protocol_inventory',
    'find_blocks',
    'protocol_tree',
    'summarize_experiment',
    'summarize_experiments',
    'plot_epoch_block_traces',
]

# Shared table CSS. Deliberately theme-neutral: colors inherit from the
# notebook, and the sticky header falls back to white only outside JupyterLab.
_CSS = """
<style>
.ra-tbl { overflow: auto; }
.ra-tbl table { border-collapse: collapse; font-size: 12.5px;
                font-variant-numeric: tabular-nums; }
.ra-tbl th { position: sticky; top: 0; z-index: 1; text-align: left;
             font-weight: 600; padding: 4px 12px 4px 0;
             border-bottom: 1px solid rgba(128,128,128,0.6);
             background: var(--jp-layout-color0, #fff); }
.ra-tbl td { padding: 2px 12px 2px 0; vertical-align: top;
             white-space: nowrap; }
.ra-tbl tr.grp > td { border-top: 1px solid rgba(128,128,128,0.28); }
.ra-tbl td.num { text-align: right; }
.ra-tbl td.lead { font-weight: 600; }
.ra-tbl summary { cursor: pointer; margin: 2px 0; }
</style>
"""


def _display_html(html: str) -> None:
    """Show HTML in the current notebook (IPython imported lazily)."""
    from IPython.display import HTML, display
    display(HTML(html))


def compact_ids(ids: Iterable[int]) -> str:
    """Collapse ints into ranges: ``[34281, 34284, 34285, 34286]`` ->
    ``'34281, 34284-34286'``. Returns '' for an empty input."""
    ids = sorted(int(i) for i in ids)
    if not ids:
        return ''
    runs, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return ', '.join(str(a) if a == b else f'{a}-{b}' for a, b in runs)


def _cells_html(values, classes) -> str:
    return ''.join(f'<td class="{c}">{v}</td>' if c else f'<td>{v}</td>'
                   for v, c in zip(values, classes))


def _render(df: pd.DataFrame, height: int, summary: str | None,
            num_cols: Sequence[str], lead_col: str | None,
            group_starts: Sequence[bool] | None) -> str:
    """Build the HTML for one table. Shared by scroll_table / tree_table."""
    from html import escape

    cls = ['num' if c in num_cols else ('lead' if c == lead_col else '')
           for c in df.columns]
    head = ''.join(f'<th>{escape(str(c))}</th>' for c in df.columns)

    body = []
    def display_value(value):
        missing = pd.isna(value)
        # Lists/arrays are legitimate consolidated table values. ``pd.isna``
        # returns an array for them, so only treat scalar missing values as blank.
        if np.isscalar(missing) and bool(missing):
            return ''
        return escape(str(value))

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [display_value(v) for v in row]
        tr = '<tr class="grp">' if group_starts is not None and group_starts[i] else '<tr>'
        body.append(tr + _cells_html(vals, cls) + '</tr>')

    table = (f'<div class="ra-tbl" style="max-height:{height}px">'
             f'<table><thead><tr>{head}</tr></thead>'
             f'<tbody>{"".join(body)}</tbody></table></div>')
    if summary:
        table = (f'<div class="ra-tbl"><details><summary>{escape(summary)}</summary>'
                 f'{table}</details></div>')
    return _CSS + table


def scroll_table(df: pd.DataFrame, height: int = 400, summary: str | None = None,
                 num_cols: Sequence[str] = (), show: bool = True) -> str:
    """Render ``df`` in a fixed-height scrollable box with a sticky header.

    Parameters
    ----------
    height : int
        Max pixel height before the box scrolls.
    summary : str, optional
        If given, the table is collapsed behind a clickable summary line.
    num_cols : sequence of str
        Columns to right-align.
    show : bool
        Display it (default). Set False to just get the HTML string back.
    """
    html = _render(df, height, summary, num_cols, None, None)
    if show:
        _display_html(html)
    return html


def tree_table(df: pd.DataFrame, levels: Sequence[str], height: int = 420,
               num_cols: Sequence[str] = (), show: bool = True) -> str:
    """Render ``df`` as a tree: repeated values in ``levels`` are blanked out.

    ``levels`` are the hierarchy columns in order (outermost first), and must
    already be sorted the way you want them grouped. A row that starts a new
    value of the outermost level gets a separator line above it.
    """
    disp = df.copy()
    levels = [c for c in levels if c in disp.columns]
    if levels:
        same = pd.Series(True, index=disp.index)
        blanks = {}
        for col in levels:
            # Blank col only when it *and* every ancestor repeat the row above.
            same = same & disp[col].eq(disp[col].shift())
            blanks[col] = same.copy()
        for col in levels:
            disp.loc[blanks[col], col] = ''
        top = levels[0]
        group_starts = list(~blanks[top])
        group_starts[0] = False  # header already separates the first row
    else:
        group_starts = None

    html = _render(disp, height, None, num_cols,
                   levels[0] if levels else None, group_starts)
    if show:
        _display_html(html)
    return html


def _as_dict(value) -> dict:
    """Return a JSON database value as a dict, or an empty dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            pass
    return {}


def _first_text(*values, default='?') -> str:
    for value in values:
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return default


def _data_owner(*paths) -> str:
    """Infer the patch-data owner from a stored source path."""
    for path in paths:
        parts = str(path or '').replace('\\', '/').lower().split('/')
        for owner in ('chris_data', 'fred_data'):
            if owner in parts:
                return owner
    return 'other_data'


def _project_label(row) -> str:
    """Best available project label from normalized or JSON metadata."""
    props = _as_dict(row.get('properties'))
    attrs = _as_dict(row.get('attributes'))
    keys = ('projectLabel', 'project_label', 'project', 'projectName')
    candidates = [row.get('project')]
    candidates.extend(props.get(k) for k in keys)
    candidates.extend(attrs.get(k) for k in keys)
    return _first_text(*candidates)


def _cell_type_short(label) -> str:
    """Final component of a Symphony cell-type path."""
    return _first_text(label).replace('/', '\\').rstrip('\\').split('\\')[-1]


def _protocol_short(name) -> str:
    """Protocol class name without its package prefix."""
    return _first_text(name).split('.protocols.')[-1].split('.')[-1]


def _experiment_catalog() -> pd.DataFrame:
    """Full experiment metadata used by both tables and cascading menus."""
    from retinanalysis.config import schema

    columns = ['data_owner', 'species', 'exp_name', 'project', 'cell_types',
               'protocols']
    ex = (schema.Experiment() & 'is_mea=0').fetch(format='frame').reset_index()
    if ex.empty:
        return pd.DataFrame(columns=columns)
    ex = ex.rename(columns={'id': 'experiment_id'})
    sc_experiments = (schema.Experiment() & 'is_mea=0').proj(experiment_id='id')

    animals = schema.Animal() & sc_experiments
    animal_df = animals.fetch(format='frame').reset_index()
    if not animal_df.empty:
        animal_df['species_display'] = animal_df.apply(
            lambda r: _first_text(r.get('species'),
                                  _as_dict(r.get('attributes')).get('label'),
                                  r.get('label')), axis=1)
        species = (animal_df.groupby('experiment_id', sort=False)['species_display']
                   .agg(_join_unique).rename('species'))
        ex = ex.merge(species, on='experiment_id', how='left')
    else:
        ex['species'] = '?'

    cells = schema.Cell() & sc_experiments
    cell_df = cells.fetch(format='frame').reset_index()
    if not cell_df.empty:
        cell_df['type_display'] = cell_df.apply(
            lambda r: _cell_type_short(_first_text(
                r.get('type'), _as_dict(r.get('properties')).get('type'))), axis=1)
        cell_types = (cell_df.groupby('experiment_id', sort=False)['type_display']
                      .agg(_join_unique).rename('cell_types'))
        ex = ex.merge(cell_types, on='experiment_id', how='left')
    else:
        ex['cell_types'] = '?'

    blocks = schema.EpochBlock() & sc_experiments
    block_df = blocks.fetch(format='frame').reset_index()
    if not block_df.empty:
        protocol_df = schema.Protocol().fetch(format='frame').reset_index()
        block_df = block_df.merge(protocol_df[['protocol_id', 'name']], on='protocol_id')
        block_df['protocol_display'] = block_df['name'].map(_protocol_short)
        protocols = (block_df[['experiment_id', 'protocol_display']]
                     .drop_duplicates()
                     .rename(columns={'protocol_display': 'protocols'}))
        ex = ex.merge(protocols, on='experiment_id', how='left')
    else:
        ex['protocols'] = '?'

    ex['data_owner'] = ex.apply(
        lambda r: _data_owner(r.get('data_file'), r.get('meta_file')), axis=1)
    ex['project'] = ex.apply(_project_label, axis=1)
    for column in ('species', 'cell_types', 'protocols'):
        ex[column] = ex[column].fillna('?')
    order = pd.Categorical(ex['data_owner'],
                           ['chris_data', 'fred_data', 'other_data'], ordered=True)
    ex = ex.assign(_owner_order=order)
    return (ex[columns + ['_owner_order']]
            .sort_values(['_owner_order', 'species', 'exp_name', 'protocols'])
            .drop(columns='_owner_order').reset_index(drop=True))


def list_experiments(show: bool = True, height: int = 400) -> pd.DataFrame:
    """Every single-cell date with project, short cell types and protocols.

    ``data_owner`` is inferred from the stored h5/meta path (``chris_data`` or
    ``fred_data``) only to render separate tables; owner and species are not
    shown or returned. Project first uses the normalized Experiment field and
    then its JSON metadata. Cell types and protocols use their short names,
    with one row per protocol for easier scanning.
    """
    catalog = _experiment_catalog()
    visible = ['exp_name', 'project', 'cell_types', 'protocols']
    df = catalog[visible].rename(columns={'protocols': 'protocol'}).copy()
    if show:
        print(f"{catalog['exp_name'].nunique()} single-cell experiments.")
        for owner in catalog['data_owner'].drop_duplicates():
            rows = (catalog.loc[catalog['data_owner'].eq(owner), visible]
                    .rename(columns={'protocols': 'protocol'}))
            print(f"\n{owner} ({rows['exp_name'].nunique()} experiments)")
            tree_table(rows.reset_index(drop=True),
                       levels=['exp_name', 'project', 'cell_types'], height=height)
    return df


def _species_group(value) -> str:
    """Normalize database species labels to primate, mouse, or other."""
    label = str(value or '').strip().lower()
    if any(token in label for token in ('primate', 'macaque', 'monkey', 'human')):
        return 'primate'
    if any(token in label for token in ('mouse', 'mice', 'mus musculus')):
        return 'mouse'
    return 'other'


def protocol_inventory(show: bool = True, height: int = 500) -> pd.DataFrame:
    """Short protocols and the unique dates recorded by species.

    Returns one row per protocol with ``primate_dates``, ``mouse_dates`` and
    ``total_dates``. Counts are unique experiment dates, not epoch blocks, so
    repeating a protocol many times on one date still contributes one.
    """
    catalog = _experiment_catalog().copy()
    if catalog.empty:
        return pd.DataFrame(columns=['protocol', 'primate_dates',
                                     'mouse_dates', 'total_dates'])
    catalog['species_group'] = catalog['species'].map(_species_group)
    rows = []
    for protocol, group in catalog.groupby('protocols', sort=False):
        rows.append({
            'protocol': protocol,
            'primate_dates': group.loc[group['species_group'].eq('primate'),
                                        'exp_name'].nunique(),
            'mouse_dates': group.loc[group['species_group'].eq('mouse'),
                                     'exp_name'].nunique(),
            'total_dates': group['exp_name'].nunique(),
        })
    result = (pd.DataFrame(rows)
              .sort_values(['total_dates', 'protocol'], ascending=[False, True],
                           ignore_index=True))
    if show:
        print(f"{len(result)} protocols across "
              f"{catalog['exp_name'].nunique()} experiment dates.")
        scroll_table(result, height=height,
                     num_cols=('primate_dates', 'mouse_dates', 'total_dates'))
    return result


_EMPTY_BLOCKS = pd.DataFrame(columns=[
    'exp_name', 'protocol', 'block_id', 'protocol_name', 'ndfs',
    'filter_wheel_ndf', 'ndf_fw',
])


def _format_ndf_fw(ndfs, filter_wheel_ndf, ndfs_recorded: bool = True) -> str:
    """Readable fixed-filter + actual-wheel setting for one epoch block."""
    from retinanalysis.utils.isomerization import split_stage_ndfs

    fixed, _embedded_fw = split_stage_ndfs(ndfs)
    parts = list(fixed)
    if pd.notna(filter_wheel_ndf):
        parts.append(f'FW{float(filter_wheel_ndf):g}')
    if parts:
        return ' + '.join(parts)
    return 'none' if ndfs_recorded else 'not recorded'


def _block_filter_settings(block_ids: Sequence[int]) -> pd.DataFrame:
    """First-epoch ``ndfs`` and actual FilterWheel NDF for each block."""
    from retinanalysis.config import schema

    ids = [int(value) for value in block_ids]
    rows = pd.DataFrame({'block_id': ids})
    if not ids:
        return rows.assign(ndfs='', filter_wheel_ndf=pd.Series(dtype=float),
                           ndfs_recorded=pd.Series(dtype=bool),
                           ndf_fw=pd.Series(dtype=str))
    epochs = schema.Epoch() & [{'parent_id': value} for value in ids]
    frame = epochs.fetch(format='frame').reset_index() if len(epochs) else pd.DataFrame()
    settings = {}
    if not frame.empty:
        for block_id, group in frame.sort_values('id').groupby('parent_id', sort=False):
            params = _as_dict(group.iloc[0].get('parameters'))
            has_ndfs = 'ndfs' in params
            raw_ndfs = params.get('ndfs', '')
            wheel = pd.to_numeric(params.get('NDF'), errors='coerce')
            settings[int(block_id)] = {
                'ndfs': raw_ndfs,
                'ndfs_recorded': has_ndfs,
                'filter_wheel_ndf': wheel,
                'ndf_fw': _format_ndf_fw(raw_ndfs, wheel, has_ndfs),
            }
    rows['ndfs'] = [settings.get(value, {}).get('ndfs', '') for value in ids]
    rows['ndfs_recorded'] = [settings.get(value, {}).get('ndfs_recorded', False)
                              for value in ids]
    rows['filter_wheel_ndf'] = [settings.get(value, {}).get('filter_wheel_ndf', float('nan'))
                                for value in ids]
    rows['ndf_fw'] = [settings.get(value, {}).get('ndf_fw', 'not recorded') for value in ids]
    return rows


def _block_light_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Combine raw Stage fixed filters with database FilterWheel readings.

    Stage configuration is authoritative when present because flattened epoch
    parameters can be overwritten by another device's empty ``ndfs`` setting.
    LED-only rigs have no Stage, so they retain the active stimulus ``ndfs``
    from the first epoch parameters.
    """
    settings = _block_filter_settings(df['block_id'])
    stage = _raw_stage_ndfs(df[['exp_name', 'block_id']])
    settings = settings.merge(stage, on='block_id', how='left')
    has_stage = settings['stage_ndfs'].fillna('').astype(str).str.strip().ne('')
    settings.loc[has_stage, 'ndfs'] = settings.loc[has_stage, 'stage_ndfs']
    settings.loc[has_stage, 'ndfs_recorded'] = True
    settings['ndf_fw'] = [
        _format_ndf_fw(ndfs, wheel, bool(recorded))
        for ndfs, wheel, recorded in zip(
            settings['ndfs'], settings['filter_wheel_ndf'], settings['ndfs_recorded'])
    ]
    return settings.drop(columns=['stage_ndfs', 'ndfs_recorded'])


def _raw_stage_ndfs(df: pd.DataFrame) -> pd.DataFrame:
    """Batch-read Stage ``ndfs``: one DB query and one H5 open per date."""
    import h5py
    from retinanalysis.config import schema
    from retinanalysis.SCutils.recording_mode import _stage_ndfs_from_group
    from retinanalysis.utils.datajoint_utils import get_h5_file

    wanted = df[['exp_name', 'block_id']].drop_duplicates().copy()
    wanted['block_id'] = wanted['block_id'].astype(int)
    result = wanted.assign(stage_ndfs='')
    ids = wanted['block_id'].tolist()
    if not ids:
        return result[['block_id', 'stage_ndfs']]
    epochs = (schema.Epoch() & [{'parent_id': value} for value in ids]).proj(
        block_id='parent_id', epoch_id='id')
    responses = epochs * schema.Response.proj(
        ..., epoch_id='parent_id', response_id='id')
    paths = responses.fetch(format='frame').reset_index() if len(responses) else pd.DataFrame()
    if paths.empty or 'h5path' not in paths:
        return result[['block_id', 'stage_ndfs']]
    paths = (paths.sort_values(['block_id', 'epoch_id', 'response_id'])
             .drop_duplicates('block_id', keep='first'))
    paths['epoch_path'] = paths['h5path'].astype(str).str.split('/responses/').str[0]
    result = result.merge(paths[['block_id', 'epoch_path']], on='block_id', how='left')
    values = {}
    for exp_name, group in result.groupby('exp_name', sort=False):
        try:
            with h5py.File(get_h5_file(str(exp_name)), 'r') as h5:
                for row in group.itertuples():
                    node = h5.get(row.epoch_path) if pd.notna(row.epoch_path) else None
                    values[int(row.block_id)] = (
                        _stage_ndfs_from_group(node) if node is not None else '')
        except Exception:
            continue
    result['stage_ndfs'] = result['block_id'].map(values).fillna('')
    return result[['block_id', 'stage_ndfs']]


def find_blocks(protocol_search: str, show: bool = True,
                show_blocks: bool = True, height: int = 400) -> pd.DataFrame:
    """Single-cell epoch blocks whose protocol name contains ``protocol_search``.

    Substring match, case-insensitive (SQL LIKE). Returns one row per block
    (``exp_name``, ``protocol``, ``block_id``, ``protocol_name``), plus fixed
    ``ndfs``, the actual numeric filter-wheel reading, and a combined
    ``ndf_fw`` label. Embedded ``FWx`` tokens in ``ndfs`` are not treated as
    wheel measurements. The display
    is one row per date — a loose search can hit thousands of blocks — with the
    per-block list collapsed behind a summary line.

    This replaces the removed ``get_datasets_from_protocol_names_sc``; MEA
    discovery still lives in ``ra.get_datasets_from_protocol_names``.
    """
    from retinanalysis.config import schema

    # An empty DataJoint query yields a (0, 0) frame with no columns, so guard
    # on len() before selecting columns.
    prot = schema.Protocol() & f'name LIKE "%{protocol_search}%"'
    if len(prot) == 0:
        print(f'No protocols match {protocol_search!r}.')
        return _EMPTY_BLOCKS.copy()
    df_prot = prot.to_pandas().reset_index()[['protocol_id', 'name']]

    blocks = schema.EpochBlock() & [f'protocol_id={p}' for p in df_prot['protocol_id']]
    if len(blocks) == 0:
        print(f'{len(df_prot)} protocol(s) match {protocol_search!r} but no blocks ran them.')
        return _EMPTY_BLOCKS.copy()

    df_exp = (schema.Experiment() & 'is_mea=0').to_pandas().reset_index()[['id', 'exp_name']]
    df = blocks.to_pandas().reset_index()[['id', 'experiment_id', 'protocol_id']]
    df = df.rename(columns={'id': 'block_id'})

    # Inner join on single-cell experiment ids drops the MEA blocks.
    df = (df.merge(df_exp, left_on='experiment_id', right_on='id')
            .merge(df_prot, on='protocol_id'))
    df['protocol'] = df['name'].str.split('.protocols.').str[-1]
    df = (df[['exp_name', 'protocol', 'block_id', 'name']]
          .rename(columns={'name': 'protocol_name'})
          .sort_values(['exp_name', 'block_id']).reset_index(drop=True))
    df = df.merge(_block_light_filters(df), on='block_id', how='left')

    if show:
        print(f"{len(df)} blocks | {df['exp_name'].nunique()} experiments | "
              f"{df['protocol'].nunique()} protocol(s) matching {protocol_search!r}")
        per_date = (df.groupby('exp_name')
                      .agg(blocks=('block_id', 'size'),
                           protocols=('protocol', _join_unique),
                           block_ids=('block_id', compact_ids))
                      .reset_index())
        scroll_table(per_date, height=height, num_cols=('blocks',))
        if show_blocks:
            display_blocks = df[['exp_name', 'protocol', 'block_id', 'ndf_fw']].rename(
                columns={'ndf_fw': 'NDF + FW'})
            scroll_table(display_blocks, height=height,
                         summary=f'all {len(df)} blocks', num_cols=('block_id',))
    return df


def _join_unique(s) -> str:
    return ', '.join(sorted(set(s)))


def protocol_tree(df_exp: pd.DataFrame, show: bool = True,
                  height: int = 420) -> pd.DataFrame:
    """Collapse a summary into cell -> epoch group -> protocol statistics.

    ``df_exp`` is what :func:`~retinanalysis.utils.datajoint_utils.get_exp_summary`
    returns, enriched with ``epochs`` by :func:`summarize_experiment`. The
    display has one row per (cell, epoch group, protocol), with block and epoch
    counts. Block ids stay in the returned frame for raw-data selection but
    are deliberately omitted from this overview.

    Rows stay in acquisition order (first appearance, i.e. ``start_time``), so
    the tree reads top-to-bottom the way the experiment was run.
    """
    df = df_exp.copy()
    missing = {'cell_label', 'cell_type', 'group_label', 'block_id'} - set(df.columns)
    if missing:
        raise ValueError(
            f'protocol_tree needs a single-cell experiment summary; missing {sorted(missing)}. '
            'get_exp_summary() returns datafile/chunk columns instead for MEA (is_mea=1) dates.')
    if 'protocol' not in df.columns:
        df['protocol'] = df['protocol_name'].str.split('.protocols.').str[-1]

    levels = ['cell', 'epoch_group']
    df['cell'] = df['cell_label'] + '  (' + df['cell_type'].fillna('?') + ')'
    df['epoch_group'] = df['group_label'].fillna('?')

    keys = levels + ['protocol']
    aggregations = {'blocks': ('block_id', 'size')}
    if 'epochs' in df.columns:
        aggregations['epochs'] = ('epochs', 'sum')
    tree = (df.groupby(keys, sort=False, dropna=False)
              .agg(**aggregations).reset_index())
    if show:
        tree_table(tree, levels=keys, height=height,
                   num_cols=('blocks', 'epochs'))
    return tree


def _epoch_counts(block_ids: Sequence[int]) -> pd.Series:
    """Count Epoch rows for each block id, preserving zero-epoch blocks."""
    from retinanalysis.config import schema

    ids = [int(i) for i in block_ids]
    counts = pd.Series(0, index=ids, dtype=int)
    if not ids:
        return counts
    epochs = schema.Epoch() & [{'parent_id': i} for i in ids]
    if len(epochs):
        frame = epochs.fetch(format='frame').reset_index()
        found = frame.groupby('parent_id').size()
        counts.loc[found.index.astype(int)] = found.astype(int).values
    return counts


def summarize_experiment(exp_name: str, show: bool = True,
                         height: int = 420) -> pd.DataFrame:
    """One single-cell experiment as a cell -> epoch group -> protocol tree.

    Returns the full per-block summary frame (``get_exp_summary`` plus a
    short protocol name and per-block epoch count) so the interactive browser
    can select an actual block. The overview omits block ids.
    """
    from retinanalysis.utils.datajoint_utils import get_exp_summary

    df_exp = get_exp_summary(exp_name)
    df_exp['protocol'] = df_exp['protocol_name'].str.split('.protocols.').str[-1]
    counts = _epoch_counts(df_exp['block_id'].tolist())
    df_exp['epochs'] = df_exp['block_id'].map(counts).fillna(0).astype(int)
    if show:
        print(f"{exp_name} | {df_exp['cell_label'].nunique()} cells | "
              f"{len(df_exp)} blocks | {df_exp['protocol'].nunique()} protocols | "
              f"{int(df_exp['epochs'].sum())} epochs")
        protocol_tree(df_exp, show=True, height=height)
    return df_exp


def plot_epoch_block_traces(exp_name: str, block_id: int, show: bool = True):
    """Load and plot every original Amp1 trace in one patch epoch block.

    Data are read directly from the h5 and are not filtered,
    baseline-subtracted or spike-detected. Returns
    ``(figure, traces, sample_rate)``.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from retinanalysis.utils.datajoint_utils import get_epochblock_amp_data

    traces, sample_rate = get_epochblock_amp_data(
        exp_name, int(block_id), verbose=False)
    traces = [np.asarray(trace, dtype=float).squeeze() for trace in traces]
    if not traces:
        raise ValueError(f'No Amp1 traces found for {exp_name}, block {block_id}.')

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for trace in traces:
        time_s = np.arange(trace.size) / float(sample_rate)
        ax.plot(time_s, trace, lw=0.65, alpha=0.55)
    ax.set(xlabel='Time (s)', ylabel='Raw Amp1 value',
           title=(f'{exp_name} | epoch block {int(block_id)} | '
                  f'{len(traces)} original epoch traces'))
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if show:
        plt.show()
    return fig, traces, float(sample_rate)


def summarize_experiments(experiments: pd.DataFrame | None = None,
                          height: int = 420):
    """Interactive single-cell browser with cascading experiment/block menus.

    The top selectors cascade through data owner, species and experiment. The
    selected experiment is summarized as cell -> epoch group -> protocol. A
    second cascade selects cell, epoch group, protocol and epoch block; its
    button loads and plots the original h5 Amp1 traces.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:
        raise ImportError('summarize_experiments requires ipywidgets.') from exc

    catalog = _experiment_catalog()
    if experiments is None:
        experiments = catalog
    else:
        experiments = experiments.copy()
        required = {'data_owner', 'species'}
        if not required.issubset(experiments.columns):
            # A frame returned by list_experiments intentionally contains only
            # visible columns. Restore hidden selector metadata by exp_name.
            experiments = catalog[catalog['exp_name'].isin(experiments['exp_name'])]
    if experiments.empty:
        raise ValueError('No single-cell experiments are available.')

    # _experiment_catalog has one row per (experiment, protocol) for the
    # Section 3/4 inventory tables. The browser must instead have exactly one
    # row per experiment, or the same date is repeated for every protocol it
    # ran. Protocols are loaded only after a date is selected.
    experiments = (experiments[['data_owner', 'species', 'exp_name']]
                   .drop_duplicates(subset='exp_name', keep='first')
                   .sort_values(['data_owner', 'species', 'exp_name'],
                                ignore_index=True))

    owner = widgets.Dropdown(description='Data:', layout=widgets.Layout(width='260px'))
    species = widgets.Dropdown(description='Species:', layout=widgets.Layout(width='300px'))
    experiment = widgets.Dropdown(description='Experiment:', layout=widgets.Layout(width='330px'))
    cell = widgets.Dropdown(description='Cell:', layout=widgets.Layout(width='330px'))
    epoch_group = widgets.Dropdown(description='Epoch group:',
                                   layout=widgets.Layout(width='360px'))
    protocol = widgets.Dropdown(description='Protocol:', layout=widgets.Layout(width='420px'))
    block = widgets.Dropdown(description='Epoch block:', layout=widgets.Layout(width='300px'))
    load = widgets.Button(description='Load original traces', icon='line-chart',
                          button_style='primary', disabled=True)
    summary_out = widgets.Output()
    raw_out = widgets.Output()
    box = widgets.VBox([
        widgets.HBox([owner, species, experiment]), summary_out,
        widgets.HTML('<b>Raw epoch-block viewer</b>'),
        widgets.HBox([cell, epoch_group]),
        widgets.HBox([protocol, block, load]), raw_out,
    ])
    box.df_exp = None
    box.selectors = {
        'data_owner': owner, 'species': species, 'experiment': experiment,
        'cell': cell, 'epoch_group': epoch_group, 'protocol': protocol,
        'block': block,
    }

    def options(frame, column):
        return [(str(v), v) for v in frame[column].dropna().drop_duplicates()]

    def set_block_options(*_):
        df = box.df_exp
        if df is None or experiment.value is None:
            block.options = []
        else:
            q = df[(df['cell_label'] == cell.value)
                   & (df['group_label'] == epoch_group.value)
                   & (df['protocol'] == protocol.value)]
            new_options = [(f"block {int(r.block_id)} ({int(r.epochs)} epochs)",
                            int(r.block_id)) for r in q.itertuples()]
            block.unobserve(set_load_enabled, names='value')
            block.options = new_options
            if new_options and block.value is None:
                block.value = new_options[0][1]
            block.observe(set_load_enabled, names='value')
        load.disabled = block.value is None

    def set_protocol_options(*_):
        df = box.df_exp
        q = (df[(df['cell_label'] == cell.value)
                & (df['group_label'] == epoch_group.value)]
             if df is not None else pd.DataFrame())
        protocol.unobserve(set_block_options, names='value')
        protocol.options = options(q, 'protocol') if not q.empty else []
        if protocol.options and protocol.value is None:
            protocol.value = protocol.options[0][1]
        protocol.observe(set_block_options, names='value')
        set_block_options()

    def set_group_options(*_):
        df = box.df_exp
        q = df[df['cell_label'] == cell.value] if df is not None else pd.DataFrame()
        epoch_group.unobserve(set_protocol_options, names='value')
        epoch_group.options = options(q, 'group_label') if not q.empty else []
        if epoch_group.options and epoch_group.value is None:
            epoch_group.value = epoch_group.options[0][1]
        epoch_group.observe(set_protocol_options, names='value')
        set_protocol_options()

    def show_experiment(*_):
        if experiment.value is None:
            return
        with summary_out:
            clear_output(wait=True)
            try:
                box.df_exp = summarize_experiment(experiment.value, show=True,
                                                  height=height)
            except Exception as exc:
                box.df_exp = None
                print(f'Could not summarize {experiment.value}: {type(exc).__name__}: {exc}')
        df = box.df_exp
        cell.unobserve(set_group_options, names='value')
        cell.options = options(df, 'cell_label') if df is not None else []
        if cell.options and cell.value is None:
            cell.value = cell.options[0][1]
        cell.observe(set_group_options, names='value')
        set_group_options()
        with raw_out:
            clear_output(wait=True)

    def set_experiment_options(*_):
        q = experiments[(experiments['data_owner'] == owner.value)
                        & (experiments['species'] == species.value)]
        experiment.unobserve(show_experiment, names='value')
        experiment.options = options(q, 'exp_name')
        if experiment.options and experiment.value is None:
            experiment.value = experiment.options[0][1]
        experiment.observe(show_experiment, names='value')
        show_experiment()

    def set_species_options(*_):
        q = experiments[experiments['data_owner'] == owner.value]
        species.unobserve(set_experiment_options, names='value')
        species.options = options(q, 'species')
        if species.options and species.value is None:
            species.value = species.options[0][1]
        species.observe(set_experiment_options, names='value')
        set_experiment_options()

    def set_load_enabled(change):
        load.disabled = change['new'] is None

    def load_raw(_):
        with raw_out:
            clear_output(wait=True)
            print(f'Loading original traces for {experiment.value}, block {block.value} ...')
            try:
                clear_output(wait=True)
                plot_epoch_block_traces(experiment.value, block.value, show=True)
            except Exception as exc:
                clear_output(wait=True)
                print(f'Could not load raw traces: {type(exc).__name__}: {exc}')

    owner.observe(set_species_options, names='value')
    species.observe(set_experiment_options, names='value')
    experiment.observe(show_experiment, names='value')
    cell.observe(set_group_options, names='value')
    epoch_group.observe(set_protocol_options, names='value')
    protocol.observe(set_block_options, names='value')
    block.observe(set_load_enabled, names='value')
    load.on_click(load_raw)

    owner.options = options(experiments, 'data_owner')
    # Some ipywidgets versions leave value=None when options are assigned
    # after observers are registered. Initialize the cascade explicitly so
    # the experiment dates are visible immediately and deterministically.
    if owner.options:
        owner.value = owner.options[0][1]
        set_species_options()
    display(box)
    return box


# --------------------------------------------------------------------------
# per-cell inspection (shared by the protocol modules)
# --------------------------------------------------------------------------

# Columns that describe a recording condition, in the order they should be
# shown. Protocol modules use different subsets, so only those present are used.
CONDITION_COLUMNS = ('onlineAnalysis', 'grating_site', 'site', 'temporalFrequency',
                     'protocols', 'bar_widths', 'aperture', 'annulus_inner',
                     'annulus_outer', 'light_setting', 'filter_wheel_ndf', 'NDF',
                     'backgroundIntensity', 'weber')


def cell_id(exp_name: str, cell_label: str) -> str:
    """The identifier used to pick a cell: ``'2026-04-23_E/Cell5'``."""
    return f'{exp_name}/{cell_label}'


def add_cell_id(groups: 'pd.DataFrame') -> 'pd.DataFrame':
    """Return a copy of a group table with a ``cell_id`` column."""
    out = groups.copy()
    out['cell_id'] = [cell_id(e, c) for e, c in zip(out['exp_name'], out['cell_label'])]
    return out


def _condition_columns(groups) -> list:
    return [c for c in CONDITION_COLUMNS if c in groups.columns]


def list_cells(groups: 'pd.DataFrame', show: bool = True, height: int = 400) -> 'pd.DataFrame':
    """One row per cell: how many recording conditions it has, and of what kind.

    Use this to find the ``cell_id`` to pass to :func:`inspect_cell`.
    """
    import pandas as pd

    g = add_cell_id(groups)
    joined = lambda s: ', '.join(sorted({str(v) for v in s}))
    agg = {'cell_type': ('cell_type_short', 'first'),
           'conditions': ('cell_id', 'size'),
           'epochs': ('epochs', 'sum')}
    for col in _condition_columns(g):
        agg[col] = (col, joined)
    out = g.groupby('cell_id', sort=False).agg(**agg).reset_index()
    out = out.sort_values(['cell_type', 'cell_id'], ignore_index=True)
    if show:
        print(f'{len(out)} cells, {len(g)} recording conditions in total')
        tree_table(out, levels=['cell_type'], height=height,
                   num_cols=('conditions', 'epochs'))
    return out


def describe_cell(groups: 'pd.DataFrame', cell: str, show: bool = True,
                  height: int = 260) -> 'pd.DataFrame':
    """What was recorded from one cell, before analyzing any of it.

    Prints the basics — cell type, how many recording conditions, how many
    blocks and epochs — and shows one row per condition. Worth a look before
    running a single recording, so you know what else that cell has and whether
    the one you picked is representative.

    ``cell`` is a ``cell_id`` (``'<experiment>/<cell label>'``); a bare cell
    label is accepted when it is unambiguous. Returns the matching rows.
    """
    g = add_cell_id(groups)
    rows = g[g['cell_id'].eq(cell)]
    if rows.empty:                      # allow a bare cell label if unambiguous
        rows = g[g['cell_label'].eq(cell)]
        if rows['cell_id'].nunique() > 1:
            raise ValueError(f'{cell!r} matches several cells: '
                             f"{sorted(rows['cell_id'].unique())} -- use the full id")
    if rows.empty:
        raise ValueError(f'no recordings for {cell!r}; '
                         f'try list_cells(groups) to see the available ids')

    cols = _condition_columns(rows)
    if show:
        n_blocks = int(rows['blocks'].sum()) if 'blocks' in rows else len(rows)
        print(f"{rows['cell_id'].iloc[0]}  |  {rows['cell_type_short'].iloc[0]}  |  "
              f"{len(rows)} recording condition(s), {n_blocks} blocks, "
              f"{int(rows['epochs'].sum())} epochs")
        varying = [c for c in cols if rows[c].astype(str).nunique() > 1]
        # Name each condition, not just how many there are.
        for n, (_, r) in enumerate(rows.iterrows()):
            label = ' | '.join(f'{c}={r[c]}' for c in (varying or cols))
            print(f'   [{n}] {label}  ({int(r.get("blocks", 1))} blocks, '
                  f'{int(r["epochs"])} epochs)')
        if not varying and len(rows) == 1:
            print('  a single condition — nothing varies across its recordings')
        scroll_table(rows[cols + ['blocks', 'epochs', 'block_ids']], height=height,
                     num_cols=('blocks', 'epochs'))
    return rows


def inspect_cell(groups: 'pd.DataFrame', cell: str, analyze, plot=None,
                 show: bool = True, height: int = 260, on_error: str = 'log',
                 **kwargs) -> list:
    """Analyze every recording of one cell, split by condition.

    Shows :func:`describe_cell` first, then analyzes each condition. ``analyze``
    and ``plot`` are the protocol module's ``analyze_group`` / ``plot_group`` —
    the protocol modules wrap this so you call
    ``<module>.inspect_cell(cell, groups)``.

    Returns the analyzed records in the order shown.
    """
    rows = describe_cell(groups, cell, show=show, height=height)
    cols = _condition_columns(rows)

    records = []
    for _, row in rows.iterrows():
        label = ' | '.join(f'{c}={row[c]}' for c in cols)
        try:
            rec = analyze(row['exp_name'], [int(b) for b in str(row['block_ids']).split(',')],
                          online_analysis=row['onlineAnalysis'], **kwargs)
            records.append(rec)
            if plot is not None:
                fig = plot(rec)
                if fig is not None and hasattr(fig, 'suptitle') and show:
                    pass  # the per-group plot already titles itself
        except Exception as e:
            if on_error != 'log':
                raise
            print(f'  FAILED {label}: {type(e).__name__}: {e}')
    if show:
        print(f'analyzed {len(records)}/{len(rows)} conditions for {cell}')
    return records
