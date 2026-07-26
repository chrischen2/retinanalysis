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

import pandas as pd

# DataJoint (and the DB connection) is imported inside the query functions, not
# here, so the display helpers stay usable — and importing this module stays
# cheap — without a database.

__all__ = [
    'compact_ids',
    'scroll_table',
    'tree_table',
    'list_experiments',
    'find_blocks',
    'protocol_tree',
    'summarize_experiment',
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
    for i, (_, row) in enumerate(df.iterrows()):
        vals = ['' if pd.isna(v) else escape(str(v)) for v in row]
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


def list_experiments(show: bool = True, height: int = 400) -> pd.DataFrame:
    """Every single-cell (``is_mea=0``) experiment, with species and rig type.

    experimenter / project / rig are not populated for patch data, so they are
    left out. ``species`` comes from the animal JSON (``attributes.label``);
    ``rig_type`` is the populated rig info ('PATCH').
    """
    from retinanalysis.config import schema

    df = (schema.Experiment() & 'is_mea=0').to_pandas()
    df['species'] = df['attributes'].apply(
        lambda d: (d if isinstance(d, dict) else json.loads(d)).get('label'))
    df = (df[['exp_name', 'species', 'rig_type']]
          .sort_values('exp_name').reset_index(drop=True))
    if show:
        print(f'{len(df)} single-cell experiments.')
        scroll_table(df, height=height)
    return df


_EMPTY_BLOCKS = pd.DataFrame(columns=['exp_name', 'protocol', 'block_id', 'protocol_name'])


def find_blocks(protocol_search: str, show: bool = True,
                show_blocks: bool = True, height: int = 400) -> pd.DataFrame:
    """Single-cell epoch blocks whose protocol name contains ``protocol_search``.

    Substring match, case-insensitive (SQL LIKE). Returns one row per block
    (``exp_name``, ``protocol``, ``block_id``, ``protocol_name``). The display
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
            scroll_table(df[['exp_name', 'protocol', 'block_id']], height=height,
                         summary=f'all {len(df)} blocks', num_cols=('block_id',))
    return df


def _join_unique(s) -> str:
    return ', '.join(sorted(set(s)))


def protocol_tree(df_exp: pd.DataFrame, show: bool = True,
                  height: int = 420) -> pd.DataFrame:
    """Collapse an experiment-summary frame into a cell -> recording -> protocol tree.

    ``df_exp`` is what :func:`~retinanalysis.utils.datajoint_utils.get_exp_summary`
    returns (one row per epoch block). The result has one row per
    (cell, recording technique, protocol) with the block count, total minutes
    and the block ids in range notation.

    Rows stay in acquisition order (first appearance, i.e. ``start_time``), so
    the tree reads top-to-bottom the way the experiment was run.
    """
    df = df_exp.copy()
    missing = {'cell_label', 'recording_technique'} - set(df.columns)
    if missing:
        raise ValueError(
            f'protocol_tree needs a single-cell experiment summary; missing {sorted(missing)}. '
            'get_exp_summary() returns datafile/chunk columns instead for MEA (is_mea=1) dates.')
    if 'protocol' not in df.columns:
        df['protocol'] = df['protocol_name'].str.split('.protocols.').str[-1]

    # group_label is 'Control' for essentially every patch experiment, so it
    # only earns a column when a date actually varies it.
    levels = ['cell', 'recording']
    df['cell'] = df['cell_label'] + '  (' + df['cell_type'].fillna('?') + ')'
    df['recording'] = df['recording_technique'].fillna('?')
    if 'pipette_solution' in df.columns and df['pipette_solution'].nunique() > 1:
        df['recording'] = df['recording'] + ', ' + df['pipette_solution'].fillna('?')
    if df['group_label'].nunique() > 1:
        df['group'] = df['group_label']
        levels.insert(1, 'group')

    keys = levels + ['protocol']
    tree = (df.groupby(keys, sort=False)
              .agg(blocks=('block_id', 'size'),
                   minutes=('duration_minutes', 'sum'),
                   block_ids=('block_id', compact_ids))
              .round({'minutes': 1}).reset_index())
    if show:
        tree_table(tree, levels=keys, height=height,
                   num_cols=('blocks', 'minutes'))
    return tree


def summarize_experiment(exp_name: str, show: bool = True,
                         height: int = 420) -> pd.DataFrame:
    """One single-cell experiment as a cell -> recording -> protocol tree.

    Returns the full per-block summary frame (``get_exp_summary`` plus a
    ``protocol`` short-name column) so downstream cells can filter it; the
    tree is only the display.
    """
    from retinanalysis.utils.datajoint_utils import get_exp_summary

    df_exp = get_exp_summary(exp_name)
    df_exp['protocol'] = df_exp['protocol_name'].str.split('.protocols.').str[-1]
    if show:
        n_min = df_exp['duration_minutes'].sum()
        print(f"{exp_name} | {df_exp['cell_label'].nunique()} cells | "
              f"{len(df_exp)} blocks | {df_exp['protocol'].nunique()} protocols | "
              f'{n_min:.0f} min')
        protocol_tree(df_exp, show=True, height=height)
    return df_exp


# --------------------------------------------------------------------------
# per-cell inspection (shared by the protocol modules)
# --------------------------------------------------------------------------

# Columns that describe a recording condition, in the order they should be
# shown. Protocol modules use different subsets, so only those present are used.
CONDITION_COLUMNS = ('onlineAnalysis', 'grating_site', 'site', 'temporalFrequency',
                     'protocols', 'bar_widths', 'light_setting', 'light_level',
                     'filter_wheel_ndf', 'NDF', 'backgroundIntensity', 'weber')


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


def inspect_cell(groups: 'pd.DataFrame', cell: str, analyze, plot=None,
                 show: bool = True, height: int = 260, on_error: str = 'log',
                 **kwargs) -> list:
    """Analyze every recording of one cell, split by condition.

    ``cell`` is a ``cell_id`` (``'<experiment>/<cell label>'``); a bare cell
    label is accepted when it is unambiguous. ``analyze`` and ``plot`` are the
    protocol module's ``analyze_group`` / ``plot_group`` — the protocol modules
    wrap this so you call ``<module>.inspect_cell(cell, groups)``.

    Returns the analyzed records in the order shown.
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
        print(f"{cell}: {len(rows)} recording condition(s), "
              f"{int(rows['epochs'].sum())} epochs, "
              f"cell type {rows['cell_type_short'].iloc[0]}")
        scroll_table(rows[cols + ['blocks', 'epochs', 'block_ids']], height=height,
                     num_cols=('blocks', 'epochs'))

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
