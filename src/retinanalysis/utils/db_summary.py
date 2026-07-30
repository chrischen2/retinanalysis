"""Read-only summaries of what is in the DataJoint database.

Answers the "what have we got?" questions that come before picking a single
experiment: how many MEA vs patch recordings, which species, which protocols
dominate the corpus, and — per date — a protocol/datafile tree with the
acquisition settings for each block.

Everything here is a bulk query joined in pandas. Nothing walks blocks one at
a time, so a whole-database summary is a handful of round trips rather than
one per experiment. The per-date tree pulls its filter-wheel readings with a
single JSON extraction pushed into SQL instead of fetching every epoch's
parameter dict.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import retinanalysis.config.schema as schema


__all__ = [
    'short_protocol',
    'recording_counts',
    'species_counts',
    'mea_protocol_counts',
    'list_experiment_dates',
    'experiment_tree',
    'browse_experiment_tree',
]


def short_protocol(name) -> str:
    """Strip the package prefix off a protocol name.

    ``'edu.washington.riekelab.turner.protocols.EyeMovementTrajectory\
AlternatingBackground'`` becomes ``'EyeMovementTrajectoryAlternatingBackground'``.
    The prefix only records which lab package the protocol came from and makes
    every table unreadable.
    """
    return str(name).rsplit('.', 1)[-1]


def _experiments() -> pd.DataFrame:
    """One row per experiment: ``id``, ``exp_name``, ``is_mea``."""
    return (schema.Experiment().proj('exp_name', 'is_mea')
            .to_pandas().reset_index())


def recording_counts() -> pd.Series:
    """How many MEA vs single-cell patch experiments are in the database."""
    exp = _experiments()
    return pd.Series({
        'MEA': int((exp['is_mea'] == 1).sum()),
        'patch': int((exp['is_mea'] == 0).sum()),
        'total': int(len(exp)),
    })


def species_counts() -> pd.DataFrame:
    """Experiments per species, split into MEA and patch columns.

    Species comes from ``Animal.species``. It is recorded for most MEA dates
    but only rarely for patch experiments, so expect a large
    ``(not recorded)`` row on the patch side rather than reading a missing
    species as a missing animal.
    """
    exp = _experiments()
    animals = (schema.Animal().proj('species', 'experiment_id')
               .to_pandas().reset_index())
    merged = (animals.merge(exp, left_on='experiment_id', right_on='id',
                            suffixes=('', '_exp'))
              .assign(rig=lambda d: np.where(d['is_mea'] == 1, 'MEA', 'patch'),
                      species=lambda d: d['species'].fillna('(not recorded)')))
    return merged.pivot_table(index='species', columns='rig',
                              values='exp_name', aggfunc='nunique',
                              fill_value=0)


def mea_protocol_counts() -> pd.DataFrame:
    """Protocols run on MEA dates, ranked by how many distinct dates ran them.

    Dates is the more useful denominator than blocks: a protocol run four
    times in one day is still one day of data. Both columns are returned
    (``n_dates``, ``n_blocks``) so the difference is visible.
    """
    exp = _experiments()
    blocks = (schema.EpochBlock().proj('protocol_id', 'experiment_id')
              .to_pandas().reset_index())
    protocols = (schema.Protocol().proj(protocol_name='name')
                 .to_pandas().reset_index())

    mea = (blocks.merge(protocols, on='protocol_id')
                 .merge(exp[exp['is_mea'] == 1], left_on='experiment_id',
                        right_on='id', suffixes=('', '_exp')))
    mea['protocol'] = mea['protocol_name'].map(short_protocol)
    return (mea.groupby('protocol')
               .agg(n_dates=('exp_name', 'nunique'),
                    n_blocks=('protocol', 'size'))
               .sort_values(['n_dates', 'n_blocks'], ascending=False))


def list_experiment_dates(is_mea: Optional[bool] = True) -> List[str]:
    """Sorted experiment names. ``is_mea=None`` returns MEA and patch alike."""
    exp = _experiments()
    if is_mea is not None:
        exp = exp[exp['is_mea'] == (1 if is_mea else 0)]
    return sorted(exp['exp_name'].unique())


def _ensure_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Guarantee ``cols`` exist so a merge on an empty result still works.

    ``to_pandas()`` on a query that matched nothing returns a frame with no
    columns at all, so selecting the columns we are about to merge on raises
    a ``KeyError``. Dates with no ``SortingChunk`` rows hit this.
    """
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _filter_wheel_by_block(exp_id: int) -> pd.Series:
    """``{block_id: NDF}`` for one experiment, from each block's first epoch.

    ``NDF`` is the filter-wheel setting (see ``populate_ndf_column``, and the
    single-cell code that renames this same field to ``filter_wheel_ndf``).
    The value is extracted from the parameters JSON inside SQL, so this is one
    small query per date rather than one parameter-dict fetch per block.
    """
    epochs = (schema.Epoch & f'experiment_id={int(exp_id)}').proj(
        block_id='parent_id', ndf="parameters->>'$.NDF'")
    df = epochs.to_pandas().reset_index()
    if df.empty:
        return pd.Series(dtype=float)
    # First epoch of each block, matching populate_ndf_column's convention.
    df = df.sort_values('id').groupby('block_id', as_index=True)['ndf'].first()
    return pd.to_numeric(df, errors='coerce')


def experiment_tree(exp_names: Sequence[str] | str) -> pd.DataFrame:
    """Date -> protocol -> datafile tree of every block on the given dates.

    Returns a DataFrame indexed by ``(exp_name, protocol, datafile_name)``.
    pandas blanks repeated index labels when it renders, which is what draws
    the nesting: each date appears once, each protocol once beneath it, and
    the datafiles that ran that protocol underneath.

    Columns are ``group_label``, ``filter_wheel_ndf``, ``chunk_name`` and
    ``duration_minutes``. Rows sort by protocol rather than by clock time so
    that every run of a protocol on a date collects under one heading.

    Only MEA experiments have a datafile per block; patch dates are skipped
    with a note. Raises ``ValueError`` if no requested date yields any rows.
    """
    if isinstance(exp_names, str):
        exp_names = [exp_names]
    exp_names = list(exp_names)

    exp = _experiments()
    protocols = (schema.Protocol().proj(protocol_name='name')
                 .to_pandas().reset_index())

    frames = []
    for name in exp_names:
        rows = exp[exp['exp_name'] == name]
        if rows.empty:
            print(f'Not in database, skipping: {name}')
            continue
        row = rows.iloc[0]
        if row['is_mea'] != 1:
            print(f'Skipping {name}: patch experiment, no per-block datafile.')
            continue
        exp_id = int(row['id'])

        groups = _ensure_cols(
            (schema.EpochGroup & f'experiment_id={exp_id}').proj(
                group_id='id', group_label='label').to_pandas().reset_index(),
            ['group_id', 'group_label'])
        blocks = (schema.EpochBlock & f'experiment_id={exp_id}').proj(
            'protocol_id', 'data_dir', 'start_time', 'end_time',
            'chunk_id', group_id='parent_id',
            block_id='id').to_pandas().reset_index()
        if blocks.empty:
            print(f'Skipping {name}: no epoch blocks.')
            continue
        # A date that was never spike-sorted has no SortingChunk rows; the
        # tree still renders, with a blank chunk_name.
        chunks = _ensure_cols(
            (schema.SortingChunk & f'experiment_id={exp_id}').proj(
                'chunk_name', chunk_id='id').to_pandas().reset_index(),
            ['chunk_id', 'chunk_name'])

        df = (blocks.merge(groups[['group_id', 'group_label']],
                           on='group_id', how='left')
                    .merge(protocols[['protocol_id', 'protocol_name']],
                           on='protocol_id', how='left')
                    .merge(chunks[['chunk_id', 'chunk_name']],
                           on='chunk_id', how='left'))

        df['exp_name'] = name
        df['protocol'] = df['protocol_name'].map(short_protocol)
        df['datafile_name'] = df['data_dir'].astype(str).str.rsplit(
            '/', n=1).str[-1]
        df['filter_wheel_ndf'] = df['block_id'].map(
            _filter_wheel_by_block(exp_id))

        start = pd.to_datetime(df['start_time'], errors='coerce')
        end = pd.to_datetime(df['end_time'], errors='coerce')
        df['duration_minutes'] = ((end - start).dt.total_seconds() / 60).round(2)
        df['_start'] = start
        frames.append(df)

    if not frames:
        raise ValueError(
            f'No MEA blocks found for any of: {", ".join(exp_names)}')

    tree = pd.concat(frames, ignore_index=True)
    cols = ['group_label', 'filter_wheel_ndf', 'chunk_name', 'duration_minutes']
    return (tree.sort_values(['exp_name', 'protocol', '_start'])
                .set_index(['exp_name', 'protocol', 'datafile_name'])[cols])


def _tree_css(font_percent: int, max_height: str) -> str:
    """Scoped stylesheet for the rendered tree table.

    Everything is namespaced under ``.ra-tree`` so it cannot leak onto other
    tables in the notebook. The header is sticky (so protocol/column names
    stay put while scrolling a long date) and takes its background from the
    Jupyter theme variable, which keeps it readable in dark mode.
    """
    return f"""<style>
.ra-tree {{ font-size: {font_percent}%; max-height: {max_height}; overflow: auto; }}
.ra-tree table {{ border-collapse: collapse; width: auto; }}
.ra-tree th, .ra-tree td {{ padding: 1px 10px 1px 4px; line-height: 1.3;
                            white-space: nowrap; }}
.ra-tree thead th {{ position: sticky; top: 0; z-index: 1;
                     background: var(--jp-layout-color0, #fff); }}
.ra-tree tbody th {{ font-weight: 600; text-align: left; vertical-align: top; }}
.ra-tree td {{ text-align: right; }}
.ra-tree td:first-of-type {{ text-align: left; }}
</style>"""


def browse_experiment_tree(exp_names: Optional[Iterable[str]] = None,
                           value: Optional[str] = None,
                           is_mea: Optional[bool] = True,
                           font_percent: int = 70,
                           max_height: str = '32em'):
    """Dropdown of experiment dates; picking one renders its block tree.

    Parameters
    ----------
    exp_names : iterable of str, optional
        Dates to offer. Defaults to every MEA date in the database.
    value : str, optional
        Date selected on open. Defaults to the most recent one.
    is_mea : bool or None
        Which dates to list when ``exp_names`` is not given. ``None`` lists
        patch dates too, though those render nothing (no per-block datafile).
    font_percent : int
        Table font size as a percentage of the notebook's. The table is
        dense and mostly short strings, so the default 70 fits a whole date
        on screen without scrolling; raise it if that reads too small.
    max_height : str
        CSS height at which the table starts scrolling internally rather
        than pushing the rest of the notebook down.

    Returns the widget box, and displays it as a side effect.
    """
    import ipywidgets as widgets
    from IPython.display import display

    dates = (list(exp_names) if exp_names is not None
             else list_experiment_dates(is_mea=is_mea))
    if not dates:
        print('No experiments to browse.')
        return None
    if value is None or value not in dates:
        value = dates[-1]

    dropdown = widgets.Dropdown(
        options=dates, value=value, description='Date:',
        layout=widgets.Layout(width='260px'))
    status = widgets.HTML()
    # Use an HTML widget — not Output — as the table surface. Setting
    # `table.value = …` atomically REPLACES the content; there is no outputs
    # list to clear and so no way for successive renders to stack. Output +
    # clear_output(wait=True) intermittently duplicates depending on the
    # Jupyter front-end, which is the same reason sorting_qc.py uses HTML.
    table = widgets.HTML(value='', layout=widgets.Layout(width='100%'))

    def render(exp_name):
        status.value = f'<small>Loading {exp_name}…</small>'
        # Any failure has to land in the status line: an exception raised
        # inside an observe callback is swallowed by ipywidgets, leaving the
        # dropdown looking hung. experiment_tree also prints when it skips a
        # date, so capture stdout rather than let it leak into the cell.
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                tree = experiment_tree([exp_name])
        except Exception as e:
            status.value = (f'<b>{exp_name}</b> — could not build tree: '
                            f'{type(e).__name__}: {e}')
            table.value = ''
            return

        n_protocols = tree.index.get_level_values('protocol').nunique()
        note = buf.getvalue().strip()
        status.value = (
            f'<b>{exp_name}</b> — {len(tree)} block(s), '
            f'{n_protocols} distinct protocol(s)'
            + (f' <small>({note})</small>' if note else ''))
        table.value = (_tree_css(font_percent, max_height)
                       + '<div class="ra-tree">'
                       + tree.to_html(border=0, na_rep='—')
                       + '</div>')

    dropdown.observe(lambda change: render(change['new']), 'value')
    render(value)

    box = widgets.VBox([widgets.HBox([dropdown, status]), table])
    display(box)
    return box
