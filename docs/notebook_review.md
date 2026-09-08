# Shared notebook browsing and review

Use `retinanalysis.utils.browse.saved_figure_review_browser` to inspect saved
PNG figures and `retinanalysis.utils.review_store.ReviewStore` to persist review
choices. Neither module loads raw recordings, imports a protocol notebook, or
requires a database connection. Widget imports occur only when building a browser.

## Responsibilities

| Layer | Responsibility |
| --- | --- |
| Notebook | Select output directory, entries, and analysis controls; display the returned widget. |
| Protocol adapter | Discover saved entries, load metadata, map figure panels to conditions, and choose review identity/scope. |
| `utils.browse` | Dropdowns, lazy loading of the selected entry, PNG display, Keep/Remove/Example controls, and visible action errors. |
| `utils.review_store` | Read and atomically update a small CSV by explicit identity columns; absent boolean flags default to false. |
| Protocol analysis | Raw loading, preprocessing, fitting, model-input persistence, and scientific population selection. |

Keep identity separate from display indices. A single-cell recording can use
experiment + cell label; a condition review can additionally use recording type,
duration, or contrast. MEA reviews may need experiment + datafile + unit ID.
Choose keys that uniquely identify the intended review unit. Use a separate
output directory per protocol to avoid sharing decisions accidentally.

“Remove” removes a review decision; it never deletes a recording or figure.
Example flags can have a different scope from Keep/Remove and do not implicitly
accept or exclude a record. Opening a browser or reading flags never writes files.

## Copyable example for another notebook

This example expects `saved_cells` to contain one row per cell with `experiment`,
`cell_label`, and `response_png` columns. All examples start as false. The example
uses one response panel; adapters can supply any number of panels and conditions.

```python
from pathlib import Path
from retinanalysis.utils.browse import saved_figure_review_browser
from retinanalysis.utils.review_store import ReviewStore

review_dir = Path('./analysis_output/my_protocol/review')
keys = ('experiment', 'cell_label')
keep_store = ReviewStore(review_dir / 'kept.csv', keys=keys, columns=keys)
example_store = ReviewStore(
    review_dir / 'examples.csv', keys=keys,
    columns=(*keys, 'is_example'), boolean_columns=('is_example',))

records = saved_cells.to_dict('records')

def identity(record):
    return {key: str(record[key]) for key in keys}

def flags(record, section):
    frame = keep_store.read()
    kept = bool(keep_store.matches(frame, identity(record)).any())
    return kept, example_store.flag(identity(record), 'is_example')

browser = saved_figure_review_browser(
    [(f"{r['experiment']} | {r['cell_label']}", i)
     for i, r in enumerate(records)],
    load_item=lambda index: records[index],
    sections=lambda record: ['response'],
    panels=('Response',),
    figure_options=lambda record, panel, section: [
        ('Mean response', record['response_png'])],
    describe=lambda record, section: (
        f"{record['experiment']} | {record['cell_label']}"),
    review_flags=flags,
    set_keep=lambda record, section, keep: keep_store.update(
        identity(record), {} if keep else None),
    set_example=lambda record, value: example_store.update(
        identity(record), {'is_example': value}),
    section_description='View:')
display(browser)
```

Only the selected record is loaded by `load_item`; use it to read a selected
entry's manifest/HDF5 metadata on demand for large archives. `describe` is plain
text and is escaped by the widget. Figure choices are `(label, PNG path)` pairs;
missing paths are omitted. The UI refreshes flags from storage after each action.
`browser.review_state` exposes selectors, buttons, images, and status for testing.

## VariableMeanNoise organization and compatibility

`analyzeVariableMeanNoise.ipynb` already delegates analysis and UI construction to
`variable_mean_noise.py`. The review implementation now delegates further:

- `build_cell_review_browser` is the protocol adapter to the shared browser.
- `_review_figure_options` maps raw, static LN, temporal LN, and decoding panels
  to VariableMeanNoise's saved figure manifest and selected recording type.
- `load_high_quality_cells` retains migration of legacy cell-level pass rows to
  saved recording types. `set_cell_visual_inspection` uses `ReviewStore` for
  writing while preserving the existing `high_quality_cells.csv` schema.
- `load_example_cells` and `set_cell_example` use a second store for the existing
  `example_cells.csv`, keyed by experiment/date + case-sensitive cell label.
- Public notebook function names and the existing output directories remain
  unchanged. Keep/Remove is cell × recording type; Example is physical cell.
- Baseline adjustment, saved LNK inputs, scientific QC, MATLAB matching, and
  population statistics remain protocol-specific. Their semantics are not UI
  responsibilities and have not been moved into the shared review layer.

`utils.browse.png_browser` still serves single-image render-on-demand views;
`lazy_tabs` still serves deferred tab rendering. Use the new browser when saved
images need multiple panels and persistent review controls.

`utils.visual_qc` is a separate existing MEA archive workflow. Its `good`/`bad`
labels and downstream filtering contract differ from the single-cell pass list;
this refactor leaves that workflow intact. Future adapters can reuse the shared
browser while continuing to use the existing visual-QC persistence API.
