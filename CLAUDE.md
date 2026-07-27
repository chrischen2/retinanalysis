# retinanalysis — Claude operating context

Auto-loaded on every Claude Code session in this repo. Captures the
operational details and conventions that recur across sessions so we
don't re-derive them each time.

## Environment

- **Python env**: conda env `retinanalysis` at
  `/Users/chrischen/opt/anaconda3/envs/retinanalysis` (Python 3.11.13).
- **Activate inline**:
  `source /Users/chrischen/opt/anaconda3/etc/profile.d/conda.sh && conda activate retinanalysis`
- **Jupyter kernel name**: `retinanalysis` (kernelspec `python3`).
- The interpreter at
  `/Users/chrischen/opt/anaconda3/envs/retinanalysis/bin/python` is
  pre-allowed; running standalone scripts via that path is the
  preferred fallback when the IDE Jupyter kernel isn't connected.

## On-disk layout

External SSD `/Volumes/ChrisProSSD`:

| Path | What lives there |
|---|---|
| `data/sorted/<exp>/<datafile>/<ss_version>/` | Kilosort output per protocol datafile |
| `analysis/<exp>/<chunk>/<ss_version>/` | Vision analysis (typing files, STAs, EIs) |
| `retinanalysis_output/<exp>/<protocol>/` | Per-experiment archive (mosaic + per-cell PNGs + `visual_qc.csv`) |
| `retinanalysis_output/<exp>/<protocol>/cells/<celltype>/cell_<id>_{raster,psth}.png` | Per-cell figures |

`ra.DATA_DIR`, `ra.ANALYSIS_DIR`, `ra.OUTPUT_DIR` resolve to these.

## Demos / notebooks

Notebooks live under `demos/`. Convention: **one notebook per
protocol** so each stays focused.

- `demos/chrisMain.ipynb` — `EyeMovementTrajectoryAlternatingBackground`
  (sections 1–12: setup §1–§3, QC §4, visual-QC GUI §5, single-date
  archive §6, sorting-QC PNGs §7, interactive sorting-QC GUI §8,
  batch archive §9, offline store §10, per-date analyses §11,
  cross-date pooling §12).
- `demos/oneDNoise.ipynb` — `monitorVariableMeanNoiseEpochs` (1-D
  temporal noise around alternating mean intensities; LN-model fits
  via cascadegraph).

When the user asks to add analysis for a new protocol, default to a
new notebook rather than extending an existing one.

## Per-experiment batch archive (§9 of chrisMain)

- Driver: `ra.analyze_experiments(dates, protocol_search=...)`. Runs
  end-to-end (build pipeline → calibration → QC → mosaic → per-cell
  rasters/PSTHs) per date. `on_error='log'` keeps the batch going past
  individual failures.
- **User controls saving via an explicit `SAVE_FIGURES` boolean at the
  top of the cell** (`True` = `overwrite=True`, `False` = skip). The
  user prefers this over auto-detect "does the PNG exist?" logic —
  don't reintroduce implicit skipping.

## Standard workflow (chrisMain §4 → §9)

Per-experiment archive + visual review is intentionally **iterative**.
The notebook lays out the conceptual order (visual QC sits before
archives because it's a *filter*), but the first-time execution order
is:

1. **§4** — compute protocol QC → writes `qc.csv`.
2. **§6** (single date) or **§9** (batch) — render every QC-passing
   cell → writes `mosaic.png`, `index.csv`, `cell_match.csv`, and
   `cells/<type>/cell_<id>_{raster,psth}.png`. *No `visual_qc.csv` yet.*
3. **§5** — `ra.browse_cells_qc(exp_name)` reads those PNGs and lets
   the user tag good/bad → writes `visual_qc.csv` (one row per click).
4. **§6 / §9 again** — `analyze_experiment` auto-detects
   `visual_qc.csv` (`respect_visual_qc=True` default) and restricts the
   per-cell PNG render to `tag == 'good'`. `cell_match.csv` and
   `qc.csv` stay comprehensive.

For a date that already has a saved archive, the user can jump straight
to §5 to keep tagging, then re-run §6/§9 to collapse the archive to
the curated set.

**Manual tags are never overwritten by archiving.** The only writer of
`visual_qc.csv` is `_save_tag()` inside the GUI; `analyze_experiment`,
`save_per_cell_plots`, `save_cell_match`, and `save_protocol_qc` are
all read-only with respect to that file. Documented in their
docstrings and enforced by an audit-style smoke test in
`tests/test_visual_qc_invariant.py`.

**Re-archive prunes stale per-cell PNGs.** `save_per_cell_plots` and
the `analyze_experiment` driver default to `prune_stale=True`: any
existing `cells/<celltype>/cell_<id>_*.png` whose `cell_id` is outside
the kept set (QC pass ∩ visual-QC `good`) is deleted. So after tagging
cells `bad` in §5 and re-running §6/§9, those cells' PNGs disappear
from disk. Stray non-canonical files in the cells dir (e.g. a README)
are not touched. Set `prune_stale=False` to keep stale PNGs.

## Population-cell selection

The default pool for population/statistical analysis is **all cells
that passed the automated protocol QC in §4** — equivalently, every
cell that has a rendered PNG pair under
`<OUTPUT_DIR>/<exp>/<protocol>/cells/.../` (one row per cell in that
experiment's `index.csv`).

- `ra.select_good_cells()` — primary entry point. Returns
  `(exp_name, cell_id, cell_type, selection_source)` rows across all
  experiments. By default (`use_visual_qc='auto'`): per-experiment, use
  `visual_qc.csv` if it exists, otherwise fall back to all QC-passers.
  Use `use_visual_qc='ignore'` to skip the visual layer entirely, or
  `'require'` to enforce it.
- `ra.browse_cells_qc(exp_name)` — **optional** ipywidgets GUI: raster
  left, PSTH right, Good/Bad/Prev/Next buttons. Each click upserts a
  row in `<OUTPUT_DIR>/<exp>/<protocol>/visual_qc.csv`. Resumable.
- `ra.load_visual_qc()` — concat raw tags across experiments (raw
  records, not selection logic).

**Default flow: don't require manual review** unless the user
explicitly asks. Always wire downstream code through
`select_good_cells()` so the visual-QC step remains optional.

## Offline data store (§10 of chrisMain)

Once a date has been through automated + visual QC, the *interesting*
data for downstream analysis is small (spike times + condition table +
STA/EI summaries). `ra.load_or_build_offline(exp, protocol_search=...)`
packages that subset into a single HDF5 at
`<OUTPUT_DIR>/<exp>/<protocol>/offline.h5` so future sessions skip
DataJoint and the SSD pipeline.

- **First call**: builds pipeline → runs QC → intersects with
  `visual_qc.csv` (good cells only) → writes HDF5. ~1–2 min/date.
- **Reload**: `ra.load_offline_data(exp)` → `OfflineDataset` in <1 s.
- **Cross-date**: `ra.load_offline_many()` returns
  `{exp_name: OfflineDataset}` for every date with `offline.h5`.

Per-protocol analyses live under
`retinanalysis.protocols.eye_movement_alt_bg`:

- `analyze_offline(ds)` — per-(cell-type × condition) mean PSTHs.
- `spike_distance_analysis(ds, window_sec=5)` — Victor-Purpura within
  vs across condition (port of `spkd_with_scr.m`).
- `movie_repeat_analysis(ds, cycle_sec=15, drop_first_sec=1)` — cycle-1
  vs cycle-2 correlation, RMSE, adaptation index; `compute_vp=True`
  adds per-trial VP timing distance (slow).
- `population_time_scale_metrics(ds)` — time-resolved population
  divergence (Cohen's d, Euclidean/cosine distance, cumulative
  |Δrate|, per-bin Mann-Whitney AUC) between the two
  `currentBackgroundScale` levels per cell type.
- `aggregate_psth_across_dates(offlines)` — pool per-cell mean PSTHs.

Per-date analysis CSVs (`spike_distance.csv`, `movie_repeat.csv`) are
written next to `offline.h5`; `load_spike_distance_many()` /
`load_movie_repeat_many()` concat them across dates.

## Database write/delete verbs

DataJoint ingest and deletion live in `utils/database_utils.py`,
auto-exported on `ra.*`:

- `ra.populate_database()` — ingest from `H5_DIR` / `META_DIR` /
  `TAGS_DIR`. Mostly append-only, but **not purely**: since 2026-07-27
  it also compares each source file's mtime against the stored
  `Experiment.date_added`, and a meta/tags `.json` newer than the row
  triggers a delete-then-re-ingest of that experiment. Returns
  `{'n_ingested', 'added', 'updated', 'skipped'}`. Only the two json
  files are watched — nothing in the DB is read from the `.h5`, so a
  re-copied raw file must not force a re-ingest; `watch_data_file=True`
  opts into that, `update_if_modified=False` restores strict
  append-only.
- `ra.list_database_experiments()` — read-only freshness view: one row
  per experiment with `date_added`, newest `source_mtime` /
  `source_file`, and `is_stale`. Use it to preview what a populate
  would refresh.
- `ra.delete_experiments(['<exp>', ...])` / `ra.purge_experiments(...)`
  — same function under two names. Drop specific experiments (a bare
  string works too). Cascades through DataJoint to every downstream
  table. No confirmation prompt.
- `ra.reload_experiment_data('<exp>')` — drop one experiment then
  re-ingest from H5. The right verb for "refresh this date".
- `ra.purge_database(confirm='YES_DELETE_ALL')` — drop every experiment.
  **Refuses to run without the literal sentinel** to prevent accidental
  catastrophic deletion.

The mtime-driven refresh inside `populate_database` was an explicit
user ask (2026-07-27). Outside of that one path, keep ingest and delete
as separate calls — don't widen `populate_*` into further
"refresh-then-insert" behaviour without another explicit ask.

Demoed in `demos/single_cell_main.ipynb` §"Populate the database" →
"Which dates are out of date?" → "Purging entries". The purge calls in
that notebook are **deliberately left commented out**; don't uncomment
them.

## Conventions

- **Stick to in-place edits.** Don't create scratch notebooks /
  scripts unless explicitly asked. Use `NotebookEdit` for notebook
  cells.
- **Commits**: only when explicitly asked. Follow the existing
  short-message style ("Fix cell 7 EI panel: …", "Resolve MEA chip
  rotation from rig config", …).
- **Background long jobs.** The per-cell archive over ~18 dates takes
  10–15 min. Run via `Bash(run_in_background=true)` and report
  filesystem-level progress (counts of `mosaic.png`, PNGs in last N
  min) rather than tailing the verbose log — joblib output is buffered.
- **MATLAB RNG parity** (see user-memory): `np.random.RandomState`
  matches MATLAB `rand()`; `randn()` needs the MATLAB engine (already
  installed in the env).
- **STA coordinate convention** (see user-memory): y-flip handled in
  `get_rf_params`; Theta is negated; pixels_per_stixel formula has a
  known fragility — don't refactor without flagging.

## Permissions

The list of pre-approved Bash patterns lives in
`.claude/settings.local.json`. Anything not on that list still
prompts. The current list covers: conda-env python invocations,
`git add/commit/push`, `python` / `python3`, `awk`, `find`, `grep`,
`ls`, `head`, `tail`, `wc`, `pgrep`, `pkill -f`, `cat /tmp/*`,
`ps -A -o ...`, the conda activate one-liner, and MCP
`ide__executeCode`. Extend that file (don't re-justify) when a new
routine command starts appearing.

## Things to verify before recommending

- A `kilosort*.classification.txt` file actually exists in the chunk
  dir before quoting one — older sorts only have `kilosort2` (not 2.5),
  and macOS leaves `._*` AppleDouble siblings that must be filtered.
- A date appears in `os.listdir(ra.ANALYSIS_DIR)` before assuming it's
  analyzable. Several recent dates (`20250306C`, `20250429C`,
  `20250514C`, …) are in the protocol registry but their sort outputs
  aren't on the SSD yet.
