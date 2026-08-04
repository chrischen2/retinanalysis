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

**Volumes are tiers, and there can be any number of them.** Each section in
`src/retinanalysis/config/config.ini` is one volume, read in file order;
`settings.py` then re-sorts so local disks are read before network mounts.
Current order: ChrisNewSSD (fred) → ChrisNewSSD (chris) → ChrisProSSD → NAS
(`/Volumes/data`). `ra.DATA_DIR` and friends name only the *top mounted*
tier, so anything that needs to sweep all volumes must go through
`ra.find_path(kind, *parts)` (one file) or `ra.tier_dirs(kind)` (all roots
for a kind) rather than listing `ra.ANALYSIS_DIR` directly.

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

**MEA protocol notebooks branch off `demos/meaAnalysisMain.ipynb`.**
That notebook is the shared front half for every protocol: §1–5 choose a
dataset, §6 builds a pipeline for one, §7 (`pipeline.inspect()`) checks
it, §8 breaks out `stim_block` / `response_block` / `analysis_chunk`. It
stops before interpreting conditions on purpose. A protocol notebook —
`demos/variableMeanDriftingGrating.ipynb` is the worked example — starts
from `(EXP_NAME, DATAFILE_NAME)` constants and builds its own pipeline,
so it runs standalone, then:

1. `ra.block_parameters(stim_block, source=...)` — **values always come
   from the recorded epochs, never from the `.m`.** A declared default is
   only what a parameter would have been had nobody touched it, and the rig
   routinely overrides them (this protocol declares 4 Hz / 800 µm and the
   analyzed block ran 2 Hz / 2000). Which parameters are condition axes is
   likewise read off the epochs — the ones that take more than one value.
   `ra.parse_protocol_source(dotted_name)` is optional and supplies one
   thing the data can't: the comment saying what each parameter means. It
   also catches name mismatches via `ra.condition_keys`
   (`variableMeanDriftingGrating` writes the misspelled `currentBarWdith`).
2. `ra.suggest_epoch_range(...)` then `ra.block_qc_metrics(...,
   epoch_range=...)` — **epochs first, then cells**, so a cell isn't
   scored against a stretch of block you were going to discard.

3. Optionally, the response in place: `ra.cell_activity_in_window(...)` +
   `ra.plot_mosaic_activity(...)` draw one epoch's per-cell firing rate on
   the RF mosaic beside the raster it was counted from, over a
   reconstructed stimulus frame (§5 of `variableMeanDriftingGrating`).
4. Sorting QC on the raw trace (§6): `ra.load_raw_window(response_block,
   epoch, (t0, t1))` reads one window shared by every cell — and returns
   `None` with a printed reason instead of raising when the NAS holding the
   raw `.bin` isn't mounted — then `ra.browse_sorting_qc(raw, ...)` puts a
   dropdown over `ra.sample_cells_by_type(qc.query('passes'), ...)`. Pass
   the QC survivors as `candidate_cell_ids` or the legend fills with
   unmatched clusters.
5. Phase alignment for any drifting-grating protocol (§7):
   `ra.phase_alignment_by_condition(pipeline, stim_block, epochs_kept,
   CONDITION_KEYS, n_shuffles=...)` runs `drift_phase_response` +
   `phase_period_scan` once per condition — **never pool epochs of
   different geometry**, drift phase is measured from stimulus onset — and
   returns `(phase_by_condition, summary)`. `ra.describe_phase_alignment`
   reports which conditions beat their shuffled null and the per-type
   residual as a latency; `ra.browse_phase_alignment` draws them. Another
   protocol reuses all of it by passing its own `geometry_fn`.

Condition labels (figure titles, dropdown entries, printouts) go through
`ra.condition_label(CONDITION_KEYS, values)`, which takes a groupby tuple
or an epoch row — otherwise one notebook grows four spellings of one label.

Two traps worth knowing, both hit while building the first one:
`block_qc_metrics` needs `t_end_ms` (without it the rate gate is NaN and
every cell silently fails), and `QCThresholds` defaults assume every epoch
is the same condition — for an alternating protocol both
`min_reliability_r` and a flat `min_frac_epochs_above_rate` of 0.8 reject
healthy cells for responding to the stimulus.

## Overlaying a stimulus on the mosaic

**RF mosaic and stimulus co-register with no fitted parameter.** The STA is
measured in the stimulus's own frame: `get_rf_params` returns centers in
stixels (already y-flipped for `imshow`), and `pixels_per_stixel =
canvasSize[0]/numXChecks` scales them to canvas pixels, which is the same
unit MATLAB specifies the stimulus in (`canvasSize/2`, `um2pix`). Draw the
frame at `extent=(0, canvas_w, canvas_h, 0)` and it lines up. **Only the
electrode overlay needs `rig_calibration`** — that one maps physical chip µm
onto the canvas and has genuine unknowns.

When porting a `createPresentation` to Python:

- **Distrust the `.m` property comments; read the code beside them.**
  `variableMeanDriftingGrating` labels `barWidths` and `apertureDiameter`
  `(pix)`, but both go through `um2pix` — they are microns, and the aperture
  is a diameter, not the radius its comment claims.
- **`um2pix` rounds** (`round(um / micronsPerPixel)`). A 50 µm bar at
  3.8 µm/px is 13 px, not 13.16, and the spatial frequency follows the
  rounded value.
- **`p.setBackgroundColor(0)` means outside the aperture is black, not
  mean.** The mean-colored rectangle carrying the circular mask is only as
  big as the aperture, so cells beyond it saw darkness — their rate is not a
  response to the stimulus.
- Stage's `Grating` takes `color = 2*mean`, giving luminance
  `mean*(1 + contrast*sin(...))`. The protocol's `phaseShift` exists to put a
  zero crossing at the quad center, so write the frame as a sine measured
  **from the canvas center** rather than reproducing that arithmetic against
  Stage's own (unavailable) texture-coordinate convention.

`ra.grating_frame` / `ra.grating_geometry`
(`regen/variable_mean_drifting_grating.py`) are the worked example.

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

- `ra.populate_database()` — ingest from **every mounted tier**, not just
  the top one: since 2026-07-30 it sweeps the `(h5, meta, tags)` triple of
  each configured volume in read-priority order (local SSDs before NAS)
  and the first drive holding a given date wins, so duplicate copies never
  re-trigger an ingest. `ra.ingest_source_dirs()` previews the triples.
  Pass an explicit `h5_dir`/`meta_dir`/`tags_dir` to restrict to one tree.
  Mostly append-only, but **not purely**: since 2026-07-27
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
