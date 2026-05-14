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
  (sections 1–17, including the per-experiment batch archive in §16
  and the visual-QC GUI in §17).
- `demos/variableMeanSpatialNoise.ipynb` — `VariableMeanSpatialNoise`.

When the user asks to add analysis for a new protocol, default to a
new notebook rather than extending an existing one.

## Per-experiment batch archive (§16 of chrisMain)

- Driver: `ra.analyze_experiments(dates, protocol_search=...)`. Runs
  end-to-end (build pipeline → calibration → QC → mosaic → per-cell
  rasters/PSTHs) per date. `on_error='log'` keeps the batch going past
  individual failures.
- **User controls saving via an explicit `SAVE_FIGURES` boolean at the
  top of the cell** (`True` = `overwrite=True`, `False` = skip). The
  user prefers this over auto-detect "does the PNG exist?" logic —
  don't reintroduce implicit skipping.

## Standard workflow (chrisMain §14 → §18)

Per-experiment archive + visual review is intentionally **iterative**.
The notebook lays out the conceptual order (visual QC sits before
archives because it's a *filter*), but the first-time execution order
is:

1. **§14** — compute protocol QC → writes `qc.csv`.
2. **§17** (single date) or **§18** (batch) — render every QC-passing
   cell → writes `mosaic.png`, `index.csv`, `cell_match.csv`, and
   `cells/<type>/cell_<id>_{raster,psth}.png`. *No `visual_qc.csv` yet.*
3. **§16** — `ra.browse_cells_qc(exp_name)` reads those PNGs and lets
   the user tag good/bad → writes `visual_qc.csv` (one row per click).
4. **§17 / §18 again** — `analyze_experiment` auto-detects
   `visual_qc.csv` (`respect_visual_qc=True` default) and restricts the
   per-cell PNG render to `tag == 'good'`. `cell_match.csv` and
   `qc.csv` stay comprehensive.

For a date that already has a saved archive, the user can jump straight
to §16 to keep tagging, then re-run §17/§18 to collapse the archive to
the curated set.

**Manual tags are never overwritten by archiving.** The only writer of
`visual_qc.csv` is `_save_tag()` inside the GUI; `analyze_experiment`,
`save_per_cell_plots`, `save_cell_match`, and `save_protocol_qc` are
all read-only with respect to that file. Documented in their
docstrings and enforced by an audit-style smoke test in
`tests/test_visual_qc_invariant.py`.

## Population-cell selection (§17 of chrisMain)

The default pool for population/statistical analysis is **all cells
that passed the automated protocol QC in §16** — equivalently, every
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

## Database write/delete verbs

DataJoint ingest and deletion live in `utils/database_utils.py`,
auto-exported on `ra.*`:

- `ra.populate_database()` — append-only ingest from `H5_DIR` /
  `META_DIR` / `TAGS_DIR`. Never deletes.
- `ra.delete_experiments(['<exp>', ...])` — drop specific experiments.
  Cascades through DataJoint to every downstream table. No confirmation
  prompt.
- `ra.reload_experiment_data('<exp>')` — drop one experiment then
  re-ingest from H5. The right verb for "refresh this date".
- `ra.purge_database(confirm='YES_DELETE_ALL')` — drop every experiment.
  **Refuses to run without the literal sentinel** to prevent accidental
  catastrophic deletion.

Keep ingest and delete as separate calls — never collapse `populate_*`
into a "refresh-then-insert" without an explicit user ask.

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
