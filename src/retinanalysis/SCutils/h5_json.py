"""Discover single-cell SSD data and update parsed Symphony JSON metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import List, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]


# A dedicated single-cell drive wins when more than one known drive is
# mounted. Layout discovery still handles renamed drives when only one fits.
DEFAULT_VOLUME_NAMES = (
    'SingleCellSSD',
    'Single Cell SSD',
    'SingleCellData',
    'Single Cell Data',
    'ChrisSSD',
    'ChrisNewSSD',
    'ChrisProSSD',
)


@dataclass(frozen=True)
class SingleCellJsonUpdate:
    """Summary returned by :func:`update_single_cell_json`."""

    volume: Path
    h5_dir: Path
    json_dir: Path
    pending: Tuple[Path, ...]
    created: Tuple[Path, ...]
    failed: Tuple[Tuple[Path, str], ...]
    dry_run: bool

    @property
    def ok(self) -> bool:
        """Whether every attempted conversion succeeded."""
        return not self.failed


def _normalized_name(value: str) -> str:
    return ''.join(character.lower() for character in value
                   if character.isalnum())


def _candidate_layouts(volume: Path, data_owner: str):
    """Known ``(h5, json)`` layouts, most specific first."""
    return (
        (volume / 'single_cell' / data_owner / 'h5',
         volume / 'single_cell' / data_owner / 'json'),
        (volume / data_owner / 'h5', volume / data_owner / 'json'),
        (volume / 'h5', volume / 'json'),
        (volume / 'data' / 'h5', volume / 'data' / 'metadata' / 'json'),
    )


def _locate_data_folders(
    volumes_root: Path,
    volume: Optional[PathLike],
    volume_names: Sequence[str],
    data_owner: str,
) -> Tuple[Path, Path, Path]:
    if volume is not None:
        explicit = Path(volume).expanduser()
        volumes = [explicit]
    else:
        if not volumes_root.is_dir():
            raise FileNotFoundError(
                f'Volume root is not mounted: {volumes_root}')
        volumes = [path for path in volumes_root.iterdir()
                   if not path.name.startswith('.')]

    matches = []
    for mounted in volumes:
        for h5_dir, json_dir in _candidate_layouts(mounted, data_owner):
            if h5_dir.is_dir():
                matches.append((mounted, h5_dir, json_dir))
                break

    if not matches:
        target = f' on {Path(volume).expanduser()}' if volume is not None else ''
        raise FileNotFoundError(
            f'No single-cell H5 folder found{target}. Expected one of: '
            f'single_cell/{data_owner}/h5, {data_owner}/h5, h5, or data/h5.')
    if len(matches) == 1:
        return matches[0]

    preferred = {_normalized_name(name): rank
                 for rank, name in enumerate(volume_names)}
    ranked = sorted(
        (preferred.get(_normalized_name(item[0].name), len(preferred)), item)
        for item in matches)
    best_rank = ranked[0][0]
    best = [item for rank, item in ranked if rank == best_rank]
    if len(best) == 1:
        return best[0]

    choices = ', '.join(str(item[0]) for item in best)
    raise RuntimeError(
        f'Multiple single-cell data drives are mounted: {choices}. '
        'Pass volume="/Volumes/<name>" to select one explicitly.')


def update_single_cell_json(
    *,
    volumes_root: PathLike = '/Volumes',
    volume: Optional[PathLike] = None,
    volume_names: Sequence[str] = DEFAULT_VOLUME_NAMES,
    data_owner: str = 'chris_data',
    stage_type: str = 'LightCrafter',
    save_h5_path: bool = False,
    dry_run: bool = False,
    continue_on_error: bool = True,
    verbose: bool = True,
) -> SingleCellJsonUpdate:
    """Parse H5 files that do not yet have matching JSON metadata.

    The connected drive is discovered automatically under ``/Volumes``. Both
    named drives (for example ``SingleCellSSD`` or ``ChrisSSD``) and the
    expected single-cell directory layout are recognized, so a harmless drive
    rename does not require changing the caller's working directory.

    Existing JSON files are never overwritten. Each new file is parsed into a
    temporary directory beside the JSON folder, validated as JSON, and moved
    into place only after parsing succeeds. Set ``dry_run=True`` to report the
    missing mappings without writing anything.

    Typical use::

        from retinanalysis import SCutils
        report = SCutils.update_single_cell_json()

    Pass ``volume='/Volumes/<name>'`` only when several suitable drives are
    connected and automatic selection is ambiguous.
    """
    from retinanalysis.utils.parse_data import (Symphony2Reader,
                                                find_new_h5_files)

    root = Path(volumes_root).expanduser()
    selected, h5_dir, json_dir = _locate_data_folders(
        root, volume, volume_names, data_owner)
    pending = tuple(find_new_h5_files(h5_dir, json_dir))

    if verbose:
        print(f'Single-cell drive: {selected}')
        print(f'H5 folder: {h5_dir}')
        print(f'JSON folder: {json_dir}')
        if pending:
            print(f'New H5 files ({len(pending)}): '
                  + ', '.join(path.name for path in pending))
        else:
            print('No new H5 files; every H5 already has matching JSON.')

    if dry_run or not pending:
        return SingleCellJsonUpdate(
            selected, h5_dir, json_dir, pending, (), (), dry_run)

    json_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    failed: List[Tuple[Path, str]] = []

    for h5_path in pending:
        output_json = json_dir / f'{h5_path.stem}.json'
        if output_json.exists():
            continue
        if verbose:
            print(f'Parsing {h5_path.name} -> {output_json.name}')
        try:
            with tempfile.TemporaryDirectory(
                    prefix=f'.{h5_path.stem}-', dir=json_dir) as temp_name:
                temp_dir = Path(temp_name)
                temp_json = temp_dir / output_json.name
                with Symphony2Reader(
                    h5_path=str(h5_path),
                    out_path=str(temp_json),
                    mea_raw_data_path=None,
                    stage_type=stage_type,
                    save_h5_path=save_h5_path,
                ) as reader:
                    reader.read_write()

                if not temp_json.is_file():
                    raise RuntimeError('parse_data did not create a JSON file')
                with temp_json.open('r') as stream:
                    json.load(stream)
                temp_json.replace(output_json)

                # parse_data also creates a human-readable summary. Preserve
                # an existing one; otherwise publish the newly generated file.
                temp_summary = temp_json.with_suffix('.txt')
                output_summary = output_json.with_suffix('.txt')
                if temp_summary.is_file() and not output_summary.exists():
                    temp_summary.replace(output_summary)
            created.append(output_json)
        except Exception as error:
            failed.append((h5_path, str(error)))
            if verbose:
                print(f'Failed {h5_path.name}: {error}')
            if not continue_on_error:
                raise

    if verbose:
        print(f'JSON update complete: {len(created)} created, '
              f'{len(failed)} failed.')
    return SingleCellJsonUpdate(
        selected, h5_dir, json_dir, pending, tuple(created), tuple(failed),
        False)


__all__ = ['DEFAULT_VOLUME_NAMES', 'SingleCellJsonUpdate',
           'update_single_cell_json']
