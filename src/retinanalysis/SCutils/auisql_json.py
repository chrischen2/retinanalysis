"""Recover retinanalysis JSON metadata from an Ovation AUISQL bundle."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import plistlib
import sqlite3
from typing import Any, Dict, Optional
import uuid
from zoneinfo import ZoneInfo

import h5py


_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
_LOCAL_ZONE = ZoneInfo('America/Los_Angeles')
_UUID_NAMESPACE = uuid.UUID('9c8bb64d-f75d-4d72-8a91-27088f9c9670')
# Legacy AUISQL exports store acquisition dates exactly two days early. This
# was verified against recordings for which both original Symphony H5 JSON and
# AUISQL representations are available.
_AUISQL_DATE_CORRECTION = timedelta(days=2)


def _synthetic_uuid(bundle_name: str, level: str, identifier: Any) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE,
                          f'{bundle_name}:{level}:{identifier}'))


def _apple_datetime(seconds: Optional[float]) -> Optional[datetime]:
    if seconds is None:
        return None
    return (_APPLE_EPOCH + timedelta(seconds=float(seconds))
            + _AUISQL_DATE_CORRECTION).astimezone(_LOCAL_ZONE)


def _time_string(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.strftime('%m/%d/%Y %H:%M:%S:%f')


def _dotnet_ticks(value: datetime) -> int:
    wall_time = value.replace(tzinfo=None)
    origin = datetime(1, 1, 1)
    delta = wall_time - origin
    return ((delta.days * 86400 + delta.seconds) * 10_000_000
            + delta.microseconds * 10)


def _attributes(identifier: str, start: Optional[datetime] = None,
                end: Optional[datetime] = None, label: Optional[str] = None):
    result: Dict[str, Any] = {'uuid': identifier}
    if label is not None:
        result['label'] = label
    offset = start.utcoffset().total_seconds() / 3600 if start else None
    if start is not None:
        result['startTimeDotNetDateTimeOffsetOffsetHours'] = offset
        result['startTimeDotNetDateTimeOffsetTicks'] = _dotnet_ticks(start)
    if end is not None:
        result['endTimeDotNetDateTimeOffsetOffsetHours'] = (
            end.utcoffset().total_seconds() / 3600)
        result['endTimeDotNetDateTimeOffsetTicks'] = _dotnet_ticks(end)
    return result


def _unarchive(blob: bytes):
    """Decode Foundation containers from an NSKeyedArchiver plist."""
    archive = plistlib.loads(blob)
    objects = archive['$objects']
    active = set()

    def resolve(value):
        if isinstance(value, plistlib.UID):
            index = value.data
            if index in active:
                return None
            active.add(index)
            try:
                return resolve(objects[index])
            finally:
                active.remove(index)
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, dict):
            return value

        class_ref = value.get('$class')
        class_name = None
        if isinstance(class_ref, plistlib.UID):
            class_obj = objects[class_ref.data]
            if isinstance(class_obj, dict):
                class_name = class_obj.get('$classname')
        if class_name in ('NSArray', 'NSMutableArray', 'NSSet',
                          'NSMutableSet'):
            return [resolve(item) for item in value.get('NS.objects', [])]
        if class_name in ('NSDictionary', 'NSMutableDictionary'):
            keys = value.get('NS.keys', [])
            vals = value.get('NS.objects', [])
            return {resolve(key): resolve(val) for key, val in zip(keys, vals)}
        if class_name in ('NSString', 'NSMutableString'):
            return resolve(value.get('NS.string'))
        return {key: resolve(child) for key, child in value.items()
                if key != '$class'}

    return resolve(archive['$top']['root'])


def _kv_value(row, entity_names):
    entity = entity_names[row['Z_ENT']]
    if entity == 'CodedKeyValuePair':
        return _unarchive(row['ZCODEDVALUE'])
    if entity == 'DateKeyValuePair':
        return _apple_datetime(row['ZDATEVALUE'])
    if entity == 'BoolKeyValuePair':
        return bool(row['ZBOOLVALUE'])
    if entity == 'DoubleKeyValuePair':
        return float(row['ZDOUBLEVALUE'])
    if entity == 'LongIntegerKeyValuePair':
        return int(row['ZLONGINTVALUE'])
    if entity == 'StringKeyValuePair':
        return row['ZSTRINGVALUE'] or None
    raise ValueError(f'Unsupported AUISQL key-value entity: {entity}')


def _split_prefixed(values, prefix):
    marker = prefix + ':'
    return {key[len(marker):]: value for key, value in values.items()
            if key.startswith(marker)}


class AuisqlReader:
    """Read an ``.auisql`` SQLite + ``.auisql.h5`` response pair."""

    def __init__(self, auisql_path, h5_path):
        self.auisql_path = Path(auisql_path)
        self.h5_path = Path(h5_path)
        self.bundle_name = self.auisql_path.name.removesuffix('.auisql')

    def _connect(self):
        uri = f'{self.auisql_path.resolve().as_uri()}?mode=ro'
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _stream_configuration(self, connection):
        row = connection.execute(
            'SELECT ZDAQCONFIGDATA FROM ZDAQCONFIGCONTAINER LIMIT 1').fetchone()
        if row is None or row[0] is None:
            return {}
        root = _unarchive(row[0])
        streams = root.get('AUIIOControllerStreamPropertiesKey', [])
        result = {}
        for stream in streams:
            identifier = stream.get('streamIdentifier', {})
            properties = stream.get('properties', {})
            key = (identifier.get('type'), identifier.get('channelNumber'))
            result[key] = properties
        return result

    def read(self):
        if not self.auisql_path.is_file():
            raise FileNotFoundError(
                f'Missing AUISQL metadata database: {self.auisql_path}')
        if not self.h5_path.is_file():
            raise FileNotFoundError(
                f'Missing AUISQL response file: {self.h5_path}')

        with self._connect() as connection:
            integrity = connection.execute('PRAGMA integrity_check').fetchone()[0]
            if integrity != 'ok':
                raise ValueError(f'AUISQL integrity check failed: {integrity}')
            entity_names = dict(connection.execute(
                'SELECT Z_ENT, Z_NAME FROM Z_PRIMARYKEY'))
            kvps_by_epoch = defaultdict(dict)
            kvps_by_stimulus = defaultdict(dict)
            for row in connection.execute('SELECT * FROM ZKEYVALUEPAIR'):
                value = _kv_value(row, entity_names)
                if row['ZEPOCH'] is not None:
                    kvps_by_epoch[row['ZEPOCH']][row['ZKEY']] = value
                elif row['ZSTIMULUS'] is not None:
                    kvps_by_stimulus[row['ZSTIMULUS']][row['ZKEY']] = value

            streams = self._stream_configuration(connection)
            cell_rows = list(connection.execute(
                'SELECT * FROM ZCELL ORDER BY ZSTARTDATE, Z_PK'))
            epoch_rows = list(connection.execute(
                'SELECT * FROM ZEPOCH ORDER BY ZSTARTDATE, Z_PK'))
            io_rows = list(connection.execute('SELECT * FROM ZIOBASE'))
            experiment_row = connection.execute(
                'SELECT * FROM ZEXPERIMENT LIMIT 1').fetchone()

        h5_keys = set()
        with h5py.File(self.h5_path, 'r') as h5_file:
            h5_keys = {key.upper() for key in h5_file.keys()}
        response_uuids = {row['ZDATAUUID'].upper() for row in io_rows
                          if row['ZDATAUUID']}
        missing = sorted(response_uuids - h5_keys)
        if missing:
            raise ValueError(
                f'AUISQL response H5 is missing {len(missing)} referenced '
                f'dataset(s), including {missing[0]}')

        response_by_epoch = defaultdict(list)
        stimulus_by_epoch = defaultdict(list)
        for row in io_rows:
            if row['ZDATAUUID']:
                response_by_epoch[row['ZEPOCH']].append(row)
            elif row['ZEPOCH1'] is not None:
                stimulus_by_epoch[row['ZEPOCH1']].append(row)

        first_values = kvps_by_epoch[epoch_rows[0]['Z_PK']] if epoch_rows else {}
        animal_properties = _split_prefixed(
            first_values, 'source:parent:parent')
        prep_properties = _split_prefixed(first_values, 'source:parent')
        prep_properties.pop('parent:label', None)
        prep_properties.pop('parent:age', None)
        prep_properties.pop('parent:darkAdaptation', None)
        prep_properties.pop('parent:description', None)
        prep_properties.pop('parent:id', None)
        prep_properties.pop('parent:sex', None)
        prep_properties.pop('parent:species', None)
        prep_properties.pop('parent:weight', None)

        animal_label = animal_properties.pop('label', None) or 'Animal'
        prep_label = prep_properties.pop('label', None) or 'Preparation'
        experiment_properties = _split_prefixed(first_values, 'experiment')
        experiment_start = _apple_datetime(
            experiment_row['ZSTARTDATE'] if experiment_row else None)
        animal_uuid = _synthetic_uuid(self.bundle_name, 'animal', 1)
        prep_uuid = _synthetic_uuid(self.bundle_name, 'preparation', 1)

        epoch_objects = {}
        block_epochs = defaultdict(list)
        block_metadata = {}
        group_metadata = {}
        for row in epoch_rows:
            epoch_pk = row['Z_PK']
            values = kvps_by_epoch[epoch_pk]
            start = _apple_datetime(row['ZSTARTDATE'])
            end = start + timedelta(seconds=float(row['ZDURATION']))
            epoch_uuid = _synthetic_uuid(self.bundle_name, 'epoch', epoch_pk)
            direct_parameters = {
                key: value for key, value in values.items()
                if ':' not in key and key != 'user:startDate'
            }
            backgrounds = defaultdict(dict)
            for key, value in values.items():
                if key.startswith('background:'):
                    _, device, property_name = key.split(':', 2)
                    backgrounds[device.replace('_', ' ')][property_name] = value
            for device, background in backgrounds.items():
                background.setdefault('sampleRate', direct_parameters.get(
                    'sampleRate'))
                background.setdefault('sampleRateUnits', 'Hz')
                background['uuid'] = _synthetic_uuid(
                    self.bundle_name, f'background:{epoch_pk}', device)

            responses = {}
            for response in response_by_epoch[epoch_pk]:
                stream = streams.get((1, response['ZCHANNELID']), {})
                device = stream.get('userDescription') or (
                    f'Input {response["ZCHANNELID"]}')
                data_uuid = response['ZDATAUUID']
                responses[device] = {
                    'sampleRate': stream.get('samplingRate')
                    or direct_parameters.get('sampleRate'),
                    'sampleRateUnits': 'Hz',
                    'uuid': data_uuid.lower(),
                    'h5path': '/' + data_uuid,
                    'inputTimeDotNetDateTimeOffsetOffsetHours': (
                        start.utcoffset().total_seconds() / 3600),
                    'inputTimeDotNetDateTimeOffsetTicks': _dotnet_ticks(start),
                }

            stimuli = {}
            for stimulus in stimulus_by_epoch[epoch_pk]:
                stream = streams.get((2, stimulus['ZCHANNELID']), {})
                device = stream.get('userDescription') or (
                    f'Output {stimulus["ZCHANNELID"]}')
                stimulus_values = kvps_by_stimulus[stimulus['Z_PK']]
                for key, value in stimulus_values.items():
                    if ':' not in key and key != 'units':
                        direct_parameters[key] = value
                stimulus_uuid = _synthetic_uuid(
                    self.bundle_name, 'stimulus', stimulus['Z_PK'])
                stimuli[device] = {
                    'durationSeconds': stimulus['ZDURATION'],
                    'sampleRate': stimulus['ZSAMPLERATE'],
                    'sampleRateUnits': 'Hz',
                    'stimulusID': stimulus['ZSTIMULUSID'],
                    'units': stimulus_values.get('units'),
                    'uuid': stimulus_uuid,
                    'h5path': '',
                    'dataStored': False,
                }

            epoch_objects[epoch_pk] = {
                'uuid': epoch_uuid,
                'protocolID': None,
                'properties': {'bathTemperature': row['ZBATHTEMPERATURE']},
                'attributes': _attributes(epoch_uuid, start, end),
                'start_time': _time_string(start),
                'end_time': _time_string(end),
                'label': None,
                'parameters': direct_parameters,
                'backgrounds': dict(backgrounds),
                'responses': responses,
                'stimuli': stimuli,
                'datetime': _time_string(start),
            }

            block_start = values.get('epochBlock:startTime') or start
            block_end = values.get('epochBlock:endTime') or end
            block_key = (row['ZCELL'], block_start.timestamp())
            block_epochs[block_key].append(epoch_objects[epoch_pk])
            block_metadata[block_key] = (
                row['ZPROTOCOLID'], block_start, block_end, direct_parameters)
            group_label = values.get('epochGroup:label') or 'Recovered'
            group_props = _split_prefixed(values, 'epochGroup')
            group_props.pop('label', None)
            group_metadata[row['ZCELL']] = (group_label, group_props)

        cells = []
        for cell_row in cell_rows:
            cell_pk = cell_row['Z_PK']
            cell_start = _apple_datetime(cell_row['ZSTARTDATE'])
            cell_epoch_values = next(
                (kvps_by_epoch[row['Z_PK']] for row in epoch_rows
                 if row['ZCELL'] == cell_pk), {})
            label = cell_row['ZLABEL'] or cell_epoch_values.get(
                'source:label') or f'Cell{cell_pk}'
            cell_type = cell_epoch_values.get('source:type')
            cell_uuid = _synthetic_uuid(self.bundle_name, 'cell', cell_pk)
            group_label, group_props = group_metadata.get(
                cell_pk, ('Recovered', {}))
            group_uuid = _synthetic_uuid(self.bundle_name, 'group', cell_pk)
            blocks = []
            for block_key in sorted(
                    (key for key in block_epochs if key[0] == cell_pk),
                    key=lambda key: key[1]):
                protocol, start, end, parameters = block_metadata[block_key]
                block_uuid = _synthetic_uuid(
                    self.bundle_name, 'block', f'{cell_pk}:{block_key[1]}')
                blocks.append({
                    'uuid': block_uuid,
                    'protocolID': protocol,
                    'label': protocol.rsplit('.', 1)[-1],
                    'properties': {},
                    'attributes': _attributes(block_uuid, start, end),
                    'start_time': _time_string(start),
                    'end_time': _time_string(end),
                    'dataFile': '',
                    'parameters': parameters,
                    'arrayPitch': None,
                    'epochs': block_epochs[block_key],
                })
            group_start = min(
                (_apple_datetime(row['ZSTARTDATE']) for row in epoch_rows
                 if row['ZCELL'] == cell_pk), default=cell_start)
            group_end = max(
                (_apple_datetime(row['ZSTARTDATE'])
                 + timedelta(seconds=float(row['ZDURATION']))
                 for row in epoch_rows if row['ZCELL'] == cell_pk),
                default=cell_start)
            epoch_group = {
                'label': group_label,
                'uuid': group_uuid,
                'notes': [],
                'properties': group_props,
                'attributes': _attributes(
                    group_uuid, group_start, group_end, group_label),
                'start_time': _time_string(group_start),
                'end_time': _time_string(group_end),
                'epoch_blocks': blocks,
            }
            cells.append({
                'label': label,
                'uuid': cell_uuid,
                'notes': [],
                'properties': {'type': cell_type},
                'attributes': _attributes(
                    cell_uuid, start=cell_start, label=label),
                'start_time': _time_string(cell_start),
                'type': cell_type,
                'epoch_groups': [epoch_group],
            })

        prep_start = min((_apple_datetime(row['ZSTARTDATE'])
                          for row in cell_rows), default=experiment_start)
        preparation = {
            'label': prep_label,
            'uuid': prep_uuid,
            'notes': [],
            'properties': prep_properties,
            'attributes': _attributes(prep_uuid, label=prep_label),
            'start_time': _time_string(prep_start),
            'bathSolution': prep_properties.get('bathSolution'),
            'preparationType': prep_properties.get('preparation'),
            'region': prep_properties.get('region'),
            'arrayPitch': None,
            'cells': cells,
        }
        animal = {
            'label': animal_label,
            'uuid': animal_uuid,
            'notes': [],
            'properties': animal_properties,
            'attributes': _attributes(animal_uuid, label=animal_label),
            'start_time': _time_string(experiment_start),
            **animal_properties,
            'preparations': [preparation],
        }
        experiment_uuid = _synthetic_uuid(self.bundle_name, 'experiment', 1)
        return {
            'label': animal_label,
            'uuid': experiment_uuid,
            'notes': [],
            'properties': animal_properties,
            'attributes': _attributes(experiment_uuid, label=animal_label),
            'start_time': _time_string(experiment_start),
            'rig_type': 'PATCH',
            **experiment_properties,
            'animals': [animal],
            'recovery_provenance': {
                'format': 'AUISQL',
                'metadata_file': self.auisql_path.name,
                'response_file': self.h5_path.name,
                'synthetic_hierarchy_uuids': True,
                'stimulus_waveforms_available': False,
                'date_correction_days': 2,
            },
        }


def _write_summary(experiment, path):
    with Path(path).open('w') as stream:
        for animal in experiment['animals']:
            stream.write(f'Animal:{animal["label"]}\n')
            for preparation in animal['preparations']:
                stream.write(f'  Preparation:{preparation["label"]}\n')
                for cell in preparation['cells']:
                    stream.write(f'    Cell:{cell["label"]}\n')
                    for group in cell['epoch_groups']:
                        stream.write(f'      Group:{group["label"]}\n')
                        protocols = dict.fromkeys(
                            block['protocolID']
                            for block in group['epoch_blocks'])
                        for protocol in protocols:
                            stream.write(f'        Protocol:{protocol}\n')


def convert_auisql_to_json(auisql_path, h5_path, out_path,
                           write_summary=True):
    """Convert one complete AUISQL bundle to canonical analysis JSON."""
    experiment = AuisqlReader(auisql_path, h5_path).read()
    output = Path(out_path)
    with output.open('w') as stream:
        json.dump(experiment, stream)
    if write_summary:
        _write_summary(experiment, output.with_suffix('.txt'))
    return experiment


__all__ = ['AuisqlReader', 'convert_auisql_to_json']
