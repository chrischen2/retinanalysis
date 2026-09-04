import h5py
import sqlite3

from retinanalysis.SCutils.auisql_json import convert_auisql_to_json


def test_convert_auisql_bundle_to_analysis_json(tmp_path):
    database = tmp_path / '2022-09-09_B.auisql'
    response_h5 = tmp_path / '2022-09-09_B.auisql.h5'
    output = tmp_path / '2022-09-09_B.json'
    with sqlite3.connect(database) as connection:
        connection.executescript('''
            CREATE TABLE Z_PRIMARYKEY (
                Z_ENT INTEGER PRIMARY KEY, Z_NAME VARCHAR);
            CREATE TABLE ZKEYVALUEPAIR (
                Z_PK INTEGER PRIMARY KEY, Z_ENT INTEGER,
                ZBOOLVALUE INTEGER, ZLONGINTVALUE INTEGER, ZEPOCH INTEGER,
                ZSTIMULUS INTEGER, ZSTREAMVALUE INTEGER,
                ZDATEVALUE TIMESTAMP, ZDOUBLEVALUE FLOAT, ZKEY VARCHAR,
                ZSTRINGVALUE VARCHAR, ZCODEDVALUE BLOB);
            CREATE TABLE ZDAQCONFIGCONTAINER (
                Z_PK INTEGER PRIMARY KEY, ZDAQCONFIGDATA BLOB);
            CREATE TABLE ZCELL (
                Z_PK INTEGER PRIMARY KEY, ZSTARTDATE TIMESTAMP,
                ZLABEL VARCHAR);
            CREATE TABLE ZEPOCH (
                Z_PK INTEGER PRIMARY KEY, ZCELL INTEGER,
                ZBATHTEMPERATURE FLOAT, ZDURATION FLOAT,
                ZSTARTDATE TIMESTAMP, ZPROTOCOLID VARCHAR);
            CREATE TABLE ZIOBASE (
                Z_PK INTEGER PRIMARY KEY, ZCHANNELID INTEGER,
                ZSAMPLERATE INTEGER, ZDURATION FLOAT, ZEPOCH INTEGER,
                ZEPOCH1 INTEGER, ZDATAUUID VARCHAR, ZSTIMULUSID VARCHAR);
            CREATE TABLE ZEXPERIMENT (
                Z_PK INTEGER PRIMARY KEY, ZSTARTDATE TIMESTAMP);
        ''')
        connection.executemany(
            'INSERT INTO Z_PRIMARYKEY VALUES (?, ?)',
            [(11, 'DateKeyValuePair'), (14, 'DoubleKeyValuePair'),
             (17, 'StringKeyValuePair')])
        connection.execute('INSERT INTO ZEXPERIMENT VALUES (1, 684274390)')
        connection.execute('INSERT INTO ZCELL VALUES (1, 684274425, "Cell1")')
        connection.execute(
            'INSERT INTO ZEPOCH VALUES '
            '(1, 1, 32.1, 50, 684279397, "VariableMeanNoise")')
        connection.execute(
            'INSERT INTO ZIOBASE VALUES '
            '(1, 0, NULL, NULL, 1, NULL, "ABC-123", NULL)')
        connection.execute(
            'INSERT INTO ZIOBASE VALUES '
            '(2, 2, 10000, 50, NULL, 1, NULL, "GaussianNoise")')
        rows = [
            (1, 17, None, None, 1, None, None, None, None,
             'source:label', 'Cell1', None),
            (2, 17, None, None, 1, None, None, None, None,
             'source:type', 'RGC\\ON-parasol', None),
            (3, 17, None, None, 1, None, None, None, None,
             'source:parent:label', 'Preparation', None),
            (4, 17, None, None, 1, None, None, None, None,
             'source:parent:parent:label', 'Primate', None),
            (5, 17, None, None, 1, None, None, None, None,
             'epochGroup:label', 'Control', None),
            (6, 11, None, None, 1, None, None, 684279395, None,
             'epochBlock:startTime', None, None),
            (7, 11, None, None, 1, None, None, 684279447, None,
             'epochBlock:endTime', None, None),
            (8, 14, None, None, 1, None, None, None, 10000,
             'sampleRate', None, None),
            (9, 14, None, None, None, 2, None, None, 0.4,
             'mean', None, None),
            (10, 14, None, None, None, 2, None, None, 0.12,
             'stDev', None, None),
            (11, 17, None, None, None, 2, None, None, None,
             'units', 'V', None),
        ]
        connection.executemany(
            'INSERT INTO ZKEYVALUEPAIR VALUES '
            '(?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    with h5py.File(response_h5, 'w') as h5_file:
        h5_file.create_dataset('ABC-123', data=[1.0, 2.0])

    result = convert_auisql_to_json(database, response_h5, output)

    cell = result['animals'][0]['preparations'][0]['cells'][0]
    epoch = cell['epoch_groups'][0]['epoch_blocks'][0]['epochs'][0]
    assert cell['label'] == 'Cell1'
    assert cell['type'] == 'RGC\\ON-parasol'
    assert epoch['parameters']['mean'] == 0.4
    assert epoch['parameters']['stDev'] == 0.12
    assert cell['start_time'].startswith('09/09/2022 ')
    assert epoch['responses']['Input 0']['h5path'] == '/ABC-123'
    assert epoch['stimuli']['Output 2']['dataStored'] is False
    assert output.is_file()
    assert output.with_suffix('.txt').is_file()
