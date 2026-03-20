from configparser import ConfigParser
import platform
import os
# import importlib.resources as ir
try:
    import importlib.resources as ir
except:
    import importlib_resources as ir # type: ignore

import retinanalysis


def load_config(config_path):
    if os.path.exists(config_path):
        config_path = os.path.abspath(config_path)
        configfile = ConfigParser()
        configfile.read(config_path)
    else:
        print()
        raise FileNotFoundError(f"No config file found at {config_path}.\nUse reset_config() and create_config() to make one.")

    if platform.system() == 'Darwin':
        DEFAULT_config = configfile['DEFAULT']
        SECONDARY_config = configfile['SECONDARY']
        TERTIARY_config = configfile['TERTIARY']
    elif platform.system() == 'Linux':
        DEFAULT_config = configfile['LINUX_DEFAULT']
        SECONDARY_config = configfile['LINUX_SECONDARY']
    else:
        DEFAULT_config = configfile['WINDOWS_DEFAULT']
        SECONDARY_config = configfile['WINDOWS_SECONDARY']
    
    mea_config = dict()
    if os.path.exists(os.path.abspath(DEFAULT_config['data'])):
        use_config = DEFAULT_config
        mea_config['primary'] = {'data' : use_config['data'], 'raw': use_config['raw'], 'analysis': use_config['analysis'],
                    'h5': use_config['h5'], 'meta': use_config['meta'], 'tags': use_config['tags'],
                    'query': use_config['query'], 'user': use_config['user']}
    if os.path.exists(os.path.abspath(SECONDARY_config['data'])):
        use_config = SECONDARY_config
        mea_config['secondary'] = {'data' : use_config['data'], 'raw': use_config['raw'], 'analysis': use_config['analysis'],
                    'h5': use_config['h5'], 'meta': use_config['meta'], 'tags': use_config['tags'],
                    'query': use_config['query'], 'user': use_config['user']}

    if os.path.exists(os.path.abspath(TERTIARY_config['data'])):
        use_config = TERTIARY_config
        mea_config['tertiary'] = {'data' : use_config['data'], 'raw': use_config['raw'], 'analysis': use_config['analysis'],
                    'h5': use_config['h5'], 'meta': use_config['meta'], 'tags': use_config['tags'],
                    'query': use_config['query'], 'user': use_config['user']}
        
    if not mea_config:
        use_config = {'data': '', 'raw': '', 'analysis': '', 'h5': '', 'meta': '', 'tags': '', 'query': '', 'user': ''}
        mea_config['primary'] = {'data' : use_config['data'], 'raw': use_config['raw'], 'analysis': use_config['analysis'],
                    'h5': use_config['h5'], 'meta': use_config['meta'], 'tags': use_config['tags'],
                    'query': use_config['query'], 'user': use_config['user']}
        print("No NAS or SSD paths found, check that one of them is connected")

    return mea_config


def create_config(config_path, config_name,
                      data_dir, analysis_dir, 
                      h5_dir, meta_dir, tags_dir, username = 'drezeanu'):

    config = ConfigParser()
    # TODO update
    config[config_name] = {'Analysis': analysis_dir,
                         'Data': data_dir,
                         'H5': h5_dir,
                         'Meta': meta_dir,
                         'Tags': tags_dir,
                         'User': username}

    if os.path.exists(config_path):
        with open(config_path, "a") as configfile:
            config.write(configfile)

    else:
        with open(config_path, "w") as configfile:
            config.write(configfile)

def reset_config(config_path):
    os.remove(config_path)
    print("Config file successfully deleted")


config_path = ir.files(retinanalysis) / os.path.join("config", "config.ini")
mea_config = load_config(config_path)

if 'primary' in mea_config:
    cfg = mea_config['primary']
elif 'secondary' in mea_config:
    cfg = mea_config['secondary']
elif 'tertiary' in mea_config:
    cfg = mea_config['tertiary']
else:
    raise RuntimeError("No valid config paths found.")
DATA_DIR = cfg['data']
RAW_DIR = cfg['raw']
ANALYSIS_DIR = cfg['analysis']
H5_DIR = cfg['h5']
META_DIR = cfg['meta']
TAGS_DIR = cfg['tags']
QUERY_DIR = cfg['query']
USER = cfg['user']