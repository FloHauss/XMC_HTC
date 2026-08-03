"""Provides utility for loading .yaml configs using pythonbox"""
import os

import yaml
from box import Box

def load_config(dataset_name, config_name):
    """Loads the config.yaml as a python box. Allows access to attributes with: config.attribute0"""
    path_config = f'../configs/{dataset_name}/{config_name}.yaml'

    if not os.path.exists(path_config):
        raise FileNotFoundError(f'Config file {config_name} not found under path: {path_config}.')

    with open(path_config, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
        config = Box(yaml_config)
        return config
