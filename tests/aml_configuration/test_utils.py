"""
ado-ml-batch-train - test_utils.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os

from azureml.core import Workspace

from ado_ai_utils.aml_configuration.utils import init_dotenv, load_configuration, get_or_create_workspace, \
    get_workspace_from_config


def test_init_dotenv():
    """ Test Init DovEnv Method """
    if os.path.isfile('.env'):
        os.remove(".env")
    init_dotenv()
    assert os.path.isfile('.env')

    init_dotenv()
    assert os.path.isfile('.env')


def test_load_configuration():
    """ Test Load Configuration Method"""
    cfg = load_configuration("../../sample_workspace_conf.yml")
    assert cfg

    assert cfg['subscription_id'] == '<>'
    assert cfg['resource_group'] == '<>'
    assert cfg['workspace_name'] == '<>'
    assert cfg['workspace_region'] == '<>'
    assert cfg['image_name'] == '<>'

    assert cfg['sql_server_name'] == '<>'
    assert cfg['sql_database_name'] == '<>'
    assert cfg['sql_username'] == '<>'
    assert cfg['sql_password'] == '<>'

    assert cfg['datastore_rg'] == '<>'
    assert cfg['container_name'] == '<>'
    assert cfg['account_name'] == '<>'
    assert cfg['account_key'] == '<>'


def test_get_or_create_workspace():
    """Test Get or Create Workspace Method"""
    cfg = load_configuration("../../workspace_conf.yml")

    get_or_create_workspace(cfg['workspace_name'], cfg['subscription_id'], cfg['resource_group'],
                            cfg['workspace_region'])

    assert os.path.isfile('../.azureml/config.json')


def test_get_workspace_from_config():
    """ Test Get Workspace From Config File"""
    cfg = load_configuration("../../workspace_conf.yml")

    get_or_create_workspace(cfg['workspace_name'], cfg['subscription_id'], cfg['resource_group'],
                            cfg['workspace_region'])

    workspace = get_workspace_from_config()
    assert type(workspace) is Workspace
