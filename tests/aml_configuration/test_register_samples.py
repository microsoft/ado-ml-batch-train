"""
ado-ml-batch-train - test_register_samples.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from ado_ai_utils.aml_configuration.register_datastores import register_blob_datastore
from ado_ai_utils.aml_configuration.register_samples import register_ai_feedback_sample, register_improvements_sample, \
    register_feedback_sample
from tests.utils import init_test_vars

CONFIG, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, WORKSPACE_REGION, WORKSPACE, SQL_DATASTORE_NAME, \
BLOB_DATASTORE_NAME = init_test_vars()

DATASTORE_RG = CONFIG['datastore_rg']
CONTAINER_NAME = CONFIG['container_name']  # Name of Azure blob container
ACCOUNT_NAME = CONFIG['account_name']  # Storage account name
ACCOUNT_KEY = CONFIG['account_key']  # Storage account key

BLOB_DATASTORE = register_blob_datastore(WORKSPACE, BLOB_DATASTORE_NAME, CONTAINER_NAME, ACCOUNT_NAME, ACCOUNT_KEY,
                                         DATASTORE_RG)


def test_register_improvements_sample():
    """ Test Register Improvements Sample """
    dataset = register_improvements_sample(BLOB_DATASTORE, WORKSPACE)
    assert dataset


def test_register_feedback_sample():
    """ Test Register Feedback Sample """
    dataset = register_feedback_sample(BLOB_DATASTORE, WORKSPACE)
    assert dataset


def test_register_ai_feedback_sample():
    """ Test Register AI Feedback Sample"""
    dataset = register_ai_feedback_sample(BLOB_DATASTORE, WORKSPACE)
    assert dataset
