"""
ado-ml-batch-train - ${FILE_NAME}

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.data import TabularDataset

from ado_ai_utils.aml_configuration.register_datasets import register_improvements, register_feedback_with_impact, \
    register_feedback_train_test
from ado_ai_utils.aml_configuration.utils import get_workspace_from_config
from ado_ai_utils.project_datastores import get_ado_sql_datastore

WORKSPACE = get_workspace_from_config()
ADO_SQL_DATASTORE = get_ado_sql_datastore(WORKSPACE)


def test_register_improvements():
    """ Test Register Improvements """
    improvements_sql_ds = register_improvements(ADO_SQL_DATASTORE)

    assert isinstance(improvements_sql_ds, TabularDataset)


def test_register_feedback_with_impact():
    """ Test Register Feedback With Impact """
    feedback_with_impact = register_feedback_with_impact(ADO_SQL_DATASTORE)

    assert isinstance(feedback_with_impact, TabularDataset)


def test_register_feedback_train_test():
    """ Test Register Feedback Train and Test Datasets"""
    feedback_with_impact = register_feedback_with_impact(ADO_SQL_DATASTORE)
    train, test = register_feedback_train_test(WORKSPACE, feedback_with_impact)

    assert isinstance(train, TabularDataset)
    assert isinstance(test, TabularDataset)
