"""
ado-ml-batch-train - test_project_datasets.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.data import TabularDataset

from ado_ai_utils.aml_configuration.utils import get_workspace_from_config
from ado_ai_utils.project_datasets import get_ai_feedback_sample

WORKSPACE = get_workspace_from_config()


def test_get_ai_feedback_sample():
    """ Test Get AI Feedback Sample Dataset"""
    dataset_name = get_ai_feedback_sample(WORKSPACE)
    assert type(dataset_name) is TabularDataset
