"""
ado-ml-batch-train - project_datasets.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.core import Dataset


def get_ai_feedback_sample(workspace):
    """
    Get Pointer to AI Feedback Sample

    :param workspace: Azure Machine Learning Workspace
    :type workspace: azureml.core.Workspace
    :return: Returns a :class:`azureml.data.TabularDataset` object, a pointer to Dataset stored in Azure Machine
    Learning Workspace
    :rtype: azureml.data.TabularDataset
    """
    dataset_name = 'ai_impact_scores'

    return Dataset.get_by_name(workspace=workspace, name=dataset_name)
